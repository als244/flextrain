"""Qwen-VL vision tower (Qwen3.5 / Qwen3.6 / Qwen3-VL).

Port of HF ``Qwen3VLVisionModel`` for flextrain's Phase 1 multimodal
input layer. Implements :class:`~flextrain.core.layer.ModalityEncoder`
in forward-only / frozen-weights mode.

Architecture (matching HF Qwen3-VL ``model.visual``):

* ``patch_embed`` -- Conv3d with stride = kernel = ``(temporal_patch_size,
  patch_size, patch_size)``. Pre-patchified input from HF's image
  processor is reshaped to ``(N, C, T, H, W)`` before the conv.
* ``pos_embed`` -- learned ``nn.Embedding(num_position_embeddings,
  hidden_size)``. Bilinearly interpolated over each image's grid
  (``fast_pos_embed_interpolate``); added once after patch embed.
* ``rotary_pos_emb`` -- 2-axis per-token RoPE on (row, col) inside each
  image (separate from the LM-text MRoPE; this is purely a vision-tower
  internal RoPE on the patch grid).
* ``depth`` transformer blocks, each:
  - ``LayerNorm`` (bias)
  - Self-attention with **halved RoPE** (HF convention) on Q, K.
    QKV is fused: a single ``(hidden, 3*hidden)`` Linear.
  - Residual
  - ``LayerNorm`` (bias)
  - MLP: ``Linear -> gelu_pytorch_tanh -> Linear`` (both with bias).
  - Residual
* ``merger`` -- ``spatial_merge_size**2`` neighbouring patches are
  flattened then MLP'd from ``hidden * merge**2`` -> ``out_hidden_size``
  (which equals the LM ``d_model`` for Qwen3.5/3.6/3-VL).

Phase 1 simplifications (documented in
``docs/internal/multimodal_session_notes.md``):

* Pure-PyTorch implementation; no Triton kernel. The encoder runs once
  per round under ``torch.inference_mode()`` so peak memory is bounded
  by one round's worth of pixel data.
* Variable-length attention across multiple images in a batched call is
  handled by iterating over images one at a time and stitching the
  outputs. Production-grade speed would batch them via FA varlen, but
  for forward-parity validation this is sufficient.
* No deepstack support yet (Qwen3.5/3.6 have ``deepstack_visual_indexes=[]``
  so it's a no-op there; Qwen3-VL proper has ``[8,16,24]`` and is
  deferred to Phase 2 -- needs backbone-layer protocol extension).
* All weights are stored as ``TensorSpec(frozen=True)``. The engine's
  existing frozen-skip in :class:`BufferManager` covers grad and
  opt-state allocation automatically.

The encoder consumes a :class:`~flextrain.core.modality.ImageInputs`
device-side bundle (packed pixel_values, pix_offsets, grid_thw) and
returns a :class:`~flextrain.core.modality.ImageEmbeddings` (ragged
embeds with per-image offsets).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, MutableMapping

import torch
import torch.nn.functional as F

from flextrain.core.activation_schema import ActivationSchema
from flextrain.core.layer import (
    ComputeCost,
    LayerContext,
    ParamSpec,
    TensorSpec,
)
from flextrain.core.modality import (
    ImageEmbeddings,
    ImageGradInputs,
    ImageInputs,
    InputsSummary,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenVLVisionConfig:
    """Vision-tower geometry for Qwen3.5 / Qwen3.6 / Qwen3-VL.

    Field names match HF ``Qwen3VLVisionConfig``. Defaults match
    Qwen3-VL-4B-Instruct; Qwen3.5 / Qwen3.6 override (smaller hidden,
    different out_hidden, ``deepstack_visual_indexes=[]``).
    """

    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    # Layer indices at which to fork off a "deepstack" feature, fed
    # back into the LM at corresponding backbone layers. Phase 1 only
    # supports empty list (Qwen3.5/3.6 behaviour); non-empty raises.
    deepstack_visual_indexes: tuple[int, ...] = ()

    compute_dtype: torch.dtype = torch.bfloat16
    # Attention implementation inside the vision tower.
    #
    # * "flash" -- flextrain's flash-attn varlen wrapper
    #   (``flextrain.ops._kernels.attention.flextrain_attention_fwd``,
    #   ``causal=False``); ONE kernel call across all images instead of
    #   the per-image Python loop SDPA uses. Auto-dispatches
    #   fa4 → fa3 → fa2 → eager. Recommended for production multi-image
    #   training; same correctness, lower kernel-launch overhead.
    # * "sdpa" (current default for parity-safety) -- one
    #   ``F.scaled_dot_product_attention`` call per image; PyTorch picks
    #   Flash / Mem-Eff / Math under the hood. Tested by the loss-parity
    #   and dataset-parity tests.
    # * "eager" -- explicit bf16 matmul + fp32 softmax + bf16 weighted
    #   sum, matching HF ``eager_attention_forward`` precision exactly.
    #   Slowest; pinned by the byte-exact encoder parity test
    #   (``tests/test_qwen_vl_vit_forward.py``).
    #
    # The math is byte-identical in fp32 (cos=1.0 exact) regardless of
    # backend choice -- this knob only affects bf16 reduction order /
    # precision behavior.
    attn_implementation: str = "sdpa"

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} not divisible by "
                f"num_heads={self.num_heads}"
            )
        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        side = int(round(math.sqrt(self.num_position_embeddings)))
        if side * side != self.num_position_embeddings:
            raise ValueError(
                f"num_position_embeddings={self.num_position_embeddings} "
                "is not a perfect square; vision pos_embed must be 2-D."
            )
        return side


# ---------------------------------------------------------------------------
# ParamSpec helper -- enumerate every encoder tensor with the modality prefix.
# ---------------------------------------------------------------------------


def _prefix(modality: str, encoder_id: int) -> str:
    """Per-encoder name prefix. The arch-loader's ``vision_embed`` /
    ``vision_layer`` ArchSpec entries must use this same convention so
    the loaded tensors land at the right dest keys."""
    return f"{modality}{encoder_id}_"


def qwen_vl_vit_param_spec(
    cfg: QwenVLVisionConfig,
    *,
    modality: str = "image",
    encoder_id: int = 0,
    frozen: bool = True,
) -> ParamSpec:
    """Construct the encoder's full :class:`ParamSpec`.

    All tensors are frozen by default; Phase 3 trainable encoders set
    ``frozen=False``.

    Tensor names follow the prefix convention ``f"{modality}{encoder_id}_..."``
    (e.g. ``"image0_patch_embed_proj_w"``, ``"image0_layer_5_norm1_w"``).
    The arch loader's vision-weight entries in
    :mod:`flextrain.io.arch.qwen3_5` reference these names verbatim.
    """
    p = _prefix(modality, encoder_id)
    bf = cfg.compute_dtype

    H = cfg.hidden_size
    H_inter = cfg.intermediate_size
    out_H = cfg.out_hidden_size
    P = cfg.patch_size
    PT = cfg.temporal_patch_size
    NP = cfg.num_position_embeddings
    M = cfg.spatial_merge_size
    merged_dim = H * (M * M)

    def C(name: str, shape: tuple[int, ...]) -> TensorSpec:
        return TensorSpec(
            name=p + name,
            shape_fn=lambda dims, _shape=shape: _shape,
            compute_dtype=bf,
            master_dtype=bf,
            grad_dtype=bf,
            frozen=frozen,
        )

    tensors: list[TensorSpec] = []

    # patch_embed.proj: nn.Conv3d weight shape (out, in, T_kernel, P, P).
    tensors.append(C("patch_embed_proj_w", (H, cfg.in_channels, PT, P, P)))
    tensors.append(C("patch_embed_proj_b", (H,)))

    # pos_embed.weight: (num_position_embeddings, hidden_size).
    tensors.append(C("pos_embed_w", (NP, H)))

    # Per-block (depth blocks).
    for i in range(cfg.depth):
        # norm1, norm2 -- LayerNorm with bias.
        tensors.append(C(f"layer_{i}_norm1_w", (H,)))
        tensors.append(C(f"layer_{i}_norm1_b", (H,)))
        tensors.append(C(f"layer_{i}_norm2_w", (H,)))
        tensors.append(C(f"layer_{i}_norm2_b", (H,)))
        # Fused QKV proj: Linear(H, 3H).
        # HF stores Linear.weight as (out, in) -- keep that layout.
        tensors.append(C(f"layer_{i}_qkv_w", (3 * H, H)))
        tensors.append(C(f"layer_{i}_qkv_b", (3 * H,)))
        # Output proj: Linear(H, H).
        tensors.append(C(f"layer_{i}_proj_w", (H, H)))
        tensors.append(C(f"layer_{i}_proj_b", (H,)))
        # MLP: Linear(H, H_inter) -> Linear(H_inter, H), both with bias.
        tensors.append(C(f"layer_{i}_mlp_fc1_w", (H_inter, H)))
        tensors.append(C(f"layer_{i}_mlp_fc1_b", (H_inter,)))
        tensors.append(C(f"layer_{i}_mlp_fc2_w", (H, H_inter)))
        tensors.append(C(f"layer_{i}_mlp_fc2_b", (H,)))

    # Merger (spatial pooling + MLP to out_H).
    tensors.append(C("merger_norm_w", (H,)))
    tensors.append(C("merger_norm_b", (H,)))
    tensors.append(C("merger_fc1_w", (merged_dim, merged_dim)))
    tensors.append(C("merger_fc1_b", (merged_dim,)))
    tensors.append(C("merger_fc2_w", (out_H, merged_dim)))
    tensors.append(C("merger_fc2_b", (out_H,)))

    return ParamSpec(tensors=tuple(tensors))


# ---------------------------------------------------------------------------
# Vision-tower internals (helpers).
# ---------------------------------------------------------------------------


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    """``torch.nn.functional.gelu(..., approximate='tanh')`` -- HF's
    ``gelu_pytorch_tanh`` is exactly this."""
    return F.gelu(x, approximate="tanh")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF rotate_half convention: (-x[..., D/2:], x[..., :D/2]) along
    the last axis. Matches the vision tower's halved RoPE layout."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_vision_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vision RoPE -- HF ``apply_rotary_pos_emb_vision``.

    cos / sin: ``(T, head_dim)`` fp32 (we cast q/k to fp32 internally
    for parity with HF then cast back).
    """
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q = q.float()
    k = k.float()
    c = cos.unsqueeze(-2).float()  # (T, 1, head_dim)
    s = sin.unsqueeze(-2).float()
    q = (q * c) + (_rotate_half(q) * s)
    k = (k * c) + (_rotate_half(k) * s)
    return q.to(orig_q_dtype), k.to(orig_k_dtype)


def _build_vision_rotary_inv_freq(
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """HF Qwen3VLVisionRotaryEmbedding.__init__ -- inv_freq over
    ``head_dim`` (since vision RoPE uses head_dim/2 freqs per axis,
    then concatenates two axes -> head_dim, but the freq curve covers
    half the head_dim per axis). Returns a length-``head_dim//4``
    tensor: the RotaryEmbedding instance in HF is constructed with
    ``head_dim // 2`` and then ``head_dim // 4`` pairs.

    HF precision-policy note: HF registers ``inv_freq`` as a
    non-persistent buffer, so when the model is cast to bf16 the
    buffer goes to bf16 too. The subsequent ``torch.outer(seq,
    inv_freq)`` runs in bf16 and the resulting freq_table / cos / sin
    are bf16 (cast to fp32 just before the actual rotation via
    ``apply_rotary_pos_emb_vision``). Pass ``dtype=torch.bfloat16``
    to match HF's effective precision; pass ``torch.float32`` for
    higher-precision (slightly more accurate but mismatches HF).
    """
    half_dim = head_dim // 2
    pair = half_dim // 2  # NOTE: vision RoPE has head_dim/2 freqs per *axis*
    # Compute in fp32 always (just like HF does in __init__), then cast.
    inv = 1.0 / (
        theta
        ** (torch.arange(0, half_dim, 2, dtype=torch.float, device=device) / half_dim)
    )
    assert inv.numel() == pair, (
        f"vision inv_freq length {inv.numel()} != head_dim//4 {pair}"
    )
    return inv.to(dtype)


def _compute_grid_pos_ids(
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build per-token 2-D position ids over the spatial-merged grid.

    Mirrors :meth:`Qwen3VLVisionModel.rot_pos_emb` (the ``pos_ids``
    computation). Returns:

    * ``pos_ids``: ``(total_tokens, 2) int64`` -- ``(row, col)`` per
      token in the full (pre-merge) resolution.
    * ``max_hw``: max of any image's grid_h or grid_w, used as the
      freq-table size.
    """
    grid_thw_list = grid_thw.tolist()
    max_hw = max(max(int(h), int(w)) for _, h, w in grid_thw_list)
    total_tokens = sum(int(t) * int(h) * int(w) for t, h, w in grid_thw_list)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
    offset = 0
    for nt, h, w in grid_thw_list:
        nt = int(nt)
        h = int(h)
        w = int(w)
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(spatial_merge_size, device=device)
        intra_col = torch.arange(spatial_merge_size, device=device)
        row_idx = (
            block_rows[:, None, None, None] * spatial_merge_size
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * spatial_merge_size
            + intra_col[None, None, None, :]
        )
        row_idx = row_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        col_idx = col_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if nt > 1:
            coords = coords.repeat(nt, 1)
        n_tok = coords.shape[0]
        pos_ids[offset : offset + n_tok] = coords
        offset += n_tok
    return pos_ids, max_hw


def _fast_pos_embed_interpolate(
    pos_embed_weight: torch.Tensor,  # (NP, H)
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    num_grid_per_side: int,
) -> torch.Tensor:
    """Bilinearly interpolate the learned 2-D pos_embed table to each
    image's grid. Mirrors HF
    :meth:`Qwen3VLVisionModel.fast_pos_embed_interpolate` exactly.
    """
    grid_thw_list = grid_thw.tolist()
    grid_ts = [int(row[0]) for row in grid_thw_list]
    grid_hs = [int(row[1]) for row in grid_thw_list]
    grid_ws = [int(row[2]) for row in grid_thw_list]
    device = pos_embed_weight.device

    idx_list: list[list[int]] = [[] for _ in range(4)]
    weight_list: list[list[float]] = [[] for _ in range(4)]

    for h, w in zip(grid_hs, grid_ws):
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w)
        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
        w_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor
        base_h = h_floor * num_grid_per_side
        base_h_ceil = h_ceil * num_grid_per_side
        indices = [
            (base_h[None].T + w_floor[None]).flatten(),
            (base_h[None].T + w_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_floor[None]).flatten(),
            (base_h_ceil[None].T + w_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(
        weight_list, dtype=pos_embed_weight.dtype, device=device
    )
    pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]
    patch_pos_embeds = (
        pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    )
    # Split per image, replicate temporally, then permute to interleave
    # the (h, w) spatial-merge ordering used by the encoder.
    splits = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
    out_chunks: list[torch.Tensor] = []
    for pos_emb, t, h, w in zip(splits, grid_ts, grid_hs, grid_ws):
        pos_emb = pos_emb.repeat(t, 1)
        pos_emb = (
            pos_emb.view(
                t,
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        out_chunks.append(pos_emb)
    return torch.cat(out_chunks, dim=0)


# ---------------------------------------------------------------------------
# The encoder.
# ---------------------------------------------------------------------------


class QwenVLVisionEncoder:
    """Qwen-VL family vision tower implementing :class:`ModalityEncoder`.

    Phase 1: forward-only, frozen weights, pure PyTorch. Backward is a
    no-op.

    The encoder is constructed by the arch's ``build_modality_encoders``
    factory in :mod:`flextrain.io.arch.qwen3_5` and consumed by
    :class:`~flextrain.nn.multimodal_input.MultimodalInputLayer`.
    """

    def __init__(
        self,
        cfg: QwenVLVisionConfig,
        *,
        modality: str = "image",
        encoder_id: int = 0,
        frozen: bool = True,
    ) -> None:
        if cfg.deepstack_visual_indexes:
            raise NotImplementedError(
                "Qwen-VL deepstack is a Phase 2 item (requires backbone "
                "layer protocol extension to accept per-layer aux inputs)."
            )
        self.cfg = cfg
        self.modality = modality
        self.encoder_id = encoder_id
        self._frozen = frozen
        self._prefix = _prefix(modality, encoder_id)
        # No per-chunk activation state.
        self.schema = ActivationSchema(fields=(), max_tier=0)
        self.param_spec = qwen_vl_vit_param_spec(
            cfg, modality=modality, encoder_id=encoder_id, frozen=frozen,
        )

    # ------------------------------------------------------------------
    # ModalityEncoder Protocol
    # ------------------------------------------------------------------

    def forward_round(
        self,
        inputs: ImageInputs,
        weights: Mapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> ImageEmbeddings:
        """Encode a round's worth of images into post-merge token
        embeddings.

        Runs entirely under ``torch.inference_mode()`` (Phase 1: all
        weights are frozen). The output is a single ragged tensor
        ``(sum_i n_tokens_post_merge_i, out_hidden_size)`` plus
        per-image offsets so the splice strategy can slice per image.
        """
        if not self._frozen:
            raise NotImplementedError(
                "Phase 1: trainable QwenVLVisionEncoder is not implemented. "
                "Set frozen=True at construction time."
            )
        with torch.inference_mode():
            return self._forward_impl(inputs, weights)

    def backward_round(
        self,
        d_embeddings: ImageGradInputs,
        inputs: ImageInputs,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> None:
        """Phase 1: no-op. The encoder is frozen; the engine does not
        allocate grad / opt-state buffers for these tensors."""
        if not self._frozen:
            raise NotImplementedError(
                "Phase 3 trainable backward_round not implemented."
            )
        # Frozen encoder: drop the grad signal. The
        # ``MultimodalInputLayer.finalize_round`` hook still calls this
        # for symmetry but expects a no-op result.
        return None

    def peak_workspace_bytes(self, summary: InputsSummary) -> int:
        """Worst-case GPU peak (params aside) during one round-level
        forward of ``summary``.

        Phase 1 estimate: dominated by per-block QKV + attention
        intermediates ``O(sum_patches * num_heads * head_dim * 5 *
        bytes)`` plus the residual ``hidden_states``. Conservative
        upper bound -- the working-set planner subtracts this from the
        GPU activation budget.
        """
        H = self.cfg.hidden_size
        bf = self.cfg.compute_dtype
        bytes_per = torch.tensor([], dtype=bf).element_size()
        # 5 × seq × H is a rough upper bound: x, q, k, v, attn_out.
        # Plus the intermediate MLP buffer (seq × intermediate).
        seq = summary.sum_patches_pre_merge
        return (
            5 * seq * H * bytes_per
            + seq * self.cfg.intermediate_size * bytes_per
        )

    def compute_cost_round(self, summary: InputsSummary) -> ComputeCost:
        """Aggregate FLOPs per round (informational; not in the DP)."""
        H = self.cfg.hidden_size
        H_inter = self.cfg.intermediate_size
        seq = summary.sum_patches_pre_merge
        # Per-block: QKV proj (2*3*H*H*seq) + attn (rough 4*seq*seq*H/n_heads,
        # but vision sequences are short so we use H*seq as a proxy) +
        # output proj (2*H*H*seq) + MLP (2 * 2*H*H_inter*seq).
        per_block = (
            2 * 3 * H * H * seq
            + 2 * H * H * seq
            + 2 * 2 * H * H_inter * seq
        )
        total = per_block * self.cfg.depth
        return ComputeCost(
            total_fwd_flops=total,
            avoided_recompute_flops=(0,),
        )

    # ------------------------------------------------------------------
    # Forward implementation (private)
    # ------------------------------------------------------------------

    def _forward_impl(
        self,
        inputs: ImageInputs,
        weights: Mapping[str, torch.Tensor],
    ) -> ImageEmbeddings:
        cfg = self.cfg
        p = self._prefix
        device = inputs.pixel_values.device
        dtype = cfg.compute_dtype

        # ----- patch embed -----
        # HF: hidden_states.view(-1, C, T, P, P) then Conv3d, then view
        # (-1, embed_dim). Our pixel_values are pre-patchified in HF's
        # processor convention: shape (n_patches, C * T * P * P).
        n_patches = inputs.pixel_values.shape[0]
        patch_x = inputs.pixel_values.view(
            -1, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size,
        ).to(dtype)
        proj_w = weights[p + "patch_embed_proj_w"]
        proj_b = weights[p + "patch_embed_proj_b"]
        # Conv3d with stride = kernel: each input "patch" produces 1 output.
        x = F.conv3d(patch_x, proj_w, proj_b, stride=proj_w.shape[2:])
        # Flatten to (n_patches, hidden_size).
        x = x.view(n_patches, cfg.hidden_size)

        # ----- pos embed (bilinear interpolation of learned table) -----
        pos_w = weights[p + "pos_embed_w"]
        pos_embeds = _fast_pos_embed_interpolate(
            pos_w, inputs.grid_thw, cfg.spatial_merge_size, cfg.num_grid_per_side,
        ).to(dtype)
        x = x + pos_embeds

        # ----- rotary position embedding (vision, 2-axis, halved) -----
        # HF precision policy via ``from_pretrained(torch_dtype=bf16)``:
        # the rotary buffer ``inv_freq`` stays **fp32** because HF's
        # ``_init_weights`` re-initializes the non-persistent buffer
        # after the bf16 cast. Standalone ``.to(bf16)`` on a fresh
        # ``Qwen3_5VisionModel()`` casts the buffer instead — this is a
        # red herring for any byte-exact reference test. Match the
        # production (``from_pretrained``) path: fp32 inv_freq → fp32
        # freq_table → cos/sin computed in fp32, downcast to dtype only
        # for the Q*cos / K*cos rotation (handled inside
        # ``_apply_vision_rope``).
        head_dim = cfg.head_dim
        inv_freq = _build_vision_rotary_inv_freq(
            head_dim, 10000.0, device, dtype=torch.float32,
        )
        pos_ids, max_hw = _compute_grid_pos_ids(
            inputs.grid_thw, cfg.spatial_merge_size, device,
        )
        seq = torch.arange(max_hw, device=device, dtype=inv_freq.dtype)
        freq_table = torch.outer(seq, inv_freq)  # (max_hw, head_dim/4) in dtype
        # rot_emb: (total_tokens, 2, head_dim/4) -> flatten -> (T, head_dim/2)
        rot_emb = freq_table[pos_ids]
        rot_emb = rot_emb.flatten(1)
        # Duplicate to cover full head_dim (halved-RoPE convention).
        emb = torch.cat((rot_emb, rot_emb), dim=-1)  # (T, head_dim)
        cos = emb.cos()
        sin = emb.sin()

        # ----- cu_seqlens for variable-length attention -----
        # Each image contributes T * H_grid * W_grid patches.
        grid_thw_list = inputs.grid_thw.tolist()
        seq_lens = []
        for nt, h, w in grid_thw_list:
            seq_lens.extend([int(h) * int(w)] * int(nt))
        # cu_seqlens: (n_seqs + 1,) -- prefix-sum.
        cu_seqlens = torch.zeros(
            len(seq_lens) + 1, dtype=torch.int32, device=device,
        )
        running = 0
        for i, L in enumerate(seq_lens):
            running += L
            cu_seqlens[i + 1] = running
        # Sanity: total should equal n_patches.
        if int(cu_seqlens[-1].item()) != n_patches:
            raise ValueError(
                f"Vision encoder: cu_seqlens total {int(cu_seqlens[-1])} != "
                f"n_patches {n_patches}; check grid_thw vs pixel_values."
            )

        # ----- transformer blocks -----
        for i in range(cfg.depth):
            x = self._block_forward(
                x,
                weights=weights,
                cu_seqlens=cu_seqlens,
                cos=cos,
                sin=sin,
                layer_idx=i,
            )

        # ----- merger (spatial merge + MLP -> out_hidden_size) -----
        x = self._merger_forward(x, weights=weights)

        # ----- pack output -----
        # After merger: ``(n_patches // spatial_merge_size**2,
        # out_hidden_size)``. Compute per-image post-merge token counts
        # to build token_offsets.
        merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size
        post_token_counts: list[int] = []
        for nt, h, w in grid_thw_list:
            post_token_counts.append(
                int(nt) * (int(h) * int(w)) // merge_unit
            )
        token_offsets = torch.zeros(
            len(post_token_counts) + 1, dtype=torch.int32, device=device,
        )
        running = 0
        for i, L in enumerate(post_token_counts):
            running += L
            token_offsets[i + 1] = running
        return ImageEmbeddings(
            embeds=x,
            token_offsets=token_offsets,
            grid_thw=inputs.grid_thw,
        )

    # --- block forward ---

    def _block_forward(
        self,
        hidden_states: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        p = self._prefix
        L = layer_idx
        H = cfg.hidden_size

        # ---- pre-attention LayerNorm ----
        ln1_w = weights[p + f"layer_{L}_norm1_w"]
        ln1_b = weights[p + f"layer_{L}_norm1_b"]
        normed = F.layer_norm(hidden_states, (H,), ln1_w, ln1_b, eps=cfg.rms_norm_eps)

        # ---- self-attention ----
        qkv_w = weights[p + f"layer_{L}_qkv_w"]
        qkv_b = weights[p + f"layer_{L}_qkv_b"]
        proj_w = weights[p + f"layer_{L}_proj_w"]
        proj_b = weights[p + f"layer_{L}_proj_b"]

        seq_len = normed.shape[0]
        qkv = F.linear(normed, qkv_w, qkv_b)  # (seq, 3*H)
        # Reshape to (seq, 3, num_heads, head_dim) then split.
        qkv = qkv.view(seq_len, 3, cfg.num_heads, cfg.head_dim)
        q, k, v = qkv.unbind(dim=1)  # each: (seq, num_heads, head_dim)

        # Apply vision RoPE on Q, K.
        q, k = _apply_vision_rope(q, k, cos, sin)

        # SDPA per-image (variable-length). Vision sequences are short
        # so this is fine for Phase 1 forward-parity validation.
        out_attn = self._sdpa_varlen(q, k, v, cu_seqlens)

        # Output projection.
        out_attn = out_attn.reshape(seq_len, H).contiguous()
        attn_out = F.linear(out_attn, proj_w, proj_b)
        hidden_states = hidden_states + attn_out

        # ---- pre-MLP LayerNorm + MLP ----
        ln2_w = weights[p + f"layer_{L}_norm2_w"]
        ln2_b = weights[p + f"layer_{L}_norm2_b"]
        normed = F.layer_norm(hidden_states, (H,), ln2_w, ln2_b, eps=cfg.rms_norm_eps)
        fc1_w = weights[p + f"layer_{L}_mlp_fc1_w"]
        fc1_b = weights[p + f"layer_{L}_mlp_fc1_b"]
        fc2_w = weights[p + f"layer_{L}_mlp_fc2_w"]
        fc2_b = weights[p + f"layer_{L}_mlp_fc2_b"]
        mlp = F.linear(normed, fc1_w, fc1_b)
        if cfg.hidden_act == "gelu_pytorch_tanh":
            mlp = _gelu_pytorch_tanh(mlp)
        else:
            raise NotImplementedError(
                f"vision activation {cfg.hidden_act!r} not supported; "
                "Qwen-VL family uses gelu_pytorch_tanh."
            )
        mlp = F.linear(mlp, fc2_w, fc2_b)
        hidden_states = hidden_states + mlp
        return hidden_states

    def _sdpa_varlen(
        self,
        q: torch.Tensor,  # (seq, n_heads, head_dim)
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Variable-length attention.

        Picks the implementation based on ``cfg.attn_implementation``:

        * ``"flash"`` -- flextrain's flash-attn varlen wrapper
          (``flextrain.ops._kernels.attention.flextrain_attention_fwd``)
          with ``causal=False``. Processes all images in one kernel
          call (no Python loop over ``cu_seqlens``). Auto-dispatches
          fa4 → fa3 → fa2 → eager via the wrapper.
        * ``"sdpa"`` (default for training) -- ``F.scaled_dot_product_attention``
          per image; lets PyTorch select Flash / Mem-Eff / Math.
        * ``"eager"`` -- explicit bf16 matmul + fp32 softmax + bf16
          weighted sum. Matches HF ``eager_attention_forward``
          precision exactly. Used by the byte-exact parity test.

        In fp32 all three paths are byte-identical; the choice only
        affects bf16 precision behavior on a 24-layer tower (different
        reduction orderings).
        """
        impl = getattr(self.cfg, "attn_implementation", "sdpa")
        if impl == "eager":
            return self._sdpa_varlen_eager(q, k, v, cu_seqlens)
        if impl == "sdpa":
            return self._sdpa_varlen_torch_sdpa(q, k, v, cu_seqlens)
        if impl == "flash":
            return self._sdpa_varlen_flash(q, k, v, cu_seqlens)
        raise ValueError(
            f"unknown attn_implementation {impl!r}; expected one of "
            "'flash' / 'sdpa' / 'eager'"
        )

    def _sdpa_varlen_flash(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Single-kernel-call varlen attention via flextrain's flash-attn
        wrapper. Replaces the per-image Python loop in
        :meth:`_sdpa_varlen_torch_sdpa` with one call over packed q/k/v.

        Non-causal (``causal=False``) because ViT attention is
        bidirectional within each image.

        Backward path note: this method is for forward only. Phase 1
        encoders are frozen, so the encoder never reaches
        ``flextrain_attention_bwd``. If the encoder is unfrozen
        (Phase 3 trainable vision tower), wire ``bwd`` analogously and
        pre-allocate ``dq/dk/dv`` scratch (the wrapper OVERWRITES
        rather than accumulates).
        """
        from flextrain.ops._kernels.attention import flextrain_attention_fwd

        total = q.shape[0]
        n_heads, head_dim = q.shape[1], q.shape[2]
        out = torch.empty_like(q)
        # softmax_lse shape per the wrapper contract: (n_q_heads, total_q)
        # fp32 -- caller-allocated; we discard after forward.
        softmax_lse = torch.empty(
            (n_heads, total), dtype=torch.float32, device=q.device,
        )
        # cu_seqlens is the standard prefix-sum layout the wrapper uses
        # as both q_seq_offsets and k_seq_offsets (vision attention is
        # self-attention so q == k offsets).
        cu_i32 = cu_seqlens.to(torch.int32)
        q_lens = (cu_i32[1:] - cu_i32[:-1]).contiguous()
        # max_seqlen drives kernel block-shape selection; pull a CPU int.
        max_seqlen = int(q_lens.max().item()) if q_lens.numel() > 0 else 0
        flextrain_attention_fwd(
            q, k, v, out, softmax_lse,
            cu_i32, cu_i32, q_lens, q_lens,
            max_seqlen, max_seqlen,
            causal=False,
        )
        return out

    def _sdpa_varlen_torch_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Per-image SDPA via ``F.scaled_dot_product_attention`` (Flash-
        compatible). The original Phase 1 implementation; kept as the
        default for training where Flash speedup matters."""
        cu = cu_seqlens.tolist()
        outputs = []
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            if e == s:
                continue
            qi = q[s:e].transpose(0, 1).unsqueeze(0)  # (1, n_heads, seq_i, head_dim)
            ki = k[s:e].transpose(0, 1).unsqueeze(0)
            vi = v[s:e].transpose(0, 1).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(
                qi, ki, vi, attn_mask=None, dropout_p=0.0, is_causal=False,
            )
            out_i = out_i.squeeze(0).transpose(0, 1)  # (seq_i, n_heads, head_dim)
            outputs.append(out_i)
        return torch.cat(outputs, dim=0)

    def _sdpa_varlen_eager(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Per-image eager attention matching HF
        ``eager_attention_forward``: bf16 QK^T, fp32 softmax, bf16
        weighted sum."""
        head_dim = q.shape[-1]
        scaling = head_dim ** -0.5
        cu = cu_seqlens.tolist()
        outputs = []
        for i in range(len(cu) - 1):
            s, e = cu[i], cu[i + 1]
            if e == s:
                continue
            qi = q[s:e].transpose(0, 1)
            ki = k[s:e].transpose(0, 1)
            vi = v[s:e].transpose(0, 1)
            attn_scores = torch.matmul(qi, ki.transpose(-1, -2)) * scaling
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(qi.dtype)
            out_i = torch.matmul(attn_weights, vi)
            out_i = out_i.transpose(0, 1)
            outputs.append(out_i)
        return torch.cat(outputs, dim=0)

    # --- merger ---

    def _merger_forward(
        self, hidden_states: torch.Tensor, weights: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.cfg
        p = self._prefix
        H = cfg.hidden_size
        merged = H * (cfg.spatial_merge_size * cfg.spatial_merge_size)

        # Pre-shuffle norm (LayerNorm over original ``H``).
        norm_w = weights[p + "merger_norm_w"]
        norm_b = weights[p + "merger_norm_b"]
        x = F.layer_norm(hidden_states, (H,), norm_w, norm_b, eps=cfg.rms_norm_eps)
        x = x.view(-1, merged)  # group spatial_merge_size**2 consecutive tokens
        fc1_w = weights[p + "merger_fc1_w"]
        fc1_b = weights[p + "merger_fc1_b"]
        fc2_w = weights[p + "merger_fc2_w"]
        fc2_b = weights[p + "merger_fc2_b"]
        x = F.linear(x, fc1_w, fc1_b)
        x = F.gelu(x)  # vanilla GELU per HF
        x = F.linear(x, fc2_w, fc2_b)
        return x

    # ------------------------------------------------------------------
    # Convenience: number of vision layers (consumed by ActiveModel.load_hf
    # to pass num_vision_layers through to load_hf_safetensors).
    # ------------------------------------------------------------------

    @property
    def num_vision_layers(self) -> int:
        return self.cfg.depth
