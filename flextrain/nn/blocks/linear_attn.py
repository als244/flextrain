"""Gated DeltaNet linear attention block (Qwen3-Next-style).

Replaces a softmax attention layer with a linear-attention recurrence.
Backed by ``flash-linear-attention`` (FLA)'s ``chunk_gated_delta_rule``
fwd/bwd primitives — we call them directly (NOT via
``torch.autograd``) so the FlexTrain engine retains full control over
activation persistence and recomputation.

Architecture (per Qwen3-Next, also Qwen 3.5 / 3.6 hybrid layers):

    x: (T, d_model)
    qkvz = x @ W_qkvz                     # (T, qkvz_dim)
    ba   = x @ W_ba                       # (T, ba_dim)
    q_pre, k_pre, v_pre, z = split(qkvz)  # heads-aware reshapes
    b, a                  = split(ba)
    mixed_qkv = silu(conv1d(cat(q_pre, k_pre, v_pre)))  # depthwise causal conv
    q, k, v   = split(mixed_qkv)
    beta = sigmoid(b)
    g    = softplus(a + dt_bias) * (-exp(A_log))         # in fp32
    o, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                                  use_qk_l2norm_in_kernel=True)
    o    = rms_norm_gated(o, z)                          # silu(z) * rmsnorm(o)
    y    = o @ W_out                                     # (T, d_model)

Activation schema (max_tier=3):

* Tier 0 — always saved (small):
    ``lin_ba`` (T, 2*n_v_heads) bf16          -- raw b|a (post-projection)
    ``lin_g_post`` (T, n_v_heads) fp32        -- post-cumsum gate
* Tier 2 — FLA-output fields, expensive to recompute:
    ``lin_A_int``     (T, n_v_heads, 64)              -- FLA intra-chunk scratch
    ``lin_core_out``  (T, n_v_heads, head_v_dim)      -- FLA output
* Tier 3 — biggest field, the projection output:
    ``lin_qkvz``      (T, 2*key_dim + 2*value_dim) bf16  -- x @ W_qkvz

Q/K/V (post-l2norm) and the l2norm rstds are NOT saved. Bwd already
runs the conv to recover pre-silu (silu_bwd input), and the same conv
with activation='silu' produces post-conv from which Q/K/V derive
trivially. Stage D2 made the conv recompute mandatory; this collapses
the schema accordingly.

post-conv (silu output) is also NOT saved (transient scratch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch
import torch.nn.functional as F

from flextrain.core.activation_schema import ActivationField
from flextrain.core.layer import (
    ChunkMeta,
    ComputeCost,
    LayerContext,
    ParamSpec,
    TensorSpec,
)


# ---------------------------------------------------------------------------
# FLA imports — required at module load time (no soft fallback for now).
# ---------------------------------------------------------------------------

try:
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_fwd,
    )
    # L2 norm helpers: HF's gated-delta linear-attn applies an L2-norm
    # to q/k before the recurrence (``use_qk_l2norm_in_kernel=True`` in
    # the high-level wrapper). The lower-level ``_fwd``/``_bwd`` calls
    # don't accept that flag, so we apply it explicitly here. Keeping
    # ``q_rstd``/``k_rstd`` for the bwd half of l2_norm.
    from fla.modules.l2norm import l2norm_fwd, l2norm_bwd
    # FLA's causal_conv1d_fwd_kernel — called directly so we can
    # supply our own output buffer (the upstream python helper
    # allocates internally with torch.empty_like).
    from fla.modules.conv.triton.ops import causal_conv1d_fwd_kernel
    # FLA's l2norm fwd kernel pair — same trick: bypass the python
    # helper's torch.empty_like to write directly into slot tensors.
    from fla.modules.l2norm import l2norm_fwd_kernel, l2norm_fwd_kernel1
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Qwen3NextLinearAttention requires `flash-linear-attention`. "
        "Install with: pip install flash-linear-attention"
    ) from e


def _fla_causal_conv1d_fwd_into(
    x_2d: torch.Tensor,                # (T, D), strided OK (kernel uses strides)
    weight: torch.Tensor,              # (D, W) bf16
    out_2d: torch.Tensor,              # (T, D) bf16, contiguous, written in-place
    *,
    cu_seqlens: torch.Tensor | None = None,
    bt: int = 64,
    activation: str | None = None,
) -> None:
    """Direct call into FLA's ``causal_conv1d_fwd_kernel`` writing into
    a caller-supplied output buffer.

    The upstream ``causal_conv1d_fwd`` python helper allocates output
    via ``torch.empty_like(x)`` and returns it; bypassing it lets us
    write the conv output directly into a caller-supplied scratch
    buffer (avoiding a (T, conv_dim) bf16 D2D memcpy).

    We mimic that helper's preprocessing (rearrange, stride read,
    chunk_indices for varlen) but skip the allocation. The output
    must already be contiguous (the kernel writes contiguously, but
    we read it back as a (T, D) view).
    """
    import triton  # local import to keep top-of-file imports light
    from fla.modules.conv.triton.ops import prepare_chunk_indices, rearrange
    # Treat as (B=1, T, D) for the kernel's grid layout.
    if x_2d.shape[-1] != weight.shape[0]:
        x_2d = rearrange(x_2d, 'b t ... -> b t (...)')
    x = x_2d.unsqueeze(0)
    y = out_2d.unsqueeze(0)
    B, T, D = x.shape[0], x.shape[1], weight.shape[0]
    W = weight.shape[1]
    stride_x_n, stride_x_t, stride_x_d = x.stride()
    BW = triton.next_power_of_2(W)
    chunk_indices = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, bt)
    NT = (
        len(chunk_indices) if cu_seqlens is not None
        else triton.cdiv(T, bt)
    )
    NB = triton.cdiv(B * T, 1024)

    def _grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, B)

    causal_conv1d_fwd_kernel[_grid](
        x=x,
        y=y,
        weight=weight,
        bias=None,
        residual=None,
        cu_seqlens=cu_seqlens,
        initial_state=None,
        chunk_indices=chunk_indices,
        B=B, T=T, D=D, W=W,
        BT=bt, BW=BW, NB=NB,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        ACTIVATION=activation,
    )


def _fla_l2norm_fwd_into(
    x: torch.Tensor,           # (T_outer, D) where T_outer is product of leading dims
    y_out: torch.Tensor,       # same shape as x; written in-place
    rstd_out: torch.Tensor,    # (T_outer,) fp32; written in-place
    *,
    eps: float = 1e-6,
) -> None:
    """Direct call into FLA's ``l2norm_fwd_kernel`` writing into
    caller-supplied output buffers.

    Mirrors the upstream python ``l2norm_fwd`` preprocessing (flatten
    leading dims, choose between the BT-tiled vs single-thread-per-row
    kernel based on D) but skips the ``torch.empty_like`` allocations
    so we can write the post-l2norm q/k directly into ``slot.lin_q``
    and ``slot.lin_k``.

    The two output buffers are written-in-place; they must already be
    contiguous and dim-matched (``y_out.shape == x.shape`` after the
    flatten, ``rstd_out`` is 1-D length ``T_outer``).
    """
    import triton  # local
    assert x.shape == y_out.shape, (
        f"x={tuple(x.shape)} y_out={tuple(y_out.shape)}"
    )
    assert y_out.is_contiguous() and rstd_out.is_contiguous()
    assert y_out.stride(-1) == 1
    T = x.shape[0]
    D = x.shape[-1]
    # Match the upstream MAX_FUSED_SIZE / BD logic.
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("l2norm: feature dim >= 64KB unsupported.")
    if D <= 512:
        # NB heuristic from upstream — see fla l2norm_fwd().
        NB = triton.cdiv(T, 2048 * 32)

        def _grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[_grid](
            x=x,
            y=y_out,
            rstd=rstd_out,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x=x,
            y=y_out,
            rstd=rstd_out,
            eps=eps,
            D=D,
            BD=BD,
        )


@dataclass(frozen=True)
class GatedDeltaNetConfig:
    """Per-instance config for :class:`GatedDeltaNetBlock`.

    ``num_v_heads`` × ``head_v_dim`` defines the value/output channel
    width; ``num_k_heads`` × ``head_k_dim`` defines key/query width
    (Qwen3-Next has GVA: ``num_v_heads >= num_k_heads``, with v repeating
    over the k dim during compute). ``conv_kernel_size`` is the depthwise
    causal-conv kernel size (4 in Qwen3-Next).
    """

    d_model: int
    num_v_heads: int
    num_k_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int = 4
    rms_norm_eps: float = 1e-6
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None

    @property
    def key_dim(self) -> int:
        return self.head_k_dim * self.num_k_heads

    @property
    def value_dim(self) -> int:
        return self.head_v_dim * self.num_v_heads

    @property
    def conv_dim(self) -> int:
        # cat(q_pre, k_pre, v_pre) along last axis (post-fix-ordering).
        return self.key_dim * 2 + self.value_dim

    @property
    def proj_qkvz_dim(self) -> int:
        # Q (key_dim) + K (key_dim) + V (value_dim) + Z (value_dim).
        return self.key_dim * 2 + self.value_dim * 2

    @property
    def proj_ba_dim(self) -> int:
        # B (n_v_heads) + A (n_v_heads).
        return self.num_v_heads * 2

    def dims(self) -> dict[str, int]:
        return {
            "d_model": self.d_model,
            "num_v_heads": self.num_v_heads,
            "num_k_heads": self.num_k_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "conv_dim": self.conv_dim,
            "proj_qkvz_dim": self.proj_qkvz_dim,
            "proj_ba_dim": self.proj_ba_dim,
            "conv_kernel_size": self.conv_kernel_size,
        }


def _split_qkvz(qkvz: torch.Tensor, cfg: GatedDeltaNetConfig):
    """Match HF's ``fix_query_key_value_ordering``.

    qkvz: (T, proj_qkvz_dim) -> reshape to (T, n_k_heads, ?) and split.

    NOTE: this is the HF-compatible decomposition. The FT-side fast-path
    (``_split_qkvz_ft``) requires the load-time column permutation
    (:func:`build_qkvz_perm`) and produces zero-copy contiguous views.
    Until the loader / exporter / bwd are all updated together, the
    block stays on this path.
    """
    T = qkvz.shape[0]
    H = cfg.num_k_heads
    HV = cfg.num_v_heads
    hk = cfg.head_k_dim
    hv = cfg.head_v_dim
    qkvz = qkvz.view(T, H, 2 * hk + 2 * (HV // H) * hv)
    parts = [hk, hk, (HV // H) * hv, (HV // H) * hv]
    q, k, v_grp, z_grp = torch.split(qkvz, parts, dim=-1)
    v = v_grp.reshape(T, HV, hv)
    z = z_grp.reshape(T, HV, hv)
    return q, k, v, z


def _split_ba(ba: torch.Tensor, cfg: GatedDeltaNetConfig):
    """HF-compatible ``ba`` decomposition. See :func:`_split_qkvz` note."""
    T = ba.shape[0]
    H = cfg.num_k_heads
    HV = cfg.num_v_heads
    grp = HV // H
    ba = ba.view(T, H, 2 * grp)
    b, a = torch.split(ba, [grp, grp], dim=-1)
    b = b.reshape(T, HV)
    a = a.reshape(T, HV)
    return b, a


def _split_qkvz_ft(qkvz: torch.Tensor, cfg: GatedDeltaNetConfig):
    """Decompose qkvz in FT's column-grouped layout (post-permutation).

    qkvz columns are ``[Q | K | V | Z]`` (block-major), each block laid
    out as ``(head, dim)`` row-major. All four returned views are
    zero-copy contiguous. Eliminates the 4 reshape-copies + cat that
    the HF-layout :func:`_split_qkvz` forces in the fwd hot path.

    Requires the load-time column permutation in
    :func:`build_qkvz_perm` (and its inverse in the exporter).
    """
    T = qkvz.shape[0]
    H = cfg.num_k_heads
    HV = cfg.num_v_heads
    hk = cfg.head_k_dim
    hv = cfg.head_v_dim
    key_dim = H * hk
    value_dim = HV * hv
    q = qkvz[:, :key_dim].view(T, H, hk)
    k = qkvz[:, key_dim:2 * key_dim].view(T, H, hk)
    v = qkvz[:, 2 * key_dim:2 * key_dim + value_dim].view(T, HV, hv)
    z = qkvz[:, 2 * key_dim + value_dim:].view(T, HV, hv)
    return q, k, v, z


def _split_ba_ft(ba: torch.Tensor, cfg: GatedDeltaNetConfig):
    """FT layout: ``[B | A]``. Zero-copy. Requires :func:`build_ba_perm`."""
    HV = cfg.num_v_heads
    return ba[:, :HV], ba[:, HV:]


def build_qkvz_perm(cfg: GatedDeltaNetConfig) -> torch.Tensor:
    """Permutation tensor mapping FT column order to HF column order.

    Use as ``W_qkvz_ft = W_qkvz_hf[:, perm].contiguous()``. Saving:
    inverse permutation is :func:`build_qkvz_perm_inverse`. Index
    tensor is on CPU; loaders / exporters are CPU-side.

    HF layout (per-K-head h, intra-head local position):
      [0..hk-1]                           : Q[h, :]
      [hk..2hk-1]                         : K[h, :]
      [2hk..2hk + grp*hv - 1]             : V slice for v-heads h*grp..(h+1)*grp-1
      [2hk + grp*hv..2hk + 2*grp*hv - 1]  : Z slice (same v-heads)

    FT target layout:
      [0..key_dim-1]                                   : Q (head varies first, dim fast)
      [key_dim..2*key_dim-1]                           : K
      [2*key_dim..2*key_dim+value_dim-1]               : V (head varies first, dim fast)
      [2*key_dim+value_dim..]                          : Z
    """
    H = cfg.num_k_heads
    HV = cfg.num_v_heads
    hk = cfg.head_k_dim
    hv = cfg.head_v_dim
    grp = HV // H
    per_k_head = 2 * hk + 2 * grp * hv
    perm: list[int] = []
    # Q block.
    for h in range(H):
        off = h * per_k_head
        for d in range(hk):
            perm.append(off + d)
    # K block.
    for h in range(H):
        off = h * per_k_head
        for d in range(hk):
            perm.append(off + hk + d)
    # V block.
    for h in range(H):
        for gh in range(grp):
            off = h * per_k_head + 2 * hk + gh * hv
            for d in range(hv):
                perm.append(off + d)
    # Z block.
    for h in range(H):
        for gh in range(grp):
            off = h * per_k_head + 2 * hk + grp * hv + gh * hv
            for d in range(hv):
                perm.append(off + d)
    return torch.tensor(perm, dtype=torch.int64)


def build_ba_perm(cfg: GatedDeltaNetConfig) -> torch.Tensor:
    """Permutation for the ``ba`` projection. HF interleaves
    ``[b_grp, a_grp]`` per K-head; FT wants ``[B | A]`` flat.
    """
    H = cfg.num_k_heads
    HV = cfg.num_v_heads
    grp = HV // H
    perm: list[int] = []
    for h in range(H):
        for gh in range(grp):
            perm.append(h * (2 * grp) + gh)
    for h in range(H):
        for gh in range(grp):
            perm.append(h * (2 * grp) + grp + gh)
    return torch.tensor(perm, dtype=torch.int64)


def build_qkvz_perm_inverse(cfg: GatedDeltaNetConfig) -> torch.Tensor:
    """Inverse of :func:`build_qkvz_perm` for HF safetensors export.
    ``W_qkvz_hf = W_qkvz_ft[:, inv_perm]``."""
    perm = build_qkvz_perm(cfg)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    return inv


def build_ba_perm_inverse(cfg: GatedDeltaNetConfig) -> torch.Tensor:
    """Inverse of :func:`build_ba_perm`."""
    perm = build_ba_perm(cfg)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    return inv


# ---------------------------------------------------------------------------
# Block class
# ---------------------------------------------------------------------------


class GatedDeltaNetBlock:
    """Linear attention via gated DeltaNet (Qwen3-Next style).

    Forward pass uses FLA's ``chunk_gated_delta_rule_fwd`` directly;
    backward pass uses ``chunk_gated_delta_rule_bwd``. The engine
    saves activations needed for bwd into the layer slot.

    The block only handles the linear-attention path itself —
    residual / norm / FFN are owned by the enclosing layer (just like
    :class:`GQAAttentionBlock`).
    """

    def __init__(self, cfg: GatedDeltaNetConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Declarations.
    # ------------------------------------------------------------------

    def fields(self) -> tuple[ActivationField, ...]:
        cfg = self.cfg
        bf = cfg.compute_dtype
        return (
            # ==================================================
            # Tier 0: small-and-always-saved fields. Recompute-helper
            # data (norm rstds, gate scalars). Sub-megabyte for any
            # realistic T.
            # ==================================================
            #
            # ``lin_ba`` is the (T, proj_ba_dim) matmul output of the
            # x @ W_lin_ba projection in the FT column-block-major
            # layout ``[B (num_v_heads) | A (num_v_heads)]``. The bwd /
            # gate-prep code accesses b/a as zero-copy slices:
            #
            #   b = slot.lin_ba[:, :num_v_heads]
            #   a = slot.lin_ba[:, num_v_heads:]
            #
            # See :func:`_split_ba_ft`.
            ActivationField(
                "lin_ba",
                lambda n, d: (n, 2 * cfg.num_v_heads),
                bf, tier=0,
            ),
            # lin_g_post: post-cumsum gate (FLA's gdn_gate_chunk_cumsum
            # output). Stage D5 dropped lin_g (pre-cumsum gate) since
            # FLA's use_gate_in_kernel path fuses softplus + dt_bias +
            # (-exp A_log) + chunk_local_cumsum into one kernel; the
            # only persisted gate state is the post-cumsum output, used
            # by the bwd's ``g`` argument and by gdn_gate_bwd to
            # backprop into raw a / A_log / dt_bias.
            ActivationField(
                "lin_g_post",
                lambda n, d: (n, cfg.num_v_heads),
                torch.float32, tier=0,
            ),
            # ==================================================
            # Tier 2: FLA scratch + FLA core output. Q/K/V/rstds are
            # NOT saved: bwd already pays a conv recompute (see Stage D2)
            # and post-conv is the input to qkv_heads, so deriving Q/K/V
            # is essentially free once post-conv is in scratch. Saves
            # ~T*(2*key_dim + value_dim) bf16 + 2*T*num_k_heads fp32
            # per layer (~270 MiB at T=16k for Qwen3.5-MoE-35B-A3B).
            # ==================================================
            ActivationField(
                "lin_A_int",
                lambda n, d: (n, cfg.num_v_heads, 64),
                bf, tier=2,
            ),
            ActivationField(
                "lin_core_out",
                lambda n, d: (n, cfg.num_v_heads, cfg.head_v_dim),
                bf, tier=2,
            ),
            # ==================================================
            # Tier 3: largest fields, recomputable from x_inp by re-
            # running the qkvz/ba matmuls + conv + l2norm.
            # ==================================================
            #
            # ``lin_qkvz`` is the (T, proj_qkvz_dim) matmul output in
            # the FT column-block-major layout
            # ``[Q | K | V | Z]`` (each block laid out (head, dim)
            # row-major). The fwd hot path views into it without
            # copying:
            #
            #   q       = qkvz[:, :key_dim].view(T, num_k_heads, head_k_dim)
            #   k       = qkvz[:, key_dim:2*key_dim].view(...)
            #   v       = qkvz[:, 2*key_dim:2*key_dim+value_dim].view(T, HV, hv)
            #   z       = qkvz[:, 2*key_dim+value_dim:].view(T, HV, hv)
            #   conv_in = qkvz[:, :conv_dim]   # contiguous slice
            #
            # Replaces the previous separate fields ``lin_a`` /
            # ``lin_b`` (tier-0; now folded into ``lin_ba``), ``lin_z``
            # (tier-1; now a view of ``lin_qkvz``), and ``lin_conv_in``
            # (tier-3; now a view of ``lin_qkvz``). The save-level DP
            # solver still has the option to drop qkvz entirely (level
            # < 3) and pay one matmul recompute via
            # ``fwd_recompute_post_conv``.
            ActivationField(
                "lin_qkvz",
                lambda n, d: (n, 2 * cfg.key_dim + 2 * cfg.value_dim),
                bf, tier=3,
            ),
            # NB: post-conv (silu output) is NOT saved. Following FLA's
            # pattern, the linear-attn bwd recomputes pre-silu by re-
            # running the conv with activation=None into a scratch
            # buffer; no separate persistent slot field is needed.
            # Saves ~T*conv_dim bf16 per layer of activation memory
            # (e.g. ~268 MiB at T=16k, conv_dim=8192).
        )

    def param_spec(self) -> ParamSpec:
        cfg = self.cfg

        def _spec(name, shape_fn, optimizer=None):
            return TensorSpec(
                name=name, shape_fn=shape_fn,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
                optimizer=optimizer,
            )

        return ParamSpec(
            tensors=(
                # Linear projections.
                _spec(
                    "w_lin_qkvz",
                    lambda d: (d["d_model"], d["proj_qkvz_dim"]),
                ),
                _spec(
                    "w_lin_ba",
                    lambda d: (d["d_model"], d["proj_ba_dim"]),
                ),
                _spec(
                    "w_lin_out",
                    lambda d: (d["value_dim"], d["d_model"]),
                ),
                # Depthwise causal conv1d weights (groups=conv_dim, so
                # weight shape is (conv_dim, 1, kernel_size)).
                _spec(
                    "w_lin_conv",
                    lambda d: (
                        d["conv_dim"], 1, d["conv_kernel_size"],
                    ),
                ),
                # Gate-rule scalars (1-D, AdamW).
                _spec(
                    "w_lin_dt_bias",
                    lambda d: (d["num_v_heads"],),
                    optimizer="adamw",
                ),
                _spec(
                    "w_lin_A_log",
                    lambda d: (d["num_v_heads"],),
                    optimizer="adamw",
                ),
                # Gated-RMSNorm weight.
                _spec(
                    "w_lin_norm",
                    lambda d: (d["head_v_dim"],),
                    optimizer="adamw",
                ),
            )
        )

    # ------------------------------------------------------------------
    # Compute.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Forward stages — split so each can be recomputed independently
    # when an upstream tier was dropped by the save-level solver.
    # Mirrors the partial-recompute pattern in
    # ``GQAAttentionBlock.fwd_recompute_qo / _attn / _o``.
    # ------------------------------------------------------------------

    def _fwd_proj_split(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
        *,
        skip_already_saved: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: ``x @ W_qkvz`` directly into ``slot.lin_qkvz`` and
        ``x @ W_ba`` directly into ``slot.lin_ba``. q/k/v/z and b/a
        are then zero-copy views of the slot tensors — no post-matmul
        memcpys.

        Tier-3 ``lin_qkvz`` and tier-0 ``lin_ba`` are the only slot
        fields written here. ``skip_already_saved=True`` from recompute
        paths uses ``slot.has("...")`` to skip the write (the slot may
        already hold valid data from the original fwd's persist).

        Returns ``(z, conv_in)`` as views into the slot tensors so the
        caller can chain into the conv / gated-RMSNorm without
        re-deriving them.
        """
        cfg = self.cfg
        T = x.shape[0]
        # Pre-condition: slot.lin_qkvz / slot.lin_ba are pre-allocated
        # in compute_dtype with the right (T, ...) shape. The matmuls
        # write directly into them via ``out=``. No casts: x is bf16,
        # weights are bf16, slots are bf16.
        if not (skip_already_saved and slot.has("lin_qkvz")):
            torch.mm(x, weights["w_lin_qkvz"], out=slot.lin_qkvz)
        if not (skip_already_saved and slot.has("lin_ba")):
            torch.mm(x, weights["w_lin_ba"], out=slot.lin_ba)
        # Zero-copy views into the saved tensors.
        _q, _k, _v, z = _split_qkvz_ft(slot.lin_qkvz, cfg)
        conv_in = slot.lin_qkvz[:, :cfg.conv_dim]
        return z, conv_in

    def _fwd_conv(
        self,
        conv_in: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> torch.Tensor:
        """Stage 2: depthwise causal conv1d + silu, fused.

        Uses FLA's ``causal_conv1d_fwd_kernel`` with ``ACTIVATION='silu'``
        — the kernel applies sigmoid(b_y)*b_y inside its epilogue, so we
        avoid a separate ~T*conv_dim bf16 silu pass over HBM.

        Returns ``post_conv`` (silu output, shape ``(T, conv_dim)``)
        allocated from scratch — no slot field for the conv output. The
        bwd path recomputes pre-silu via a conv with activation=None
        when it needs to silu_bwd.
        """
        cfg = self.cfg
        # FLA weight shape is (D, W); our slot weight is (D, 1, W) for
        # depthwise compatibility with torch.conv1d. Squeeze the middle.
        w = weights["w_lin_conv"].squeeze(1).contiguous()
        T = conv_in.shape[0]
        post_conv = ctx.scratch((T, cfg.conv_dim), cfg.compute_dtype)
        _fla_causal_conv1d_fwd_into(
            x_2d=conv_in,
            weight=w,
            out_2d=post_conv,
            activation="silu",
        )
        return post_conv

    def _fwd_qkv_heads(
        self, post_conv: torch.Tensor, ctx: LayerContext,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Stage 3: split post_conv → Q/K/V heads + l2norm on q/k.

        Allocates Q/K/rstds from scratch; v_h is a zero-copy view of
        post_conv. None of these are saved to slot — bwd recomputes
        them from saved ``lin_qkvz`` via the same conv + qkv_heads path
        (the conv runs in bwd anyway for silu_bwd's pre-silu input).

        Returns ``(q_n, k_n, v_h, q_rstd, k_rstd)`` for the FLA stage.

        **No GVA repeat_interleave**. FLA's
        ``chunk_gated_delta_rule_fwd_h`` / ``chunk_fwd_o`` kernels do the
        GVA index mapping themselves: each v-head ``i_h`` reads from
        k-head ``i_h // (HV // H)`` via raw pointer arithmetic
        (see fla/ops/common/chunk_o.py:76, chunk_delta_h.py). Passing
        un-expanded ``(T, num_k_heads, head_k_dim)`` q/k saves the
        ~2 GiB ``repeat_interleave`` materialization at
        T=131072 H=32 K=128."""
        cfg = self.cfg
        T = post_conv.shape[0]
        q_p, k_p, v_p = torch.split(
            post_conv,
            [cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1,
        )
        # Under the FT layout, post_conv is contiguous per the conv kernel
        # output, and q_p/k_p/v_p are contiguous slices: each .reshape
        # below is a free view (no copy).
        q_h = q_p.reshape(T, cfg.num_k_heads, cfg.head_k_dim)
        k_h = k_p.reshape(T, cfg.num_k_heads, cfg.head_k_dim)
        v_h = v_p.reshape(T, cfg.num_v_heads, cfg.head_v_dim)
        # Allocate l2norm outputs from scratch (Q/K post-l2norm + rstds).
        q_n = ctx.scratch(
            (T, cfg.num_k_heads, cfg.head_k_dim), cfg.compute_dtype,
        )
        k_n = ctx.scratch(
            (T, cfg.num_k_heads, cfg.head_k_dim), cfg.compute_dtype,
        )
        q_rstd = ctx.scratch((T, cfg.num_k_heads), torch.float32)
        k_rstd = ctx.scratch((T, cfg.num_k_heads), torch.float32)
        # Strided l2norm (q_h/k_h are non-contig slices of post_conv).
        from flextrain.ops import flextrain_l2norm_fwd_into
        flextrain_l2norm_fwd_into(q_h, q_n, q_rstd)
        flextrain_l2norm_fwd_into(k_h, k_n, k_rstd)
        return q_n, k_n, v_h, q_rstd, k_rstd

    def _fwd_beta(self, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Compute beta = sigmoid(b) in compute_dtype.

        After Stage D5 the gate ``g = -exp(A_log) * softplus(a+dt_bias)``
        + chunk_local_cumsum is fused inside FLA via
        ``use_gate_in_kernel=True``; we no longer compute g separately.
        Only beta still needs a one-line sigmoid pass.
        """
        return b.float().sigmoid().to(dtype)

    def _fwd_fla(
        self, q_n, k_n, v_h, a, beta, A_log, dt_bias, slot,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 5: FLA chunk-gated-delta-rule with fused gate (Stage D5).

        Passes ``use_gate_in_kernel=True`` so FLA fuses
        ``softplus(a + dt_bias) * (-exp(A_log))`` plus chunk_local_cumsum
        into one kernel (gdn_gate_chunk_cumsum_scalar_kernel). ``a`` is
        the raw projection output ``slot.lin_ba[:, HV:]``; FLA reads
        A_log / dt_bias and applies them inside the kernel. Replaces the
        previous python pipeline that called our flextrain_gate_prep_fwd
        kernel + FLA's standalone chunk_local_cumsum.

        Saves ``lin_core_out``, ``lin_A_int``, ``lin_g_post`` to slot.
        Returns ``core_out`` for the gated norm + out projection
        downstream.

        FLA's kernels read via hardcoded ``stride=(H*K, 1)`` block-ptr
        math; only v_h needs an explicit ``.contiguous()`` because its
        token stride is conv_dim (from post_conv slice), not HV*hv.
        """
        cfg = self.cfg
        bf = cfg.compute_dtype
        scale = cfg.head_k_dim ** -0.5

        g_post, o, A_int, _, _, _ = chunk_gated_delta_rule_fwd(
            q_n.unsqueeze(0),
            k_n.unsqueeze(0),
            v_h.unsqueeze(0).contiguous(),
            a.unsqueeze(0),
            beta.unsqueeze(0),
            scale=scale, initial_state=None,
            output_final_state=False, cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
        )
        o = o.squeeze(0)
        slot.lin_g_post.copy_(g_post.squeeze(0))
        slot.lin_A_int.copy_(A_int.squeeze(0).to(bf))
        slot.lin_core_out.copy_(o.to(bf))
        return o

    def _fwd_norm_out(
        self, core_out, z, weights: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Stage 6: gated RMSNorm + out projection. Returns ``y``
        (T, d_model). Doesn't save into the slot (the layer-level
        residual fold writes into the layer output)."""
        cfg = self.cfg
        T = core_out.shape[0]
        o_normed = self._gated_rmsnorm_fwd(
            core_out, z, weights["w_lin_norm"], cfg.rms_norm_eps,
        )
        return o_normed.reshape(T, cfg.value_dim) @ weights["w_lin_out"]

    def fwd(
        self,
        x: torch.Tensor,           # (T, d_model)
        weights: Mapping[str, torch.Tensor],
        slot,                      # ActivationSlot
        ctx: LayerContext,
        chunk: ChunkMeta | None = None,
    ) -> torch.Tensor:
        """Forward pass. Saves activations into ``slot``; returns
        the linear-attention output of shape ``(T, d_model)``.

        Composed from the per-stage helpers
        (``_fwd_proj_split`` → ``_fwd_conv`` → ``_fwd_qkv_heads`` →
        ``_fwd_gate_and_beta`` → ``_fwd_fla`` → ``_fwd_norm_out``).
        :meth:`fwd_recompute_*` re-runs only the missing stages.

        ``chunk`` carries per-sequence offsets (``q_seq_offsets``)
        which we forward to FLA as ``cu_seqlens`` so its recurrent
        state resets at packed-sequence boundaries inside the chunk.
        Optional only for callers that don't have a chunk handy
        (e.g. unit tests on a single sequence); production layers
        always pass it.
        """
        cu_seqlens = chunk.q_seq_offsets_i64 if chunk is not None else None
        chunk_indices = (
            chunk.fla_chunk_indices_64 if chunk is not None else None
        )
        z, conv_in = self._fwd_proj_split(x, weights, slot)
        post_conv = self._fwd_conv(conv_in, weights, ctx)
        q_n, k_n, v_h, _q_rstd, _k_rstd = self._fwd_qkv_heads(post_conv, ctx)
        b, a = _split_ba_ft(slot.lin_ba, self.cfg)
        beta = self._fwd_beta(b, self.cfg.compute_dtype)
        core_out = self._fwd_fla(
            q_n, k_n, v_h, a, beta,
            weights["w_lin_A_log"], weights["w_lin_dt_bias"], slot,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )
        return self._fwd_norm_out(core_out, z, weights)

    def _gated_rmsnorm_fwd(
        self,
        o: torch.Tensor,        # (T, n_v_heads, head_v_dim)
        z: torch.Tensor,        # (T, n_v_heads, head_v_dim)
        weight: torch.Tensor,   # (head_v_dim,)
        eps: float,
    ) -> torch.Tensor:
        """RMSNormGated forward: ``silu(z) * rmsnorm(o, w) * w``.

        Delegates to the fused :func:`flextrain_gated_rmsnorm_fwd`
        Triton kernel — keeps all per-(T, H, D) intermediates inside
        SRAM and only touches HBM once for input read and once for
        output write. Replaces ~12 unfused python ops that round-trip
        the largest per-token tensor in this block through HBM.

        On RTX 3090 this is ~10x faster than the python path (84% of
        peak HBM BW vs 9%) and bit-equivalent within bf16 noise.
        """
        from flextrain.ops import flextrain_gated_rmsnorm_fwd
        y, _rstd = flextrain_gated_rmsnorm_fwd(o, z, weight, eps)
        return y

    # ------------------------------------------------------------------
    # Forward-recompute helpers — mirror ``GQAAttentionBlock`` /
    # ``SwiGLUFFN`` partial-tier recomputes. Each method recomputes
    # only what's missing at the current save level.
    # ------------------------------------------------------------------

    def fwd_recompute_proj(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
    ) -> None:
        """Tier-3 recompute: re-run ``x @ W_qkvz`` and ``x @ W_ba`` to
        repopulate slot.lin_qkvz / slot.lin_ba when they weren't
        persisted by the save-level DP. ``skip_already_saved=True``
        short-circuits per-tensor when the slot already holds valid
        data from the original fwd's persist."""
        self._fwd_proj_split(x, weights, slot, skip_already_saved=True)

    def fwd_recompute_fla(
        self,
        q_n: torch.Tensor,
        k_n: torch.Tensor,
        v_h: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
        chunk: ChunkMeta | None = None,
    ) -> torch.Tensor:
        """Tier-2 recompute (FLA half): re-run FLA fwd from supplied
        Q/K/V (typically scratch-allocated via _fwd_qkv_heads on a
        recomputed post_conv) plus saved raw ``a`` (= slot.lin_ba[:, HV:])
        and weights ``w_lin_A_log`` / ``w_lin_dt_bias`` for FLA's fused
        gate path. Repopulates ``lin_core_out`` / ``lin_A_int`` /
        ``lin_g_post`` in slot.

        ``chunk`` is forwarded so the recompute uses the same
        ``cu_seqlens`` as the original fwd — otherwise saved and
        recomputed ``core_out`` would diverge across packed-seq
        boundaries inside the chunk."""
        b, a = _split_ba_ft(slot.lin_ba, self.cfg)
        beta = self._fwd_beta(b, self.cfg.compute_dtype)
        cu_seqlens = chunk.q_seq_offsets_i64 if chunk is not None else None
        chunk_indices = (
            chunk.fla_chunk_indices_64 if chunk is not None else None
        )
        return self._fwd_fla(
            q_n, k_n, v_h, a, beta,
            weights["w_lin_A_log"], weights["w_lin_dt_bias"], slot,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )

    def _gated_rmsnorm_bwd(
        self,
        do_normed: torch.Tensor,   # (T, n_v_heads, head_v_dim)
        o: torch.Tensor,           # (T, n_v_heads, head_v_dim) — saved core_out
        z: torch.Tensor,           # (T, n_v_heads, head_v_dim) — saved
        weight: torch.Tensor,      # (head_v_dim,)
        eps: float,
        *,
        dz_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward of ``o_normed = silu(z) * rmsnorm(o, w) * w``.

        Returns ``(do, dz, dw)``. ``dz_out`` (optional) lets the caller
        supply a pre-allocated buffer for dz so its writes can land
        directly in (e.g.) the Z slice of d_qkvz, avoiding a copy.

        Delegates to the fused :func:`flextrain_gated_rmsnorm_bwd`
        Triton kernel — keeps all per-(T, H, D) intermediates inside
        SRAM and only writes the three outputs back to HBM.

        Math (verified by ``tests/test_gated_rmsnorm_bwd.py`` against
        ``torch.autograd.grad`` on a pure-pytorch reference):

            d_normed = dy * silu(z) * w
            dot      = sum_d d_normed * o
            do       = rstd * d_normed - (rstd^3 / D) * o * dot
            dz       = dy * normed * w * silu'(z)
            dw[d]    = sum_{T,H} dy * silu(z) * normed[..., d]
        """
        from flextrain.ops import flextrain_gated_rmsnorm_bwd
        return flextrain_gated_rmsnorm_bwd(
            do_normed, o, z, weight, eps, dz_out=dz_out,
        )

    def _fla_autograd_fn(self, cu_seqlens: torch.Tensor | None = None):
        """A ``torch.autograd.Function`` wrapper around FLA's
        chunk-gated-delta-rule fwd/bwd. Used only inside :meth:`bwd`
        so we can run one autograd.backward over the local subgraph.

        Returns the class (not an instance — the class itself has
        ``apply``). ``cu_seqlens`` is captured by closure so the bwd
        sees the same packed-sequence boundaries as fwd.
        """
        cfg = self.cfg

        class _GatedDeltaRuleFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx_a, q_b, k_b, v_b, g_b, beta_b):
                # Direct FLA fwd; save what bwd needs.
                scale = cfg.head_k_dim ** -0.5
                g_chunk, o, A_int, _, _, _ = chunk_gated_delta_rule_fwd(
                    q_b, k_b, v_b, g_b, beta_b,
                    scale=scale, initial_state=None,
                    output_final_state=False, cu_seqlens=cu_seqlens,
                )
                ctx_a.save_for_backward(
                    q_b, k_b, v_b, g_b, beta_b, A_int, g_chunk,
                )
                return o

            @staticmethod
            def backward(ctx_a, do):
                q_b, k_b, v_b, g_b, beta_b, A_int, g_chunk = ctx_a.saved_tensors
                scale = cfg.head_k_dim ** -0.5
                dq, dk, dv, db, dg, _, _, _ = chunk_gated_delta_rule_bwd(
                    q=q_b, k=k_b, v=v_b, g=g_chunk, beta=beta_b, A=A_int,
                    scale=scale, initial_state=None,
                    do=do, dht=None, cu_seqlens=cu_seqlens,
                )
                return dq, dk, dv, dg, db

        return _GatedDeltaRuleFn

    def bwd(
        self,
        dy: torch.Tensor,            # (T, d_model) -- upstream grad
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
        chunk: ChunkMeta | None = None,
        *,
        skip_grads: frozenset[str] = frozenset(),
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Backward through the Gated DeltaNet block.

        **No forward recomputation.** All bwd math operates on saved
        activations:

        * ``slot.x_inp`` — input to the projections.
        * ``slot.lin_qkvz`` — full ``x @ W_qkvz`` output in FT
          ``[Q | K | V | Z]`` column-block layout. We view into it
          for ``conv_in`` (``[:, :conv_dim]``) and ``z``
          (``[:, 2*key_dim+value_dim:].view(T, HV, hv)``).
        * ``slot.lin_ba`` — full ``x @ W_ba`` output in FT ``[B | A]``
          layout. b/a are zero-copy slices.
        * (post-conv pre-silu is NOT saved; recomputed via conv-without-
          activation into scratch at the top of bwd. FLA's pattern.)
        * (lin_q / lin_k / lin_v are NOT saved; recomputed from saved
          lin_qkvz via conv → silu → split → l2norm into scratch at
          the top of bwd. Stage D2.5.)
        * (lin_g is no longer in the schema — Stage D5; FLA's
          gdn_gate_bwd recomputes it from raw a + A_log + dt_bias.)
        * ``slot.lin_core_out`` — FLA output before the gated norm.
        * ``slot.aux["lin_A_int"]`` — FLA's intra-chunk attention scratch.

        Sequence of bwd primitives (mirroring fwd in reverse):

        1. ``y = o_normed @ W_out``         → linear bwd → do_normed, dW_out
        2. gated-RMSNorm bwd (hand-rolled)  → do, dz, dw_norm
        3. ``chunk_gated_delta_rule_bwd``   → dq_h, dk_h, dv_h, dbeta, dg
        4. gate bwd (g, beta)               → da_via_g, db_via_beta, dA_log, ddt_bias
        5. GVA reverse repeat_interleave    → dq_p, dk_p, dv_p (pre-repeat)
        6. silu bwd                          → d_post_conv_pre_silu
        7. depthwise conv1d bwd             → d_conv_in, dW_conv
        8. cat reverse + reshapes           → dqkvz / dba parts
        9. linear bwd for ``x @ W_qkvz``    → dx_via_qkvz, dW_qkvz
        10. linear bwd for ``x @ W_ba``     → dx_via_ba, dW_ba
        11. ``return dx_via_qkvz + dx_via_ba``
        """
        cfg = self.cfg
        T = dy.shape[0]
        dtype = dy.dtype
        device = dy.device

        # Pull saved tensors. ``lin_qkvz`` and ``lin_ba`` are the full
        # matmul outputs in FT block-major column layout; we view into
        # them for the per-component pieces (zero-copy).
        x = slot.x_inp                                       # (T, d_model)
        b, a = _split_ba_ft(slot.lin_ba, cfg)                # both (T, num_v_heads)
        _q_pre, _k_pre, _v_pre, z = _split_qkvz_ft(slot.lin_qkvz, cfg)
        # z: (T, num_v_heads, head_v_dim) view of slot.lin_qkvz
        # Stage D5 dropped lin_g (pre-cumsum); raw a is in slot.lin_ba.
        core_out = slot.lin_core_out                         # (T, n_v_heads, head_v_dim)
        conv_in = slot.lin_qkvz[:, :cfg.conv_dim]            # (T, conv_dim) view
        # Q/K/V post-l2norm + rstds are no longer saved (Stage D2.5).
        # Recompute them here by re-running the conv (no activation,
        # giving us pre-silu for silu_bwd later) then silu+l2norm. The
        # conv was already mandatory in bwd since Stage D2 (FLA pattern).
        # The added work vs pre-D2.5 is one silu + two strided l2norms.
        post_conv_pre_silu = ctx.scratch(
            (T, cfg.conv_dim), cfg.compute_dtype,
        )
        _fla_causal_conv1d_fwd_into(
            x_2d=conv_in,
            weight=weights["w_lin_conv"].squeeze(1).contiguous(),
            out_2d=post_conv_pre_silu,
            activation=None,
        )
        # post_conv = silu(post_conv_pre_silu); allocate fresh scratch
        # since silu_bwd later needs pre-silu intact.
        post_conv = F.silu(post_conv_pre_silu)
        q_n, k_n, v_h, q_rstd, k_rstd = self._fwd_qkv_heads(post_conv, ctx)
        # FLA outputs from fwd. ``lin_A_int`` shape is
        # (T, n_v_heads, 64); add batch dim for FLA. lin_A_int is bf16
        # in the slot but FLA's bwd accepts bf16 directly.
        A_int = slot.lin_A_int.unsqueeze(0)                  # (1, T, H, 64)
        g_post = slot.lin_g_post.unsqueeze(0)                # (1, T, H), fp32

        W_out = weights["w_lin_out"]
        W_qkvz = weights["w_lin_qkvz"]
        W_ba = weights["w_lin_ba"]
        W_conv = weights["w_lin_conv"]
        W_norm = weights["w_lin_norm"]
        W_A_log = weights["w_lin_A_log"]
        W_dt_bias = weights["w_lin_dt_bias"]

        # 1. Out projection bwd: y = o_normed @ W_out
        # Need o_normed for dW_out. Cheap recompute: gated RMSNorm fwd
        # over saved core_out and z (no FLA, no conv).
        o_normed = self._gated_rmsnorm_fwd(
            core_out, z, W_norm, cfg.rms_norm_eps,
        )                                                    # (T, n_v_heads, head_v_dim)
        o_normed_2d = o_normed.reshape(T, cfg.value_dim)
        # dW_out = o_normed^T @ dy (skip-able: LoRA fast path).
        if "g_lin_out" in skip_grads:
            if capture_xy is not None:
                capture_xy["g_lin_out"] = (o_normed_2d, dy)
        else:
            g_w_out = grads.get("g_lin_out")
            if g_w_out is not None:
                # Fused bf16 @ bf16 -> fp32 accumulate via cuBLAS
                # ``addmm`` (alpha*A@B + beta*C, fp32 internal accum
                # for bf16 inputs). Avoids materializing fp32 copies of
                # ``o_normed_2d`` (T*value_dim*4 bytes) and ``dy`` —
                # these are large at hybrid linear+full backbones with
                # big chunks (e.g. ~6 GiB transient at chunk=131072 on
                # Qwen3.6-35B-A3B).
                torch.addmm(
                    g_w_out, o_normed_2d.T, dy,
                    alpha=1.0, beta=1.0, out=g_w_out,
                )
        # do_normed = dy @ W_out^T. Keep matmul in compute_dtype (bf16)
        # to avoid materializing a fp32 copy of the (potentially frozen)
        # weight matrix on every backward pass.
        do_normed_2d = (dy.to(dtype) @ W_out.T)              # (T, value_dim)
        do_normed = do_normed_2d.reshape(
            T, cfg.num_v_heads, cfg.head_v_dim,
        )

        # 2. Gated-RMSNorm bwd: o_normed = silu(z) * rmsnorm(core_out, W_norm) * W_norm.
        # Pre-allocate d_qkvz_2d here so dz_out can write directly into
        # its Z slice — avoids the dz portion of the final cat (~T*value_dim
        # bf16 = ~134 MiB at T=16k saved). Q/K/V slices are populated
        # later from FLA's conv_bwd output (still needs one D2D copy).
        d_qkvz_2d = ctx.scratch((T, cfg.proj_qkvz_dim), dtype)
        d_z_view = d_qkvz_2d[:, cfg.conv_dim:].view(
            T, cfg.num_v_heads, cfg.head_v_dim,
        )
        do, dz, dw_norm = self._gated_rmsnorm_bwd(
            do_normed, core_out, z, W_norm, cfg.rms_norm_eps,
            dz_out=d_z_view,
        )
        # dz IS d_z_view (same buffer); no need to use it again.
        g_w_norm = grads.get("g_lin_norm")
        if g_w_norm is not None:
            g_w_norm.add_(dw_norm.to(g_w_norm.dtype))

        # 3. FLA bwd. ``q_n``/``k_n`` are already post-l2norm and
        # per-K-head; FLA's chunk_gated_delta_rule_bwd accepts GVA
        # natively (q.shape[2]=H_k, v.shape[2]=H_v with H_v % H_k == 0;
        # the kernel reads each v-head from k-head ``i_h // (HV//H)``
        # via raw pointer arithmetic — see
        # fla/ops/common/chunk_o.py:76, chunk_delta_h.py). No reverse
        # GVA repeat_interleave needed; the kernel returns dq/dk
        # already shaped ``(B, T, n_k_heads, head_k_dim)``.
        # q_rstd / k_rstd come from _fwd_qkv_heads above (scratch).
        # Pass the POST-cumsum g (= g_post saved from fwd) so FLA's
        # internal state matches what it computed during fwd. FLA's bwd
        # applies a reverse-cumsum at the end so the returned ``dg`` is
        # in raw pre-cumsum g_input space — i.e. ∂L/∂(g_input).
        # FLA's kernels use hardcoded ``stride=(H*K, 1)`` block-ptr math
        # (e.g. fla/ops/common/chunk_o.py:86) so token stride MUST equal
        # H*K. q_n / k_n are scratch contiguous from _fwd_qkv_heads
        # (stride = H*hk), beta is allocated contiguous, do is from
        # gated_rmsnorm_bwd (torch.empty_like(o), contiguous). Only v_h
        # is strided (token stride = conv_dim from the post_conv slice)
        # so it alone needs .contiguous(). The stride[0]=0 from the
        # bare ``.unsqueeze(0)`` is benign because FLA's kernels compute
        # ``bos = i_b*T = 0`` for B=1.
        q_b = q_n.unsqueeze(0)
        k_b = k_n.unsqueeze(0)
        v_b = v_h.unsqueeze(0).contiguous()
        # Cache sig_b in fp32 — needed twice: once cast-to-dtype as
        # beta for FLA bwd, once for sigmoid bwd (db_via_beta below).
        sig_b = b.float().sigmoid()
        beta = sig_b.to(dtype)
        beta_b = beta.unsqueeze(0)
        do_b = do.unsqueeze(0)
        scale = cfg.head_k_dim ** -0.5
        cu_seqlens = chunk.q_seq_offsets_i64 if chunk is not None else None
        chunk_indices = (
            chunk.fla_chunk_indices_64 if chunk is not None else None
        )
        # Stage D5: use_gate_in_kernel=True — FLA fuses the gate fwd
        # (softplus + dt_bias + (-exp A_log)) and its bwd (gdn_gate_bwd)
        # inside the chunk-gated-delta-rule kernels. Pass g_input = raw a
        # (= slot.lin_ba[:, HV:]) and the gate weights; FLA returns dg
        # already in raw-a space plus dA_log / ddt_bias. Replaces ~10
        # python elementwise ops on (T, HV) tensors with one fused
        # kernel pair.
        a_b = a.unsqueeze(0)
        dq_n, dk_n, dv_h, dbeta, dg, _, dA_log_fla, ddt_bias_fla = (
            chunk_gated_delta_rule_bwd(
                q=q_b, k=k_b, v=v_b, g=g_post, beta=beta_b, A=A_int,
                scale=scale, initial_state=None, do=do_b, dht=None,
                cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                use_gate_in_kernel=True, g_input=a_b,
                A_log=W_A_log, dt_bias=W_dt_bias,
            )
        )
        dq_n = dq_n.squeeze(0)                               # (T, n_k_heads, head_k_dim)
        dk_n = dk_n.squeeze(0)
        dv_h = dv_h.squeeze(0)                               # (T, n_v_heads, head_v_dim)
        dbeta = dbeta.squeeze(0)                             # (T, n_v_heads)
        dg = dg.squeeze(0)                                   # (T, n_v_heads) — in raw-a space
        # 3b. Back-propagate through the l2 norm. Our strided-input
        # bwd kernel takes (T, H, D) views directly. dq_n / dk_n come
        # from FLA's bwd via q.new_empty(B, T, H, K) (chunk_o.py:737)
        # so they're contiguous after .squeeze(0); no .contiguous()
        # needed. We allocate dq_h/dk_h as views of d_post_conv's Q/K
        # slices so the bwd output lands directly in the conv-bwd input
        # layout — no separate cat-and-copy step downstream.
        from flextrain.ops import flextrain_l2norm_bwd_into
        d_post_conv = ctx.scratch((T, cfg.conv_dim), dtype)
        d_post_conv_q = d_post_conv[:, :cfg.key_dim].view(
            T, cfg.num_k_heads, cfg.head_k_dim,
        )
        d_post_conv_k = d_post_conv[:, cfg.key_dim:2 * cfg.key_dim].view(
            T, cfg.num_k_heads, cfg.head_k_dim,
        )
        flextrain_l2norm_bwd_into(q_n, q_rstd, dq_n, d_post_conv_q)
        flextrain_l2norm_bwd_into(k_n, k_rstd, dk_n, d_post_conv_k)
        # V slice: dv_h is contig (T, HV, hv) from FLA; copy into
        # d_post_conv's V slice via a strided view. One D2D pass over
        # T*value_dim bytes, vs the cat which copied all 3 of Q+K+V.
        d_post_conv[:, 2 * cfg.key_dim:].view(
            T, cfg.num_v_heads, cfg.head_v_dim,
        ).copy_(dv_h)

        # 4. Gate bwd.
        # beta = sigmoid(b) — sigmoid bwd: db = dbeta * sig_b * (1 - sig_b).
        # sig_b reused from beta computation above; no redundant pass.
        db_via_beta = (dbeta.float() * sig_b * (1.0 - sig_b)).to(dtype)
        # da_via_g comes directly from FLA's gdn_gate_bwd (called inside
        # chunk_gated_delta_rule_bwd because use_gate_in_kernel=True).
        # FLA returns dg already in raw-a space; just cast to dtype.
        da_via_g = dg.to(dtype)
        # dA_log / ddt_bias also come from FLA's gdn_gate_bwd.
        if grads.get("g_lin_A_log") is not None and dA_log_fla is not None:
            grads["g_lin_A_log"].add_(
                dA_log_fla.to(grads["g_lin_A_log"].dtype)
            )
        if grads.get("g_lin_dt_bias") is not None and ddt_bias_fla is not None:
            grads["g_lin_dt_bias"].add_(
                ddt_bias_fla.to(grads["g_lin_dt_bias"].dtype)
            )

        # 5. d_post_conv is already assembled in step 3b — l2norm_bwd
        # writes Q/K grads into views of d_post_conv directly, and we
        # copy dv_h into the V slice. The torch.cat the previous
        # implementation did (allocate (T, conv_dim) bf16 + copy 3
        # tensors into it) is gone; we paid one D2D pass over the
        # V slice instead of three over Q+K+V.

        # 6. silu bwd applied to post_conv_pre_silu via fused Triton
        # kernel. Computes d_post_conv_pre_silu = d_post_conv *
        # silu'(post_conv_pre_silu) without materializing the
        # ``sig_zs`` / ``dsilu`` intermediates that the python path
        # would (~3 GiB at T=32768 conv_dim=8192). All math is in fp32
        # in SRAM; output written directly to HBM in compute_dtype.
        from flextrain.ops import flextrain_silu_bwd
        d_post_conv_pre_silu = flextrain_silu_bwd(
            post_conv_pre_silu, d_post_conv,
        )

        # 7. Depthwise causal conv1d bwd via FLA's causal_conv1d_bwd
        # Triton kernel — drop-in replacement for the previous
        # torch.nn.grad.conv1d_input + conv1d_weight pair, which were
        # very slow at large T (aten convolution_backward fell off a
        # cliff at T>=32768 on RTX 3090, 86 ms at T=65536) and also
        # required materializing d_post_conv_btc + d_post_padded
        # transient buffers. FLA's bwd takes (x: (B,T,D), dy: (B,T,D),
        # weight: (D,W)) directly with no transpose churn and produces
        # dx + dw in one launch.
        from fla.modules.conv.triton.ops import causal_conv1d_bwd
        # Squeeze the depthwise-compat (D, 1, W) into FLA's (D, W).
        W_fla = W_conv.squeeze(1).contiguous()
        x_fla = conv_in.unsqueeze(0)                          # (1, T, D)
        dy_fla = d_post_conv_pre_silu.unsqueeze(0)            # (1, T, D)
        dx_fla, dw_fla, _db, _dr, _dh0 = causal_conv1d_bwd(
            x=x_fla, dy=dy_fla, dht=None,
            weight=W_fla, bias=None, residual=None,
            initial_state=None, activation=None,
        )
        # FLA's conv_bwd allocates dx via torch.empty_like; .squeeze(0)
        # is a contiguous view (no copy). Copy [Q|K|V] into d_qkvz_2d's
        # first conv_dim columns; Z slice was already written above by
        # _gated_rmsnorm_bwd. One D2D pass over (T, conv_dim) bf16 vs
        # the previous torch.cat which paid (T, proj_qkvz_dim) bf16
        # write — saves the dz portion (~T*value_dim bf16).
        d_qkvz_2d[:, :cfg.conv_dim].copy_(dx_fla.squeeze(0))

        if grads.get("g_lin_conv") is not None:
            # FLA returns dW with shape (D, W); reshape back to our
            # depthwise-compat (D, 1, W) and accumulate in grads dtype.
            grads["g_lin_conv"].add_(
                dw_fla.unsqueeze(1).to(grads["g_lin_conv"].dtype)
            )

        # 10. Assemble d_ba in the FT block-major layout ``[B | A]``.
        # db = db_via_beta (sigmoid bwd of b), da = da_via_g (gate bwd).
        d_ba_2d = torch.cat(
            [db_via_beta.reshape(T, cfg.num_v_heads),
             da_via_g.reshape(T, cfg.num_v_heads)],
            dim=-1,
        )                                                          # (T, proj_ba_dim)

        # 10. Linear bwd for x @ W_qkvz and x @ W_ba.
        # Wgrad addmms are skip-able (LoRA fast path); the dx accumulations
        # below always run (they're dgrad).
        x_2d_dt = x.reshape(T, cfg.d_model)  # native dtype copy for capture
        if "g_lin_qkvz" in skip_grads:
            if capture_xy is not None:
                capture_xy["g_lin_qkvz"] = (x_2d_dt, d_qkvz_2d)
        elif grads.get("g_lin_qkvz") is not None:
            # Fused bf16 @ bf16 -> fp32 accumulate. cuBLAS uses fp32
            # internal accumulators for bf16 tensor-core matmuls, so
            # the numeric result is identical to (x.float() @
            # d_qkvz.float()) but avoids materializing fp32 copies of
            # ``x_2d_dt`` (T*d_model*4) and ``d_qkvz_2d``
            # (T*proj_qkvz_dim*4). For Qwen3.6-35B-A3B at chunk=131072
            # that's ~7 GiB of transient fp32 per linear-attn layer
            # per chunk, eliminated.
            torch.addmm(
                grads["g_lin_qkvz"], x_2d_dt.T, d_qkvz_2d.to(x_2d_dt.dtype),
                alpha=1.0, beta=1.0, out=grads["g_lin_qkvz"],
            )
        if "g_lin_ba" in skip_grads:
            if capture_xy is not None:
                capture_xy["g_lin_ba"] = (x_2d_dt, d_ba_2d)
        elif grads.get("g_lin_ba") is not None:
            torch.addmm(
                grads["g_lin_ba"], x_2d_dt.T, d_ba_2d.to(x_2d_dt.dtype),
                alpha=1.0, beta=1.0, out=grads["g_lin_ba"],
            )
        # dx via base matmul. Keep in compute_dtype (typically bf16) to
        # avoid materializing a fp32 copy of the (frozen, big) weight
        # matrix on every backward pass — under LoRA that costs ~hidden
        # x proj_qkvz_dim x 4 bytes per layer in flight, which dwarfs
        # all other transient buffers.
        dx_via_qkvz = (d_qkvz_2d.to(dtype) @ W_qkvz.T)
        dx_via_ba = (d_ba_2d.to(dtype) @ W_ba.T)
        dx = (dx_via_qkvz + dx_via_ba).view_as(x)
        return dx

    # ------------------------------------------------------------------
    # FLOP estimate (very approximate).
    # ------------------------------------------------------------------

    def compute_cost(self, chunk: ChunkMeta, max_tier: int) -> ComputeCost:
        cfg = self.cfg
        T = chunk.total_q
        # Forward FLOP decomposition:
        #   proj      : x @ W_qkvz + x @ W_ba   (matmul, factor 2)
        #   conv      : depthwise conv1d         (mul + add, factor 2)
        #   fla       : chunk-gated-delta-rule  (rough constant factor)
        #   out_proj  : core_out @ W_out        (matmul, factor 2)
        # Out-proj and gated-RMSNorm are NOT in any recompute path — bwd
        # uses saved ``lin_core_out`` and saved ``lin_z`` directly to
        # compute their gradients without re-running the forward op,
        # so they don't appear in ``avoided``.
        proj = (
            2 * T * cfg.d_model * cfg.proj_qkvz_dim
            + 2 * T * cfg.d_model * cfg.proj_ba_dim
        )
        conv = 2 * T * cfg.conv_dim * cfg.conv_kernel_size
        fla = T * cfg.num_v_heads * cfg.head_k_dim * cfg.head_v_dim * 8
        out_proj = 2 * T * cfg.value_dim * cfg.d_model
        total = proj + conv + fla + out_proj

        # avoided_recompute_flops[L] = forward FLOPs we DON'T have to
        # rerun in bwd if we saved at tier L. Recompute paths:
        #   Tier 0/1 dropped: not allowed -- z, lin_q_rstd etc. are
        #     correctness-required, no recompute path exists.
        #   Tier 2 saved: ``fwd_recompute_fla`` skipped (FLA not redone).
        #   Tier 3 saved: also ``fwd_recompute_post_conv`` skipped
        #     (projections + conv1d + silu not redone).
        avoided = [0] * (max_tier + 1)
        if max_tier >= 2:
            avoided[2] = fla
        if max_tier >= 3:
            avoided[3] = fla + conv + proj
        # Make monotone non-decreasing for any tiers in between (the DP
        # solver requires this; tier 1 inherits tier 0's value).
        for t in range(1, max_tier + 1):
            if avoided[t] < avoided[t - 1]:
                avoided[t] = avoided[t - 1]
        return ComputeCost(
            total_fwd_flops=total,
            avoided_recompute_flops=tuple(avoided),
        )
