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

* Tier 0 — always saved (small ``(T, n_v_heads)`` scratches):
    ``lin_a`` (T, n_v_heads), ``lin_b`` (T, n_v_heads)        -- raw a/b
    ``lin_g`` / ``lin_g_post`` (T, n_v_heads) fp32           -- gate scalars
    ``lin_q_rstd`` / ``lin_k_rstd`` (T, n_v_heads) fp32       -- l2-norm rstds
* Tier 1 — recomputable via ``fwd_recompute_post_conv`` (re-runs the
  projection-split stage); saved by default to avoid the small
  recompute cost when host budget allows:
    ``lin_z`` (T, n_v_heads, head_v_dim)                      -- gated-RMSNorm gate
* Tier 2 — recomputable via ``fwd_recompute_fla`` from tier-3 + Q/K/V:
    ``lin_q`` (T, n_k_heads, head_k_dim) -- post-l2norm; per-K-head.
    ``lin_k`` (T, n_k_heads, head_k_dim) -- post-l2norm; per-K-head.
    ``lin_v`` (T, n_v_heads, head_v_dim)
    ``lin_A_int``                                             -- FLA scratch
    ``lin_core_out``                                          -- FLA output
* Tier 3 — recomputable via ``fwd_recompute_post_conv`` from x_inp:
    ``lin_conv_in``                                           -- pre-conv qkv concat
    ``lin_post_conv_pre_silu``                                -- conv output (pre-silu)
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
) -> None:
    """Direct call into FLA's ``causal_conv1d_fwd_kernel`` writing into
    a caller-supplied output buffer.

    The upstream ``causal_conv1d_fwd`` python helper allocates output
    via ``torch.empty_like(x)`` and returns it; for our use case we
    want the kernel to write directly into ``slot.lin_post_conv_pre_silu``
    so the post-conv ``slot.copy_(...)`` D2D memcpy can be eliminated.

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
        ACTIVATION=None,
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
            ActivationField(
                "lin_g",
                lambda n, d: (n, cfg.num_v_heads),
                torch.float32, tier=0,
            ),
            ActivationField(
                "lin_g_post",
                lambda n, d: (n, cfg.num_v_heads),
                torch.float32, tier=0,
            ),
            # L2-norm reciprocal-stddev per (token, k_head). HF's linear
            # attention uses ``use_qk_l2norm_in_kernel=True`` which wraps
            # FLA's recurrence with ``l2norm_fwd``+``l2norm_bwd``; the
            # lower-level kernel we drive directly doesn't, so we run
            # l2 norm explicitly. We save only the rstd (small, fp32)
            # and recompute the normalized q/k in bwd from the saved
            # un-normalized q_h/k_h.
            ActivationField(
                "lin_q_rstd",
                lambda n, d: (n, cfg.num_k_heads),
                torch.float32, tier=0,
            ),
            ActivationField(
                "lin_k_rstd",
                lambda n, d: (n, cfg.num_k_heads),
                torch.float32, tier=0,
            ),
            # ==================================================
            # Tier 2: post-conv Q/K/V, FLA scratch, FLA core output.
            # All recomputable from saved qkvz via the projection /
            # conv / l2norm stages.
            # ==================================================
            ActivationField(
                "lin_q",
                lambda n, d: (n, cfg.num_k_heads, cfg.head_k_dim),
                bf, tier=2,
            ),
            ActivationField(
                "lin_k",
                lambda n, d: (n, cfg.num_k_heads, cfg.head_k_dim),
                bf, tier=2,
            ),
            ActivationField(
                "lin_v",
                lambda n, d: (n, cfg.num_v_heads, cfg.head_v_dim),
                bf, tier=2,
            ),
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
            ActivationField(
                "lin_post_conv_pre_silu",
                lambda n, d: (n, cfg.conv_dim),
                bf, tier=3,
            ),
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
        slot,
        *,
        skip_already_saved: bool = False,
    ) -> torch.Tensor:
        """Stage 2: depthwise causal conv1d. Saves ``lin_post_conv_pre_silu``.

        Uses FLA's ``causal_conv1d_fwd`` Triton kernel (~16x faster
        than torch's ``F.conv1d`` at our sizes, e.g. 20.8 ms -> 1.27
        ms at T=32768 conv_dim=8192 on RTX 3090). FLA expects
        ``x: (B, T, D)`` and ``weight: (D, W)``, matching our (T, D)
        layout after a single ``unsqueeze(0)`` — no transpose needed.

        We pass ``activation=None`` here (we still need to save the
        pre-silu intermediate for tier-3 recompute compatibility), then
        apply silu separately. The activation fusion in FLA is ~5us at
        these sizes — negligible vs the ~1ms conv body — so leaving it
        out has no perf cost.

        Returns ``post_conv`` (silu output, shape ``(T, conv_dim)``).

        We call FLA's ``causal_conv1d_fwd_kernel`` directly through a
        thin wrapper (:func:`_fla_causal_conv1d_fwd_into`) instead of
        the python ``causal_conv1d_fwd`` helper so we can supply
        ``slot.lin_post_conv_pre_silu`` as the output buffer — saving
        a (T, conv_dim) bf16 D2D memcpy per layer per fwd. (FLA's
        helper does ``y = torch.empty_like(x)`` internally, then we'd
        have to ``slot.copy_(y)`` after.)
        """
        cfg = self.cfg
        # FLA weight shape is (D, W); our slot weight is (D, 1, W) for
        # depthwise compatibility with torch.conv1d. Squeeze the middle.
        w = weights["w_lin_conv"].squeeze(1).contiguous()
        # cu_seqlens for varlen (None when no chunk metadata available).
        # Currently the only caller of _fwd_conv is the fwd path which
        # invokes us without chunk; the fla bwd recovers cu_seqlens
        # via its own re-fwd call site so we don't need to pass here.
        # Production callers thread chunk through _fwd_fla; conv runs
        # the same depthwise op regardless.
        if not (skip_already_saved and slot.has("lin_post_conv_pre_silu")):
            _fla_causal_conv1d_fwd_into(
                x_2d=conv_in,
                weight=w,
                out_2d=slot.lin_post_conv_pre_silu,
            )
        # F.silu over a contiguous slot tensor returns a fresh contiguous
        # output; the trailing .contiguous() that used to be here was a
        # no-op pass-through.
        return F.silu(slot.lin_post_conv_pre_silu)

    def _fwd_qkv_heads(self, post_conv: torch.Tensor, slot) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor,
    ]:
        """Stage 3: split post_conv → Q/K/V heads + l2norm on q/k.

        Saves tier-2 ``lin_q`` / ``lin_k`` (post-l2norm, per-K-head shape
        ``(T, num_k_heads, head_k_dim)``) and ``lin_v``
        (``(T, num_v_heads, head_v_dim)``), plus tier-0 ``lin_q_rstd`` /
        ``lin_k_rstd`` (per-(t, k_head) reciprocal-stddev needed by
        ``l2norm_bwd`` in the bwd path).

        **No GVA repeat_interleave**. FLA's
        ``chunk_gated_delta_rule_fwd_h`` / ``chunk_fwd_o`` kernels do the
        GVA index mapping themselves: each v-head ``i_h`` reads from
        k-head ``i_h // (HV // H)`` via raw pointer arithmetic
        (see fla/ops/common/chunk_o.py:76, chunk_delta_h.py). Passing
        un-expanded ``(T, num_k_heads, head_k_dim)`` q/k saves the
        ~2 GiB ``repeat_interleave`` materialization at
        T=131072 H=32 K=128.

        Saving the post-l2norm q/k (instead of the un-normalized values
        and recomputing in bwd) eliminates a 2 GiB fp32 transient
        (the ``q_h.float() * q_rstd.float()`` promotion in the old bwd)
        and one redundant elementwise pass per layer per chunk.

        Returns ``(q_n, k_n, v_h, q_rstd, k_rstd)`` for the FLA stage."""
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
        # L2 norm per-(token, k_head) row, written directly into slot
        # tensors. Our flextrain_l2norm_fwd_into accepts strided 3D
        # input via runtime token/head strides, so q_h / k_h (which are
        # strided slices of the conv output: stride(0)=conv_dim,
        # stride(1)=head_k_dim, stride(2)=1) feed in zero-copy. The
        # FLA wrapper required contiguous (T*H, D) input and forced a
        # ~T*key_dim bf16 D2D copy per call before this change.
        from flextrain.ops import flextrain_l2norm_fwd_into
        flextrain_l2norm_fwd_into(q_h, slot.lin_q, slot.lin_q_rstd)
        flextrain_l2norm_fwd_into(k_h, slot.lin_k, slot.lin_k_rstd)
        # v has no l2norm; it just needs to land in slot.lin_v. v_h is
        # a free view of post_conv (contiguous slice), but slot.lin_v
        # is its own buffer — we still need the copy here. Cheap (one
        # contig-to-contig D2D pass over T*value_dim bf16).
        slot.lin_v.copy_(v_h)
        # Return slot views so the rest of the fwd avoids re-deriving.
        return (
            slot.lin_q, slot.lin_k, v_h,
            slot.lin_q_rstd, slot.lin_k_rstd,
        )

    def _fwd_gate_and_beta(
        self,
        a: torch.Tensor,                       # (T, n_v_heads) bf16
        b: torch.Tensor,                       # (T, n_v_heads) bf16
        weights: Mapping[str, torch.Tensor],
        slot,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 4: produce ``g`` (fp32) and ``beta`` (bf16) for FLA.

        Math:
            g[t, h]    = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h])
            beta[t, h] = sigmoid(b[t, h])

        Single fused Triton kernel replaces ~9 elementwise launches in
        the python pipeline. Saves ``lin_g`` to the slot.
        """
        from flextrain.ops import flextrain_gate_prep_fwd
        # Write g directly into slot.lin_g; beta is a transient (not
        # saved separately — bwd recomputes via gate_prep on saved b).
        g, beta = flextrain_gate_prep_fwd(
            a, b, weights["w_lin_A_log"], weights["w_lin_dt_bias"],
            g_out=slot.lin_g,
        )
        return g, beta

    def _fwd_fla(
        self, q_n, k_n, v_h, g, beta, slot,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 5: FLA chunk-gated-delta-rule. Saves ``lin_core_out``,
        ``lin_A_int``, ``lin_g_post``. Returns ``core_out`` for the
        gated norm + out projection downstream.

        ``q_n``/``k_n`` are already l2-normalized (done in
        ``_fwd_qkv_heads``); ``beta`` is precomputed by
        ``_fwd_gate_and_beta``. The l2norm rstds are saved into the
        slot during ``_fwd_qkv_heads`` for the bwd's ``l2norm_bwd``.

        ``cu_seqlens`` (shape ``[N+1]``, int64) tells FLA where each
        packed sequence ends inside this chunk's flattened token axis;
        without it, FLA would let the recurrent state leak across
        sequence boundaries (treats the whole chunk as one sequence).

        ``chunk_indices`` (shape ``(num_64_chunks, 2)``, int64) is FLA's
        per-(seq, intra-seq-chunk) lookup table at chunk_size=64. When
        passed, FLA skips its internal ``prepare_chunk_indices`` call
        whose ``.tolist()`` is a D->H sync; we precompute host-side in
        ``ChunkMeta.build``.
        """
        cfg = self.cfg
        bf = cfg.compute_dtype
        scale = cfg.head_k_dim ** -0.5

        g_post, o, A_int, _, _, _ = chunk_gated_delta_rule_fwd(
            q_n.unsqueeze(0).contiguous(),
            k_n.unsqueeze(0).contiguous(),
            v_h.unsqueeze(0).contiguous(),
            g.unsqueeze(0).contiguous(),
            beta.unsqueeze(0).contiguous(),
            scale=scale, initial_state=None,
            output_final_state=False, cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
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
        post_conv = self._fwd_conv(conv_in, weights, slot)
        q_n, k_n, v_h, _q_rstd, _k_rstd = self._fwd_qkv_heads(post_conv, slot)
        b, a = _split_ba_ft(slot.lin_ba, self.cfg)
        g, beta = self._fwd_gate_and_beta(a, b, weights, slot)
        core_out = self._fwd_fla(
            q_n, k_n, v_h, g, beta, slot,
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

    def fwd_recompute_post_conv(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
    ) -> torch.Tensor:
        """Tier-3 recompute: re-run ``x @ W_qkvz`` (and ``x @ W_ba``)
        plus the conv stage when the saved ``lin_qkvz`` /
        ``lin_post_conv_pre_silu`` weren't persisted by the save-level
        DP. ``slot.has("lin_qkvz")`` / ``slot.has("lin_ba")`` short-
        circuit when the slot already holds valid data from the original
        fwd's persist; same for ``lin_post_conv_pre_silu``.

        Returns the post-silu ``post_conv`` tensor (T, conv_dim) for
        the caller to feed into stage 3."""
        _z, conv_in = self._fwd_proj_split(
            x, weights, slot, skip_already_saved=True,
        )
        return self._fwd_conv(conv_in, weights, slot, skip_already_saved=True)

    def fwd_recompute_qkv_heads(self, slot) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Tier-2 recompute: re-derive Q/K/V from saved
        ``lin_post_conv_pre_silu``. Cheaper than tier-3 (no conv1d
        re-run, just silu + reshape + repeat-interleave).

        Returns ``(q_h, k_h, v_h)``."""
        cfg = self.cfg
        # Reapply silu to saved pre-silu.
        post_conv = F.silu(slot.lin_post_conv_pre_silu.float()).to(
            slot.lin_post_conv_pre_silu.dtype
        )
        return self._fwd_qkv_heads(post_conv, slot)

    def fwd_recompute_fla(
        self,
        weights: Mapping[str, torch.Tensor],
        slot,
        chunk: ChunkMeta | None = None,
    ) -> torch.Tensor:
        """Tier-2 recompute (FLA half): re-run FLA fwd from saved
        Q/K/V/g/b to repopulate ``lin_core_out`` / ``lin_A_int`` /
        ``lin_g_post``. ``lin_g`` (raw) and ``lin_ba`` (b/a slots)
        must already be present (tier-0 — always saved).

        Beta is recomputed from saved ``b`` (= ``slot.lin_ba[:, :HV]``)
        via the fused gate-prep kernel (which also recomputes g; we
        discard the recomputed g since slot.lin_g is already valid).

        ``chunk`` is forwarded to ``_fwd_fla`` so the recompute uses
        the same ``cu_seqlens`` as the original fwd — otherwise saved
        and recomputed ``core_out`` would diverge across packed-seq
        boundaries inside the chunk."""
        from flextrain.ops import flextrain_gate_prep_fwd
        b, a = _split_ba_ft(slot.lin_ba, self.cfg)
        _g, beta = flextrain_gate_prep_fwd(
            a, b,
            weights["w_lin_A_log"], weights["w_lin_dt_bias"],
        )
        cu_seqlens = chunk.q_seq_offsets_i64 if chunk is not None else None
        chunk_indices = (
            chunk.fla_chunk_indices_64 if chunk is not None else None
        )
        return self._fwd_fla(
            slot.lin_q, slot.lin_k, slot.lin_v,
            slot.lin_g, beta, slot,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )

    def _gated_rmsnorm_bwd(
        self,
        do_normed: torch.Tensor,   # (T, n_v_heads, head_v_dim)
        o: torch.Tensor,           # (T, n_v_heads, head_v_dim) — saved core_out
        z: torch.Tensor,           # (T, n_v_heads, head_v_dim) — saved
        weight: torch.Tensor,      # (head_v_dim,)
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward of ``o_normed = silu(z) * rmsnorm(o, w) * w``.

        Returns ``(do, dz, dw)``. Delegates to the fused
        :func:`flextrain_gated_rmsnorm_bwd` Triton kernel — keeps all
        per-(T, H, D) intermediates inside SRAM and only writes the
        three outputs back to HBM, avoiding the ~10 fp32 (T, H, D)
        intermediates the python path materialized.

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
            do_normed, o, z, weight, eps,
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
        * ``slot.lin_post_conv_pre_silu`` — conv1d output (silu input).
        * ``slot.lin_q``, ``lin_k``, ``lin_v`` — post-conv-silu Q/K/V
          (the inputs to FLA).
        * ``slot.lin_g`` — pre-cumsum gate.
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

        # Pull saved tensors. ``lin_q`` / ``lin_k`` are POST-l2norm and
        # per-K-head (shape ``(T, num_k_heads, head_k_dim)``) — see
        # ``_fwd_qkv_heads``. ``lin_qkvz`` and ``lin_ba`` are the full
        # matmul outputs in FT block-major column layout; we view into
        # them for the per-component pieces (zero-copy).
        x = slot.x_inp                                       # (T, d_model)
        b, a = _split_ba_ft(slot.lin_ba, cfg)                # both (T, num_v_heads)
        _q_pre, _k_pre, _v_pre, z = _split_qkvz_ft(slot.lin_qkvz, cfg)
        # z: (T, num_v_heads, head_v_dim) view of slot.lin_qkvz
        q_n = slot.lin_q                                     # (T, n_k_heads, head_k_dim) post-l2norm
        k_n = slot.lin_k                                     # (T, n_k_heads, head_k_dim) post-l2norm
        v_h = slot.lin_v                                     # (T, n_v_heads, head_v_dim)
        g = slot.lin_g                                       # (T, n_v_heads) fp32
        core_out = slot.lin_core_out                         # (T, n_v_heads, head_v_dim)
        conv_in = slot.lin_qkvz[:, :cfg.conv_dim]            # (T, conv_dim) view
        post_conv_pre_silu = slot.lin_post_conv_pre_silu     # (T, conv_dim)
        # FLA outputs from fwd. ``lin_A_int`` shape is
        # (T, n_v_heads, 64); add batch dim for FLA. lin_A_int is bf16
        # in the slot but FLA's bwd accepts bf16 directly.
        A_int = slot.lin_A_int.unsqueeze(0)                  # (1, T, H, 64)
        g_post = slot.lin_g_post.unsqueeze(0).contiguous()   # (1, T, H), fp32

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
        do, dz, dw_norm = self._gated_rmsnorm_bwd(
            do_normed, core_out, z, W_norm, cfg.rms_norm_eps,
        )
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
        q_rstd = slot.lin_q_rstd                             # (T, n_k_heads)
        k_rstd = slot.lin_k_rstd
        # Pass the POST-cumsum g (= g_post saved from fwd) so FLA's
        # internal state matches what it computed during fwd. FLA's bwd
        # applies a reverse-cumsum at the end so the returned ``dg`` is
        # in raw pre-cumsum g_input space — i.e. ∂L/∂(g_input).
        # Force ``.contiguous()`` on every kernel input -- FLA's kernels
        # use raw pointer arithmetic and silently produce wrong results
        # on non-contiguous strides.
        q_b = q_n.unsqueeze(0).contiguous()
        k_b = k_n.unsqueeze(0).contiguous()
        v_b = v_h.unsqueeze(0).contiguous()
        beta = b.float().sigmoid().to(dtype)
        beta_b = beta.unsqueeze(0).contiguous()
        do_b = do.unsqueeze(0).contiguous()
        scale = cfg.head_k_dim ** -0.5
        cu_seqlens = chunk.q_seq_offsets_i64 if chunk is not None else None
        chunk_indices = (
            chunk.fla_chunk_indices_64 if chunk is not None else None
        )
        dq_n, dk_n, dv_h, dbeta, dg, _, _, _ = chunk_gated_delta_rule_bwd(
            q=q_b, k=k_b, v=v_b, g=g_post, beta=beta_b, A=A_int,
            scale=scale, initial_state=None, do=do_b, dht=None,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )
        dq_n = dq_n.squeeze(0)                               # (T, n_k_heads, head_k_dim)
        dk_n = dk_n.squeeze(0)
        dv_h = dv_h.squeeze(0)                               # (T, n_v_heads, head_v_dim)
        dbeta = dbeta.squeeze(0)                             # (T, n_v_heads)
        dg = dg.squeeze(0)                                   # (T, n_v_heads)
        # 3b. Back-propagate through the l2 norm. Our strided-input
        # bwd kernel takes (T, H, D) views directly so the saved q_n /
        # k_n / q_rstd / k_rstd (slot tensors, contiguous) and the
        # dq_n / dk_n FLA outputs (squeezed views, last-axis-contig)
        # all feed in zero-copy.
        from flextrain.ops import flextrain_l2norm_bwd_into
        dq_h = torch.empty_like(q_n)
        dk_h = torch.empty_like(k_n)
        flextrain_l2norm_bwd_into(q_n, q_rstd, dq_n.contiguous(), dq_h)
        flextrain_l2norm_bwd_into(k_n, k_rstd, dk_n.contiguous(), dk_h)

        # 4. Gate bwd.
        # beta = sigmoid(b). dbeta -> db_via_beta.
        sig_b = b.float().sigmoid()
        db_via_beta = (dbeta.float() * sig_b * (1.0 - sig_b)).to(dtype)
        # Add dz contribution to slot positions corresponding to the gated z
        # (handled in step 7 below — z came from qkvz split).
        # g = -exp(A_log) * softplus(a + dt_bias).
        # dg/d(A_log)  = -exp(A_log) * softplus(...) = g (per-head, summed over T).
        # dg/d(a + dt_bias) = -exp(A_log) * sigmoid(a + dt_bias).
        a_f32 = a.float()
        A_log_f32 = W_A_log.float()
        dt_bias_f32 = W_dt_bias.float()
        sig_apdt = (a_f32 + dt_bias_f32).sigmoid()
        neg_exp_A = -A_log_f32.exp()                         # (n_v_heads,)
        # da via g: dg * (-exp(A_log)) * sigmoid(a + dt_bias)
        da_via_g = (dg.float() * neg_exp_A.unsqueeze(0) * sig_apdt).to(dtype)
        # dA_log = sum_t (g * dg)  (since g_t = -exp(A_log) * softplus(...) = g)
        # Actually: dg/d A_log = g  (chain rule with g as the per-t value).
        # So dA_log = sum_t (dg * g).
        d_A_log = (dg.float() * g.float()).sum(dim=0)         # (n_v_heads,)
        # ddt_bias = sum_t (-exp(A_log)) * sigmoid(a+dt_bias) * dg
        d_dt_bias = (
            dg.float() * neg_exp_A.unsqueeze(0) * sig_apdt
        ).sum(dim=0)                                          # (n_v_heads,)
        if grads.get("g_lin_A_log") is not None:
            grads["g_lin_A_log"].add_(d_A_log.to(grads["g_lin_A_log"].dtype))
        if grads.get("g_lin_dt_bias") is not None:
            grads["g_lin_dt_bias"].add_(d_dt_bias.to(grads["g_lin_dt_bias"].dtype))

        # 5. dq_h / dk_h are already per-K-head (no reverse expand
        # needed — FLA's GVA-aware bwd returned them already-collapsed).
        # Flatten to (T, key_dim) / (T, key_dim) / (T, value_dim).
        d_q_p = dq_h.reshape(T, cfg.key_dim)
        d_k_p = dk_h.reshape(T, cfg.key_dim)
        d_v_p = dv_h.reshape(T, cfg.value_dim)
        # cat reverse → d_post_conv (T, conv_dim).
        d_post_conv = torch.cat([d_q_p, d_k_p, d_v_p], dim=-1)

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
        d_conv_in = dx_fla.squeeze(0).contiguous()           # (T, conv_dim)

        if grads.get("g_lin_conv") is not None:
            # FLA returns dW with shape (D, W); reshape back to our
            # depthwise-compat (D, 1, W) and accumulate in grads dtype.
            grads["g_lin_conv"].add_(
                dw_fla.unsqueeze(1).to(grads["g_lin_conv"].dtype)
            )

        # 8. Split d_conv_in into d_q_flat / d_k_flat / d_v_flat. Under
        # the FT layout these are contiguous slices that match the
        # ``[Q | K | V]`` blocks of qkvz at the front of the column axis;
        # no reshape-copies needed.
        d_q_flat, d_k_flat, d_v_flat = torch.split(
            d_conv_in,
            [cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1,
        )

        # 9. Assemble d_qkvz in the FT block-major column layout
        # ``[Q | K | V | Z]``. d_z (gated-rmsnorm bwd output) becomes
        # the Z block. All four pieces are already in the right
        # head-major layout per FT convention.
        # NB: dz comes from ``_gated_rmsnorm_bwd`` with shape
        # (T, num_v_heads, head_v_dim); reshape to (T, value_dim).
        d_z_flat = dz.reshape(T, cfg.value_dim).to(dtype)
        d_qkvz_2d = torch.cat(
            [d_q_flat, d_k_flat, d_v_flat, d_z_flat], dim=-1,
        )                                                          # (T, proj_qkvz_dim)

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
