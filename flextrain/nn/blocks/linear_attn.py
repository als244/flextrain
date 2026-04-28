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

* Tier 0 — always saved (correctness-required, no recompute path):
    ``lin_a`` (T, n_v_heads), ``lin_b`` (T, n_v_heads)        -- raw a/b
    ``lin_g`` / ``lin_g_post`` (T, n_v_heads) fp32           -- gate scalars
    ``lin_q_rstd`` / ``lin_k_rstd`` (T, n_v_heads) fp32       -- l2-norm rstds
    ``lin_z`` (T, n_v_heads, head_v_dim)                      -- gated-RMSNorm gate
* Tier 2 — recomputable via ``fwd_recompute_fla`` from tier-3 + Q/K/V:
    ``lin_q``, ``lin_k``, ``lin_v`` (post-GVA, post-l2norm-input)
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
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Qwen3NextLinearAttention requires `flash-linear-attention`. "
        "Install with: pip install flash-linear-attention"
    ) from e


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
    """
    T = qkvz.shape[0]
    H = cfg.num_k_heads
    HV = cfg.num_v_heads
    hk = cfg.head_k_dim
    hv = cfg.head_v_dim
    # The qkvz row layout is interleaved per-K-head with v-grouping.
    qkvz = qkvz.view(T, H, 2 * hk + 2 * (HV // H) * hv)
    parts = [hk, hk, (HV // H) * hv, (HV // H) * hv]
    q, k, v_grp, z_grp = torch.split(qkvz, parts, dim=-1)
    # q, k: (T, H, hk).  v, z: (T, H * (HV/H), hv) = (T, HV, hv) after reshape.
    v = v_grp.reshape(T, HV, hv)
    z = z_grp.reshape(T, HV, hv)
    return q, k, v, z


def _split_ba(ba: torch.Tensor, cfg: GatedDeltaNetConfig):
    T = ba.shape[0]
    H = cfg.num_k_heads
    HV = cfg.num_v_heads
    grp = HV // H
    ba = ba.view(T, H, 2 * grp)
    b, a = torch.split(ba, [grp, grp], dim=-1)
    b = b.reshape(T, HV)
    a = a.reshape(T, HV)
    return b, a


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
            # Tier 0: small-and-cheap.  Per FT convention these
            # are always saved; they're the "context needed for
            # update-forward-context" or recompute-helper data
            # (norm rstds, gate scalars). Sub-megabyte for any
            # realistic T.
            # ==================================================
            ActivationField(
                "lin_a",
                lambda n, d: (n, cfg.num_v_heads),
                bf, tier=0,
            ),
            ActivationField(
                "lin_b",
                lambda n, d: (n, cfg.num_v_heads),
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
            # After GVA repeat_interleave, q_h / k_h have shape
            # ``(T, num_v_heads, head_k_dim)`` so l2norm_fwd produces
            # rstds of shape ``(T, num_v_heads)``. (For Qwen3.5-2B with
            # grp=1 this happens to equal num_k_heads; for 9B with
            # grp=2 it's num_v_heads.)
            ActivationField(
                "lin_q_rstd",
                lambda n, d: (n, cfg.num_v_heads),
                torch.float32, tier=0,
            ),
            ActivationField(
                "lin_k_rstd",
                lambda n, d: (n, cfg.num_v_heads),
                torch.float32, tier=0,
            ),
            # Gated-RMSNorm gate. ``z`` flows through gated-RMSNorm
            # bwd directly and there is no recompute path for it
            # (it's a separate projection ``x @ W_qkvz`` slice that
            # we'd need to redo, but the recompute helpers don't
            # produce z). So tier 0 — required-for-correctness, not
            # an optional save.
            ActivationField(
                "lin_z",
                lambda n, d: (n, cfg.num_v_heads, cfg.head_v_dim),
                bf, tier=0,
            ),
            # ==================================================
            # Tier 2: post-conv Q/K/V, FLA scratch, FLA core output.
            # All recomputable from x_inp via the block fwd.
            # ==================================================
            ActivationField(
                "lin_q",
                lambda n, d: (n, cfg.num_v_heads, cfg.head_k_dim),
                bf, tier=2,
            ),
            ActivationField(
                "lin_k",
                lambda n, d: (n, cfg.num_v_heads, cfg.head_k_dim),
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
            # Tier 3: largest fields, recomputable.
            # ``conv_in`` and ``post_conv_pre_silu`` each have
            # shape (T, conv_dim) where conv_dim = 2*key_dim +
            # value_dim — the largest activations in this block.
            # ==================================================
            ActivationField(
                "lin_conv_in",
                lambda n, d: (n, cfg.conv_dim),
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: x @ W_qkvz, x @ W_ba, splits, and cat → conv_in.

        Saves tier-0 ``lin_a``, ``lin_b`` and tier-1 ``lin_z``;
        saves tier-3 ``lin_conv_in``.

        Returns ``(z, conv_in)`` (z and conv_in are also re-derivable
        from saved fields, but returning saves a re-cast).
        """
        cfg = self.cfg
        bf = cfg.compute_dtype
        T = x.shape[0]
        qkvz = x @ weights["w_lin_qkvz"]
        ba = x @ weights["w_lin_ba"]
        q_pre, k_pre, v_pre, z = _split_qkvz(qkvz, cfg)
        b, a = _split_ba(ba, cfg)
        q_flat = q_pre.reshape(T, cfg.key_dim)
        k_flat = k_pre.reshape(T, cfg.key_dim)
        v_flat = v_pre.reshape(T, cfg.value_dim)
        conv_in = torch.cat([q_flat, k_flat, v_flat], dim=-1)  # (T, conv_dim)
        slot.lin_conv_in.copy_(conv_in.to(bf))
        slot.lin_a.copy_(a.to(bf))
        slot.lin_b.copy_(b.to(bf))
        slot.lin_z.copy_(z.to(bf))
        return z, conv_in

    def _fwd_conv(
        self,
        conv_in: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
    ) -> torch.Tensor:
        """Stage 2: depthwise causal conv1d. Saves ``lin_post_conv_pre_silu``.

        Returns ``post_conv`` (silu output, shape ``(T, conv_dim)``)."""
        cfg = self.cfg
        bf = cfg.compute_dtype
        T = conv_in.shape[0]
        K = cfg.conv_kernel_size
        conv_x = conv_in.transpose(0, 1).unsqueeze(0)  # (1, conv_dim, T)
        post_conv_pre_silu = F.conv1d(
            conv_x, weights["w_lin_conv"], bias=None,
            padding=K - 1, groups=cfg.conv_dim,
        )[..., :T]
        slot.lin_post_conv_pre_silu.copy_(
            post_conv_pre_silu.squeeze(0).transpose(0, 1).contiguous().to(bf)
        )
        post_conv = F.silu(post_conv_pre_silu)
        return post_conv.squeeze(0).transpose(0, 1).contiguous()

    def _fwd_qkv_heads(self, post_conv: torch.Tensor, slot) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Stage 3: split post_conv → Q/K/V heads + repeat-interleave for GVA.

        Saves tier-2 ``lin_q``, ``lin_k``, ``lin_v``."""
        cfg = self.cfg
        bf = cfg.compute_dtype
        T = post_conv.shape[0]
        q_p, k_p, v_p = torch.split(
            post_conv,
            [cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1,
        )
        q_h = q_p.reshape(T, cfg.num_k_heads, cfg.head_k_dim)
        k_h = k_p.reshape(T, cfg.num_k_heads, cfg.head_k_dim)
        v_h = v_p.reshape(T, cfg.num_v_heads, cfg.head_v_dim)
        if cfg.num_v_heads // cfg.num_k_heads > 1:
            rep = cfg.num_v_heads // cfg.num_k_heads
            q_h = q_h.repeat_interleave(rep, dim=1)
            k_h = k_h.repeat_interleave(rep, dim=1)
        slot.lin_q.copy_(q_h.to(bf))
        slot.lin_k.copy_(k_h.to(bf))
        slot.lin_v.copy_(v_h.to(bf))
        return q_h, k_h, v_h

    def _fwd_gate(
        self, a: torch.Tensor, weights: Mapping[str, torch.Tensor], slot,
    ) -> torch.Tensor:
        """Stage 4: g = -exp(A_log) * softplus(a + dt_bias). Saves ``lin_g``.

        Returns the raw pre-cumsum ``g`` (fp32)."""
        a_f32 = a.float()
        A_log = weights["w_lin_A_log"].float()
        dt_bias = weights["w_lin_dt_bias"].float()
        g = -A_log.exp() * F.softplus(a_f32 + dt_bias)
        slot.lin_g.copy_(g.detach().clone())
        return g

    def _fwd_fla(
        self, q_h, k_h, v_h, g, b, slot,
    ) -> torch.Tensor:
        """Stage 5: FLA chunk-gated-delta-rule. Saves ``lin_core_out``,
        ``lin_A_int``, ``lin_g_post``. Returns ``core_out`` for the
        gated norm + out projection downstream.

        L2-normalizes q/k per-(token, k_head) before the recurrence to
        match HF's ``use_qk_l2norm_in_kernel=True``. ``lin_q_rstd`` /
        ``lin_k_rstd`` are saved (tier 0) so bwd can apply the
        complementary ``l2norm_bwd`` to the gradients FLA returns.
        """
        cfg = self.cfg
        bf = cfg.compute_dtype
        beta = b.float().sigmoid().to(bf)
        scale = cfg.head_k_dim ** -0.5

        # L2 norm on q/k. l2norm_fwd treats the last dim as the
        # normalization axis; q_h / k_h have shape (T, num_k_heads,
        # head_k_dim), so per-(t, h) row of length head_k_dim is
        # normalized as expected. l2norm_fwd internally calls
        # ``.view(-1, last_dim)`` which requires contiguous input.
        # FLA's chunk kernel reads with elementwise pointer arithmetic and
        # silently produces wrong results on non-contiguous inputs.
        # ``torch.split`` + ``.reshape`` upstream can yield non-contiguous
        # views, so force contiguity on every kernel input here.
        q_n, q_rstd = l2norm_fwd(q_h.contiguous())
        k_n, k_rstd = l2norm_fwd(k_h.contiguous())

        g_post, o, A_int, _, _, _ = chunk_gated_delta_rule_fwd(
            q_n.unsqueeze(0).contiguous(),
            k_n.unsqueeze(0).contiguous(),
            v_h.unsqueeze(0).contiguous(),
            g.unsqueeze(0).contiguous(),
            beta.unsqueeze(0).contiguous(),
            scale=scale, initial_state=None,
            output_final_state=False, cu_seqlens=None,
        )
        o = o.squeeze(0)
        slot.lin_g_post.copy_(g_post.squeeze(0))
        slot.lin_A_int.copy_(A_int.squeeze(0).to(bf))
        slot.lin_core_out.copy_(o.to(bf))
        # Save rstds (tier 0). Bwd will recompute q_n / k_n from the
        # saved un-normalized q_h / k_h (in slot.lin_q / slot.lin_k)
        # and these rstds. q_rstd has shape (T, num_k_heads); make
        # contiguous for the slot copy.
        slot.lin_q_rstd.copy_(q_rstd.contiguous())
        slot.lin_k_rstd.copy_(k_rstd.contiguous())
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
    ) -> torch.Tensor:
        """Forward pass. Saves activations into ``slot``; returns
        the linear-attention output of shape ``(T, d_model)``.

        Composed from the per-stage helpers
        (``_fwd_proj_split`` → ``_fwd_conv`` → ``_fwd_qkv_heads`` →
        ``_fwd_gate`` → ``_fwd_fla`` → ``_fwd_norm_out``).
        :meth:`fwd_recompute_*` re-runs only the missing stages.
        """
        z, conv_in = self._fwd_proj_split(x, weights, slot)
        post_conv = self._fwd_conv(conv_in, weights, slot)
        q_h, k_h, v_h = self._fwd_qkv_heads(post_conv, slot)
        a = slot.lin_a
        b = slot.lin_b
        g = self._fwd_gate(a, weights, slot)
        core_out = self._fwd_fla(q_h, k_h, v_h, g, b, slot)
        return self._fwd_norm_out(core_out, z, weights)

    def _gated_rmsnorm_fwd(
        self,
        o: torch.Tensor,        # (T, n_v_heads, head_v_dim)
        z: torch.Tensor,        # (T, n_v_heads, head_v_dim)
        weight: torch.Tensor,   # (head_v_dim,)
        eps: float,
    ) -> torch.Tensor:
        """Reference impl of FLA's RMSNormGated: silu(z) * rmsnorm(o) * w."""
        # PyTorch is fine here — small tensors.
        o_f = o.float()
        rms = (o_f * o_f).mean(dim=-1, keepdim=True).add_(eps).rsqrt_()
        normed = (o_f * rms).to(o.dtype)
        return normed * weight * F.silu(z.float()).to(o.dtype)

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
        """Tier-3 recompute: ``lin_conv_in`` and ``lin_post_conv_pre_silu``
        weren't saved, so re-run projections + conv from ``x``.
        Tier-0 ``lin_a/b`` and tier-1 ``lin_z`` are also re-derived
        and copied (cheap; same matmul that produces conv_in).

        Returns the post-silu ``post_conv`` tensor (T, conv_dim) for
        the caller to feed into stage 3."""
        _z, conv_in = self._fwd_proj_split(x, weights, slot)
        return self._fwd_conv(conv_in, weights, slot)

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
    ) -> torch.Tensor:
        """Tier-2 recompute (FLA half): re-run FLA fwd from saved
        Q/K/V/g/b to repopulate ``lin_core_out`` / ``lin_A_int`` /
        ``lin_g_post``. ``lin_g`` (raw) and ``lin_a`` / ``lin_b``
        must already be present (tier-0 — always saved)."""
        return self._fwd_fla(
            slot.lin_q, slot.lin_k, slot.lin_v,
            slot.lin_g, slot.lin_b, slot,
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

        Returns ``(do, dz, dw)``. Pure tensor math, no autograd. Used by
        :meth:`bwd` to avoid re-running the gated-norm forward.
        """
        D = o.shape[-1]
        o_f = o.float()
        z_f = z.float()
        do_f = do_normed.float()
        w_f = weight.float()

        # Forward intermediates (cheap, no kernels involved).
        rms_sqr = (o_f * o_f).mean(dim=-1, keepdim=True).add_(eps)
        rstd = rms_sqr.rsqrt()
        normed = o_f * rstd                           # rmsnorm(o)
        sig_z = z_f.sigmoid()
        silu_z = z_f * sig_z                          # silu(z)
        # o_normed = silu(z) * normed * w (broadcast over heads).
        # We have do_normed; split through the silu(z) gate first.
        d_silu_z = do_f * (normed * w_f)              # (T, H, D)
        d_normed_w = do_f * silu_z                    # (T, H, D)
        # silu(z) bwd: silu' = sigmoid(z) * (1 + z*(1-sigmoid(z)))
        dsig = sig_z * (1.0 + z_f * (1.0 - sig_z))
        dz_f = d_silu_z * dsig
        # rmsnorm w bwd: o_normed_pre_silu = w * normed.
        # d_normed_w = do_f * silu(z); we then split:
        #   dw += sum_{T,H} (d_normed_w * normed)
        #   d_normed = d_normed_w * w
        dw_per_dim = (d_normed_w * normed).sum(dim=(0, 1))     # (head_v_dim,)
        d_normed = d_normed_w * w_f                            # (T, H, D)
        # rmsnorm fwd: normed = o * rstd.  o_f -> normed.
        # d/do  normed_i = rstd * δ_ij - rstd^3 * o_i * o_j / D
        # so do = rstd * d_normed - (rstd^3 / D) * o * sum(d_normed * o)
        dot = (d_normed * o_f).sum(dim=-1, keepdim=True)
        rstd3 = rstd * rstd * rstd
        do_f_out = rstd * d_normed - (rstd3 / D) * o_f * dot
        return do_f_out.to(o.dtype), dz_f.to(z.dtype), dw_per_dim.to(weight.dtype)

    def _fla_autograd_fn(self):
        """A ``torch.autograd.Function`` wrapper around FLA's
        chunk-gated-delta-rule fwd/bwd. Used only inside :meth:`bwd`
        so we can run one autograd.backward over the local subgraph.

        Returns the class (not an instance — the class itself has
        ``apply``).
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
                    output_final_state=False, cu_seqlens=None,
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
                    do=do, dht=None, cu_seqlens=None,
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
        *,
        skip_grads: frozenset[str] = frozenset(),
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Backward through the Gated DeltaNet block.

        **No forward recomputation.** All bwd math operates on saved
        activations:

        * ``slot.x_inp`` — input to the projections.
        * ``slot.lin_conv_in`` — pre-conv qkv concat.
        * ``slot.lin_post_conv_pre_silu`` — conv1d output (silu input).
        * ``slot.lin_q``, ``lin_k``, ``lin_v`` — post-conv-silu Q/K/V
          (the inputs to FLA).
        * ``slot.lin_g``, ``slot.lin_a``, ``slot.lin_b`` — pre-cumsum
          gate, raw a/b scalars.
        * ``slot.lin_z`` — gate vector for the gated RMSNorm.
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

        # Pull saved tensors.
        x = slot.x_inp                                       # (T, d_model)
        a = slot.lin_a                                       # (T, n_v_heads)
        b = slot.lin_b                                       # (T, n_v_heads)
        z = slot.lin_z                                       # (T, n_v_heads, head_v_dim)
        q_h = slot.lin_q                                     # (T, n_v_heads, head_k_dim)
        k_h = slot.lin_k                                     # (T, n_v_heads, head_k_dim)
        v_h = slot.lin_v                                     # (T, n_v_heads, head_v_dim)
        g = slot.lin_g                                       # (T, n_v_heads) fp32
        core_out = slot.lin_core_out                         # (T, n_v_heads, head_v_dim)
        conv_in = slot.lin_conv_in                           # (T, conv_dim)
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
                g_w_out.add_(
                    (o_normed_2d.T.float() @ dy.float()).to(g_w_out.dtype)
                )
        # do_normed = dy @ W_out^T
        do_normed_2d = dy.float() @ W_out.float().T          # (T, value_dim)
        do_normed = do_normed_2d.to(dtype).reshape(
            T, cfg.num_v_heads, cfg.head_v_dim,
        )

        # 2. Gated-RMSNorm bwd: o_normed = silu(z) * rmsnorm(core_out, W_norm) * W_norm.
        do, dz, dw_norm = self._gated_rmsnorm_bwd(
            do_normed, core_out, z, W_norm, cfg.rms_norm_eps,
        )
        g_w_norm = grads.get("g_lin_norm")
        if g_w_norm is not None:
            g_w_norm.add_(dw_norm.to(g_w_norm.dtype))

        # 3. FLA bwd. Fwd l2-normalized q/k before feeding to the FLA
        # kernel; bwd has to consume the SAME (post-l2-norm) q_n / k_n
        # as the kernel saw and return gradients in that space, which we
        # then back-prop through l2norm_bwd to get ∂L/∂q_h / ∂L/∂k_h.
        # Recompute q_n / k_n from saved un-normalized q_h / k_h plus
        # saved rstds (cheap: one elementwise mul per element).
        q_rstd = slot.lin_q_rstd
        k_rstd = slot.lin_k_rstd
        q_n = (q_h.float() * q_rstd.float().unsqueeze(-1)).to(dtype).contiguous()
        k_n = (k_h.float() * k_rstd.float().unsqueeze(-1)).to(dtype).contiguous()
        # Pass the POST-cumsum g (= g_post saved from fwd) so FLA's
        # internal state matches what it computed during fwd. FLA's bwd
        # applies a reverse-cumsum at the end so the returned ``dg`` is
        # in raw pre-cumsum g_input space — i.e. ∂L/∂(g_input).
        # Force ``.contiguous()`` on every kernel input -- FLA's kernels
        # use raw pointer arithmetic and silently produce wrong results
        # on non-contiguous strides (same issue as the fwd path).
        q_b = q_n.unsqueeze(0).contiguous()
        k_b = k_n.unsqueeze(0).contiguous()
        v_b = v_h.unsqueeze(0).contiguous()
        beta = b.float().sigmoid().to(dtype)
        beta_b = beta.unsqueeze(0).contiguous()
        do_b = do.unsqueeze(0).contiguous()
        scale = cfg.head_k_dim ** -0.5
        dq_n, dk_n, dv_h, dbeta, dg, _, _, _ = chunk_gated_delta_rule_bwd(
            q=q_b, k=k_b, v=v_b, g=g_post, beta=beta_b, A=A_int,
            scale=scale, initial_state=None, do=do_b, dht=None,
            cu_seqlens=None,
        )
        dq_n = dq_n.squeeze(0)                               # (T, n_v_heads, head_k_dim)
        dk_n = dk_n.squeeze(0)
        dv_h = dv_h.squeeze(0)
        dbeta = dbeta.squeeze(0)                             # (T, n_v_heads)
        dg = dg.squeeze(0)                                   # (T, n_v_heads)
        # 3b. Back-propagate through the l2 norm: dq_h = l2norm_bwd(q_n,
        # q_rstd, dq_n). l2norm_bwd takes the *normalized* tensor (y),
        # the saved rstd, and the upstream gradient (dy), and returns
        # the gradient in input-space.
        dq_h = l2norm_bwd(
            q_n.contiguous(), q_rstd.contiguous(), dq_n.contiguous(),
            eps=1e-6,
        )
        dk_h = l2norm_bwd(
            k_n.contiguous(), k_rstd.contiguous(), dk_n.contiguous(),
            eps=1e-6,
        )

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

        # 5. GVA reverse repeat_interleave for q_h, k_h: pre-repeat had
        # cfg.num_k_heads heads; we repeated each by `rep`.
        rep = cfg.num_v_heads // cfg.num_k_heads
        if rep > 1:
            # Sum over each group of `rep` heads.
            dq_pre = dq_h.reshape(T, cfg.num_k_heads, rep, cfg.head_k_dim).sum(dim=2)
            dk_pre = dk_h.reshape(T, cfg.num_k_heads, rep, cfg.head_k_dim).sum(dim=2)
        else:
            dq_pre = dq_h
            dk_pre = dk_h
        dv_pre = dv_h                                          # already n_v_heads

        # Flatten back to (T, key_dim) / (T, key_dim) / (T, value_dim).
        d_q_p = dq_pre.reshape(T, cfg.key_dim)
        d_k_p = dk_pre.reshape(T, cfg.key_dim)
        d_v_p = dv_pre.reshape(T, cfg.value_dim)
        # cat reverse → d_post_conv (T, conv_dim).
        d_post_conv = torch.cat([d_q_p, d_k_p, d_v_p], dim=-1)

        # 6. silu bwd applied to post_conv_pre_silu.
        # silu' = sigmoid(z) * (1 + z*(1-sigmoid(z)))
        z_silu = post_conv_pre_silu.float()
        sig_zs = z_silu.sigmoid()
        dsilu = sig_zs * (1.0 + z_silu * (1.0 - sig_zs))
        d_post_conv_pre_silu = (d_post_conv.float() * dsilu).to(dtype)

        # 7. Depthwise conv1d bwd.  fwd: post_conv = conv1d(conv_in.T)[..., :T].
        # We pad on the left by K-1 then truncate to T at the right (causal).
        # dW_conv via grad_input formulation; d_conv_in via convolve grad-output
        # with weight. PyTorch's grad helpers do this:
        K = cfg.conv_kernel_size
        # We need d_post_conv_pre_silu in (1, conv_dim, T) layout.
        d_post_TC = d_post_conv_pre_silu                      # (T, conv_dim)
        d_post_conv_btc = d_post_TC.transpose(0, 1).unsqueeze(0)  # (1, conv_dim, T)
        # Pad d_post on the right by (K-1) to mirror fwd's left-pad-then-truncate.
        d_post_padded = F.pad(d_post_conv_btc, (0, K - 1))     # (1, conv_dim, T+K-1)

        # dW_conv: depthwise conv1d input-gradient form = conv(conv_in_padded, d_post)
        # Easiest: use torch.nn.grad helpers.
        conv_in_btc = conv_in.transpose(0, 1).unsqueeze(0)     # (1, conv_dim, T)
        dW_conv = torch.nn.grad.conv1d_weight(
            input=conv_in_btc,
            weight_size=W_conv.shape,
            grad_output=d_post_padded,
            stride=1, padding=K - 1, dilation=1,
            groups=cfg.conv_dim,
        )
        if grads.get("g_lin_conv") is not None:
            grads["g_lin_conv"].add_(dW_conv.to(grads["g_lin_conv"].dtype))

        # d_conv_in: conv1d transpose with weight.
        d_conv_in_btc = torch.nn.grad.conv1d_input(
            input_size=conv_in_btc.shape,
            weight=W_conv,
            grad_output=d_post_padded,
            stride=1, padding=K - 1, dilation=1,
            groups=cfg.conv_dim,
        )
        d_conv_in = d_conv_in_btc.squeeze(0).transpose(0, 1).contiguous()  # (T, conv_dim)

        # 8. cat → d_q_flat, d_k_flat, d_v_flat (each pre-conv).
        d_q_flat, d_k_flat, d_v_flat = torch.split(
            d_conv_in,
            [cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1,
        )
        # Pre-conv q_pre/k_pre were (T, n_k_heads, head_k_dim); v_pre was
        # (T, n_v_heads, head_v_dim). Reshape grads back.
        d_q_pre = d_q_flat.reshape(T, cfg.num_k_heads, cfg.head_k_dim)
        d_k_pre = d_k_flat.reshape(T, cfg.num_k_heads, cfg.head_k_dim)
        d_v_pre = d_v_flat.reshape(T, cfg.num_v_heads, cfg.head_v_dim)

        # 9. Reverse the qkvz / ba splits to assemble dqkvz, dba.
        # qkvz layout: per-K-head, [q (head_k_dim), k (head_k_dim),
        #                            v_grp ((HV/H)*hv), z_grp ((HV/H)*hv)].
        H = cfg.num_k_heads
        HV = cfg.num_v_heads
        grp = HV // H
        # v_pre and z came from (T, H, grp*hv) before reshape to (T, HV, hv).
        d_v_grp = d_v_pre.reshape(T, H, grp * cfg.head_v_dim)
        d_z_grp = dz.reshape(T, H, grp * cfg.head_v_dim).to(dtype)
        # Assemble dqkvz per-K-head row.
        d_qkvz = torch.cat(
            [d_q_pre, d_k_pre, d_v_grp, d_z_grp], dim=-1,
        )                                                          # (T, H, qkvz_per_head)
        d_qkvz_2d = d_qkvz.reshape(T, cfg.proj_qkvz_dim)

        # ba layout: per-K-head, [b (grp), a (grp)].
        # db = db_via_beta (sigmoid bwd of b), da = da_via_g (gate bwd).
        d_b_grp = db_via_beta.reshape(T, H, grp)
        d_a_grp = da_via_g.reshape(T, H, grp)
        d_ba = torch.cat([d_b_grp, d_a_grp], dim=-1)                # (T, H, 2*grp)
        d_ba_2d = d_ba.reshape(T, cfg.proj_ba_dim)

        # 10. Linear bwd for x @ W_qkvz and x @ W_ba.
        # Wgrad addmms are skip-able (LoRA fast path); the dx accumulations
        # below always run (they're dgrad).
        x_2d_f = x.reshape(T, cfg.d_model).float()
        x_2d_dt = x.reshape(T, cfg.d_model)  # native dtype copy for capture
        if "g_lin_qkvz" in skip_grads:
            if capture_xy is not None:
                capture_xy["g_lin_qkvz"] = (x_2d_dt, d_qkvz_2d)
        elif grads.get("g_lin_qkvz") is not None:
            grads["g_lin_qkvz"].add_(
                (x_2d_f.T @ d_qkvz_2d.float())
                .to(grads["g_lin_qkvz"].dtype)
            )
        if "g_lin_ba" in skip_grads:
            if capture_xy is not None:
                capture_xy["g_lin_ba"] = (x_2d_dt, d_ba_2d)
        elif grads.get("g_lin_ba") is not None:
            grads["g_lin_ba"].add_(
                (x_2d_f.T @ d_ba_2d.float())
                .to(grads["g_lin_ba"].dtype)
            )
        dx_via_qkvz = (d_qkvz_2d.float() @ W_qkvz.float().T).to(dtype)
        dx_via_ba = (d_ba_2d.float() @ W_ba.float().T).to(dtype)
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
