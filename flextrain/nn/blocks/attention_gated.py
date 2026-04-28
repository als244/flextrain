"""Gated GQA attention block (Qwen3-Next / Qwen3.5 / Qwen3.6 full-attention variant).

Standalone block — does NOT subclass ``GQAAttentionBlock``. Shares
helpers via plain function calls (``apply_rope_fwd/bwd``,
``flextrain_attention_fwd/bwd``, ``dispatcher.matmul``).

The only structural differences vs :class:`GQAAttentionBlock`:

1. ``w_q`` projects ``(d_model → attn_dim * 2)``. The output is split
   into ``(query, gate)`` halves along the last dim. ``query`` flows
   through QK-norm + RoPE + flash-attn as usual; ``gate`` is the
   per-head pre-sigmoid gating signal saved for the bwd chain rule.

2. After flash-attn fwd, the output is multiplied element-wise by
   ``sigmoid(gate)`` BEFORE the ``w_o`` projection: ``xo = (attn_result *
   sigmoid(gate)) @ w_o + x_resid``. The gate is broadcast across each
   token's full ``attn_dim`` axis (i.e., per-head-element, not per-head).

Used by Qwen3-Next, Qwen3.5, Qwen3.6 full-attention layers (HF
``Qwen3NextAttention``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import ActivationField
from flextrain.core.layer import (
    ChunkMeta,
    ComputeCost,
    LayerContext,
    ParamSpec,
    TensorSpec,
)
from flextrain.ops import (
    flextrain_attention_bwd,
    flextrain_attention_fwd,
    dispatcher,
)

from .attention import GQAAttentionConfig
from .rope import (
    apply_rope_bwd,
    apply_rope_fwd,
    apply_rope_partial_bwd,
    apply_rope_partial_fwd,
    build_partial_rope_inv_freq,
    build_rope_inv_freq,
)


@dataclass(frozen=True)
class GQAAttentionGatedConfig(GQAAttentionConfig):
    """Config for :class:`GQAAttentionGatedBlock`.

    Adds ``partial_rotary_factor`` (Qwen3-Next/3.5/3.6 default 0.25):
    fraction of ``head_dim`` that receives RoPE; the remaining channels
    pass through unrotated. ``1.0`` (default) reproduces full RoPE.
    """

    partial_rotary_factor: float = 1.0


class GQAAttentionGatedBlock:
    """GQA attention with a sigmoid-gated output (Qwen3-Next / 3.5 / 3.6).

    Pipeline::

        qproj   = attn_norm_output @ w_q              # (T, attn_dim*2)
        Q, gate = chunk(qproj, 2, dim=-1)             # (T, attn_dim) each
        Q       = qk_norm(Q)         (if cfg.qk_norm)
        Q       = rope(Q)            (in-place)
        flash-attn fwd → attn_result                 # (T, n_heads, head_dim)
        gated   = attn_result * sigmoid(gate)         # (T, attn_dim)
        xo      = gated @ w_o + x_resid               (fused)

    Activation fields owned (matches :class:`GQAAttentionBlock` plus
    ``attn_gate``):

    * ``xk``         tier 0  — pre-norm K projection (per-chunk)
    * ``xv``         tier 0  — V projection (per-chunk)
    * ``attn_result`` tier 1 — flash-attn fwd output
    * ``softmax_lse`` tier 1
    * ``xq``         tier 2  — Q (post-norm/RoPE)
    * ``attn_gate``  tier 2  — pre-sigmoid gate (saved for bwd)
    * ``xo``         tier 2  — final output (residual-folded)
    """

    def __init__(self, cfg: GQAAttentionConfig) -> None:
        # Accept either GQAAttentionConfig or GQAAttentionGatedConfig.
        # The output gate is implicit from the block class.
        self.cfg = cfg
        self._rope_inv_freq_cache: torch.Tensor | None = None
        # QK-norm hooks (Qwen3 / Qwen3-Next per-head). Attached by the
        # enclosing layer when cfg.qk_norm=True.
        self.q_norm = None
        self.k_norm = None

    def set_qk_norm(self, q_norm, k_norm) -> None:
        if not self.cfg.qk_norm:
            raise ValueError(
                "GQAAttentionGatedBlock: cfg.qk_norm must be True to attach QK-norm"
            )
        self.q_norm = q_norm
        self.k_norm = k_norm

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def fields(self) -> tuple[ActivationField, ...]:
        cfg = self.cfg
        bf = cfg.compute_dtype
        return (
            ActivationField(
                "xk",
                lambda n, d: (n, cfg.n_kv_heads, cfg.head_dim),
                bf, tier=0,
            ),
            ActivationField(
                "xv",
                lambda n, d: (n, cfg.n_kv_heads, cfg.head_dim),
                bf, tier=0,
            ),
            ActivationField(
                "attn_result",
                lambda n, d: (n, cfg.n_heads, cfg.head_dim),
                bf, tier=1,
            ),
            ActivationField(
                "softmax_lse",
                lambda n, d: (cfg.n_heads, n),
                torch.float32, tier=1, token_axis=1,
            ),
            ActivationField(
                "xq",
                lambda n, d: (n, cfg.n_heads, cfg.head_dim),
                bf, tier=2,
            ),
            ActivationField(
                "attn_gate",
                lambda n, d: (n, cfg.attn_dim),
                bf, tier=2,
            ),
            ActivationField(
                "xo",
                lambda n, d: (n, cfg.d_model),
                bf, tier=2,
            ),
        )

    def param_spec(self) -> ParamSpec:
        """``w_q`` has DOUBLED out-dim (Q + gate concat). Other tensors
        match :class:`GQAAttentionBlock`."""
        cfg = self.cfg
        tensors: list[TensorSpec] = [
            TensorSpec(
                "w_q",
                lambda d: (cfg.d_model, cfg.attn_dim * 2),  # ← doubled
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            TensorSpec(
                "w_k",
                lambda d: (cfg.d_model, cfg.kv_dim),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            TensorSpec(
                "w_v",
                lambda d: (cfg.d_model, cfg.kv_dim),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            TensorSpec(
                "w_o",
                lambda d: (cfg.attn_dim, cfg.d_model),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
        ]
        if cfg.qkv_bias:
            tensors.extend([
                TensorSpec(
                    "b_q",
                    lambda d: (cfg.attn_dim * 2,),  # ← matches doubled w_q
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "b_k",
                    lambda d: (cfg.kv_dim,),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "b_v",
                    lambda d: (cfg.kv_dim,),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
            ])
        return ParamSpec(tensors=tuple(tensors))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _rot_dim(self) -> int:
        """Number of head channels that receive RoPE (rest pass through).
        Equal to ``head_dim`` for full rotary, ``head_dim/4`` for
        Qwen3-Next/3.5/3.6 partial rotary (factor 0.25)."""
        prf = getattr(self.cfg, "partial_rotary_factor", 1.0)
        rot_dim = int(self.cfg.head_dim * prf)
        # Must be even (RoPE pairs channels).
        if rot_dim % 2 != 0:
            raise ValueError(
                f"GQAAttentionGatedBlock: partial_rotary_factor={prf} × "
                f"head_dim={self.cfg.head_dim} = {rot_dim}, must be even."
            )
        return rot_dim

    @property
    def _is_partial_rotary(self) -> bool:
        return self._rot_dim != self.cfg.head_dim

    def _rope_theta(self, device: torch.device) -> torch.Tensor:
        if (
            self._rope_inv_freq_cache is None
            or self._rope_inv_freq_cache.device != device
        ):
            if self._is_partial_rotary:
                inv_freq_cpu = build_partial_rope_inv_freq(
                    rot_dim=self._rot_dim,
                    rope_base=self.cfg.rope_base,
                    rope_scaling=self.cfg.rope_scaling,
                )
            else:
                inv_freq_cpu = build_rope_inv_freq(
                    head_dim=self.cfg.head_dim,
                    rope_base=self.cfg.rope_base,
                    rope_scaling=self.cfg.rope_scaling,
                )
            self._rope_inv_freq_cache = inv_freq_cpu.to(
                device=device, dtype=torch.float32,
            )
        return self._rope_inv_freq_cache

    def _rope_fwd(self, tensors, seq_positions, rope_theta):
        if self._is_partial_rotary:
            return apply_rope_partial_fwd(
                tensors, seq_positions, rope_theta, self._rot_dim,
            )
        return apply_rope_fwd(tensors, seq_positions, rope_theta)

    def _rope_bwd(self, grad_tensors, seq_positions, rope_theta):
        if self._is_partial_rotary:
            return apply_rope_partial_bwd(
                grad_tensors, seq_positions, rope_theta, self._rot_dim,
            )
        return apply_rope_bwd(grad_tensors, seq_positions, rope_theta)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def fwd(
        self,
        x_resid: torch.Tensor,
        attn_norm_output: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        cfg = self.cfg
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        total_q = chunk.total_q
        total_k = chunk.total_k

        # Q projection produces (Q | gate) concatenated. Use scratch
        # because the saved fields (slot.xq, slot.attn_gate) are
        # individually attn_dim-wide; can't write the doubled output
        # into them directly.
        qproj = ctx.scratch(
            (slot.xq.shape[0], cfg.attn_dim * 2), slot.xq.dtype,
        )
        torch.matmul(attn_norm_output, weights["w_q"], out=qproj)
        if cfg.qkv_bias:
            qproj.add_(weights["b_q"])
        # Split into Q (first half) and gate (second half).
        slot.xq.view(-1, cfg.attn_dim).copy_(qproj[:, :cfg.attn_dim])
        slot.attn_gate.copy_(qproj[:, cfg.attn_dim:])

        # K, V projections (unchanged from GQAAttentionBlock).
        torch.matmul(
            attn_norm_output, weights["w_k"],
            out=slot.xk.view(-1, cfg.kv_dim),
        )
        torch.matmul(
            attn_norm_output, weights["w_v"],
            out=slot.xv.view(-1, cfg.kv_dim),
        )
        if cfg.qkv_bias:
            slot.xk.view(-1, cfg.kv_dim).add_(weights["b_k"])
            slot.xv.view(-1, cfg.kv_dim).add_(weights["b_v"])

        # QK-norm (per-head RMSNorm on Q, K).
        if cfg.qk_norm:
            xq2d = slot.xq.view(-1, cfg.attn_dim)
            xk2d = slot.xk.view(-1, cfg.kv_dim)
            self.q_norm.fwd(
                xq2d, weights, getattr(slot, self.q_norm.rstd_name),
                output=xq2d,
            )
            self.k_norm.fwd(
                xk2d, weights, getattr(slot, self.k_norm.rstd_name),
                output=xk2d,
            )

        # RoPE in-place on Q and K (full or partial per cfg). The gate
        # path is NOT rotated — gate is a per-head-element scalar signal
        # not a positional vector.
        rope_theta = self._rope_theta(x_resid.device)
        self._rope_fwd([slot.xq, slot.xk], chunk.seq_positions, rope_theta)

        # KV-cache write.
        kv = ctx.kv_cache
        start = (
            chunk.prior_seq_offsets_host[0] + chunk.prior_seq_lens_host[0]
        )
        kv.k[start : start + total_q, :].copy_(slot.xk)
        kv.v[start : start + total_q, :].copy_(slot.xv)

        # Flash-attn varlen fwd.
        flextrain_attention_fwd(
            slot.xq.view(-1, n_heads, head_dim),
            kv.k[:total_k, :],
            kv.v[:total_k, :],
            slot.attn_result,
            slot.softmax_lse,
            chunk.q_seq_offsets,
            chunk.k_seq_offsets,
            chunk.q_seq_lens,
            chunk.k_seq_lens,
            chunk.max_seqlen_q,
            chunk.max_seqlen_k,
            causal=cfg.is_causal,
            window_size=(cfg.window_size_left, cfg.window_size_right),
            softcap=cfg.attn_logit_softcap,
        )

        # Apply the sigmoid gate then fold residual via fused o_proj.
        # gated = attn_result * sigmoid(gate). We materialize ``gated``
        # in scratch (same shape as attn_result) so we don't clobber
        # attn_result (which the bwd needs). Could be optimized into a
        # fused kernel later; for now PyTorch element-wise is fine.
        gated = ctx.scratch(
            slot.attn_result.shape, slot.attn_result.dtype,
        )
        torch.mul(
            slot.attn_result.view(-1, cfg.attn_dim),
            torch.sigmoid(slot.attn_gate.float()).to(slot.attn_gate.dtype),
            out=gated.view(-1, cfg.attn_dim),
        )

        # Fused O-projection + residual: xo = gated @ w_o + x_resid.
        stream_ptr = torch.cuda.current_stream().cuda_stream
        return dispatcher.matmul(
            stream_ptr,
            A=gated.view(-1, cfg.attn_dim),
            B=weights["w_o"],
            C=x_resid,
            D=slot.xo,
            alpha=1.0, beta=1.0,
        )

    # ------------------------------------------------------------------
    # Forward-recompute helpers
    # ------------------------------------------------------------------

    def fwd_recompute_qo(
        self,
        attn_norm_output: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        slot,
        x_resid: torch.Tensor,
    ) -> torch.Tensor:
        """Tier<2: recompute Q (and re-RoPE), gate, and xo."""
        cfg = self.cfg
        n_heads, head_dim = cfg.n_heads, cfg.head_dim
        # Re-project the doubled Q and split into Q + gate.
        qproj = torch.matmul(attn_norm_output, weights["w_q"])  # (T, attn_dim*2)
        if cfg.qkv_bias:
            qproj.add_(weights["b_q"])
        slot.xq.view(-1, cfg.attn_dim).copy_(qproj[:, :cfg.attn_dim])
        slot.attn_gate.copy_(qproj[:, cfg.attn_dim:])
        if cfg.qk_norm:
            xq2d = slot.xq.view(-1, cfg.attn_dim)
            self.q_norm.fwd_from_rstd(
                xq2d, weights, getattr(slot, self.q_norm.rstd_name),
                output=xq2d,
            )
        rope_theta = self._rope_theta(attn_norm_output.device)
        self._rope_fwd([slot.xq], chunk.seq_positions, rope_theta)
        return slot.xq

    def fwd_recompute_attn(
        self,
        chunk: ChunkMeta,
        slot,
        ctx: LayerContext,
    ) -> None:
        cfg = self.cfg
        n_heads, head_dim = cfg.n_heads, cfg.head_dim
        total_k = chunk.total_k
        kv = ctx.kv_cache
        flextrain_attention_fwd(
            slot.xq.view(-1, n_heads, head_dim),
            kv.k[:total_k, :],
            kv.v[:total_k, :],
            slot.attn_result,
            slot.softmax_lse,
            chunk.q_seq_offsets,
            chunk.k_seq_offsets,
            chunk.q_seq_lens,
            chunk.k_seq_lens,
            chunk.max_seqlen_q,
            chunk.max_seqlen_k,
            causal=cfg.is_causal,
            window_size=(cfg.window_size_left, cfg.window_size_right),
            softcap=cfg.attn_logit_softcap,
        )

    def fwd_recompute_o(
        self,
        x_resid: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
    ) -> torch.Tensor:
        """Tier<2 xo recompute. Includes the gate multiplication."""
        cfg = self.cfg
        # Materialize gated attn_result, then addmm with residual.
        sig_gate = torch.sigmoid(slot.attn_gate.float()).to(slot.attn_gate.dtype)
        gated = slot.attn_result.view(-1, cfg.attn_dim) * sig_gate
        return torch.addmm(
            x_resid,
            gated,
            weights["w_o"],
            out=slot.xo,
        )

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def bwd(
        self,
        dx_resid: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
        attn_norm_output: torch.Tensor,
        *,
        skip_grads: frozenset[str] = frozenset(),
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        num_tokens = dx_resid.shape[0]
        total_q = chunk.total_q
        total_k = chunk.total_k

        attn_dim = cfg.attn_dim
        # Saved attn_result (pre-gate). Sigmoid(gate) recomputed once.
        attn_result_2d = slot.attn_result.view(num_tokens, attn_dim)
        gate_2d = slot.attn_gate.view(num_tokens, attn_dim)
        sig_gate = torch.sigmoid(gate_2d.float()).to(gate_2d.dtype)
        # The "gated" output that fed o_proj.
        gated_2d = attn_result_2d * sig_gate

        # 1. g_o += gated^T @ dx_resid
        if "g_o" in skip_grads:
            if capture_xy is not None:
                # Clone gated_2d (built from saved tensors -- safe but
                # minimal cost) and dx_resid (caller will overwrite).
                capture_xy["g_o"] = (gated_2d.clone(), dx_resid.clone())
        else:
            torch.addmm(
                grads["g_o"], gated_2d.T, dx_resid,
                alpha=1.0, beta=1.0, out=grads["g_o"],
            )

        # 2. d(gated) = dx_resid @ w_o^T   (T, attn_dim)
        d_gated = torch.matmul(dx_resid, weights["w_o"].T)

        # 3. Split into d(attn_result) and d(gate):
        #    gated = attn_result * sigmoid(gate)
        #    d(attn_result) = d(gated) * sigmoid(gate)
        #    d(gate)        = d(gated) * attn_result * sigmoid(gate) * (1 - sigmoid(gate))
        d_attn_result_2d = d_gated * sig_gate
        d_gate_2d = (
            d_gated * attn_result_2d * sig_gate * (1.0 - sig_gate)
        )

        # 4. flash-attn bwd consumes d_attn_result, writes dq + dk/dv
        #    into the running KV-grad ring.
        d_attn_result_3d = d_attn_result_2d.view(num_tokens, n_heads, head_dim)
        dq = ctx.scratch(d_attn_result_3d.shape, d_attn_result_3d.dtype)
        dq.zero_()
        flextrain_attention_bwd(
            d_attn_result_3d,
            slot.xq.view(-1, n_heads, head_dim),
            ctx.kv_cache.k[:total_k, :],
            ctx.kv_cache.v[:total_k, :],
            slot.attn_result,
            slot.softmax_lse,
            dq,
            ctx.kv_cache.dk[:total_k, :],
            ctx.kv_cache.dv[:total_k, :],
            chunk.q_seq_offsets,
            chunk.k_seq_offsets,
            chunk.q_seq_lens,
            chunk.k_seq_lens,
            chunk.max_seqlen_q,
            chunk.max_seqlen_k,
            causal=cfg.is_causal,
            window_size=(cfg.window_size_left, cfg.window_size_right),
            softcap=cfg.attn_logit_softcap,
        )

        # 5. Pull this chunk's local dK / dV from the bwd ring + zero the
        #    consumed positions so the prior chunk doesn't double-count.
        start = (
            chunk.prior_seq_offsets_host[0] + chunk.prior_seq_lens_host[0]
        )
        local_dk = ctx.scratch(slot.xk.shape, slot.xk.dtype)
        local_dv = ctx.scratch(slot.xv.shape, slot.xv.dtype)
        local_dk.copy_(ctx.kv_cache.dk[start : start + total_q, :])
        local_dv.copy_(ctx.kv_cache.dv[start : start + total_q, :])
        ctx.kv_cache.dk[start : start + total_q, :].zero_()
        ctx.kv_cache.dv[start : start + total_q, :].zero_()

        # 6. RoPE bwd in-place on dq + local_dk (full or partial per cfg).
        rope_theta = self._rope_theta(dx_resid.device)
        dq_view, local_dk_view = self._rope_bwd(
            [dq.view(-1, n_heads, head_dim), local_dk.view(-1, n_kv, head_dim)],
            chunk.seq_positions,
            rope_theta,
        )

        # 6b. Qwen3-style per-head QK-norm bwd. We need pre-norm Q/K which
        #     are recomputed from the FIRST half of (attn_norm_output @
        #     w_q) and (attn_norm_output @ w_k). Same as
        #     GQAAttentionBlock.bwd, but Q's recompute reads only the
        #     first half of w_q.
        if cfg.qk_norm:
            if attn_norm_output is None:
                raise ValueError(
                    "GQAAttentionGatedBlock.bwd requires attn_norm_output "
                    "when cfg.qk_norm=True"
                )
            # Recompute pre-norm Q (first half of w_q only).
            w_q_first_half = weights["w_q"][:, :attn_dim]
            xq_pre_norm_2d = torch.matmul(
                attn_norm_output, w_q_first_half,
            ).contiguous()
            xk_pre_norm_2d = torch.matmul(
                attn_norm_output, weights["w_k"],
            ).contiguous()
            dq_pre_norm_2d, _ = self.q_norm.bwd(
                dq_view.view(num_tokens, -1),
                xq_pre_norm_2d,
                weights, grads,
                getattr(slot, self.q_norm.rstd_name),
                dx_accumulator=None,
                recompute_output=False,
            )
            dk_pre_norm_2d, _ = self.k_norm.bwd(
                local_dk_view.view(num_tokens, -1),
                xk_pre_norm_2d,
                weights, grads,
                getattr(slot, self.k_norm.rstd_name),
                dx_accumulator=None,
                recompute_output=False,
            )
            dq_view = dq_pre_norm_2d.view(num_tokens, n_heads, head_dim)
            local_dk_view = dk_pre_norm_2d.view(num_tokens, n_kv, head_dim)

        # 6c. Qwen2-style QKV bias grads. Bias on doubled w_q means the
        #     bias has shape (attn_dim*2,); concat (dq | d_gate) for its
        #     bias-grad sum.
        if cfg.qkv_bias:
            dqg_for_bias = torch.cat(
                [dq_view.view(num_tokens, -1), d_gate_2d], dim=-1,
            )
            grads["g_b_q"].add_(dqg_for_bias.sum(dim=0))
            grads["g_b_k"].add_(local_dk_view.view(num_tokens, -1).sum(dim=0))
            grads["g_b_v"].add_(local_dv.view(num_tokens, -1).sum(dim=0))

        # 7. dx_attn_norm_up = dqg @ w_q^T + local_dk @ w_k^T + local_dv @ w_v^T
        #    where dqg = concat(dq_pre, d_gate, dim=-1)  (T, attn_dim*2)
        dqg_2d = torch.cat(
            [dq_view.view(num_tokens, -1), d_gate_2d], dim=-1,
        )
        dx_attn_norm_up = torch.matmul(dqg_2d, weights["w_q"].T)
        torch.addmm(
            dx_attn_norm_up, local_dk_view.view(num_tokens, -1),
            weights["w_k"].T, alpha=1.0, beta=1.0, out=dx_attn_norm_up,
        )
        torch.addmm(
            dx_attn_norm_up, local_dv.view(num_tokens, -1),
            weights["w_v"].T, alpha=1.0, beta=1.0, out=dx_attn_norm_up,
        )

        # Stash dqg + local_dk + local_dv for bwd_accumulate_qkv_grads
        # (the layer routes them after RMSNorm bwd recomputes
        # attn_norm_output, which is the left operand).
        slot.aux["bwd_dqg"] = dqg_2d
        slot.aux["bwd_local_dk"] = local_dk_view.view(num_tokens, -1)
        slot.aux["bwd_local_dv"] = local_dv.view(num_tokens, -1)

        return dx_attn_norm_up

    def bwd_accumulate_qkv_grads(
        self,
        attn_norm_output: torch.Tensor,
        grads: MutableMapping[str, torch.Tensor],
        slot,
        *,
        skip_grads: frozenset[str] = frozenset(),
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        """Second half of bwd: weight-grad matmuls against the
        recomputed ``attn_norm_output``. ``g_q`` accumulates the FULL
        ``(d_model, attn_dim*2)`` weight-grad (Q + gate halves).

        ``skip_grads`` / ``capture_xy``: see
        :class:`flextrain.nn.blocks.attention.GQAAttentionBlock` --
        same LoRA fast-path contract."""
        for name, dy_key in (
            ("g_v", "bwd_local_dv"),
            ("g_k", "bwd_local_dk"),
            ("g_q", "bwd_dqg"),
        ):
            dy = slot.aux[dy_key]
            if name in skip_grads:
                if capture_xy is not None:
                    capture_xy[name] = (attn_norm_output, dy)
            else:
                torch.addmm(
                    grads[name], attn_norm_output.T, dy,
                    alpha=1.0, beta=1.0, out=grads[name],
                )
        del slot.aux["bwd_dqg"]
        del slot.aux["bwd_local_dk"]
        del slot.aux["bwd_local_dv"]

    # ------------------------------------------------------------------
    # FLOP accounting (mirrors GQAAttentionBlock + small gate ops).
    # ------------------------------------------------------------------

    def compute_cost(
        self, chunk: ChunkMeta, max_tier: int
    ) -> ComputeCost:
        cfg = self.cfg
        avoided = [0] * (max_tier + 1)
        total = 0
        for seq_len, prior_len in zip(
            chunk.seq_lens_host, chunk.prior_seq_lens_host
        ):
            kv_proj = 2 * seq_len * cfg.d_model * cfg.kv_dim * 2
            total += kv_proj
            # Q + gate projection (doubled w_q) + O projection.
            qo = (
                2 * seq_len * cfg.d_model * (cfg.attn_dim * 2)  # qproj (Q+gate)
                + 2 * seq_len * cfg.d_model * cfg.attn_dim       # o_proj
            )
            total += qo
            if max_tier >= 2:
                for L in range(2, max_tier + 1):
                    avoided[L] += qo
            attn_prior = 4 * seq_len * prior_len * cfg.attn_dim
            if cfg.is_causal:
                attn_current = 2 * seq_len * seq_len * cfg.attn_dim
            else:
                attn_current = 4 * seq_len * seq_len * cfg.attn_dim
            attn = attn_prior + attn_current
            total += attn
            if max_tier >= 1:
                for L in range(1, max_tier + 1):
                    avoided[L] += attn
            # Element-wise gate multiplication is small but count it.
            total += seq_len * cfg.attn_dim
        return ComputeCost(
            total_fwd_flops=total,
            avoided_recompute_flops=tuple(avoided),
        )
