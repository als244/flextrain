"""Mixture-of-Experts SwiGLU FFN with optional shared experts.

Standalone block that adds a parallel "shared experts" path on top of
the existing routed MoE — used by Qwen3-Next / Qwen3.5 / Qwen3.6 (S=1)
and DeepSeek-V2/V3 (S>1).

Math (S = num_shared_experts >= 1)
----------------------------------
Forward, given residual stream ``x`` (T, d_model):

    # Routed path (delegates to MoESwiGLUFFN, residual deferred):
    routed_out = MoESwiGLUFFN(x)              # (T, d_model)

    # Shared path:
    sh_pre   = einsum("td, sdf -> tsf", x, w_shared_up)         # (T, S, 2 F_s)
    sh_act   = up_half * silu(gate_half)                         # (T, S, F_s)
    sh_each  = einsum("tsf, sfd -> tsd", sh_act, w_shared_down)  # (T, S, d_model)

    # Per-shared-expert sigmoid gate:
    sh_gate_pre = x @ w_shared_expert_gate                       # (T, S)
    sh_gate     = sigmoid(sh_gate_pre)                           # (T, S)
    shared_out  = (sh_gate.unsqueeze(-1) * sh_each).sum(dim=1)   # (T, d_model)

    # Combine (additive — NOT a (1-σ)*routed + σ*shared mixture):
    final_out = routed_out + shared_out + residual

S = 1 special case
------------------
For Qwen3-Next / 3.5 / 3.6, S = 1 — the leading-1 dim is no-op (bmm /
reductions on size-1 dim are essentially free). We use the (T, S, ...)
shapes uniformly so the bwd code doesn't branch.

Param layout
------------
* ``w_router``               (d_model, num_experts)
* ``w_up``                   (num_experts, d_model, 2 * expert_dim)
* ``w_down``                 (num_experts, expert_dim, d_model)
* ``w_shared_up``            (S, d_model, 2 * shared_expert_dim)
* ``w_shared_down``          (S, shared_expert_dim, d_model)
* ``w_shared_expert_gate``   (d_model, S)

Convention: ``w_shared_up``'s last dim is packed ``[up, gate]``
(value first, gate second), matching the routed ``w_up`` packing for
consistency. Loaders concatenate HF's separate ``up_proj`` / ``gate_proj``
weights in this order.

Activation schema (additive over MoESwiGLUFFN)
----------------------------------------------
* ``x_shared_pre``    (T, S, 2 * shared_expert_dim)  bf16  tier 3
   Pre-SwiGLU intermediate; recomputable from x + w_shared_up.
* ``x_shared_gate``   (T, S)                          bf16  tier 0
   Pre-sigmoid gate values; tiny.
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
from .ffn_moe import MoESwiGLUConfig, MoESwiGLUFFN


@dataclass(frozen=True)
class MoESwiGLUSharedExpertConfig(MoESwiGLUConfig):
    """Config for :class:`MoESwiGLUSharedExpertFFN`.

    Inherits all fields of :class:`MoESwiGLUConfig` (routed-expert
    setup) and adds shared-expert fields:

    * ``num_shared_experts`` (S) -- typically 1 for Qwen3-Next/3.5/3.6;
      can be >1 for DeepSeek-style designs.
    * ``shared_expert_dim`` (F_s) -- per-shared-expert intermediate
      dimension. HF: ``shared_expert_intermediate_size``.

    Setting ``num_shared_experts == 0`` is not supported here — use
    plain :class:`MoESwiGLUFFN` for routed-only.
    """

    num_shared_experts: int = 1
    shared_expert_dim: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_shared_experts < 1:
            raise ValueError(
                "MoESwiGLUSharedExpertConfig requires num_shared_experts >= 1; "
                "use MoESwiGLUFFN for routed-only configs."
            )
        if self.shared_expert_dim <= 0:
            raise ValueError(
                f"MoESwiGLUSharedExpertConfig requires shared_expert_dim > 0; "
                f"got {self.shared_expert_dim}."
            )


class MoESwiGLUSharedExpertFFN:
    """MoE SwiGLU FFN with ``num_shared_experts`` always-active dense
    experts in addition to the top-K routed experts.

    Standalone block — composes :class:`MoESwiGLUFFN` for the routed
    half (does not subclass it). The routed call is invoked with a
    zero residual; the wrapper folds in the layer's actual residual
    once at the end (avoids double-counting).
    """

    def __init__(self, cfg: MoESwiGLUSharedExpertConfig) -> None:
        self.cfg = cfg
        # Build the inner routed block from the routed-only fields. The
        # shared block's grads + activations live alongside in the same
        # weights/grads/slot dicts; the inner block only reads/writes
        # routed-named keys (w_router, w_up, w_down, etc.) so there's
        # no name collision.
        routed_cfg = MoESwiGLUConfig(
            d_model=cfg.d_model,
            expert_dim=cfg.expert_dim,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            load_balance_coef=cfg.load_balance_coef,
            routing_mode=cfg.routing_mode,
            compute_dtype=cfg.compute_dtype,
            master_dtype=cfg.master_dtype,
            grad_dtype=cfg.grad_dtype,
        )
        self._routed_ffn = MoESwiGLUFFN(routed_cfg)

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def fields(self) -> tuple[ActivationField, ...]:
        cfg = self.cfg
        bf = cfg.compute_dtype
        S = cfg.num_shared_experts
        Fs = cfg.shared_expert_dim
        # Routed fields + shared fields. Tier 3 for the heavy
        # (T, S, 2F_s) tensor (recomputable); tier 0 for the tiny gate.
        return self._routed_ffn.fields() + (
            ActivationField(
                "x_shared_pre",
                lambda n, d, S=S, Fs=Fs: (n, S, 2 * Fs),
                bf,
                tier=3,
            ),
            ActivationField(
                "x_shared_gate",
                lambda n, d, S=S: (n, S),
                bf,
                tier=0,
            ),
        )

    def param_spec(self) -> ParamSpec:
        cfg = self.cfg
        S = cfg.num_shared_experts
        Fs = cfg.shared_expert_dim
        # Inherit routed param spec, then append shared params.
        routed_specs = self._routed_ffn.param_spec().tensors
        # Per-shared-expert SwiGLU up: (S, d_model, 2 * F_s) — packed [up, gate].
        # Per-shared-expert SwiGLU down: (S, F_s, d_model).
        # Per-shared-expert per-token scalar gate: (d_model, S).
        shared_specs = (
            TensorSpec(
                "w_shared_up",
                lambda d, S=S, Fs=Fs: (S, d["d_model"], 2 * Fs),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            TensorSpec(
                "w_shared_down",
                lambda d, S=S, Fs=Fs: (S, Fs, d["d_model"]),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            TensorSpec(
                "w_shared_expert_gate",
                lambda d, S=S: (d["d_model"], S),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
                # Single output per shared expert (per token); same role as
                # the router gate. Use AdamW like the router.
                optimizer="adamw",
            ),
        )
        return ParamSpec(tensors=routed_specs + shared_specs)

    # ------------------------------------------------------------------
    # Compute helpers
    # ------------------------------------------------------------------

    def _shared_swiglu_fwd(
        self,
        x_2d: torch.Tensor,           # (T, d_model)
        weights: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute shared-expert SwiGLU output (pre-gate, pre-sum).

        Returns:
          * ``x_shared_pre``: (T, S, 2 * F_s) — pre-SwiGLU. Caller saves to slot.
          * ``sh_each``:      (T, S, d_model)  — per-shared-expert output, pre-gate.
        """
        cfg = self.cfg
        S = cfg.num_shared_experts
        Fs = cfg.shared_expert_dim
        T = x_2d.shape[0]

        # x @ w_shared_up: (T, d) × (S, d, 2F) → (T, S, 2F)
        # Use bmm by broadcasting x to (S, T, d) and using matmul, OR
        # einsum. einsum is most readable; PyTorch dispatches to bmm.
        x_shared_pre = torch.einsum(
            "td,sdf->tsf", x_2d.float(), weights["w_shared_up"].float()
        ).to(x_2d.dtype)

        # Split into [up, gate]; SwiGLU.
        up_half = x_shared_pre[..., :Fs]
        gate_half = x_shared_pre[..., Fs:]
        sh_act = up_half * F.silu(gate_half.float()).to(x_2d.dtype)  # (T, S, F_s)

        # sh_act @ w_shared_down: (T, S, F_s) × (S, F_s, d) → (T, S, d)
        sh_each = torch.einsum(
            "tsf,sfd->tsd", sh_act.float(), weights["w_shared_down"].float()
        ).to(x_2d.dtype)

        return x_shared_pre, sh_each

    def _shared_gate_fwd(
        self, x_2d: torch.Tensor, weights: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute pre-sigmoid gate ``x @ w_shared_expert_gate``  (T, S)."""
        return x_2d @ weights["w_shared_expert_gate"]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def fwd(
        self,
        ffn_norm_output: torch.Tensor,           # (T, d_model)
        weights: Mapping[str, torch.Tensor],
        attn_output_with_residual: torch.Tensor, # (T, d_model) — layer's pre-FFN residual
        out_tensor: torch.Tensor,                # (T, d_model) — destination
        slot,
        ctx: LayerContext,
        chunk: ChunkMeta,
        *,
        layer_id: int,
    ) -> torch.Tensor:
        """Forward pass.

        Step layout:
          1. Routed FFN with ZERO residual (so we can fold residual + shared
             once at the end).
          2. Shared-expert SwiGLU + per-shared-expert sigmoid gate.
          3. Final: ``out_tensor = routed_out + shared_summed + residual``.

        Activations saved: routed fields + ``x_shared_pre`` (tier 3) +
        ``x_shared_gate`` (tier 0).
        """
        cfg = self.cfg
        T = ffn_norm_output.shape[0]
        d_model = cfg.d_model

        # 1) Routed path with ZERO residual.
        # Allocate a zero scratch the same shape as the residual. Cheap;
        # the routed kernel reads it inside flextrain_moe_gather and the
        # add is memory-bandwidth-bound but has zero math impact.
        zero_resid = ctx.scratch(attn_output_with_residual.shape, attn_output_with_residual.dtype)
        zero_resid.zero_()
        routed_out = ctx.scratch(out_tensor.shape, out_tensor.dtype)
        self._routed_ffn.fwd(
            ffn_norm_output, weights,
            attn_output_with_residual=zero_resid,
            out_tensor=routed_out,
            slot=slot, ctx=ctx, chunk=chunk,
            layer_id=layer_id,
        )

        # 2) Shared-expert path: SwiGLU + gate.
        x_2d = ffn_norm_output.view(T, d_model)
        x_shared_pre, sh_each = self._shared_swiglu_fwd(x_2d, weights)
        slot.x_shared_pre.copy_(x_shared_pre)

        sh_gate_pre = self._shared_gate_fwd(x_2d, weights)            # (T, S)
        slot.x_shared_gate.copy_(sh_gate_pre)

        sh_gate = torch.sigmoid(sh_gate_pre.float()).to(x_2d.dtype)   # (T, S)
        # (T, S, 1) × (T, S, d) → sum over S → (T, d)
        shared_out = (sh_gate.unsqueeze(-1) * sh_each).sum(dim=1)

        # 3) Combine.
        torch.add(routed_out, shared_out, out=out_tensor)
        out_tensor.add_(attn_output_with_residual)
        return out_tensor

    # ------------------------------------------------------------------
    # Forward-recompute helper (mirrors routed fwd_recompute_x_up)
    # ------------------------------------------------------------------

    def fwd_recompute_x_shared_pre(
        self,
        ffn_norm_output: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
    ) -> None:
        """Recompute ``x_shared_pre`` from ``ffn_norm_output`` and
        ``w_shared_up``. Called when save_level < 3 dropped this field."""
        x_shared_pre, _ = self._shared_swiglu_fwd(
            ffn_norm_output.view(ffn_norm_output.shape[0], -1), weights,
        )
        slot.x_shared_pre.copy_(x_shared_pre)

    def fwd_recompute_x_up(
        self,
        ffn_norm_output: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
        chunk: ChunkMeta,
        *,
        layer_id: int,
    ) -> None:
        """Tier-3 recompute: routed ``x_up`` AND shared ``x_shared_pre``.

        Layers compose this block via ``self.ffn = MoESwiGLUSharedExpertFFN(...)``
        and call ``self.ffn.fwd_recompute_x_up(...)`` from their
        ``forward_recompute``; this delegates the routed-half recompute
        to the inner routed FFN and also re-derives ``x_shared_pre``.
        """
        self._routed_ffn.fwd_recompute_x_up(
            ffn_norm_output, weights, slot, chunk, layer_id=layer_id,
        )
        self.fwd_recompute_x_shared_pre(ffn_norm_output, weights, slot)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def bwd(
        self,
        dy_resid: torch.Tensor,                  # (T, d_model)
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
        chunk: ChunkMeta,
        *,
        layer_id: int,
    ) -> torch.Tensor:
        """Backward.

        Inputs ``dy_resid`` is the gradient at the layer's final output
        (i.e. ``out_tensor``). Both routed and shared paths are summed
        additively into ``out_tensor``, so each receives ``dy_resid``
        unchanged.

        Returns ``dx_ffn_norm_up`` — gradient w.r.t. the FFN-norm output
        (sum of contributions from routed, shared MLP, and shared gate).
        """
        cfg = self.cfg
        S = cfg.num_shared_experts
        Fs = cfg.shared_expert_dim
        d_model = cfg.d_model
        num_tokens = dy_resid.shape[0]

        # Caller has placed the recomputed ffn_norm_output at
        # slot.aux["recompute_ffn_norm_output"] (mirrors routed convention).
        ffn_norm_output = slot.aux.get("recompute_ffn_norm_output", None)
        if ffn_norm_output is None:
            raise RuntimeError(
                "MoESwiGLUSharedExpertFFN.bwd requires the caller to provide "
                "ffn_norm_output via slot.aux['recompute_ffn_norm_output']."
            )
        x_2d = ffn_norm_output.view(num_tokens, d_model)

        # ------------------------------------------------------------------
        # Shared-expert bwd. Reverse of:
        #   sh_pre        = x @ w_shared_up           (T, S, 2F)
        #   up, gate      = split(sh_pre, dim=-1)
        #   sh_act        = up * silu(gate)            (T, S, F)
        #   sh_each       = einsum("tsf, sfd -> tsd")  (T, S, d)
        #   sh_gate_pre   = x @ w_shared_expert_gate   (T, S)
        #   sh_gate       = sigmoid(sh_gate_pre)
        #   shared_out    = sum_s sh_gate[:, s] * sh_each[:, s, :]
        # ------------------------------------------------------------------
        sig_gate = torch.sigmoid(slot.x_shared_gate.float()).to(slot.x_shared_gate.dtype)  # (T, S)
        # Recompute sh_each from saved x_shared_pre + w_shared_down (cheap;
        # no need to save sh_each separately).
        x_shared_pre = slot.x_shared_pre                                    # (T, S, 2F)
        up_half = x_shared_pre[..., :Fs]                                     # (T, S, F)
        gate_half = x_shared_pre[..., Fs:]                                   # (T, S, F)
        sh_act = up_half * F.silu(gate_half.float()).to(up_half.dtype)       # (T, S, F)
        sh_each = torch.einsum(
            "tsf,sfd->tsd", sh_act.float(), weights["w_shared_down"].float()
        ).to(up_half.dtype)                                                  # (T, S, d)

        # d/d(shared_out) = dy_resid; each shared expert's contribution is
        # sh_gate[:,s] * sh_each[:,s,:] summed over s.
        dy_2d = dy_resid                                                     # (T, d)
        # d/d(sh_each[t, s, :]) = dy_resid[t, :] * sh_gate[t, s]
        d_sh_each = dy_2d.unsqueeze(1) * sig_gate.unsqueeze(-1)              # (T, S, d)
        # d/d(sh_gate[t, s]) = sum_d (dy_resid[t,d] * sh_each[t,s,d])
        d_sh_gate = (dy_2d.unsqueeze(1) * sh_each).sum(dim=-1)               # (T, S)
        # d/d(sh_gate_pre) = d_sh_gate * sigmoid(g) * (1 - sigmoid(g))
        d_sh_gate_pre = (
            d_sh_gate.float() * sig_gate.float() * (1.0 - sig_gate.float())
        ).to(d_sh_gate.dtype)                                                # (T, S)

        # w_shared_expert_gate grad: x.T @ d_sh_gate_pre  (d_model, S)
        if grads.get("g_shared_expert_gate") is not None:
            grads["g_shared_expert_gate"].add_(
                (x_2d.float().T @ d_sh_gate_pre.float()).to(grads["g_shared_expert_gate"].dtype)
            )
        # dx via gate path: d_sh_gate_pre @ w_shared_expert_gate.T  (T, d_model)
        dx_via_gate = (
            d_sh_gate_pre.float() @ weights["w_shared_expert_gate"].float().T
        ).to(dy_resid.dtype)

        # Backprop through the down-proj einsum:
        # sh_each[t,s,d] = sum_f sh_act[t,s,f] * w_down[s,f,d]
        # d/d(w_down[s,f,d]) = sum_t sh_act[t,s,f] * d_sh_each[t,s,d]
        # d/d(sh_act[t,s,f]) = sum_d w_down[s,f,d] * d_sh_each[t,s,d]
        if grads.get("g_shared_down") is not None:
            g_w_shared_down = torch.einsum(
                "tsf,tsd->sfd", sh_act.float(), d_sh_each.float()
            ).to(grads["g_shared_down"].dtype)
            grads["g_shared_down"].add_(g_w_shared_down)
        d_sh_act = torch.einsum(
            "tsd,sfd->tsf",
            d_sh_each.float(), weights["w_shared_down"].float(),
        ).to(sh_act.dtype)                                                   # (T, S, F)

        # SwiGLU bwd: sh_act = up * silu(gate)
        # d/d(up)   = d_sh_act * silu(gate)
        # d/d(gate) = d_sh_act * up * silu'(gate)
        # silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
        gate_f = gate_half.float()
        sig_g = gate_f.sigmoid()
        silu_gate = (gate_f * sig_g).to(up_half.dtype)
        dsilu = (sig_g * (1.0 + gate_f * (1.0 - sig_g))).to(up_half.dtype)
        d_up = d_sh_act * silu_gate
        d_gate = d_sh_act * up_half * dsilu
        # Concat into d_x_shared_pre  (T, S, 2F)
        d_x_shared_pre = torch.cat([d_up, d_gate], dim=-1)

        # Backprop through the up-proj einsum:
        # sh_pre[t,s,f] = sum_d x[t,d] * w_up[s,d,f]
        # d/d(w_up[s,d,f]) = sum_t x[t,d] * d_sh_pre[t,s,f]
        # d/d(x[t,d])      = sum_{s,f} w_up[s,d,f] * d_sh_pre[t,s,f]
        if grads.get("g_shared_up") is not None:
            g_w_shared_up = torch.einsum(
                "td,tsf->sdf", x_2d.float(), d_x_shared_pre.float()
            ).to(grads["g_shared_up"].dtype)
            grads["g_shared_up"].add_(g_w_shared_up)
        dx_via_shared_mlp = torch.einsum(
            "tsf,sdf->td",
            d_x_shared_pre.float(), weights["w_shared_up"].float(),
        ).to(dy_resid.dtype)

        # ------------------------------------------------------------------
        # Routed path bwd — delegate. The inner block accumulates
        # g_router/g_up/g_down into ``grads`` and returns dx_via_routed.
        # ------------------------------------------------------------------
        dx_via_routed = self._routed_ffn.bwd(
            dy_resid, weights, grads, slot, ctx, chunk, layer_id=layer_id,
        )

        # Combine all three dx contributions.
        return dx_via_routed + dx_via_shared_mlp + dx_via_gate

    # ------------------------------------------------------------------
    # FLOP estimate
    # ------------------------------------------------------------------

    def compute_cost(self, chunk: ChunkMeta, max_tier: int) -> ComputeCost:
        cfg = self.cfg
        # Routed cost from inner block.
        routed = self._routed_ffn.compute_cost(chunk, max_tier)
        S = cfg.num_shared_experts
        Fs = cfg.shared_expert_dim
        d = cfg.d_model
        # Shared-path FLOPs: per-token, S × (2*d*2F + 2*F*d + d) (up + down + gate)
        shared_per_tok = 2 * S * (2 * d * (2 * Fs) + 2 * Fs * d) + 2 * d * S
        shared_total = sum(seq_len * shared_per_tok for seq_len in chunk.seq_lens_host)
        # Tier 3 saves x_shared_pre — recomputable cost = up matmul.
        avoided = list(routed.avoided_recompute_flops)
        if max_tier >= 3:
            shared_recompute_flops = sum(
                seq_len * (2 * d * S * 2 * Fs)
                for seq_len in chunk.seq_lens_host
            )
            avoided[3] += shared_recompute_flops
        return ComputeCost(
            total_fwd_flops=routed.total_fwd_flops + shared_total,
            avoided_recompute_flops=tuple(avoided),
        )
