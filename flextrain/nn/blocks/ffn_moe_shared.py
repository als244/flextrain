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

    def __init__(
        self, cfg: MoESwiGLUSharedExpertConfig,
        *,
        expert_compute=None,  # MoEExpertCompute | None
    ) -> None:
        self.cfg = cfg
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
        self._routed_ffn = MoESwiGLUFFN(routed_cfg, expert_compute=expert_compute)

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

        Math: ``up || gate = x @ W_shared_up``, ``sh_act = up * silu(gate)``,
        ``sh_each = sh_act @ W_shared_down``. All matmuls bf16; SwiGLU
        fused via ``flextrain_swiglu_fwd``. Pack order on the last dim
        of ``W_shared_up`` is ``[up | gate]`` (loader convention; see
        ``flextrain/io/arch/qwen3_5_moe.py``).

        Returns:
          * ``x_shared_pre``: (T, S, 2 * F_s) — pre-SwiGLU bf16; saved to slot.
          * ``sh_each``:      (T, S, d_model) — per-shared-expert output (bf16).
        """
        from flextrain.ops import flextrain_swiglu_fwd

        cfg = self.cfg
        S = cfg.num_shared_experts
        Fs = cfg.shared_expert_dim

        if S == 1:
            # Fast path (Qwen3-Next/3.5/3.6): 2D bf16 matmul instead of bmm.
            w_up = weights["w_shared_up"][0]                    # (d, 2F)
            w_down = weights["w_shared_down"][0]                # (F, d)
            x_shared_pre_2d = x_2d @ w_up                       # (T, 2F) bf16
            up_half = x_shared_pre_2d[..., :Fs]
            gate_half = x_shared_pre_2d[..., Fs:]
            # flextrain_swiglu_fwd(x1, x3) = silu(x1) * x3. Math wants
            # ``up * silu(gate)``, so pass x1=gate, x3=up.
            sh_act_2d = flextrain_swiglu_fwd(gate_half, up_half)  # (T, F)
            sh_each_2d = sh_act_2d @ w_down                       # (T, d)
            # Re-introduce the leading S=1 dim so callers (and the slot
            # field shape) see the canonical (T, S, ...) layout.
            x_shared_pre = x_shared_pre_2d.unsqueeze(1)           # (T, 1, 2F)
            sh_each = sh_each_2d.unsqueeze(1)                     # (T, 1, d)
            return x_shared_pre, sh_each

        # Generic S>1 path (DeepSeek-style). bf16 bmm via einsum;
        # PyTorch dispatches to a tiled bmm which keeps everything in
        # the compute dtype (no fp32 promotion).
        x_shared_pre = torch.einsum(
            "td,sdf->tsf", x_2d, weights["w_shared_up"]
        )                                                          # (T, S, 2F) bf16
        up_half = x_shared_pre[..., :Fs].contiguous()
        gate_half = x_shared_pre[..., Fs:].contiguous()
        sh_act = flextrain_swiglu_fwd(gate_half, up_half)          # (T, S, F)
        sh_each = torch.einsum(
            "tsf,sfd->tsd", sh_act, weights["w_shared_down"]
        )                                                          # (T, S, d)
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
        ctx: LayerContext,
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
            ffn_norm_output, weights, slot, chunk, ctx, layer_id=layer_id,
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
        lora_capture: dict | None = None,
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
        #   up, gate      = split(sh_pre, dim=-1)     pack order: [up | gate]
        #   sh_act        = up * silu(gate)            (T, S, F)
        #   sh_each       = sh_act @ w_shared_down     (T, S, d)
        #   sh_gate_pre   = x @ w_shared_expert_gate   (T, S)
        #   sh_gate       = sigmoid(sh_gate_pre)
        #   shared_out    = sum_s sh_gate[:, s] * sh_each[:, s, :]
        #
        # All matmul/SwiGLU bwd ops here run in bf16. The only
        # surviving fp32 promotion is the sigmoid bwd
        # ``sig * (1 - sig)`` on the (T, S) gate scalars — it's
        # ~T*S elements (tiny, S=1 in production) and the math is
        # genuinely sensitive to round-off on near-saturated gates.
        # ------------------------------------------------------------------
        from flextrain.ops import flextrain_swiglu_bwd

        x_shared_pre = slot.x_shared_pre                                    # (T, S, 2F) bf16
        up_half = x_shared_pre[..., :Fs]                                     # (T, S, F)
        gate_half = x_shared_pre[..., Fs:]                                   # (T, S, F)

        if S == 1:
            # Fast path (Qwen3-Next/3.5/3.6).
            up_2d = up_half.squeeze(1).contiguous()                          # (T, F)
            gate_2d = gate_half.squeeze(1).contiguous()                      # (T, F)
            w_down = weights["w_shared_down"][0]                             # (F, d)
            w_up = weights["w_shared_up"][0]                                 # (d, 2F)

            # sigmoid(g) and the sigmoid-derivative term: kept fp32
            # only on the (T, 1) tensor for accuracy near saturation.
            x_shared_gate_f = slot.x_shared_gate.float()                     # (T, 1)
            sig_gate_f = x_shared_gate_f.sigmoid()
            sig_gate = sig_gate_f.to(slot.x_shared_gate.dtype)

            # Recompute sh_each via bf16 matmul (used for d_sh_gate).
            # Cost: same shape as a single fwd down matmul.
            # Avoids saving the (T, d) sh_each tensor.
            # Path is: sh_act = silu(gate)*up; sh_each = sh_act @ w_down.
            from flextrain.ops import flextrain_swiglu_fwd
            sh_act_2d = flextrain_swiglu_fwd(gate_2d, up_2d)                 # (T, F)
            sh_each_2d = sh_act_2d @ w_down                                  # (T, d) bf16

            dy_2d = dy_resid                                                 # (T, d)
            # d/d(sh_each) = dy * sig_gate.
            d_sh_each_2d = dy_2d * sig_gate                                  # (T, d)
            # d/d(sig_gate) = sum_d dy * sh_each. (T, 1) result.
            d_sh_gate_2d = (dy_2d * sh_each_2d).sum(dim=-1, keepdim=True)    # (T, 1)
            # d/d(sh_gate_pre) = d_sh_gate * sig*(1-sig). fp32-safe scalar op.
            d_sh_gate_pre = (
                d_sh_gate_2d.float() * sig_gate_f * (1.0 - sig_gate_f)
            ).to(dy_resid.dtype)                                             # (T, 1)

            # w_shared_expert_gate grad / dx via gate. Shape (d, 1) and
            # (T, d). bf16 throughout.
            if grads.get("g_shared_expert_gate") is not None:
                grads["g_shared_expert_gate"].addmm_(x_2d.T, d_sh_gate_pre)
            dx_via_gate = d_sh_gate_pre @ weights["w_shared_expert_gate"].T  # (T, d) bf16

            # Down-proj bwd: sh_each = sh_act @ w_down.
            # d_sh_act = d_sh_each @ w_down.T;
            # g_w_down += sh_act.T @ d_sh_each.
            if grads.get("g_shared_down") is not None:
                # Stored shape: (S=1, F, d).
                grads["g_shared_down"][0].addmm_(sh_act_2d.T, d_sh_each_2d)
            d_sh_act_2d = d_sh_each_2d @ w_down.T                            # (T, F) bf16

            # SwiGLU bwd: returns d_gate, d_up, and recomputed sh_act
            # (which we already have, so discard).
            d_gate_2d, d_up_2d = flextrain_swiglu_bwd(
                gate_2d, up_2d, d_sh_act_2d,
            )

            # Up-proj bwd: sh_pre = x @ w_up. Pack order [up | gate].
            # d_sh_pre = cat([d_up, d_gate], dim=-1) (T, 2F).
            # g_w_up += x.T @ d_sh_pre; dx += d_sh_pre @ w_up.T.
            d_x_shared_pre_2d = torch.cat([d_up_2d, d_gate_2d], dim=-1)      # (T, 2F)
            if grads.get("g_shared_up") is not None:
                grads["g_shared_up"][0].addmm_(x_2d.T, d_x_shared_pre_2d)
            dx_via_shared_mlp = d_x_shared_pre_2d @ w_up.T                   # (T, d) bf16

        else:
            # Generic S>1 path (DeepSeek-style). bf16 bmm/einsum throughout.
            x_shared_gate_f = slot.x_shared_gate.float()                     # (T, S)
            sig_gate_f = x_shared_gate_f.sigmoid()
            sig_gate = sig_gate_f.to(slot.x_shared_gate.dtype)               # (T, S)

            # Recompute sh_act + sh_each in bf16.
            from flextrain.ops import flextrain_swiglu_fwd
            up_c = up_half.contiguous()
            gate_c = gate_half.contiguous()
            sh_act = flextrain_swiglu_fwd(gate_c, up_c)                      # (T, S, F)
            sh_each = torch.einsum(
                "tsf,sfd->tsd", sh_act, weights["w_shared_down"]
            )                                                                # (T, S, d)

            dy_2d = dy_resid                                                 # (T, d)
            d_sh_each = dy_2d.unsqueeze(1) * sig_gate.unsqueeze(-1)          # (T, S, d)
            d_sh_gate = (dy_2d.unsqueeze(1) * sh_each).sum(dim=-1)           # (T, S)
            d_sh_gate_pre = (
                d_sh_gate.float() * sig_gate_f * (1.0 - sig_gate_f)
            ).to(dy_resid.dtype)                                             # (T, S)

            if grads.get("g_shared_expert_gate") is not None:
                grads["g_shared_expert_gate"].addmm_(x_2d.T, d_sh_gate_pre)
            dx_via_gate = d_sh_gate_pre @ weights["w_shared_expert_gate"].T  # (T, d)

            if grads.get("g_shared_down") is not None:
                grads["g_shared_down"].add_(
                    torch.einsum("tsf,tsd->sfd", sh_act, d_sh_each)
                )
            d_sh_act = torch.einsum(
                "tsd,sfd->tsf", d_sh_each, weights["w_shared_down"]
            )                                                                # (T, S, F)

            # SwiGLU bwd, fused.
            d_gate, d_up = flextrain_swiglu_bwd(gate_c, up_c, d_sh_act)
            d_x_shared_pre = torch.cat([d_up, d_gate], dim=-1)               # (T, S, 2F)

            if grads.get("g_shared_up") is not None:
                grads["g_shared_up"].add_(
                    torch.einsum("td,tsf->sdf", x_2d, d_x_shared_pre)
                )
            dx_via_shared_mlp = torch.einsum(
                "tsf,sdf->td", d_x_shared_pre, weights["w_shared_up"]
            )                                                                # (T, d)

        # ------------------------------------------------------------------
        # Routed path bwd — delegate. The inner block accumulates
        # g_router/g_up/g_down into ``grads`` and returns dx_via_routed.
        # ``lora_capture`` (when not None) is forwarded so the routed
        # backend can stage per-expert grouped intermediates for the
        # downstream LoRA wgrad finalize.
        # ------------------------------------------------------------------
        dx_via_routed = self._routed_ffn.bwd(
            dy_resid, weights, grads, slot, ctx, chunk, layer_id=layer_id,
            lora_capture=lora_capture,
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
