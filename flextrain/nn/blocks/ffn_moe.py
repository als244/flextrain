"""Mixture-of-Experts FFN block.

Ports the MoE feedforward from ``orig/awsm_transformer/moe_layer.py``
into a composable algorithmic block that layers can plug in place of
:class:`SwiGLUFFN`. Matches the OLMoE / Qwen3-MoE / GPT-OSS family of
MoE routings: top-k expert selection, softmax gating, scatter-gather
compute pattern.

Param spec
----------
* ``w_router`` -- ``(d_model, num_experts)`` gating linear
* ``w_up``     -- ``(num_experts, 2 * expert_dim, d_model)`` -- stacked
                  per-expert SwiGLU up (out, in) order. The 2F axis is
                  CHUNKED ``[up_first_F, gate_second_F]``.
* ``w_down``   -- ``(num_experts, d_model, expert_dim)`` -- stacked
                  per-expert SwiGLU down (out, in) order.

Activation schema
-----------------
Tier 0 (always saved):
* ``x_router``                -- ``(T, num_experts)``  bf16 -- gate logits
* ``expert_counts``           -- ``(num_experts,)``    int32
* ``router_weights``          -- ``(T, top_k)``        bf16 -- softmax(topk(logits))
* ``chosen_experts``          -- ``(T, top_k)``        int32
* ``scattered_router_weights``-- ``(T*top_k, 1)``      bf16 -- router weights in scatter order

Tier 3 (saved at max level):
* ``x_up``                    -- ``(T*top_k, 2*expert_dim)`` bf16 -- per-slot
                                 pre-SwiGLU activations. Mirrors orig's
                                 flat activation buffer; views indexed
                                 by sort order at bwd time.

The x1 / x3 (saved at tier 3 for dense SwiGLUFFN) are REPLACED by a
single ``x_up`` that holds ``concat(x1, x3)`` per expert slot.

Load-balance loss (optional)
---------------------------
Qwen3-MoE and OLMoE both train with a load-balance auxiliary loss
proportional to ``num_experts * sum(fraction_dispatched *
mean_router_prob)``. The backward of this loss is handled by
``flextrain_load_balance_bwd`` which adds into ``dlogits`` before the
router-weight gradient is computed. See ``forward_backward`` below.

Non-trivial invariants
----------------------
* ``token_index_mapping[layer_id]`` and ``expert_counts_host[layer_id]``
  are provided by the engine per-chunk; the block consumes them from
  ``chunk.moe_mapping`` / ``chunk.moe_expert_counts`` (added to
  :class:`ChunkMeta` when any MoE layer is present).
* Scatter is *T*K* rows in ``input`` order; expert ``e`` gets indices
  ``[cum[e-1]:cum[e]]`` in the sorted view. The kernels
  ``flextrain_moe_scatter`` / ``flextrain_moe_gather`` take ``indices`` (T*K,)
  as the mapping from output row to input (T, K) (row, k) pair.
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
from flextrain.ops.full_moe import (
    routed_swiglu_moe_bwd,
    routed_swiglu_moe_fwd,
    routed_swiglu_moe_recompute_x_up,
)
from flextrain.ops.moe_backend import (
    FlextrainMoEExpertCompute,
    MoEExpertCompute,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoESwiGLUConfig:
    """Configuration for :class:`MoESwiGLUFFN`.

    ``num_experts`` is the total number of routed experts; ``top_k`` is
    how many each token is dispatched to (with ``sum(router_weights)=1``
    for each token after softmax).

    ``load_balance_coef`` controls the auxiliary load-balance loss
    (``aux_loss = lbc * E * sum(f_i * p_i)``, summed over experts,
    where ``f_i`` is fraction of tokens dispatched to expert ``i`` and
    ``p_i`` is the mean router probability assigned to expert ``i``).
    Set to 0 to disable. OLMoE's default is 0.01; Qwen3-MoE uses 0.001.

    Dtypes follow the same convention as :class:`SwiGLUConfig`:
    compute-dtype is used for the stacked expert weights; master /
    grad dtypes may differ (default: same as compute).
    """

    d_model: int
    expert_dim: int
    num_experts: int
    top_k: int
    load_balance_coef: float = 0.0
    # Routing mode. Two variants observed in the wild:
    # * ``"topk_then_softmax"`` — pick top-K logits, softmax over K.
    #   Weights sum to 1. Matches Qwen3-MoE default and OLMoE with
    #   ``norm_topk_prob=True`` (Qwen3-MoE). This is what the fused
    #   Triton kernel implements.
    # * ``"softmax_then_topk"`` — softmax over all E logits, keep top-K
    #   weights as-is (NOT renormalized). Matches OLMoE-1B-7B
    #   (``norm_topk_prob=False``); weights sum to <= 1 and are small.
    routing_mode: str = "topk_then_softmax"
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.routing_mode not in (
            "topk_then_softmax", "softmax_then_topk"
        ):
            raise ValueError(
                f"Unknown routing_mode: {self.routing_mode!r}"
            )


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class MoESwiGLUFFN:
    """Mixture-of-Experts SwiGLU FFN.

    Forward:
        gate_logits = h @ w_router
        router_w, expert_ids = topk_softmax(gate_logits)       # (T,K)
        indices, counts = sort_tokens_by_expert(expert_ids)    # (T*K,)
        x_sorted = scatter(h, indices)                         # (T*K, d_model)
        for each expert e in [0, E):
            x_pre = x_sorted[range_e] @ w_up[e]                # (T_e, 2F)
            x_act = swiglu(x_pre)                              # (T_e, F)
            out[range_e] = x_act @ w_down[e]                   # (T_e, d_model)
        layer_out = residual + gather(out, indices, router_w)  # (T, d_model)

    Backward mirrors the forward pipeline inverse; see ``bwd`` below.
    """

    def __init__(
        self, cfg: MoESwiGLUConfig,
        *,
        expert_compute: MoEExpertCompute | None = None,
    ) -> None:
        self.cfg = cfg
        # Backend that owns scatter + per-expert MLP + gather. Defaults
        # to flextrain. ScatterMoE / sonic backends are added later.
        self.expert_compute: MoEExpertCompute = (
            expert_compute if expert_compute is not None
            else FlextrainMoEExpertCompute()
        )

    # ------------------------------------------------------------------
    # Declarations consumed by the layer / engine.
    # ------------------------------------------------------------------

    def fields(self) -> tuple[ActivationField, ...]:
        """Concatenate shared block fields + backend-private fields.

        Shared (router state + pre-SwiGLU activations) are declared
        here; the backend declares its private scatter bookkeeping
        (e.g., flextrain's ``index_mapping``, ``expert_counts``,
        ``scattered_router_weights``).
        """
        cfg = self.cfg
        bf = cfg.compute_dtype
        shared = (
            # Tier 0 (always saved): router state needed by router-bwd.
            ActivationField(
                "x_router",
                lambda n, d: (n, cfg.num_experts),
                bf,
                tier=0,
            ),
            ActivationField(
                "router_weights",
                lambda n, d: (n, cfg.top_k),
                bf,
                tier=0,
            ),
            ActivationField(
                "chosen_experts",
                lambda n, d: (n, cfg.top_k),
                torch.int32,
                tier=0,
            ),
            # Tier 3 (max level): per-slot pre-SwiGLU activations.
            # Recomputable via expert_compute.fwd_recompute when tier <3.
            ActivationField(
                "x_up",
                lambda n, d: (n * cfg.top_k, 2 * cfg.expert_dim),
                bf,
                tier=3,
                token_axis=None,
            ),
        )
        backend_private = self.expert_compute.activation_fields(
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            expert_dim=cfg.expert_dim,
            d_model=cfg.d_model,
            compute_dtype=bf,
        )
        return shared + backend_private

    def param_spec(self) -> ParamSpec:
        cfg = self.cfg
        return ParamSpec(
            tensors=(
                TensorSpec(
                    "w_router",
                    lambda d: (cfg.d_model, cfg.num_experts),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "w_up",
                    # (E, out=2F, in=d) — matches HF gate_up_proj shape and
                    # Megatron TEGroupedMLP / vLLM fused_moe / scattermoe
                    # convention. Allows sonic to drop its per-call axis
                    # transpose and scattermoe to drop its bwd
                    # `.permute(0, 2, 1).contiguous()`. Per-expert dispatcher
                    # in flextrain backend uses cuBLAS transpose-on-fly via
                    # `w_up_e.T`. Gate/up packing within the 2F axis is
                    # CHUNKED [up_first_F, gate_second_F] (matches HF +
                    # ecosystem). Sonic does an interleave at its boundary.
                    lambda d: (cfg.num_experts, 2 * cfg.expert_dim, cfg.d_model),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "w_down",
                    # (E, out=d, in=F) — matches HF down_proj. No transpose
                    # needed at load.
                    lambda d: (cfg.num_experts, cfg.d_model, cfg.expert_dim),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Compute.
    # ------------------------------------------------------------------

    def fwd(
        self,
        ffn_norm_output: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        attn_output_with_residual: torch.Tensor,
        out_tensor: torch.Tensor,
        slot,
        ctx: LayerContext,
        chunk: ChunkMeta,
        *,
        layer_id: int,
    ) -> torch.Tensor:
        """Compute the MoE SwiGLU forward.

        Thin caller of :func:`flextrain.ops.full_moe.routed_swiglu_moe_fwd`.
        Runs the router (always flextrain-owned) then delegates the
        scatter + per-expert MLP + gather to ``self.expert_compute``.
        Writes router state into tier-0 slot fields, per-slot pre-SwiGLU
        activations into tier-3 ``slot.x_up`` (or whatever the backend
        declares), and the residual-added output into ``out_tensor``.
        """
        cfg = self.cfg
        routed_swiglu_moe_fwd(
            ffn_norm_output, weights,
            out_tensor=out_tensor,
            residual=attn_output_with_residual,
            slot=slot,
            chunk_extra=chunk.extra,
            layer_id=layer_id,
            top_k=cfg.top_k,
            num_experts=cfg.num_experts,
            routing_mode=cfg.routing_mode,
            primary_stream=ctx.stream,
            secondary_stream=ctx.secondary_stream,
            scratch_fn=ctx.scratch,
            expert_compute=self.expert_compute,
        )
        return out_tensor

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
        """Refill tier-3 ``slot.x_up`` (or whatever the backend
        declares) by re-running the dropped fwd work. Called by the
        enclosing layer's ``forward_recompute`` when save_level < 3.

        Stashes the backend's recompute handoff (e.g., a rescatter
        buffer) on ``slot.aux["moe_recompute_handoff"]`` so the
        subsequent bwd can reuse it.
        """
        handoff = routed_swiglu_moe_recompute_x_up(
            ffn_norm_output, weights, slot, chunk.extra, layer_id,
            primary_stream=ctx.stream,
            secondary_stream=ctx.secondary_stream,
            scratch_fn=ctx.scratch,
            expert_compute=self.expert_compute,
        )
        slot.aux["moe_recompute_handoff"] = handoff

    def bwd(
        self,
        dy_resid: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
        chunk: ChunkMeta,
        *,
        layer_id: int,
        lora_capture: dict | None = None,
    ) -> torch.Tensor:
        """MoE backward. Thin caller of
        :func:`flextrain.ops.full_moe.routed_swiglu_moe_bwd`.

        ``lora_capture`` (when not None) is a caller-owned dict the
        backend populates with per-expert grouped intermediates that
        the LoRA wrapper's ``backward_wgrad`` consumes via
        grouped_mm-batched dA/dB accumulation. See the
        ``MoEExpertCompute.bwd`` protocol docstring for the dict
        contract.
        """
        cfg = self.cfg
        ffn_norm_output = slot.aux.get("recompute_ffn_norm_output", None)
        if ffn_norm_output is None:
            raise RuntimeError(
                "MoE backward requires caller to provide ffn_norm_output "
                "via slot.aux['recompute_ffn_norm_output'] (same "
                "convention as SwiGLUFFN dense)."
            )
        recompute_handoff = slot.aux.pop("moe_recompute_handoff", None)
        return routed_swiglu_moe_bwd(
            dy_resid, weights, grads, slot, chunk.extra, layer_id,
            ffn_norm_output=ffn_norm_output,
            top_k=cfg.top_k,
            num_experts=cfg.num_experts,
            routing_mode=cfg.routing_mode,
            load_balance_coef=cfg.load_balance_coef,
            total_tokens_per_step=ctx.total_tokens_per_step,
            primary_stream=ctx.stream,
            secondary_stream=ctx.secondary_stream,
            scratch_fn=ctx.scratch,
            expert_compute=self.expert_compute,
            scattered_x_recompute=recompute_handoff,
            lora_capture=lora_capture,
        )

    # ------------------------------------------------------------------
    # FLOP accounting.
    # ------------------------------------------------------------------

    def compute_cost(self, chunk: ChunkMeta, max_tier: int) -> ComputeCost:
        """FLOP estimate. Only counts the data-path (router is trivial).

        Each token activates ``top_k`` experts, so per-token FLOPs =
        ``top_k * (2 * d_model * 2 * expert_dim + 2 * expert_dim * d_model)``
        -- the same as a dense SwiGLU sized at ``top_k * expert_dim``
        per token, modulo the small router matmul we ignore here.

        Tier-3 save avoids the ``w_up`` recompute (gate+up concatenated
        into x_up).
        """
        cfg = self.cfg
        avoided = [0] * (max_tier + 1)
        total = 0
        top_k = cfg.top_k

        for seq_len in chunk.seq_lens_host:
            up_gate = 2 * seq_len * top_k * cfg.d_model * (2 * cfg.expert_dim)
            total += up_gate
            if max_tier >= 3:
                avoided[3] += up_gate
            down = 2 * seq_len * top_k * cfg.expert_dim * cfg.d_model
            total += down

        # Small router matmul (kept for completeness; not tier-dependent).
        for seq_len in chunk.seq_lens_host:
            total += 2 * seq_len * cfg.d_model * cfg.num_experts

        return ComputeCost(
            total_fwd_flops=total,
            avoided_recompute_flops=tuple(avoided),
        )
