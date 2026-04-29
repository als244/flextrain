"""Mixture-of-Experts FFN block.

Ports the MoE feedforward from ``orig/awsm_transformer/moe_layer.py``
into a composable algorithmic block that layers can plug in place of
:class:`SwiGLUFFN`. Matches the OLMoE / Qwen3-MoE / GPT-OSS family of
MoE routings: top-k expert selection, softmax gating, scatter-gather
compute pattern.

Param spec
----------
* ``w_router`` -- ``(d_model, num_experts)`` gating linear
* ``w_up``     -- ``(num_experts, d_model, 2 * expert_dim)`` -- stacked
                  per-expert SwiGLU up (gate+value concatenated)
* ``w_down``   -- ``(num_experts, expert_dim, d_model)`` -- stacked
                  per-expert SwiGLU down

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
from flextrain.ops import (
    flextrain_copy_expert_counts,
    flextrain_fused_topk_softmax,
    flextrain_load_balance_bwd,
    flextrain_moe_gather,
    flextrain_moe_router_gate_bwd,
    flextrain_moe_scatter,
    flextrain_moe_scatter_routing_weights,
    flextrain_moe_sort,
    dispatcher,
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

    def __init__(self, cfg: MoESwiGLUConfig) -> None:
        self.cfg = cfg
        # Maintained across steps for diagnostic/logging purposes
        # (orig stores the same on CPU). Set to None until first forward
        # so the fallback logic in backward can distinguish "never ran".
        self._expert_hist: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Declarations consumed by the layer / engine.
    # ------------------------------------------------------------------

    def fields(self) -> tuple[ActivationField, ...]:
        cfg = self.cfg
        bf = cfg.compute_dtype
        return (
            # Tier 0 (always saved): router state.
            ActivationField(
                "x_router",
                lambda n, d: (n, cfg.num_experts),
                bf,
                tier=0,
            ),
            ActivationField(
                "expert_counts",
                lambda n, d: (cfg.num_experts,),
                torch.int32,
                tier=0,
                token_axis=None,  # shape independent of num_tokens
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
            ActivationField(
                "scattered_router_weights",
                lambda n, d: (n * cfg.top_k, 1),
                bf,
                tier=0,
                token_axis=None,  # (T*K, 1) — slices handled by block
            ),
            # Tier 3 (max level): per-slot pre-SwiGLU activations.
            ActivationField(
                "x_up",
                lambda n, d: (n * cfg.top_k, 2 * cfg.expert_dim),
                bf,
                tier=3,
                token_axis=None,
            ),
        )

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
                    lambda d: (cfg.num_experts, cfg.d_model, 2 * cfg.expert_dim),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "w_down",
                    lambda d: (cfg.num_experts, cfg.expert_dim, cfg.d_model),
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

        Writes router state into tier-0 slot fields and per-slot
        pre-SwiGLU activations into tier-3 ``slot.x_up``. Returns the
        residual-added output written into ``out_tensor``.

        ``chunk.meta.extra`` must contain ``moe_token_index_mapping``
        and ``moe_expert_counts_host`` dicts (engine allocates when
        layers declare ``moe_chunk_config``). The per-layer entries
        survive between fwd and bwd for this chunk.
        """
        cfg = self.cfg
        num_tokens = ffn_norm_output.shape[0]
        top_k = cfg.top_k
        num_experts = cfg.num_experts
        expert_dim = cfg.expert_dim
        d_model = cfg.d_model

        # 1) Router logits + top-k / softmax routing.
        gate_logits = torch.matmul(
            ffn_norm_output, weights["w_router"], out=slot.x_router
        )
        router_weights, topk_ids = flextrain_fused_topk_softmax(
            gate_logits,
            top_k=top_k,
            topk_ids_out=slot.chosen_experts,
            topk_weights_out=slot.router_weights,
            mode=cfg.routing_mode,
        )

        # 2) Sort tokens by expert. ``indices`` is a (T, top_k) int32
        # mapping: sorted slot → (token, k) pair. Written here, read
        # verbatim in bwd via chunk.extra.
        index_mapping = chunk.extra["moe_token_index_mapping"][layer_id]
        expert_counts_gpu = slot.expert_counts
        indices, expert_counts_gpu = flextrain_moe_sort(
            topk_ids,
            num_experts=num_experts,
            indices=index_mapping,
            expert_counts_gpu=expert_counts_gpu,
        )

        # 3) Scatter inputs + router weights into sorted layout.
        # ``scattered_router_weights`` and ``x_up`` are sized for
        # max_chunk_size * top_k at slot-allocation time; narrow to
        # the actual num_tokens * top_k here.
        TK = num_tokens * top_k
        scattered_x = ctx.scratch(
            (TK, d_model), ffn_norm_output.dtype
        )
        x_sorted = flextrain_moe_scatter(ffn_norm_output, indices, out=scattered_x)
        srw = slot.scattered_router_weights[:TK, :]
        flextrain_moe_scatter_routing_weights(
            router_weights, indices, out=srw
        )

        # 4) Copy expert_counts to pinned host. The engine allocated a
        # layer-specific pinned host tensor that survives across chunks.
        # ``flextrain_copy_expert_counts`` writes directly into the mapped
        # host memory on the compute stream, bypassing a D->H DMA.
        expert_counts_cpu = chunk.extra["moe_expert_counts_host"][layer_id]
        flextrain_copy_expert_counts(expert_counts_gpu, expert_counts_cpu)
        torch.cuda.current_stream().synchronize()  # CPU needs the counts
        if self._expert_hist is None:
            self._expert_hist = torch.zeros(
                num_experts, dtype=torch.int64, device="cpu"
            )
        self._expert_hist.add_(expert_counts_cpu)

        # 5) Expert loop. Per-expert SwiGLU fwd, alternating across
        # primary/secondary streams when available. Delegated to
        # ``flextrain.ops.full_moe.swiglu_expert_loop_fwd`` so the
        # expert-compute backend is hot-swappable independently of the
        # router/scatter/gather routing surrounds.
        from flextrain.ops.full_moe import swiglu_expert_loop_fwd
        max_exp_tokens = int(expert_counts_cpu.max())
        primary_stream = ctx.stream
        secondary_stream = ctx.secondary_stream
        use_secondary = secondary_stream is not None
        x_act_even = ctx.scratch(
            (max_exp_tokens, expert_dim), ffn_norm_output.dtype,
        )
        x_act_odd = (
            ctx.scratch((max_exp_tokens, expert_dim), ffn_norm_output.dtype)
            if use_secondary else x_act_even
        )
        swiglu_expert_loop_fwd(
            scattered_x=scattered_x,
            x_preact_buf=slot.x_up,
            w_up=weights["w_up"],
            w_down=weights["w_down"],
            expert_counts_cpu=expert_counts_cpu,
            primary_stream=primary_stream,
            secondary_stream=secondary_stream,
            x_act_even=x_act_even,
            x_act_odd=x_act_odd,
        )

        # 6) Gather back to (T, d_model), scaling by router weights and
        # adding the block's residual input (attn_output_with_residual).
        # This writes directly into out_tensor to avoid a DtoD copy.
        layer_output = flextrain_moe_gather(
            scattered_x,
            indices,
            residual=attn_output_with_residual.view(-1, d_model),
            weights=router_weights,
            out=out_tensor,
        )
        return layer_output

    def fwd_recompute_x_up(
        self,
        ffn_norm_output: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
        chunk: ChunkMeta,
        *,
        layer_id: int,
    ) -> None:
        """Refill tier-3 ``slot.x_up`` by re-running the expert up-projection.

        Called by the enclosing layer's ``forward_recompute`` when the
        layer's save level was < 3 (so x_up wasn't saved at fwd time).
        Mirrors ``orig/moe_layer.forward_moe_recompute`` — only the
        up-projection; the down-projection is handled inside
        ``bwd`` via ``swiglu_moe_bwd``.

        Reads ``index_mapping`` and ``expert_counts_host`` from
        ``chunk.extra`` (saved during fwd). Writes ``slot.x_up`` in
        sorted-by-expert order.
        """
        cfg = self.cfg
        num_tokens = ffn_norm_output.shape[0]
        top_k = cfg.top_k
        num_experts = cfg.num_experts
        d_model = cfg.d_model

        index_mapping = chunk.extra["moe_token_index_mapping"][layer_id]
        expert_counts_cpu = chunk.extra["moe_expert_counts_host"][layer_id]

        # Re-scatter inputs.
        scattered_x = torch.empty(
            (num_tokens * top_k, d_model),
            dtype=ffn_norm_output.dtype, device=ffn_norm_output.device,
        )
        x_sorted = flextrain_moe_scatter(
            ffn_norm_output, index_mapping, out=scattered_x
        )
        # Per-expert up-projection. Single-stream — one matmul per
        # expert is too small to benefit from secondary-stream overlap.
        from flextrain.ops.full_moe import swiglu_expert_loop_recompute_x_up
        stream_ptr = torch.cuda.current_stream().cuda_stream
        swiglu_expert_loop_recompute_x_up(
            scattered_x=scattered_x,
            x_preact_buf=slot.x_up,
            w_up=weights["w_up"],
            expert_counts_cpu=expert_counts_cpu,
            stream_ptr=stream_ptr,
        )
        # Save scattered_x on the slot aux so bwd can reuse it (avoids
        # a second scatter). Matches orig's
        # ``fwd_act_slot["scattered_x"] = scattered_x`` stash.
        slot.aux["moe_scattered_x"] = scattered_x

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
        skip_grads: frozenset[str] = frozenset(),
        lora_per_expert_callback: object | None = None,
    ) -> torch.Tensor:
        """MoE backward. Accumulates ``g_router / g_up / g_down`` and
        returns ``dx_ffn_norm_up``.

        Load-balance loss coefficient is read from ``self.cfg``; the
        ``total_tokens_per_step`` needed by the aux-loss kernel is
        read from ``ctx.total_tokens_per_step``. Mirrors
        ``orig/moe_layer.backward_moe``.

        ``skip_grads`` may include ``g_up``, ``g_down``, ``g_router``
        to gate the corresponding per-expert (or full-router) ``addmm``.
        Skipped projections do not allocate / write into ``grads[name]``;
        the caller (LoRA wrapper) supplies a
        ``lora_per_expert_callback(name, eid, X, dY)`` that fires inside
        the per-expert loop with the same ``(X, dY)`` the addmm would
        have used. The callback typically does rank-r matmuls into the
        wrapper's ``g_a/g_b`` accumulators, never materializing the
        per-expert ``dW``. ``g_router`` is callback-fired with eid=-1.
        """
        cfg = self.cfg
        num_tokens = dy_resid.shape[0]
        top_k = cfg.top_k
        num_experts = cfg.num_experts
        expert_dim = cfg.expert_dim
        d_model = cfg.d_model

        index_mapping = chunk.extra["moe_token_index_mapping"][layer_id]
        expert_counts_cpu = chunk.extra["moe_expert_counts_host"][layer_id]
        total_tokens_per_step = ctx.total_tokens_per_step

        # 1) Scatter upstream gradient by the saved sort indices (same
        # mapping used at fwd). Produces (T*K, d_model) of per-slot grads.
        scattered_upstream = torch.zeros(
            (num_tokens * top_k, d_model), dtype=dy_resid.dtype, device=dy_resid.device,
        )
        flextrain_moe_scatter(dy_resid, index_mapping, out=scattered_upstream)

        # 2) For the w_up weight-grad we need the scattered pre-ffn-norm
        # inputs. Caller has recomputed ffn_norm_output and placed it on
        # ``slot.aux["recompute_ffn_norm_output"]``. If this layer's
        # save level was < 3 and ``fwd_recompute_x_up`` ran earlier in
        # this bwd iter, it stashed ``scattered_x`` too — reuse it
        # instead of re-scattering.
        ffn_norm_output = slot.aux.get("recompute_ffn_norm_output", None)
        if ffn_norm_output is None:
            raise RuntimeError(
                "MoE backward requires caller to provide ffn_norm_output "
                "via slot.aux['recompute_ffn_norm_output'] (same "
                "convention as SwiGLUFFN dense)."
            )
        scattered_x = slot.aux.pop("moe_scattered_x", None)
        if scattered_x is None:
            # No pre-stashed scatter: compute now.
            scattered_x = torch.zeros(
                (num_tokens * top_k, d_model),
                dtype=dy_resid.dtype, device=dy_resid.device,
            )
            flextrain_moe_scatter(ffn_norm_output, index_mapping, out=scattered_x)

        # 3) Expert backward loop. Delegated to
        # ``flextrain.ops.full_moe.swiglu_expert_loop_bwd`` — same
        # extraction as fwd; this stays as a single hot-swap point
        # for alternative expert-compute backends (sonic-MoE etc.).
        # The loop accumulates ``g_up`` / ``g_down`` (or routes per-
        # expert tiles to ``lora_per_expert_callback`` when those
        # names are in ``skip_grads``) and overwrites
        # ``scattered_upstream`` with per-slot dx in-place. Returned
        # ``dprobs`` carries per-slot d_router_weight for the
        # router-grad math below.
        from flextrain.ops.full_moe import swiglu_expert_loop_bwd
        TK = num_tokens * top_k
        srw = slot.scattered_router_weights[:TK, :]
        dprobs = torch.zeros_like(srw)
        primary_stream = ctx.stream
        primary_stream_ptr = primary_stream.cuda_stream
        secondary_stream = ctx.secondary_stream
        swiglu_expert_loop_bwd(
            scattered_upstream=scattered_upstream,
            scattered_x=scattered_x,
            x_preact_buf=slot.x_up,
            srw=srw,
            dprobs=dprobs,
            w_up=weights["w_up"],
            w_down=weights["w_down"],
            grads=grads,
            expert_counts_cpu=expert_counts_cpu,
            expert_dim=expert_dim,
            primary_stream=primary_stream,
            secondary_stream=secondary_stream,
            skip_grads=skip_grads,
            lora_per_expert_callback=lora_per_expert_callback,
        )

        del scattered_x

        # 4) Gather the per-slot input gradients back to per-token space.
        # Result is the FFN-norm upstream gradient contribution from the
        # data path; router path adds on top below.
        ffn_norm_upstream = torch.zeros_like(dy_resid)
        flextrain_moe_gather(scattered_upstream, index_mapping, out=ffn_norm_upstream)

        # 5) Router gate gradient: per-token d_logit contribution from
        # the d_router_weight values computed above. Shape (T, E).
        # The kernel's Jacobian branches on routing_mode:
        # * topk_then_softmax: softmax was over only K selected positions,
        #   so dlogits is nonzero on K positions; K-local Jacobian.
        # * softmax_then_topk: softmax was over all E; dlogits nonzero on
        #   all E, full-E Jacobian (recomputes probs from saved logits).
        dlogits = torch.zeros(
            (num_tokens, num_experts), dtype=dy_resid.dtype, device=dy_resid.device,
        )
        router_weights = slot.router_weights
        chosen_experts = slot.chosen_experts
        flextrain_moe_router_gate_bwd(
            router_weights, dprobs, index_mapping, chosen_experts,
            dlogits=dlogits,
            mode=cfg.routing_mode,
            logits=slot.x_router,
        )

        # 6) Optional load-balance loss gradient (added into dlogits).
        if cfg.load_balance_coef > 0.0 and total_tokens_per_step is not None:
            flextrain_load_balance_bwd(
                logits=slot.x_router,
                expert_counts=slot.expert_counts,
                num_experts=num_experts,
                alpha=cfg.load_balance_coef,
                tokens_per_step=total_tokens_per_step,
                top_k=top_k,
                dlogits=dlogits,
            )

        # 7) Router weight gradient + downstream FFN-norm-upstream
        # accumulation. These run AFTER the secondary-stream join above,
        # always on the primary stream/dispatcher.
        # ffn_norm_upstream += dlogits @ w_router.T (always runs -- dgrad).
        dispatcher.matmul(
            primary_stream_ptr,
            A=dlogits, B=weights["w_router"].T,
            C=ffn_norm_upstream, D=ffn_norm_upstream,
            beta=1.0, alpha=1.0,
        )
        # g_router += ffn_norm_output.T @ dlogits (Wgrad -- skip-able).
        if "g_router" in skip_grads:
            if lora_per_expert_callback is not None:
                # eid=-1 marks "non-per-expert"; the LoRA wrapper does
                # a 2-D rank-r matmul rather than a per-expert 3-D one.
                lora_per_expert_callback(
                    "g_router", -1, ffn_norm_output, dlogits,
                )
        elif grads.get("g_router") is not None:
            dispatcher.matmul(
                primary_stream_ptr,
                A=ffn_norm_output.T, B=dlogits,
                C=grads["g_router"], D=grads["g_router"],
                beta=1.0, alpha=1.0,
            )
        # else: w_router frozen (LoRA on attn only). No wgrad to write.

        return ffn_norm_upstream

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
