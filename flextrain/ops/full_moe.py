"""High-level MoE composition ops.

The routed MoE forward/backward is composed of three stages:

1. **Routing** — router-gate matmul + top-k + softmax → ``(T, K)``
   expert ids and weights.
2. **Dispatch / scatter** — sort tokens by expert into a flat
   ``(T*K, d_model)`` buffer; bookkeeping for the per-expert ranges.
3. **Expert compute** — per-expert SwiGLU fwd/bwd (cuBLAS dispatcher
   matmuls + Triton ``flextrain_swiglu_moe_*`` kernels).
4. **Combine / gather** — weighted sum back to ``(T, d_model)``,
   add residual.

This module exposes the routed-SwiGLU pipeline at three levels of
granularity so different MoE backends can hot-swap whichever stage
they implement differently:

End-to-end ops (the typical entry point — :class:`MoESwiGLUFFN` is a
thin caller of these):

* :func:`routed_swiglu_moe_fwd` — runs all four stages, populates
  the activation slot's tier-0 fields and tier-3 ``x_up``, and
  writes the post-residual output into ``out_tensor``.
* :func:`routed_swiglu_moe_bwd` — mirror; consumes the tier-0
  router state, runs the expert bwd, gather, router-gate-bwd,
  optional load-balance bwd, and the router-Wgrad / FFN-norm-
  upstream addmms.
* :func:`routed_swiglu_moe_recompute_x_up` — tier-3 recompute hook
  for save_level<3; rescatter + per-expert up-projection only.

Phase ops (compose your own pipeline). Backends that don't use
scatter/gather (e.g. sonic-MoE with a different dispatch shape) skip
:func:`dispatch_scatter` / :func:`combine_gather` and substitute
their own, but can still reuse :func:`route_topk_softmax` and the
expert-loop ops below.

* :func:`route_topk_softmax`
* :func:`dispatch_scatter`
* :func:`combine_gather`

Expert-compute ops (the SwiGLU backend per se — swap this when
plugging in a different expert kernel):

* :func:`swiglu_expert_loop_fwd`
* :func:`swiglu_expert_loop_bwd`
* :func:`swiglu_expert_loop_recompute_x_up`
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping

import torch

from flextrain.ops import (
    flextrain_copy_expert_counts,
    flextrain_fused_topk_softmax,
    flextrain_load_balance_bwd,
    flextrain_moe_gather,
    flextrain_moe_router_gate_bwd,
    flextrain_moe_scatter,
    flextrain_moe_scatter_routing_weights,
    flextrain_moe_sort,
    flextrain_swiglu_moe_bwd,
    flextrain_swiglu_moe_fwd,
)
from flextrain.ops._kernels._matmul_dispatchers import (
    dispatcher,
    dispatcher_secondary,
)


# ---------------------------------------------------------------------------
# Phase ops.
# ---------------------------------------------------------------------------


def route_topk_softmax(
    x: torch.Tensor,                       # (T, d_model)
    w_router: torch.Tensor,                # (d_model, num_experts)
    *,
    top_k: int,
    routing_mode: str,
    gate_logits_out: torch.Tensor,         # (T, num_experts) -- written
    topk_ids_out: torch.Tensor,            # (T, top_k)       -- written
    topk_weights_out: torch.Tensor,        # (T, top_k)       -- written
) -> tuple[torch.Tensor, torch.Tensor]:
    """Router projection + top-k softmax.

    Computes ``gate_logits = x @ w_router`` and then runs the fused
    Triton top-k + softmax kernel. All three output buffers must be
    pre-allocated by the caller (typically slot tier-0 fields), and
    are written in place.

    Returns ``(router_weights, topk_ids)`` views into the provided
    buffers — same tensors as ``topk_weights_out`` / ``topk_ids_out``,
    returned for ergonomic chaining.
    """
    torch.matmul(x, w_router, out=gate_logits_out)
    router_weights, topk_ids = flextrain_fused_topk_softmax(
        gate_logits_out,
        top_k=top_k,
        topk_ids_out=topk_ids_out,
        topk_weights_out=topk_weights_out,
        mode=routing_mode,
    )
    return router_weights, topk_ids


def dispatch_scatter(
    x: torch.Tensor,                       # (T, d_model)
    router_weights: torch.Tensor,          # (T, top_k)
    topk_ids: torch.Tensor,                # (T, top_k)
    *,
    num_experts: int,
    index_mapping: torch.Tensor,           # (T, top_k) int32 -- written by sort
    expert_counts_gpu: torch.Tensor,       # (E,) int32 -- written by sort
    expert_counts_cpu: torch.Tensor,       # (E,) int32 pinned host -- written by D->H copy
    scattered_x_out: torch.Tensor,         # (T*K, d_model) -- written
    scattered_router_weights_out: torch.Tensor,  # (T*K, 1) bf16 -- written
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort tokens by expert; scatter inputs + router weights into
    sorted layout; copy expert counts to pinned host and sync.

    On return, ``scattered_x_out`` holds ``x`` permuted by expert
    (each expert's tokens contiguous), ``scattered_router_weights_out``
    holds the per-slot router weights in matching order, and
    ``expert_counts_cpu`` contains the per-expert token count. The
    caller can read the CPU counts immediately (we sync inside).

    Returns ``(indices, expert_counts_cpu)``: ``indices`` is the
    ``(T, top_k)`` int32 mapping from original (token, k-slot) to
    sorted-position; needed at bwd time.
    """
    indices, _expert_counts_gpu = flextrain_moe_sort(
        topk_ids,
        num_experts=num_experts,
        indices=index_mapping,
        expert_counts_gpu=expert_counts_gpu,
    )
    flextrain_moe_scatter(x, indices, out=scattered_x_out)
    flextrain_moe_scatter_routing_weights(
        router_weights, indices, out=scattered_router_weights_out,
    )
    flextrain_copy_expert_counts(expert_counts_gpu, expert_counts_cpu)
    # Sync so the python loop in swiglu_expert_loop_fwd can read the
    # per-expert counts on the CPU.
    torch.cuda.current_stream().synchronize()
    return indices, expert_counts_cpu


def combine_gather(
    scattered_y: torch.Tensor,             # (T*K, d_model) post-expert outputs
    indices: torch.Tensor,                 # (T, top_k) int32
    *,
    router_weights: torch.Tensor,          # (T, top_k) bf16 -- ORIGINAL order, NOT scattered
    residual: torch.Tensor,                # (T, d_model) bf16 -- added in-kernel
    out_tensor: torch.Tensor,              # (T, d_model) bf16 -- written
) -> torch.Tensor:
    """Weighted gather + residual add.

    Thin kwargs-only wrapper around :func:`flextrain_moe_gather` for
    callers who always pass weights and residual.
    """
    return flextrain_moe_gather(
        scattered_y, indices,
        residual=residual,
        weights=router_weights,
        out=out_tensor,
    )


# ---------------------------------------------------------------------------
# End-to-end ops. Block fwd/bwd reduce to a single call to one of these.
# ---------------------------------------------------------------------------


def routed_swiglu_moe_fwd(
    ffn_norm_output: torch.Tensor,         # (T, d_model)
    weights: Mapping[str, torch.Tensor],   # {w_router, w_up, w_down}
    *,
    out_tensor: torch.Tensor,              # (T, d_model) -- written
    residual: torch.Tensor,                # (T, d_model) -- added during gather
    slot: Any,                             # ActivationSlot
    chunk_extra: MutableMapping[str, Any], # chunk.meta.extra
    layer_id: int,
    top_k: int,
    num_experts: int,
    routing_mode: str,
    primary_stream: torch.cuda.Stream,
    secondary_stream: torch.cuda.Stream | None,
    scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    expert_compute,                        # MoEExpertCompute backend
) -> None:
    """End-to-end routed-SwiGLU MoE forward.

    Runs the router (always flextrain-owned) then delegates the
    scatter + per-expert MLP + gather pipeline to ``expert_compute``.
    Writes router state into tier-0 slot fields and the residual-added
    output into ``out_tensor``.
    """
    # 1) Routing.
    router_weights, topk_ids = route_topk_softmax(
        ffn_norm_output, weights["w_router"],
        top_k=top_k,
        routing_mode=routing_mode,
        gate_logits_out=slot.x_router,
        topk_ids_out=slot.chosen_experts,
        topk_weights_out=slot.router_weights,
    )

    # Backend-parity diagnostic dump (env-var-gated; no-op otherwise).
    # Captures router state + fwd input; out is dumped after step 2.
    from flextrain.ops._moe_dump import dump_tensor
    dump_tensor("ffn_norm_output", ffn_norm_output, layer_id=layer_id, phase="fwd")
    dump_tensor("x_router", slot.x_router, layer_id=layer_id, phase="fwd")
    dump_tensor("chosen_experts", slot.chosen_experts, layer_id=layer_id, phase="fwd")
    dump_tensor("router_weights", slot.router_weights, layer_id=layer_id, phase="fwd")

    # 2) Backend-owned MoE block (scatter + experts + gather).
    expert_compute.fwd(
        ffn_norm_output, router_weights, topk_ids, weights,
        out=out_tensor,
        residual=residual if expert_compute.supports_residual_in_gather else None,
        slot=slot,
        chunk_extra=chunk_extra,
        layer_id=layer_id,
        primary_stream=primary_stream,
        secondary_stream=secondary_stream,
        scratch_fn=scratch_fn,
    )
    if not expert_compute.supports_residual_in_gather and residual is not None:
        out_tensor.add_(residual.view(-1, ffn_norm_output.shape[-1]))

    # Dump the MoE block's output (with residual added) — this is the
    # transition into the next layer.
    dump_tensor("out", out_tensor, layer_id=layer_id, phase="fwd")


def routed_swiglu_moe_bwd(
    dy_resid: torch.Tensor,                # (T, d_model)
    weights: Mapping[str, torch.Tensor],
    grads: MutableMapping[str, torch.Tensor],
    slot: Any,
    chunk_extra: MutableMapping[str, Any],
    layer_id: int,
    *,
    ffn_norm_output: torch.Tensor,         # caller-recomputed (T, d_model)
    top_k: int,
    num_experts: int,
    routing_mode: str,
    load_balance_coef: float,
    total_tokens_per_step: int | None,
    primary_stream: torch.cuda.Stream,
    secondary_stream: torch.cuda.Stream | None,
    scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    expert_compute,                        # MoEExpertCompute backend
    scattered_x_recompute: torch.Tensor | None = None,
    skip_grads: frozenset[str] = frozenset(),
    lora_per_expert_callback: Callable | None = None,
) -> torch.Tensor:
    """End-to-end routed-SwiGLU MoE backward. Returns
    ``ffn_norm_upstream`` (the d-loss/d-input gradient).

    The scatter + per-expert bwd + gather happens inside
    ``expert_compute.bwd``; the router-gate-bwd, load-balance, and
    w_router wgrad steps stay flextrain-owned and run here.

    ``scattered_x_recompute`` is the caller-side handoff from
    :func:`routed_swiglu_moe_recompute_x_up` (when save_level<3 and
    that ran earlier in this bwd iter); ``None`` causes the backend
    to re-scatter from ``ffn_norm_output``.

    LoRA: ``skip_grads`` and ``lora_per_expert_callback`` are
    flextrain-only kwargs forwarded to the backend; non-flextrain
    backends will reject non-empty ``skip_grads``.
    """
    num_tokens, d_model = dy_resid.shape
    primary_stream_ptr = primary_stream.cuda_stream

    # Backend-parity diagnostic dump (env-var-gated; no-op otherwise).
    from flextrain.ops._moe_dump import dump_tensor
    dump_tensor("dy", dy_resid, layer_id=layer_id, phase="bwd")

    # 1-4) Backend-owned MoE block bwd (scatter + experts + gather).
    # Returns dx (the gradient at the FFN-norm output); writes scattered
    # dprobs into slot.aux["moe_dprobs"] for step 5.
    backend_bwd_kwargs: dict[str, Any] = {
        "slot": slot,
        "chunk_extra": chunk_extra,
        "layer_id": layer_id,
        "primary_stream": primary_stream,
        "secondary_stream": secondary_stream,
        "scratch_fn": scratch_fn,
        "recompute_handoff": scattered_x_recompute,
    }
    # LoRA-only (flextrain backend accepts these as extra kwargs).
    if skip_grads or lora_per_expert_callback is not None:
        backend_bwd_kwargs["skip_grads"] = skip_grads
        backend_bwd_kwargs["lora_per_expert_callback"] = lora_per_expert_callback
    ffn_norm_upstream = expert_compute.bwd(
        dy_resid, ffn_norm_output, weights, grads, **backend_bwd_kwargs,
    )

    # 5) Router gate gradient: per-token d_logit from per-slot dprobs.
    dprobs = slot.aux.pop("moe_dprobs")
    dlogits = torch.zeros(
        (num_tokens, num_experts),
        dtype=dy_resid.dtype, device=dy_resid.device,
    )
    # Flextrain layout: dprobs is scattered (TK, 1); the kernel
    # unpermutes internally. When other backends are added we'll
    # generalize this.
    flextrain_moe_router_gate_bwd(
        slot.router_weights, dprobs, slot.index_mapping, slot.chosen_experts,
        dlogits=dlogits,
        mode=routing_mode,
        logits=slot.x_router,
    )

    # 6) Optional load-balance loss gradient (added into dlogits).
    if load_balance_coef > 0.0 and total_tokens_per_step is not None:
        flextrain_load_balance_bwd(
            logits=slot.x_router,
            expert_counts=expert_compute.expert_counts_gpu(slot),
            num_experts=num_experts,
            alpha=load_balance_coef,
            tokens_per_step=total_tokens_per_step,
            top_k=top_k,
            dlogits=dlogits,
        )

    # 7) Router weight gradient + downstream FFN-norm-upstream
    # accumulation. dgrad always runs; wgrad is skip-able under LoRA.
    dispatcher.matmul(
        primary_stream_ptr,
        A=dlogits, B=weights["w_router"].T,
        C=ffn_norm_upstream, D=ffn_norm_upstream,
        beta=1.0, alpha=1.0,
    )
    if "g_router" in skip_grads:
        if lora_per_expert_callback is not None:
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

    # Diagnostic dump of post-bwd grad accumulators (env-var-gated).
    if grads.get("g_up") is not None:
        dump_tensor("g_up", grads["g_up"], layer_id=layer_id, phase="bwd")
    if grads.get("g_down") is not None:
        dump_tensor("g_down", grads["g_down"], layer_id=layer_id, phase="bwd")
    if grads.get("g_router") is not None:
        dump_tensor("g_router", grads["g_router"], layer_id=layer_id, phase="bwd")

    return ffn_norm_upstream


def routed_swiglu_moe_recompute_x_up(
    ffn_norm_output: torch.Tensor,         # (T, d_model)
    weights: Mapping[str, torch.Tensor],
    slot: Any,
    chunk_extra: MutableMapping[str, Any],
    layer_id: int,
    *,
    primary_stream: torch.cuda.Stream,
    secondary_stream: torch.cuda.Stream | None,
    scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    expert_compute,                        # MoEExpertCompute backend
) -> Any:
    """Tier-(<max) recompute hook. Delegates to ``expert_compute.fwd_recompute``,
    which repopulates whatever slot fields were dropped at the chosen
    save tier (typically ``slot.x_up``) and returns an opaque handoff
    the backend's bwd will pick up (e.g., a rescatter buffer the bwd
    can reuse instead of re-scattering)."""
    return expert_compute.fwd_recompute(
        ffn_norm_output, weights,
        slot=slot,
        chunk_extra=chunk_extra,
        layer_id=layer_id,
        primary_stream=primary_stream,
        secondary_stream=secondary_stream,
        scratch_fn=scratch_fn,
    )


def swiglu_expert_loop_fwd(
    scattered_x: torch.Tensor,         # (TK, d_model) bf16; scatter input → expert output, in-place
    x_preact_buf: torch.Tensor,        # (TK, 2 * F) bf16; per-slot pre-SwiGLU (saved for bwd)
    w_up: torch.Tensor,                # (E, d_model, 2 * F) bf16
    w_down: torch.Tensor,              # (E, F, d_model) bf16
    expert_counts_cpu: torch.Tensor,   # (E,) int (host, pinned) — count per expert
    *,
    primary_stream: torch.cuda.Stream,
    secondary_stream: torch.cuda.Stream | None,
    x_act_even: torch.Tensor,          # (max_exp_tokens, F) bf16 scratch
    x_act_odd: torch.Tensor,           # (max_exp_tokens, F) bf16 scratch (=x_act_even when no 2nd stream)
) -> None:
    """SwiGLU expert-compute fwd loop.

    For each expert ``e`` with ``T_e = expert_counts_cpu[e]`` slots:
      1. ``x_preact[start:end] = scattered_x[start:end] @ w_up[e]``
      2. ``x_act = SwiGLU(x_preact[start:end])`` (Triton fused, no fp32 promote)
      3. ``scattered_x[start:end] = x_act @ w_down[e]`` (overwrites input)

    When ``secondary_stream`` is not None, alternate odd/even experts
    across the two streams to overlap their cuBLAS matmuls. The two
    ``x_act_*`` scratches must be distinct in this case so writes
    don't race; pass the same buffer for both when no secondary
    stream is used.
    """
    num_experts = w_up.shape[0]
    primary_stream_ptr = primary_stream.cuda_stream
    use_secondary = secondary_stream is not None
    if use_secondary:
        secondary_stream_ptr = secondary_stream.cuda_stream
        secondary_stream.wait_stream(primary_stream)
    else:
        secondary_stream_ptr = primary_stream_ptr

    cur_offset = 0
    for eid in range(num_experts):
        n_exp_tokens = int(expert_counts_cpu[eid].item())
        if n_exp_tokens == 0:
            continue
        start = cur_offset
        end = cur_offset + n_exp_tokens
        cur_offset = end

        x_inp = scattered_x[start:end, :]
        x_preact = x_preact_buf[start:end, :]
        w_up_e = w_up[eid, :, :]
        w_down_e = w_down[eid, :, :]

        if use_secondary and (eid % 2 == 1):
            cur_dispatcher = dispatcher_secondary
            cur_stream_ptr = secondary_stream_ptr
            cur_stream = secondary_stream
            x_act = x_act_odd[:n_exp_tokens, :]
        else:
            cur_dispatcher = dispatcher
            cur_stream_ptr = primary_stream_ptr
            cur_stream = primary_stream
            x_act = x_act_even[:n_exp_tokens, :]

        with torch.cuda.stream(cur_stream):
            cur_dispatcher.matmul(
                cur_stream_ptr, A=x_inp, B=w_up_e, D=x_preact,
            )
            flextrain_swiglu_moe_fwd(x_preact, out=x_act)
            cur_dispatcher.matmul(
                cur_stream_ptr, A=x_act, B=w_down_e, D=x_inp,
            )

    if use_secondary:
        primary_stream.wait_stream(secondary_stream)


def swiglu_expert_loop_bwd(
    scattered_upstream: torch.Tensor,    # (TK, d_model) bf16; bwd input → dx output, in-place
    scattered_x: torch.Tensor,           # (TK, d_model) bf16; saved fwd input (for g_up wgrad)
    x_preact_buf: torch.Tensor,          # (TK, 2F) bf16; saved fwd pre-SwiGLU
    srw: torch.Tensor,                   # (TK, 1) bf16; saved scattered router weights
    dprobs: torch.Tensor,                # (TK, 1) bf16; OUTPUT: per-slot d_router_weight
    w_up: torch.Tensor,                  # (E, d_model, 2F) bf16
    w_down: torch.Tensor,                # (E, F, d_model) bf16
    grads: MutableMapping[str, torch.Tensor],  # accumulator dict
    expert_counts_cpu: torch.Tensor,     # (E,) int host — count per expert
    *,
    expert_dim: int,
    primary_stream: torch.cuda.Stream,
    secondary_stream: torch.cuda.Stream | None,
    skip_grads: frozenset[str] = frozenset(),
    lora_per_expert_callback: Callable | None = None,
) -> None:
    """SwiGLU expert-compute bwd loop.

    Steps per expert ``e`` (``T_e = expert_counts_cpu[e]``):
      a) ``dx_act_up = scattered_upstream[start:end] @ w_down[e].T``
      b) ``dx_up_up, dprobs[start:end] = swiglu_moe_bwd(...)`` — Triton
         fused; rescales by saved ``srw[start:end]`` and computes
         per-slot d_router_weight (dot product with recomputed
         post-SwiGLU activation).
      c) Optionally accumulate ``g_down[e] += fwd_act.T @ scattered_upstream[start:end]``
         (cuBLAS addmm). Skipped when ``"g_down" in skip_grads`` —
         the LoRA wrapper handles the rank-r path via callback.
      d) Overwrite ``scattered_upstream[start:end] = dx_up_up @ w_up[e].T``.
      e) Optionally accumulate ``g_up[e] += scattered_x[start:end].T @ dx_up_up``.

    Steps a/d are dgrads (always run). Steps c/e are wgrads and may be
    skipped via ``skip_grads``; in that case ``lora_per_expert_callback``
    fires per-expert with ``(name, eid, X, dY)`` so the LoRA wrapper
    can do its own rank-r accumulation without ever materializing the
    full per-expert ``dW``.
    """
    num_experts = w_up.shape[0]
    max_exp_tokens = int(expert_counts_cpu.max())
    primary_stream_ptr = primary_stream.cuda_stream
    use_secondary = secondary_stream is not None

    # Scratch carving: each expert needs (T_e, F) dx_act_up + (T_e, 2F) dx_up_up
    # + (T_e, F) fwd_act = (T_e, 4F) total per stream.
    bf = scattered_upstream.dtype
    device = scattered_upstream.device
    X_temp_even = torch.zeros(
        max_exp_tokens * (4 * expert_dim), dtype=bf, device=device,
    )
    if use_secondary:
        X_temp_odd = torch.zeros(
            max_exp_tokens * (4 * expert_dim), dtype=bf, device=device,
        )
        secondary_stream_ptr = secondary_stream.cuda_stream
        secondary_stream.wait_stream(primary_stream)
    else:
        X_temp_odd = X_temp_even
        secondary_stream_ptr = primary_stream_ptr

    # LoRA scratch: per-stream (max_T_e, r) buffers for the dY_B and X_A
    # rank-r intermediates. Untracked, tiny (~MiB), keeps the per-expert
    # callback free of allocator/dispatcher hops.
    if lora_per_expert_callback is not None:
        max_rank = getattr(lora_per_expert_callback, "max_rank", 0)
        lora_dY_B_even = torch.empty(max_exp_tokens, max_rank, dtype=bf, device=device)
        lora_X_A_even  = torch.empty(max_exp_tokens, max_rank, dtype=bf, device=device)
        if use_secondary:
            lora_dY_B_odd = torch.empty(max_exp_tokens, max_rank, dtype=bf, device=device)
            lora_X_A_odd  = torch.empty(max_exp_tokens, max_rank, dtype=bf, device=device)
        else:
            lora_dY_B_odd = lora_dY_B_even
            lora_X_A_odd  = lora_X_A_even

    cur_offset = 0
    for eid in range(num_experts):
        n_exp_tokens = int(expert_counts_cpu[eid].item())
        if n_exp_tokens == 0:
            continue
        start = cur_offset
        end = cur_offset + n_exp_tokens
        cur_offset = end

        exp_upstream = scattered_upstream[start:end, :]
        x_preact = x_preact_buf[start:end, :]
        exp_probs = srw[start:end]
        exp_dprobs = dprobs[start:end]
        w_up_e = w_up[eid, :, :]
        w_down_e = w_down[eid, :, :]

        if use_secondary and (eid % 2 == 1):
            cur_dispatcher = dispatcher_secondary
            cur_stream_ptr = secondary_stream_ptr
            cur_stream = secondary_stream
            X_temp = X_temp_odd
            lora_dY_B = lora_dY_B_odd if lora_per_expert_callback is not None else None
            lora_X_A  = lora_X_A_odd  if lora_per_expert_callback is not None else None
        else:
            cur_dispatcher = dispatcher
            cur_stream_ptr = primary_stream_ptr
            cur_stream = primary_stream
            X_temp = X_temp_even
            lora_dY_B = lora_dY_B_even if lora_per_expert_callback is not None else None
            lora_X_A  = lora_X_A_even  if lora_per_expert_callback is not None else None

        toff = 0
        dx_act_up = X_temp[toff : toff + n_exp_tokens * expert_dim].view(
            n_exp_tokens, expert_dim
        )
        toff += n_exp_tokens * expert_dim
        dx_up_up = X_temp[toff : toff + n_exp_tokens * 2 * expert_dim].view(
            n_exp_tokens, 2 * expert_dim
        )
        toff += n_exp_tokens * 2 * expert_dim
        fwd_act = X_temp[toff : toff + n_exp_tokens * expert_dim].view(
            n_exp_tokens, expert_dim
        )

        with torch.cuda.stream(cur_stream):
            # a) dx_act_up = exp_upstream @ w_down.T
            cur_dispatcher.matmul_fast(
                cur_stream_ptr, A=exp_upstream, B=w_down_e.T, D=dx_act_up,
            )
            # b) SwiGLU bwd: rescale + d_router_weight + recomputed fwd_act
            dx_up_up, exp_dprobs = flextrain_swiglu_moe_bwd(
                dx_act_up, x_preact, exp_probs,
                dx=dx_up_up, dw=exp_dprobs, fwd_act=fwd_act,
            )
            # c) g_down[e] += fwd_act.T @ exp_upstream  (or LoRA callback)
            if "g_down" in skip_grads:
                if lora_per_expert_callback is not None:
                    lora_per_expert_callback(
                        "g_down", eid, fwd_act, exp_upstream,
                        cur_dispatcher, cur_stream_ptr,
                        lora_dY_B, lora_X_A,
                    )
            elif grads.get("g_down") is not None:
                g_down_e = grads["g_down"][eid, :, :]
                cur_dispatcher.matmul_fast(
                    cur_stream_ptr,
                    A=fwd_act.T, B=exp_upstream,
                    C=g_down_e, D=g_down_e,
                    beta=1.0, alpha=1.0,
                )
            # d) dx_pre = dx_up_up @ w_up.T (overwrites exp_upstream)
            cur_dispatcher.matmul_fast(
                cur_stream_ptr, A=dx_up_up, B=w_up_e.T, D=exp_upstream,
            )
            # e) g_up[e] += scattered_x[start:end].T @ dx_up_up  (or LoRA callback)
            exp_inp = scattered_x[start:end, :]
            if "g_up" in skip_grads:
                if lora_per_expert_callback is not None:
                    lora_per_expert_callback(
                        "g_up", eid, exp_inp, dx_up_up,
                        cur_dispatcher, cur_stream_ptr,
                        lora_dY_B, lora_X_A,
                    )
            elif grads.get("g_up") is not None:
                g_up_e = grads["g_up"][eid, :, :]
                cur_dispatcher.matmul_fast(
                    cur_stream_ptr,
                    A=exp_inp.T, B=dx_up_up,
                    C=g_up_e, D=g_up_e,
                    beta=1.0, alpha=1.0,
                )

    if use_secondary:
        primary_stream.wait_stream(secondary_stream)


def swiglu_expert_loop_recompute_x_up(
    scattered_x: torch.Tensor,         # (TK, d_model) bf16; saved fwd input
    x_preact_buf: torch.Tensor,        # (TK, 2F) bf16; OUTPUT: per-slot pre-SwiGLU refilled
    w_up: torch.Tensor,                # (E, d_model, 2F)
    expert_counts_cpu: torch.Tensor,   # (E,) int host
    *,
    primary_stream: torch.cuda.Stream,
    secondary_stream: torch.cuda.Stream | None,
) -> None:
    """Tier-3 recompute: refill ``x_preact_buf`` from saved scattered
    input by re-running each expert's up-projection.

    Same primary/secondary alternation as :func:`swiglu_expert_loop_fwd`:
    even-id experts on primary, odd-id on secondary, with
    :meth:`wait_stream` book-ends so callers see strict
    primary-stream ordering on entry/exit. The secondary stream is
    optional — pass ``None`` for single-stream execution.
    """
    num_experts = w_up.shape[0]
    primary_stream_ptr = primary_stream.cuda_stream
    use_secondary = secondary_stream is not None
    if use_secondary:
        secondary_stream_ptr = secondary_stream.cuda_stream
        secondary_stream.wait_stream(primary_stream)
    else:
        secondary_stream_ptr = primary_stream_ptr

    cur_offset = 0
    for eid in range(num_experts):
        n_exp_tokens = int(expert_counts_cpu[eid].item())
        if n_exp_tokens == 0:
            continue
        start = cur_offset
        end = cur_offset + n_exp_tokens
        cur_offset = end
        x_inp = scattered_x[start:end, :]
        w_up_e = w_up[eid, :, :]
        x_preact = x_preact_buf[start:end, :]
        if use_secondary and (eid % 2 == 1):
            cur_dispatcher = dispatcher_secondary
            cur_stream_ptr = secondary_stream_ptr
        else:
            cur_dispatcher = dispatcher
            cur_stream_ptr = primary_stream_ptr
        cur_dispatcher.matmul(
            cur_stream_ptr, A=x_inp, B=w_up_e, D=x_preact,
        )

    if use_secondary:
        primary_stream.wait_stream(secondary_stream)


__all__ = (
    # End-to-end ops (typical entry point).
    "routed_swiglu_moe_fwd",
    "routed_swiglu_moe_bwd",
    "routed_swiglu_moe_recompute_x_up",
    # Phase ops (compose your own pipeline).
    "route_topk_softmax",
    "dispatch_scatter",
    "combine_gather",
    # Expert-compute backend.
    "swiglu_expert_loop_fwd",
    "swiglu_expert_loop_bwd",
    "swiglu_expert_loop_recompute_x_up",
)
