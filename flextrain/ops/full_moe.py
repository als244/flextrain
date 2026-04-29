"""High-level MoE composition ops.

This module separates the *expert-compute* step (per-expert SwiGLU
fwd/bwd via cuBLAS dispatcher matmuls + ``flextrain_swiglu_moe_*``
Triton kernels) from the surrounding *routing infrastructure*
(router projection, top-k softmax, sort, scatter, gather, gate-grad).

The routing infrastructure stays in :mod:`flextrain.nn.blocks.ffn_moe`'s
:class:`MoESwiGLUFFN`. The expert-compute step is hot-swappable —
this module provides the SwiGLU expert backend; an alternative
backend (e.g. sonic-MoE) can replace these two functions while
keeping the rest of the block untouched.

API
---
:func:`swiglu_expert_loop_fwd` — given scattered input + per-expert
counts, run all experts (with optional secondary-stream
double-buffering), write per-slot pre-SwiGLU activations into
``x_preact_buf`` and overwrite ``scattered_x`` with each expert's
post-SwiGLU output.

:func:`swiglu_expert_loop_bwd` — given upstream grads + per-expert
counts, run all experts' bwd, accumulating ``g_up`` / ``g_down``
weight grads (or routing per-expert tiles to a LoRA callback) and
writing per-slot input grads back into ``scattered_upstream``.
"""
from __future__ import annotations

from typing import Callable, Mapping, MutableMapping

import torch

from flextrain.ops import (
    flextrain_swiglu_moe_bwd,
    flextrain_swiglu_moe_fwd,
)
from flextrain.ops._kernels._matmul_dispatchers import (
    dispatcher,
    dispatcher_secondary,
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
        else:
            cur_dispatcher = dispatcher
            cur_stream_ptr = primary_stream_ptr
            cur_stream = primary_stream
            X_temp = X_temp_even

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
            cur_dispatcher.matmul(
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
                    )
            elif grads.get("g_down") is not None:
                g_down_e = grads["g_down"][eid, :, :]
                cur_dispatcher.matmul(
                    cur_stream_ptr,
                    A=fwd_act.T, B=exp_upstream,
                    C=g_down_e, D=g_down_e,
                    beta=1.0, alpha=1.0,
                )
            # d) dx_pre = dx_up_up @ w_up.T (overwrites exp_upstream)
            cur_dispatcher.matmul(
                cur_stream_ptr, A=dx_up_up, B=w_up_e.T, D=exp_upstream,
            )
            # e) g_up[e] += scattered_x[start:end].T @ dx_up_up  (or LoRA callback)
            exp_inp = scattered_x[start:end, :]
            if "g_up" in skip_grads:
                if lora_per_expert_callback is not None:
                    lora_per_expert_callback(
                        "g_up", eid, exp_inp, dx_up_up,
                    )
            elif grads.get("g_up") is not None:
                g_up_e = grads["g_up"][eid, :, :]
                cur_dispatcher.matmul(
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
    stream_ptr: int,
) -> None:
    """Tier-3 recompute: refill ``x_preact_buf`` from saved scattered
    input by re-running each expert's up-projection. Single-stream;
    one matmul per expert is too small to benefit from secondary-stream
    overlap.
    """
    num_experts = w_up.shape[0]
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
        dispatcher.matmul(stream_ptr, A=x_inp, B=w_up_e, D=x_preact)


__all__ = (
    "swiglu_expert_loop_fwd",
    "swiglu_expert_loop_bwd",
    "swiglu_expert_loop_recompute_x_up",
)
