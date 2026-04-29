"""Fused gate-prep kernel for GatedDeltaNet (Qwen3-Next / Qwen3.5* / Qwen3.6*).

Replaces the elementwise pipeline at the entry of FLA's chunk-gated-
delta-rule with a single Triton kernel that produces both ``g`` (fp32,
the cumsum-state input) and ``beta`` (bf16, the per-token gate scalar).

Math
----
Per (token t, head h):

    g[t, h]    = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h])    # fp32
    beta[t, h] = sigmoid(b[t, h])                                   # bf16

The python path runs ~9 elementwise kernels for these (a.float(),
A_log.float(), dt_bias.float(), A_log.exp(), neg, softplus(.+.),
mul, b.float(), sigmoid, .to(bf)). Each touches small (T, n_v_heads)
tensors but the cumulative launch overhead matters: 9x40_layers = 360
launches per chunk on Qwen3.6-MoE.

This kernel reads a, b, A_log, dt_bias once and writes g, beta once.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_prep_kernel(
    A_ptr,           # (T, H) bf16 — raw a
    B_ptr,           # (T, H) bf16 — raw b
    A_log_ptr,       # (H,) bf16/fp32 — w_lin_A_log
    Dt_bias_ptr,     # (H,) bf16/fp32 — w_lin_dt_bias
    G_ptr,           # (T, H) fp32 — output g
    Beta_ptr,        # (T, H) bf16 — output beta = sigmoid(b)
    stride_a_t, stride_b_t,
    stride_g_t, stride_beta_t,
    H,
    TOTAL_ELEMS,     # T * H
    BLOCK: tl.constexpr,
):
    """One program processes BLOCK contiguous (token, head) pairs.

    Loads neg_exp(A_log) and dt_bias gathered by head once via per-
    element index, then fuses the rest of the pointwise pipeline."""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS

    # Decompose flat index -> (t, h)
    t = offs // H
    h = offs % H

    # Per-(t, h) loads of a, b. Layout is contiguous (T, H), so the
    # flat index gives the right element.
    a = tl.load(A_ptr + t * stride_a_t + h, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + t * stride_b_t + h, mask=mask, other=0.0).to(tl.float32)

    # Per-head loads of A_log and dt_bias (small lookup, no stride needed).
    A_log = tl.load(A_log_ptr + h, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(Dt_bias_ptr + h, mask=mask, other=0.0).to(tl.float32)

    # softplus(x) = log1p(exp(x)) — but use the numerically stable form
    # for x>0: softplus(x) = max(0,x) + log1p(exp(-|x|)).
    s = a + dt_bias
    abs_s = tl.abs(s)
    softplus = tl.maximum(s, 0.0) + tl.log(1.0 + tl.exp(-abs_s))

    g = -tl.exp(A_log) * softplus

    # beta = sigmoid(b)
    beta = tl.sigmoid(b)

    tl.store(G_ptr + t * stride_g_t + h, g, mask=mask)
    tl.store(
        Beta_ptr + t * stride_beta_t + h,
        beta.to(Beta_ptr.dtype.element_ty),
        mask=mask,
    )


def flextrain_gate_prep_fwd(
    a: torch.Tensor,                # (T, H) bf16
    b: torch.Tensor,                # (T, H) bf16
    A_log: torch.Tensor,            # (H,) any
    dt_bias: torch.Tensor,          # (H,) any
    *,
    g_out: torch.Tensor | None = None,
    beta_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute g (fp32) and beta (compute_dtype of ``a``) in one kernel.

    Returns
    -------
    (g, beta) where:
        g[t, h]    = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h])
        beta[t, h] = sigmoid(b[t, h])
    """
    assert a.shape == b.shape, f"a={a.shape}, b={b.shape}"
    assert a.dim() == 2, f"a must be 2-D (T, H); got {a.shape}"
    T, H = a.shape
    assert A_log.shape == (H,), f"A_log shape {A_log.shape} != (H={H},)"
    assert dt_bias.shape == (H,), f"dt_bias shape {dt_bias.shape} != (H={H},)"
    assert a.is_cuda and b.is_cuda

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if not A_log.is_contiguous():
        A_log = A_log.contiguous()
    if not dt_bias.is_contiguous():
        dt_bias = dt_bias.contiguous()

    if g_out is None:
        g_out = torch.empty(T, H, dtype=torch.float32, device=a.device)
    if beta_out is None:
        beta_out = torch.empty(T, H, dtype=a.dtype, device=a.device)

    total = T * H
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _gate_prep_kernel[grid](
        a, b, A_log, dt_bias, g_out, beta_out,
        a.stride(0), b.stride(0),
        g_out.stride(0), beta_out.stride(0),
        H, total,
        BLOCK=BLOCK,
        num_warps=2,
    )
    return g_out, beta_out
