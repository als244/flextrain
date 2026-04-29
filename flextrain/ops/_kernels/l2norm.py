"""Strided L2-norm fwd/bwd kernels.

Mirrors fla.modules.l2norm but takes runtime strides for the per-token
and per-head axes so callers can pass non-contiguous (T, H, D) views
directly. The fla version hardcodes ``(D, 1)`` strides in
``tl.make_block_ptr`` and its python wrapper does
``x.view(-1, x.shape[-1])`` which requires contiguous input — forcing a
``.contiguous()`` materialization at each call site that wants to
l2-norm a strided slice.

Used by linear_attn fwd/bwd to l2-norm the per-(token, k_head) Q and K
slices of post_conv. After Stage A's column permutation, post_conv has
rows ``[Q_concat | K_concat | V_concat]``; Q is a strided 3D view
``post_conv[:, :key_dim].reshape(T, H, hk)`` with strides
``(conv_dim, hk, 1)`` — non-contiguous along the T axis.

Each row of length D = head_k_dim is independently l2-normalized; rstd
is per-(token, head) so output shape is (T, H).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _l2norm_fwd_kernel(
    X_ptr, Y_ptr, Rstd_ptr,
    stride_x_t, stride_x_h,        # input row strides
    stride_y_t, stride_y_h,        # output row strides
    stride_rstd_t,                 # rstd is (T, H) shaped
    H,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    """One program per (token, head) row.

    Grid: ``(T * H,)``. Program ``pid`` handles row
    ``(pid // H, pid % H)``: l2-norm over the D-axis.
    """
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H

    cols = tl.arange(0, BD)
    col_mask = cols < D

    x_row = X_ptr + t * stride_x_t + h * stride_x_h
    y_row = Y_ptr + t * stride_y_t + h * stride_y_h

    b_x = tl.load(x_row + cols, mask=col_mask, other=0.0).to(tl.float32)
    b_sumsq = tl.sum(b_x * b_x)
    b_rstd = 1.0 / tl.sqrt(b_sumsq + 1e-6)
    b_y = b_x * b_rstd

    tl.store(y_row + cols, b_y.to(Y_ptr.dtype.element_ty), mask=col_mask)
    tl.store(Rstd_ptr + t * stride_rstd_t + h, b_rstd)


@triton.jit
def _l2norm_bwd_kernel(
    Y_ptr, Rstd_ptr, DY_ptr, DX_ptr,
    stride_y_t, stride_y_h,
    stride_dy_t, stride_dy_h,
    stride_dx_t, stride_dx_h,
    stride_rstd_t,
    H,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    """L2-norm backward, per (token, head) row.

    Math (in fp32):
        y    = x * rstd
        dx   = dy * rstd - sum(dy * y) * y * rstd
    """
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H

    cols = tl.arange(0, BD)
    col_mask = cols < D

    y_row = Y_ptr + t * stride_y_t + h * stride_y_h
    dy_row = DY_ptr + t * stride_dy_t + h * stride_dy_h
    dx_row = DX_ptr + t * stride_dx_t + h * stride_dx_h

    b_y = tl.load(y_row + cols, mask=col_mask, other=0.0).to(tl.float32)
    b_dy = tl.load(dy_row + cols, mask=col_mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(Rstd_ptr + t * stride_rstd_t + h).to(tl.float32)

    b_dot = tl.sum(b_dy * b_y)
    b_dx = b_dy * b_rstd - b_dot * b_y * b_rstd

    tl.store(dx_row + cols, b_dx.to(DX_ptr.dtype.element_ty), mask=col_mask)


def flextrain_l2norm_fwd_into(
    x: torch.Tensor,            # (T, H, D), last-axis stride 1
    y_out: torch.Tensor,        # (T, H, D), last-axis stride 1
    rstd_out: torch.Tensor,     # (T, H) fp32
    *,
    eps: float = 1e-6,
) -> None:
    """L2-norm the last axis of ``x`` per (T, H) row.

    ``x`` may be a non-contiguous slice (e.g.
    ``post_conv[:, :key_dim].reshape(T, H, hk)``). The kernel uses
    runtime strides for the T and H axes; only the last axis must have
    stride 1.
    """
    assert x.shape == y_out.shape, f"{x.shape} vs {y_out.shape}"
    assert x.dim() == 3
    assert x.stride(-1) == 1
    assert y_out.stride(-1) == 1
    assert rstd_out.shape == x.shape[:2], (
        f"rstd_out {rstd_out.shape} vs (T, H) {x.shape[:2]}"
    )
    assert eps == 1e-6, "eps != 1e-6 not currently supported"
    T, H, D = x.shape
    BD = triton.next_power_of_2(D)
    grid = (T * H,)
    _l2norm_fwd_kernel[grid](
        x, y_out, rstd_out,
        x.stride(0), x.stride(1),
        y_out.stride(0), y_out.stride(1),
        rstd_out.stride(0),
        H, T, D, BD,
        num_warps=4,
    )


def flextrain_l2norm_bwd_into(
    y: torch.Tensor,            # (T, H, D) saved fwd output
    rstd: torch.Tensor,         # (T, H) fp32 saved rstd
    dy: torch.Tensor,           # (T, H, D) upstream grad
    dx_out: torch.Tensor,       # (T, H, D) output grad
    *,
    eps: float = 1e-6,
) -> None:
    """L2-norm backward."""
    assert y.shape == dy.shape == dx_out.shape
    assert y.dim() == 3
    assert y.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dx_out.stride(-1) == 1
    assert rstd.shape == y.shape[:2]
    T, H, D = y.shape
    BD = triton.next_power_of_2(D)
    grid = (T * H,)
    _l2norm_bwd_kernel[grid](
        y, rstd, dy, dx_out,
        y.stride(0), y.stride(1),
        dy.stride(0), dy.stride(1),
        dx_out.stride(0), dx_out.stride(1),
        rstd.stride(0),
        H, T, D, BD,
        num_warps=4,
    )
