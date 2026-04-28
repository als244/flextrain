"""Fused Triton kernels for the gated-RMSNorm operator used inside
GatedDeltaNet (Qwen3-Next / Qwen3.5 / Qwen3.5-MoE / Qwen3.6).

Forward
-------

Given:
    o:       (T, H, D)  bf16    -- saved core_out
    z:       (T, H, D)  bf16    -- saved gate
    weight:  (D,)       bf16    -- per-head_v_dim weight, broadcast over H
    eps:     float

The fwd computes (per row, where one row = one (t, h) position):

    rstd     = 1 / sqrt(mean(o^2, dim=-1) + eps)         shape (T, H)
    normed   = o * rstd                                  shape (T, H, D)
    o_normed = silu(z) * normed * weight                 shape (T, H, D)

with ``silu(z) = z * sigmoid(z)``.

We DO NOT save ``rstd`` from the forward — the gated-RMSNorm wrapper in
``linear_attn`` recomputes it inside the bwd kernel to keep the API
symmetric with the existing python helper. (The cost of one extra
``mean(o^2)`` reduction per row is small compared to the matmul-bound
work that dominates linear-attn bwd.)

Backward
--------

Given upstream ``dy = do_normed: (T, H, D)``, derive:

    dy = silu(z) * normed * weight.

dw[d] = sum_{T,H} (dy * silu(z) * normed)[..., d]                shape (D,)
dz    = dy * normed * weight * silu'(z)                          shape (T, H, D)
        where silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
do (= dcore_out): chain through normed = o * rstd. Per-row Jacobian
        d(normed_i)/d(o_j) = rstd * δ_ij - rstd^3 * o_i * o_j / D
    so do_i = rstd * d_normed_i - (rstd^3 / D) * o_i * sum_j (d_normed_j * o_j)
    where d_normed = dy * silu(z) * weight.

Memory model
------------

The python implementation in ``linear_attn._gated_rmsnorm_bwd`` (now
replaced) materialized ~10 ``(T, H, D)`` fp32 intermediates. At
``T=32768, H=32, D=128`` each is 4 GiB — OOMs a 24 GiB GPU.

These kernels keep all storage at ``(T, H, D)`` bf16 (same as inputs/
outputs) and accumulate the per-row reductions (``rstd``, ``dot``) and
the cross-row ``dw`` reduction in fp32 inside SRAM.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ===================================================================
# Backward dX/dZ kernel: one program per (t, h) row.
# ===================================================================


@triton.jit
def gated_rmsnorm_bwd_dxdz_kernel(
    DY_ptr,          # (T, H, D) bf16  -- upstream do_normed
    O_ptr,           # (T, H, D) bf16  -- saved core_out
    Z_ptr,           # (T, H, D) bf16  -- saved gate
    W_ptr,           # (D,)      bf16  -- weight, broadcast over H
    DO_ptr,          # (T, H, D) bf16  -- output: do (gradient w.r.t. core_out)
    DZ_ptr,          # (T, H, D) bf16  -- output: dz
    stride_dy_t, stride_dy_h,
    stride_o_t, stride_o_h,
    stride_z_t, stride_z_h,
    stride_do_t, stride_do_h,
    stride_dz_t, stride_dz_h,
    D,                          # head_v_dim
    EPSILON: tl.float32,
    BLOCK_SIZE_D: tl.constexpr,  # next-pow2(D)
):
    """One program instance handles one (t, h) row of length D.

    Computes dO and dZ from saved O, Z, W and upstream DY in one
    pass, with the per-row reductions (rstd, dot) accumulated in fp32
    SRAM.
    """
    pid_t = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # Row pointers.
    dy_row = DY_ptr + pid_t * stride_dy_t + pid_h * stride_dy_h
    o_row = O_ptr + pid_t * stride_o_t + pid_h * stride_o_h
    z_row = Z_ptr + pid_t * stride_z_t + pid_h * stride_z_h
    do_row = DO_ptr + pid_t * stride_do_t + pid_h * stride_do_h
    dz_row = DZ_ptr + pid_t * stride_dz_t + pid_h * stride_dz_h

    offs = tl.arange(0, BLOCK_SIZE_D)
    mask = offs < D

    # Load row tensors and the (per-row, broadcast over H) weight in fp32.
    dy = tl.load(dy_row + offs, mask=mask, other=0.0).to(tl.float32)
    o = tl.load(o_row + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_row + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Recompute rstd from o (cheap: one row reduction).
    var = tl.sum(o * o, axis=0) / D
    rstd = tl.rsqrt(var + EPSILON)

    # silu(z), silu'(z), normed, d_normed.
    sig_z = tl.sigmoid(z)
    silu_z = z * sig_z
    silu_prime = sig_z * (1.0 + z * (1.0 - sig_z))
    normed = o * rstd                                  # (D,)

    # d_normed = dy * silu(z) * w; this is also what's needed for dot.
    d_normed = dy * silu_z * w
    # dot = sum_j (d_normed_j * o_j)
    dot = tl.sum(d_normed * o, axis=0)
    rstd3 = rstd * rstd * rstd
    coef = rstd3 / D * dot                             # scalar per row

    # do = rstd * d_normed - coef * o
    do_out = rstd * d_normed - coef * o
    # dz = dy * normed * w * silu'(z)
    dz_out = dy * normed * w * silu_prime

    tl.store(do_row + offs, do_out.to(DO_ptr.dtype.element_ty), mask=mask)
    tl.store(dz_row + offs, dz_out.to(DZ_ptr.dtype.element_ty), mask=mask)


# ===================================================================
# Backward dW kernel: Split-K reduction across (T, H) rows.
# ===================================================================


@triton.jit
def gated_rmsnorm_bwd_dw_kernel(
    DY_ptr,          # (T*H, D) bf16  -- upstream do_normed flattened
    O_ptr,           # (T*H, D) bf16  -- saved core_out flattened
    Z_ptr,           # (T*H, D) bf16  -- saved gate flattened
    DW_ptr,          # (D,)     fp32  -- output dw accumulator (must be pre-zeroed
                                    #              if ACCUMULATE_DW=False)
    stride_dy_row, stride_o_row, stride_z_row,
    TOTAL_ROWS,      # T * H
    D,               # head_v_dim
    EPSILON: tl.float32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Computes dw[d] = sum_{T,H} dy * silu(z) * normed for d in [0, D).

    Split-K over rows, atomic-add into dw. Each program processes
    BLOCK_M rows × BLOCK_N columns (BLOCK_N = D, since D is small).
    """
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    num_split_k = tl.num_programs(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < D

    rows_per_split = tl.cdiv(TOTAL_ROWS, num_split_k)
    start_row = pid_k * rows_per_split
    end_row = tl.minimum(start_row + rows_per_split, TOTAL_ROWS)

    dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for row_idx in range(start_row, end_row, BLOCK_M):
        offs_m = row_idx + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end_row
        mask_mn = mask_m[:, None] & mask_n[None, :]

        dy_ptrs = DY_ptr + offs_m[:, None] * stride_dy_row + offs_n[None, :]
        o_ptrs = O_ptr + offs_m[:, None] * stride_o_row + offs_n[None, :]
        z_ptrs = Z_ptr + offs_m[:, None] * stride_z_row + offs_n[None, :]

        dy = tl.load(dy_ptrs, mask=mask_mn, other=0.0).to(tl.float32)
        o = tl.load(o_ptrs, mask=mask_mn, other=0.0).to(tl.float32)
        z = tl.load(z_ptrs, mask=mask_mn, other=0.0).to(tl.float32)

        # Recompute rstd per row inside the loop. We always read full rows
        # (mask_n only blanks out trailing pad if D isn't a multiple of
        # BLOCK_N, but in our schema D is small enough that BLOCK_N == D
        # and there's no padding). The sum-of-squares reduction is over
        # the full row, computed from the loaded slice.
        var_row = tl.sum(o * o, axis=1) / D                 # (BLOCK_M,)
        rstd_row = tl.rsqrt(var_row + EPSILON)              # (BLOCK_M,)

        normed = o * rstd_row[:, None]
        sig_z = tl.sigmoid(z)
        silu_z = z * sig_z
        # dw contribution: dy * silu(z) * normed, summed over rows.
        dw_acc += tl.sum(dy * silu_z * normed, axis=0)

    dw_out_ptr = DW_ptr + offs_n
    tl.atomic_add(dw_out_ptr, dw_acc, mask=mask_n)


# ===================================================================
# Python wrapper.
# ===================================================================


def flextrain_gated_rmsnorm_bwd(
    do_normed: torch.Tensor,    # (T, H, D)
    o: torch.Tensor,            # (T, H, D)
    z: torch.Tensor,            # (T, H, D)
    weight: torch.Tensor,       # (D,)
    eps: float,
    *,
    do_out: torch.Tensor | None = None,
    dz_out: torch.Tensor | None = None,
    dw_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (do, dz, dw) for ``y = silu(z) * rmsnorm(o, w) * w``.

    Numerically equivalent (modulo bf16 reorder noise) to the python
    helper this replaces. ``do`` and ``dz`` come back in ``o``'s dtype;
    ``dw`` in ``weight``'s dtype.
    """
    assert do_normed.shape == o.shape == z.shape, (
        f"shape mismatch: do_normed={do_normed.shape}, o={o.shape}, z={z.shape}"
    )
    assert do_normed.is_cuda and o.is_cuda and z.is_cuda
    assert weight.is_cuda
    assert weight.shape == (o.shape[-1],), (
        f"weight must be ({o.shape[-1]},); got {tuple(weight.shape)}"
    )
    if not do_normed.is_contiguous():
        do_normed = do_normed.contiguous()
    if not o.is_contiguous():
        o = o.contiguous()
    if not z.is_contiguous():
        z = z.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    T, H, D = o.shape

    if do_out is None:
        do_out = torch.empty_like(o)
    if dz_out is None:
        dz_out = torch.empty_like(z)
    # dw is fp32 internally for accumulation accuracy; we cast at the end.
    if dw_out is None:
        dw_acc = torch.zeros(D, dtype=torch.float32, device=o.device)
    else:
        dw_acc = torch.zeros(D, dtype=torch.float32, device=o.device)

    # ---- dX/dZ kernel (one program per (t, h)) ----
    BLOCK_SIZE_D = triton.next_power_of_2(D)
    grid_xz = (T, H)
    gated_rmsnorm_bwd_dxdz_kernel[grid_xz](
        do_normed, o, z, weight,
        do_out, dz_out,
        do_normed.stride(0), do_normed.stride(1),
        o.stride(0), o.stride(1),
        z.stride(0), z.stride(1),
        do_out.stride(0), do_out.stride(1),
        dz_out.stride(0), dz_out.stride(1),
        D, eps,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=4,
    )

    # ---- dW kernel (Split-K reduction) ----
    # Flatten (T, H) into TOTAL_ROWS for the dw kernel's row sum.
    do_normed_flat = do_normed.view(T * H, D)
    o_flat = o.view(T * H, D)
    z_flat = z.view(T * H, D)
    TOTAL_ROWS = T * H
    BLOCK_M = 64
    BLOCK_N = BLOCK_SIZE_D
    # Split-K factor: pick enough splits to keep each block busy. At
    # TOTAL_ROWS=1M rows with BLOCK_M=64, ~16k rows per split keeps a
    # 64-block GPU well-fed; cap at the total row budget.
    split_k = max(1, min(64, triton.cdiv(TOTAL_ROWS, BLOCK_M * 16)))
    grid_dw = (triton.cdiv(D, BLOCK_N), split_k)
    gated_rmsnorm_bwd_dw_kernel[grid_dw](
        do_normed_flat, o_flat, z_flat, dw_acc,
        do_normed_flat.stride(0), o_flat.stride(0), z_flat.stride(0),
        TOTAL_ROWS, D, eps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    if dw_out is None:
        dw_out_final = dw_acc.to(weight.dtype)
    else:
        # Accumulate into provided dw_out buffer.
        dw_out.add_(dw_acc.to(dw_out.dtype))
        dw_out_final = dw_out
    return do_out, dz_out, dw_out_final
