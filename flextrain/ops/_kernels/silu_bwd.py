"""Fused silu-bwd kernel.

Computes ``d_in = d_out * silu'(z)`` where ``silu'(z) = sigmoid(z) *
(1 + z * (1 - sigmoid(z)))``. Saves the python implementation in
``linear_attn.bwd`` from materializing 3-4 ``(T, conv_dim)``
intermediates (~1 GiB each at the Qwen3.5-MoE-35B linear-attn shape
T=32768 conv_dim=8192) — kernel keeps everything in SRAM.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def silu_bwd_kernel(
    Z_PTR,              # saved input (the silu argument)
    DOUT_PTR,           # upstream d/d(silu(z))
    DIN_PTR,            # output: d/dz
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    z = tl.load(Z_PTR + offsets, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(DOUT_PTR + offsets, mask=mask, other=0.0).to(tl.float32)

    sig_z = tl.sigmoid(z)
    dsilu = sig_z * (1.0 + z * (1.0 - sig_z))
    din = dout * dsilu

    tl.store(DIN_PTR + offsets, din.to(DIN_PTR.dtype.element_ty), mask=mask)


def flextrain_silu_bwd(
    z: torch.Tensor,
    dout: torch.Tensor,
    din: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``d/dz [silu(z)] * dout`` element-wise.

    Inputs and outputs share dtype (typically bf16). Reductions are
    done in fp32 in SRAM to match the original python path's accuracy.
    """
    assert z.shape == dout.shape, (
        f"shape mismatch: z={tuple(z.shape)}, dout={tuple(dout.shape)}"
    )
    assert z.is_cuda and dout.is_cuda
    if not z.is_contiguous():
        z = z.contiguous()
    if not dout.is_contiguous():
        dout = dout.contiguous()
    if din is None:
        din = torch.empty_like(z)

    n = z.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    silu_bwd_kernel[grid](
        z, dout, din, n,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return din
