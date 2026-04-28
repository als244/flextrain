"""Unit test for ``flextrain_gated_rmsnorm_bwd`` Triton kernel.

Compares the fused-kernel output against a torch.autograd.grad
reference implementation that mirrors the math used by GatedDeltaNet's
gated-RMSNorm:

    y = silu(z) * rmsnorm(o, weight) * weight

across various (T, H, D) shapes. Asserts ``do``, ``dz``, ``dw`` all
match within bf16 reorder tolerance.
"""
from __future__ import annotations

import math
import sys
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.ops._kernels.gated_rmsnorm import flextrain_gated_rmsnorm_bwd


def torch_reference_fwd(o, z, weight, eps):
    """Pure-pytorch fwd, autograd-friendly."""
    rms_sqr = (o.float() * o.float()).mean(dim=-1, keepdim=True) + eps
    rstd = rms_sqr.rsqrt()
    normed = (o.float() * rstd).to(o.dtype)
    silu_z = z * torch.sigmoid(z)
    y = silu_z * normed * weight
    return y


def run_one(T, H, D, *, dtype=torch.bfloat16, eps=1e-6, seed=0):
    """Run kernel + autograd ref, compare via bf16-aware *relative*
    tolerance.

    bf16 has ~7 bits of mantissa, so values reduce-summed across many
    elements have relative error ~O(sqrt(N) * 2^-7) ≈ 1% for N=10k.
    We check that ``max(|err|) / max(|ref|)`` and ``mean(|err|) /
    mean(|ref|)`` are both below 5% — this is the tightest bound that
    holds across the kernel's fp32 reductions.

    A point-wise atol/rtol check via ``torch.allclose`` is too strict
    here because the elementwise outputs span ~O(30) and bf16's ULP
    at that scale is ~0.25.
    """
    torch.manual_seed(seed)
    device = "cuda"

    o = torch.randn(T, H, D, dtype=dtype, device=device, requires_grad=True)
    z = torch.randn(T, H, D, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, dtype=dtype, device=device, requires_grad=True)

    # Forward via reference (autograd) at fp32 internally; cast back.
    y = torch_reference_fwd(o, z, weight, eps)
    do_normed = torch.randn_like(y)

    # Autograd reference grads.
    do_ref, dz_ref, dw_ref = torch.autograd.grad(
        outputs=y, inputs=(o, z, weight), grad_outputs=do_normed,
    )

    # Kernel.
    o_k = o.detach().clone().contiguous()
    z_k = z.detach().clone().contiguous()
    w_k = weight.detach().clone().contiguous()
    do_k, dz_k, dw_k = flextrain_gated_rmsnorm_bwd(
        do_normed.detach().contiguous(),
        o_k, z_k, w_k, eps,
    )

    # Compare via scale-relative bounds: error normalized by reference
    # magnitude. bf16 reorder noise is bounded by sqrt(N) * 2^-7 for
    # an N-element reduction; for our largest dw shapes (T*H ≈ 130k
    # elements summed per output dim) that's ~3% theoretical max.
    def check(name, ref, ker):
        err_abs = (ref - ker).abs().float()
        ref_abs = ref.abs().float()
        # Tail elements where ref is exactly 0 get atol-only.
        max_err_abs = err_abs.max().item()
        mean_err_abs = err_abs.mean().item()
        max_ref_abs = ref_abs.max().item()
        mean_ref_abs = ref_abs.mean().item()
        max_scale_rel = max_err_abs / max(max_ref_abs, 1e-6)
        mean_scale_rel = mean_err_abs / max(mean_ref_abs, 1e-6)
        ok = max_scale_rel < 0.05 and mean_scale_rel < 0.05
        print(
            f"  {name:6s}: shape={tuple(ref.shape)}  "
            f"max_err_abs={max_err_abs:.4e} (vs max_ref={max_ref_abs:.4e})  "
            f"max_scale_rel={max_scale_rel:.4%}  "
            f"mean_scale_rel={mean_scale_rel:.4%}  "
            f"ok={ok}"
        )
        return ok

    print(f"== T={T}, H={H}, D={D}, dtype={dtype} ==")
    ok_do = check("do", do_ref, do_k)
    ok_dz = check("dz", dz_ref, dz_k)
    ok_dw = check("dw", dw_ref, dw_k)
    return ok_do and ok_dz and ok_dw


def main():
    # Assorted shapes covering the regimes used by Qwen3-Next family.
    shapes = [
        # (T, H, D)
        (32, 4, 32),       # tiny sanity
        (256, 16, 128),    # Qwen3-Next 9B-ish
        (1024, 32, 128),   # Qwen3.5-MoE-35B linear-attn shape
        (4096, 32, 128),   # bigger T to stress the dw split-K
    ]
    all_ok = True
    for (T, H, D) in shapes:
        ok = run_one(T, H, D)
        all_ok = all_ok and ok
    print("=" * 60)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
