"""Unit test for ``flextrain_silu_bwd``: pointwise ``d/dz silu(z) * dout``.

Compares against ``torch.autograd.grad`` on a pure-pytorch reference.
"""
from __future__ import annotations

import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.ops._kernels.silu_bwd import flextrain_silu_bwd


def silu_ref(z):
    return z * torch.sigmoid(z)


def run_one(shape, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    z = torch.randn(*shape, dtype=dtype, device="cuda", requires_grad=True)
    y = silu_ref(z)
    dout = torch.randn_like(y)
    (din_ref,) = torch.autograd.grad(y, z, dout)

    din_k = flextrain_silu_bwd(z.detach().contiguous(), dout.detach().contiguous())

    err_abs = (din_ref - din_k).abs().float()
    ref_abs = din_ref.abs().float()
    max_err = err_abs.max().item()
    max_ref = ref_abs.max().item()
    rel = max_err / max(max_ref, 1e-6)
    ok = rel < 0.02   # bf16 pointwise — tight tolerance
    print(
        f"  shape={shape}: max_err={max_err:.4e} max_ref={max_ref:.4e} "
        f"rel={rel:.4%} ok={ok}"
    )
    return ok


def main():
    shapes = [
        (32, 32),
        (256, 1024),
        (1024, 8192),
        (4096, 8192),         # Qwen3.5-MoE-35B linear-attn shape
        (32768, 8192),        # full MoE-35B chunk
    ]
    all_ok = True
    for s in shapes:
        all_ok = run_one(s) and all_ok
    print("=" * 60)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
