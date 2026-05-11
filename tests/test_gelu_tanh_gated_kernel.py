"""Kernel-level parity for ``flextrain_gelu_tanh_gated_{fwd,bwd}``.

Independent oracle — compares the Triton kernel against
``torch.autograd`` operating on ``F.gelu(..., approximate='tanh') * x3``.
This is the test that would have caught the original "Gemma uses SiLU"
bug: the block-parity test alone couldn't, because its hand-rolled
reference replicated whatever activation flextrain happened to use.
Here the reference is a pure ``F.gelu`` call with autograd computing
the derivative — there's no shared logic with the kernel.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.ops import (
    flextrain_gelu_tanh_gated_bwd,
    flextrain_gelu_tanh_gated_fwd,
)
from tests.test_gemma3_block_parity import _compare


DEVICE = "cuda:0"

# Single-op kernel parity — bf16 quantum is the noise floor. These
# thresholds are tighter than the block-parity ones because there's no
# compounding chain.
COS_TOL = 0.9995
SIGN_TOL = 0.999
REL_L2_TOL = 1e-2


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="kernel parity requires CUDA",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(64, 256), (32, 1152), (8, 6912)])
def test_gelu_tanh_gated_fwd_matches_autograd(shape, dtype) -> None:
    """``flextrain_gelu_tanh_gated_fwd(x1, x3) ≈ F.gelu(x1, tanh) * x3``."""
    torch.manual_seed(7)
    x1 = torch.randn(shape, device=DEVICE, dtype=dtype)
    x3 = torch.randn(shape, device=DEVICE, dtype=dtype)

    out_ft = flextrain_gelu_tanh_gated_fwd(x1, x3)

    # Reference: pure F.gelu in fp32, multiplied in compute-dtype to
    # match the kernel's cast policy.
    gate_ref = F.gelu(x1.float(), approximate="tanh").to(dtype)
    out_ref = gate_ref * x3

    _compare(
        f"fwd[{shape}-{dtype}]", out_ft, out_ref,
        cos_tol=COS_TOL, sign_tol=SIGN_TOL, rel_l2_tol=REL_L2_TOL,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(64, 256), (32, 1152), (8, 6912)])
def test_gelu_tanh_gated_bwd_matches_autograd(shape, dtype) -> None:
    """``flextrain_gelu_tanh_gated_bwd(x1, x3, dout) ≈ autograd grads``."""
    torch.manual_seed(11)
    x1 = torch.randn(shape, device=DEVICE, dtype=dtype)
    x3 = torch.randn(shape, device=DEVICE, dtype=dtype)
    dout = torch.randn(shape, device=DEVICE, dtype=dtype)

    # FT bwd: kernel produces dx1, dx3 (and optionally recomputed fwd act).
    dx1_ft, dx3_ft, act_ft = flextrain_gelu_tanh_gated_bwd(
        x1, x3, dout, store_activations=True,
    )

    # Reference via torch.autograd. Compute in fp32 so the reference is
    # not dragged by the same bf16 quantum the kernel rounds to; the
    # comparison is against an authoritative gradient.
    x1_ref = x1.detach().float().requires_grad_(True)
    x3_ref = x3.detach().float().requires_grad_(True)
    gate_ref = F.gelu(x1_ref, approximate="tanh")
    out_ref = gate_ref * x3_ref
    out_ref.backward(dout.float())

    _compare(
        f"dx1[{shape}-{dtype}]", dx1_ft, x1_ref.grad.to(dtype),
        cos_tol=COS_TOL, sign_tol=SIGN_TOL, rel_l2_tol=REL_L2_TOL,
    )
    _compare(
        f"dx3[{shape}-{dtype}]", dx3_ft, x3_ref.grad.to(dtype),
        cos_tol=COS_TOL, sign_tol=SIGN_TOL, rel_l2_tol=REL_L2_TOL,
    )
    # The recomputed forward activation is what the layer-level bwd
    # uses as the left operand for ``w_2``'s wgrad — must match too.
    _compare(
        f"act[{shape}-{dtype}]", act_ft, out_ref.detach().to(dtype),
        cos_tol=COS_TOL, sign_tol=SIGN_TOL, rel_l2_tol=REL_L2_TOL,
    )
