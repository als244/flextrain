"""Test that Qwen3-Next's RMSNorm convention is correctly handled at load time.

HF's ``Qwen3NextRMSNorm.forward`` computes ``x * rsqrt(mean(x²)+eps) * (1 + weight)``.
The stored weight is ``γ - 1`` (initialized to zeros via ``init.zeros_(weight)``).

FT's ``RMSNormBlock.fwd`` computes ``x * rsqrt(mean(x²)+eps) * γ`` where γ
is the loaded value directly.

So the loader for Qwen3-Next MUST shift every RMSNorm γ by +1 at load time,
otherwise outputs differ wildly from HF's.

This test does NOT exercise the FT engine; it just verifies that:

* HF's ``Qwen3NextRMSNorm(weight=zeros).forward(x)`` returns ``x.normed`` (i.e.
  the canonical γ=1 RMSNorm).
* If we apply the +1 shift to the stored weight then feed it into FT's
  ``flextrain_rmsnorm_fwd``, we get the SAME output (within bf16 noise).

If the +1 shift were missing, FT would multiply by ~0 and produce ~0 outputs,
clearly diverging from HF.

This test is a smoke check. The real proof is end-to-end logit parity, which
requires the loader to apply the shift. We track the shift status in the
Qwen3-Next ArchSpec's ``post_load_hook``.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def main():
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm
    from flextrain.ops import flextrain_rmsnorm_fwd

    torch.manual_seed(0)
    T, D = 16, 128
    eps = 1e-6

    x = torch.randn(T, D, dtype=DTYPE, device=DEVICE) * 0.5

    # HF: stored γ random (around the (1+w) scheme), w ~ small.
    weight_stored = torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.05

    # HF reference fwd (uses (1 + weight)).
    hf_norm = Qwen3NextRMSNorm(D, eps=eps).to(DEVICE, dtype=DTYPE)
    hf_norm.weight.data.copy_(weight_stored)
    with torch.no_grad():
        y_hf = hf_norm(x)

    # FT: applies the shift at load time → the FT-internal weight is
    # canonical γ = 1 + weight_stored. Then flextrain_rmsnorm_fwd does
    # ``x * rsqrt(...) * γ`` with no extra shift.
    canonical_gamma = (1.0 + weight_stored.float()).to(DTYPE)
    rstd_buf = torch.empty(T, dtype=torch.float32, device=DEVICE)
    out_ft, _ = flextrain_rmsnorm_fwd(
        x, W=canonical_gamma, rms_norm_eps=eps,
    )

    delta = (y_hf.float() - out_ft.float()).abs()
    print(f"FT (post-shift) vs HF Qwen3NextRMSNorm: max|Δ| = {float(delta.max().item()):.3e}, mean|Δ| = {float(delta.mean().item()):.3e}")
    assert float(delta.max().item()) < 5e-2, "FT vs HF Qwen3-Next RMSNorm diverges"

    # Also verify that applying NO shift would fail visibly.
    out_no_shift, _ = flextrain_rmsnorm_fwd(
        x, W=weight_stored, rms_norm_eps=eps,
    )
    delta_unshifted = (y_hf.float() - out_no_shift.float()).abs().max().item()
    assert delta_unshifted > 0.5, (
        "Unshifted FT RMSNorm should differ a lot from HF (it doesn't, "
        "test setup may be wrong)."
    )
    print(f"Unshifted FT vs HF: max|Δ| = {delta_unshifted:.3e} (expected: large, ≥0.5)")
    print("\n✓ RMSNorm (1+weight) shift convention test PASSED")


if __name__ == "__main__":
    main()
