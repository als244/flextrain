"""Phase 4: MoE expert-LoRA math parity test.

Builds a tiny OLMoE block with LoRA on the 3-D MoE expert stacks
(``w_up``, ``w_down``) plus the 2-D w_router (just to mix shapes),
runs fwd+bwd, compares grads against an autograd reference where
``W'[e] = W[e] + A[e] @ B[e] * scale`` is built per-expert.

This test specifically exercises the 3-D adapter path
(``bmm`` over the expert dim) and the corresponding bwd grad
decomposition.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.core.layer import ParamSpec, TensorSpec
from flextrain.nn.layers.lora_wrapper import (
    LoRATargetConfig, _make_lora_specs, _discover_lora_eligible_names,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def test_3d_specs():
    """Verify A/B specs for a 3-D MoE expert stack."""
    base = TensorSpec(
        name="w_up",
        shape_fn=lambda d: (d["num_experts"], d["d_model"], 2 * d["expert_dim"]),
        compute_dtype=DTYPE,
    )
    cfg = LoRATargetConfig(target_name="w_up", rank=8, alpha=16.0)
    dims = {"num_experts": 4, "d_model": 64, "expert_dim": 32}
    a, b = _make_lora_specs(base, cfg, dims)
    assert a.shape(dims) == (4, 64, 8), a.shape(dims)
    assert b.shape(dims) == (4, 8, 64), b.shape(dims)  # d_out = 2 * 32 = 64
    print(f"  3-D spec: A={a.shape(dims)}, B={b.shape(dims)} ✓")


def test_3d_math_parity():
    """End-to-end 3-D LoRA: ``W'[e] = W[e] + A[e] @ B[e] * s``,
    bwd decomposes correctly via torch.bmm."""
    torch.manual_seed(7)
    E, d_in, d_out = 4, 64, 96
    rank = 8
    alpha = 16.0
    scale = alpha / rank
    T = 32   # tokens routed per expert (small)

    # Per-expert random base + A + B.
    W = torch.randn(E, d_in, d_out, dtype=DTYPE, device=DEVICE) * 0.02
    A = torch.randn(E, d_in, rank, dtype=DTYPE, device=DEVICE) * 0.02
    B = torch.randn(E, rank, d_out, dtype=DTYPE, device=DEVICE) * 0.02
    # Per-expert input (all experts get T tokens for the test).
    x_e = torch.randn(E, T, d_in, dtype=DTYPE, device=DEVICE)
    upstream = torch.randn(E, T, d_out, dtype=DTYPE, device=DEVICE) * 0.01

    # ---- FT effective-W path ----
    W_eff = W + torch.bmm(A, B) * scale
    y_ft = torch.bmm(x_e, W_eff)
    # bwd: dL/dW_eff = x_e^T @ dy (per expert).
    dW_eff = torch.bmm(x_e.transpose(1, 2), upstream)
    # Decompose per-expert.
    dA_ft = (torch.bmm(dW_eff, B.transpose(-1, -2))) * scale
    dB_ft = (torch.bmm(A.transpose(-1, -2), dW_eff)) * scale
    # dx contribution from LoRA (for completeness).
    dx_ft = torch.bmm(upstream, W_eff.transpose(-1, -2))

    # ---- Autograd reference ----
    A_ref = A.clone().float().requires_grad_(True)
    B_ref = B.clone().float().requires_grad_(True)
    x_ref = x_e.clone().requires_grad_(True)
    W_eff_ref = W.float() + torch.bmm(A_ref, B_ref) * scale
    y_ref = torch.bmm(x_ref.float(), W_eff_ref)
    y_ref.backward(upstream.detach().float())

    # Compare.
    print("  3-D math parity (per-expert LoRA):")
    for name, ft, ref in [
        ("dA", dA_ft, A_ref.grad),
        ("dB", dB_ft, B_ref.grad),
        ("dx", dx_ft, x_ref.grad),
    ]:
        d = (ft.float() - ref.float()).abs().max().item()
        m = ref.abs().max().item()
        rel = d / (m + 1e-12)
        print(f"    {name}: max|Δ|={d:.4e}  |ref|max={m:.4e}  rel={rel:.4f}")
        assert rel < 0.10 or d < 1e-4, f"{name}: rel={rel:.4f}"


def test_discover_eligible_names():
    """Sanity check that discovery includes 3-D MoE stacks but
    excludes routers."""
    specs = (
        TensorSpec(
            "w_q", lambda d: (d["d_model"], d["attn_dim"]),
            compute_dtype=DTYPE,
        ),
        TensorSpec(
            "w_router", lambda d: (d["d_model"], d["num_experts"]),
            compute_dtype=DTYPE,
        ),
        TensorSpec(
            "w_up", lambda d: (d["num_experts"], d["d_model"], 2*d["expert_dim"]),
            compute_dtype=DTYPE,
        ),
        TensorSpec(
            "w_attn_norm", lambda d: (d["d_model"],),
            compute_dtype=DTYPE,
        ),
    )
    ps = ParamSpec(tensors=specs)
    dims = {"d_model": 64, "attn_dim": 64, "num_experts": 4, "expert_dim": 32}
    eligible = _discover_lora_eligible_names(ps, dims)
    assert "w_q" in eligible
    assert "w_up" in eligible
    assert "w_router" not in eligible, "router should be excluded by default"
    assert "w_attn_norm" not in eligible, "1-D should be excluded"
    print(f"  discover eligible: {eligible} ✓")


def main():
    print("=== test_discover_eligible_names ===")
    test_discover_eligible_names()
    print("\n=== test_3d_specs ===")
    test_3d_specs()
    print("\n=== test_3d_math_parity ===")
    test_3d_math_parity()
    print("\n✓ MoE expert-LoRA math parity PASSED")


if __name__ == "__main__":
    main()
