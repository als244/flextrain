"""LoRA block parity vs autograd reference + frozen-tensor allocation test.

Two test groups:

1. **Math parity**: ``lora_linear_fwd`` + ``lora_linear_bwd`` vs full
   autograd reference. Catches indexing / scaling / transpose bugs.

2. **Frozen allocation**: build a LoRA-config ParamSpec, run it through
   the buffer allocator, verify that the frozen base has master + compute
   buffers but NO grad / no opt-state allocations.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.core.layer import ParamSpec, TensorSpec
from flextrain.nn.blocks.lora import (
    LoRALinearConfig, lora_init, lora_linear_bwd, lora_linear_fwd,
    lora_param_spec,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def test_math_parity():
    torch.manual_seed(7)
    cfg = LoRALinearConfig(
        name="w_q", d_in_dim_name="d_model", d_out_dim_name="attn_dim",
        rank=8, alpha=16.0,
    )
    d_model, attn_dim, T = 64, 96, 32

    # Random weights.
    W = torch.randn(d_model, attn_dim, dtype=DTYPE, device=DEVICE) * 0.02
    A = torch.randn(d_model, cfg.rank, dtype=DTYPE, device=DEVICE) * 0.02
    B = torch.randn(cfg.rank, attn_dim, dtype=DTYPE, device=DEVICE) * 0.02
    x = torch.randn(T, d_model, dtype=DTYPE, device=DEVICE)

    weights = {
        "w_q": W,
        cfg.a_name: A.clone(),
        cfg.b_name: B.clone(),
    }
    # FT fwd.
    y_ft = lora_linear_fwd(x, weights, cfg)

    # Autograd reference.
    A_ref = A.clone().requires_grad_(True)
    B_ref = B.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)
    y_ref = x_ref @ W + (x_ref @ A_ref) @ B_ref * cfg.scale
    delta_fwd = (y_ft.float() - y_ref.float()).abs().max().item()
    print(f"  fwd parity: max |Δ| = {delta_fwd:.4e}")
    assert delta_fwd < 1e-2, f"fwd diverges: {delta_fwd}"

    # Backward parity.
    upstream = torch.randn_like(y_ref) * 0.01
    y_ref.backward(upstream)
    ref_dA = A_ref.grad.clone()
    ref_dB = B_ref.grad.clone()
    ref_dx = x_ref.grad.clone()

    grads = {
        "g_" + cfg.a_name: torch.zeros_like(A, dtype=torch.float32),
        "g_" + cfg.b_name: torch.zeros_like(B, dtype=torch.float32),
    }
    dx_ft = lora_linear_bwd(upstream, x, weights, grads, cfg)

    da_max = (ref_dA.float() - grads["g_" + cfg.a_name].float()).abs().max().item()
    db_max = (ref_dB.float() - grads["g_" + cfg.b_name].float()).abs().max().item()
    dx_max = (ref_dx.float() - dx_ft.float()).abs().max().item()
    print(f"  bwd dA: max |Δ| = {da_max:.4e}")
    print(f"  bwd dB: max |Δ| = {db_max:.4e}")
    print(f"  bwd dx: max |Δ| = {dx_max:.4e}")
    # Note: ref_dx has contributions from BOTH the base path (x @ W) AND
    # the LoRA path. Our `dx_ft` returns ONLY the LoRA path's contribution
    # — the base path's grad is added by the calling block (it does the
    # base matmul so dy @ W.T is its responsibility). Construct the
    # expected LoRA-only dx and compare against THAT:
    lora_only_dx = (upstream @ B.T * cfg.scale) @ A.T
    dx_lora_max = (lora_only_dx.float() - dx_ft.float()).abs().max().item()
    print(f"  bwd dx (LoRA only): max |Δ| = {dx_lora_max:.4e}")
    assert dx_lora_max < 1e-3, f"dx LoRA mismatch: {dx_lora_max}"
    assert da_max < 1e-3, f"dA mismatch: {da_max}"
    assert db_max < 1e-3, f"dB mismatch: {db_max}"
    print("  ✓ LoRA math parity PASSED")


def test_frozen_allocation():
    """Build a ParamSpec with one frozen base + LoRA A/B; allocate
    through the engine's buffer manager; verify only A/B get grads."""
    from flextrain.engine.buffers import _alloc_dict_on_host

    cfg = LoRALinearConfig(
        name="w_q", d_in_dim_name="d_model", d_out_dim_name="attn_dim",
        rank=4,
    )
    ps = ParamSpec(tensors=lora_param_spec(cfg))
    dims = {"d_model": 64, "attn_dim": 64}

    # Use a stub backend that just returns an empty cpu tensor.
    class StubBackend:
        def allocate_tensor(self, shape, dtype):
            return torch.empty(shape, dtype=dtype)

    backend = StubBackend()
    masters = _alloc_dict_on_host(ps, dims, role="master", backend=backend)
    grads = _alloc_dict_on_host(ps, dims, role="grad", backend=backend)
    opt_states = _alloc_dict_on_host(ps, dims, role="opt_state", backend=backend)

    # All three params have masters.
    assert "w_q" in masters
    assert cfg.a_name in masters
    assert cfg.b_name in masters
    print("  master alloc: all three present ✓")

    # Only A and B have grads (frozen base elided).
    assert "g_q" not in grads, "frozen w_q got a grad allocation"
    assert "g_" + cfg.a_name[2:] in grads or f"g_{cfg.a_name}" in grads
    assert "g_" + cfg.b_name[2:] in grads or f"g_{cfg.b_name}" in grads
    print(f"  grad alloc: only A/B (no g_q for frozen w_q) — keys: {list(grads.keys())} ✓")

    # Same for opt_state.
    assert "w_q" not in opt_states
    print("  opt_state alloc: w_q skipped ✓")


def test_init_scheme():
    cfg = LoRALinearConfig(
        name="w_q", d_in_dim_name="d_model", d_out_dim_name="attn_dim",
        rank=4,
    )
    weights = {
        cfg.a_name: torch.empty(64, 4, dtype=DTYPE, device=DEVICE),
        cfg.b_name: torch.empty(4, 64, dtype=DTYPE, device=DEVICE),
    }
    lora_init(weights, cfg, seed=42)
    # B should be exactly zero (so the LoRA delta is zero on first fwd).
    assert (weights[cfg.b_name] == 0).all(), "B should be zero-init"
    # A should have nonzero variance.
    a_std = weights[cfg.a_name].float().std().item()
    assert 0.005 < a_std < 0.05, f"A std {a_std} outside expected range"
    print(f"  init: B all zeros, A std = {a_std:.4f} ✓")


def main():
    print("=== LoRA math parity ===")
    test_math_parity()
    print("\n=== Frozen allocation ===")
    test_frozen_allocation()
    print("\n=== Init scheme ===")
    test_init_scheme()
    print("\n✓ All LoRA tests PASSED")


if __name__ == "__main__":
    main()
