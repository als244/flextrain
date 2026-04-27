"""Partial-rotary RoPE parity test.

Verifies the new ``flextrain_rope_partial_fwd/bwd`` kernels against a
plain-PyTorch reference. The reference applies rotation only to the
first ``rot_dim`` channels per head and leaves the remaining channels
untouched (HF Qwen3-Next / Qwen3.5 / Qwen3.6 partial-rotary semantics).

Tested:

* Forward: FT in-place rotation matches reference within bf16 noise.
* Pass-through: channels ``[rot_dim : head_dim]`` are bit-identical
  before and after the kernel call.
* Backward: applying bwd after fwd recovers the original tensor
  (within bf16 noise) — i.e. the rotation is invertible.
* Variable T: kernel works correctly across multiple T values per call
  (the use case is variable-length chunks).
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


def _ref_partial_rope(
    x: torch.Tensor,           # (T, n_heads, head_dim)
    pos: torch.Tensor,         # (T,) int32
    rope_base: float,
    rot_dim: int,
) -> torch.Tensor:
    """Pure-PyTorch reference for FT's partial RoPE in pair-interleave layout.

    Mirrors the kernel's precision exactly: cos/sin stay in fp32, the
    multiply runs at fp32, and only the final store casts to ``x.dtype``.
    This way the kernel and the reference round at the SAME boundary,
    so any |Δ| above bf16-store noise indicates a math bug.
    """
    T, H, D = x.shape
    half = rot_dim // 2
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, half, dtype=torch.float32, device=x.device) * 2.0 / rot_dim))
    p_f = pos.to(torch.float32)
    freqs = torch.einsum("t,d->td", p_f, inv_freq)  # (T, rot_dim/2) fp32
    cos = freqs.cos()                                # fp32 (matches kernel)
    sin = freqs.sin()                                # fp32

    out = x.clone()
    rot_view = out[..., :rot_dim]                    # (T, H, rot_dim) bf16
    even = rot_view[..., 0::2].float()               # promote to fp32 for the multiply
    odd  = rot_view[..., 1::2].float()
    rot_even = even * cos.unsqueeze(1) - odd * sin.unsqueeze(1)   # fp32
    rot_odd  = even * sin.unsqueeze(1) + odd * cos.unsqueeze(1)   # fp32
    # Cast back to bf16 on store (matches the kernel's tl.store rounding).
    rot_view[..., 0::2] = rot_even.to(x.dtype)
    rot_view[..., 1::2] = rot_odd.to(x.dtype)
    return out


def _ref_partial_rope_inv(
    x: torch.Tensor, pos: torch.Tensor, rope_base: float, rot_dim: int,
) -> torch.Tensor:
    """Inverse rotation: cos(-θ)=cos(θ), sin(-θ)=-sin(θ). Same precision
    handling as ``_ref_partial_rope``."""
    T, H, D = x.shape
    half = rot_dim // 2
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, half, dtype=torch.float32, device=x.device) * 2.0 / rot_dim))
    p_f = pos.to(torch.float32)
    freqs = torch.einsum("t,d->td", p_f, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    out = x.clone()
    rot_view = out[..., :rot_dim]
    even = rot_view[..., 0::2].float()
    odd  = rot_view[..., 1::2].float()
    inv_even = even * cos.unsqueeze(1) + odd * sin.unsqueeze(1)
    inv_odd  = -even * sin.unsqueeze(1) + odd * cos.unsqueeze(1)
    rot_view[..., 0::2] = inv_even.to(x.dtype)
    rot_view[..., 1::2] = inv_odd.to(x.dtype)
    return out


def _check(name, a, b, tol=5e-3):
    delta = (a.float() - b.float()).abs()
    print(f"  {name:30s} max|Δ|={delta.max().item():.3e}  mean|Δ|={delta.mean().item():.3e}")
    assert delta.max().item() <= tol, f"{name} max|Δ| {delta.max().item():.3e} > {tol}"


def main():
    from flextrain.nn.blocks.rope import (
        apply_rope_partial_fwd, apply_rope_partial_bwd,
        build_partial_rope_inv_freq,
    )

    torch.manual_seed(0)

    # ----- Test 1: small (T, head_dim, rot_dim) — exercise everything -----
    print("=== Test 1: T=16, head_dim=64, rot_dim=16, n_heads_q=4, n_kv=2 ===")
    T = 16
    head_dim = 64
    rot_dim = 16
    n_q = 4
    n_kv = 2
    rope_base = 500_000.0

    q = torch.randn(T, n_q, head_dim, dtype=DTYPE, device=DEVICE) * 0.5
    k = torch.randn(T, n_kv, head_dim, dtype=DTYPE, device=DEVICE) * 0.5
    pos = torch.arange(T, dtype=torch.int32, device=DEVICE).reshape(-1, 1)

    # Build inv_freq via the block-level helper (so test exercises the
    # full plumbing).
    inv_freq = build_partial_rope_inv_freq(
        rot_dim=rot_dim, rope_base=rope_base, rope_scaling=None,
    ).to(DEVICE)

    # FT in-place fwd.
    q_ft = q.clone()
    k_ft = k.clone()
    apply_rope_partial_fwd(
        [q_ft, k_ft], pos, inv_freq, rot_dim,
    )

    # Reference fwd.
    q_ref = _ref_partial_rope(q, pos.squeeze(-1), rope_base, rot_dim)
    k_ref = _ref_partial_rope(k, pos.squeeze(-1), rope_base, rot_dim)

    _check("Q fwd parity", q_ft, q_ref)
    _check("K fwd parity", k_ft, k_ref)

    # Pass-through: channels [rot_dim:] should match the original input.
    _check("Q pass-through bit-identical", q_ft[..., rot_dim:], q[..., rot_dim:], tol=0.0)
    _check("K pass-through bit-identical", k_ft[..., rot_dim:], k[..., rot_dim:], tol=0.0)

    # ----- Test 2: bwd is inverse of fwd within bf16 noise -----
    print("\n=== Test 2: bwd ∘ fwd ≈ identity ===")
    q_round = q.clone()
    k_round = k.clone()
    apply_rope_partial_fwd([q_round, k_round], pos, inv_freq, rot_dim)
    apply_rope_partial_bwd([q_round, k_round], pos, inv_freq, rot_dim)
    _check("Q round-trip", q_round, q, tol=2e-2)  # bf16 round-trip noise
    _check("K round-trip", k_round, k, tol=2e-2)

    # ----- Test 3: bwd matches independent reference inverse -----
    print("\n=== Test 3: bwd matches reference inverse ===")
    q_ft_bwd = q.clone()
    k_ft_bwd = k.clone()
    apply_rope_partial_bwd([q_ft_bwd, k_ft_bwd], pos, inv_freq, rot_dim)
    q_ref_bwd = _ref_partial_rope_inv(q, pos.squeeze(-1), rope_base, rot_dim)
    k_ref_bwd = _ref_partial_rope_inv(k, pos.squeeze(-1), rope_base, rot_dim)
    _check("Q bwd parity", q_ft_bwd, q_ref_bwd)
    _check("K bwd parity", k_ft_bwd, k_ref_bwd)

    # ----- Test 4: full rotary case (rot_dim == head_dim) — should match
    # the full-rotary kernel exactly. -----
    print("\n=== Test 4: rot_dim == head_dim equivalence ===")
    rot_dim_full = head_dim
    inv_freq_full = build_partial_rope_inv_freq(
        rot_dim=rot_dim_full, rope_base=rope_base,
    ).to(DEVICE)
    q_full = q.clone()
    apply_rope_partial_fwd([q_full], pos, inv_freq_full, rot_dim_full)
    # Compare to full-rotary kernel.
    from flextrain.nn.blocks.rope import apply_rope_fwd, build_rope_inv_freq
    q_full_ref = q.clone()
    inv_freq_full_ref = build_rope_inv_freq(
        head_dim=head_dim, rope_base=rope_base,
    ).to(DEVICE)
    apply_rope_fwd([q_full_ref], pos, inv_freq_full_ref)
    _check("Full-rotary equivalence", q_full, q_full_ref)

    # ----- Test 5: variable T per call (different chunks) -----
    print("\n=== Test 5: variable T per call ===")
    for T_test in (1, 7, 32, 100, 257):
        q_t = torch.randn(T_test, n_q, head_dim, dtype=DTYPE, device=DEVICE) * 0.5
        k_t = torch.randn(T_test, n_kv, head_dim, dtype=DTYPE, device=DEVICE) * 0.5
        pos_t = torch.arange(T_test, dtype=torch.int32, device=DEVICE).reshape(-1, 1)
        q_ft_t = q_t.clone(); k_ft_t = k_t.clone()
        apply_rope_partial_fwd([q_ft_t, k_ft_t], pos_t, inv_freq, rot_dim)
        q_ref_t = _ref_partial_rope(q_t, pos_t.squeeze(-1), rope_base, rot_dim)
        k_ref_t = _ref_partial_rope(k_t, pos_t.squeeze(-1), rope_base, rot_dim)
        _check(f"T={T_test} Q", q_ft_t, q_ref_t)
        _check(f"T={T_test} K", k_ft_t, k_ref_t)

    print("\n✓ partial-RoPE kernel parity PASSED")


if __name__ == "__main__":
    main()
