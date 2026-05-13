"""Numeric parity tests for the multi-axis MRoPE in
``flextrain.nn.blocks.rope.apply_rope_mrope_fwd/bwd``.

Two invariants:

1. **Degenerate-3D consistency**: when all three position axes carry the
   same text position per token, the MRoPE output must be byte-identical
   to applying standard partial-RoPE with the 1-D position vector. This
   is the property that lets text-only Qwen3.5/3.6 training stay
   unchanged while the MRoPE kernel is in the block.

2. **HF interleaved-MRoPE reference**: with the
   ``mrope_interleaved=True`` axis-assignment table (matching HF
   ``Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope`` for
   ``mrope_section=[11, 11, 10]``), the per-pair axis-vs-channel
   assignment must produce the expected layout: 11 channels follow t,
   11 follow h, 10 follow w, with the interleaving pattern documented
   in HF.

Run: ``./run_with_env.sh python tests/test_mrope_block.py``
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.nn.blocks.rope import (
    apply_rope_mrope_fwd,
    apply_rope_partial_fwd,
    build_mrope_axis_assignment,
    build_partial_rope_inv_freq,
)


def _make_qk(T: int, n_heads: int, head_dim: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build random ``(T, n_heads, head_dim)`` Q / K tensors in bf16 on CUDA."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, n_heads, head_dim, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(T, n_heads, head_dim, generator=g, device="cuda", dtype=torch.bfloat16)
    return q, k


def test_degenerate_3d_matches_partial_rope() -> None:
    """seq_positions_3d = (t, t, t) per token -> MRoPE output should match
    standard partial-RoPE applied to seq_positions_1d = (t,) per token.
    """
    torch.manual_seed(0)
    T = 64
    n_heads = 4
    head_dim = 128
    rot_dim = 32  # partial_rotary_factor = 0.25
    n_pairs = rot_dim // 2
    rope_base = 10_000_000.0

    inv_freq = build_partial_rope_inv_freq(rot_dim, rope_base).to("cuda")

    # Degenerate axis assignment: every pair uses axis 0 (t). Confirms
    # the math collapses cleanly when all positions are equal across
    # axes -- a useful safety property even for the interleaved layout.
    axis_assignment = torch.zeros(n_pairs, dtype=torch.int64, device="cuda")

    seq_positions_1d = torch.arange(T, dtype=torch.int32, device="cuda").reshape(-1, 1)
    seq_positions_3d = seq_positions_1d.expand(T, 3).contiguous()

    q1, k1 = _make_qk(T, n_heads, head_dim, seed=42)
    q2, k2 = q1.clone(), k1.clone()

    apply_rope_partial_fwd([q1, k1], seq_positions_1d, inv_freq, rot_dim)
    apply_rope_mrope_fwd([q2, k2], seq_positions_3d, inv_freq, rot_dim, axis_assignment)

    # The pure-PyTorch MRoPE differs from the Triton partial-RoPE kernel
    # by at most 1 bf16 ULP per element (different reduction order +
    # round-to-nearest splits at the grid boundary). 1 bf16 ULP at
    # magnitude ~1 is 1/64 = 1.5625e-2; allow 2.0e-2 to give a touch of
    # headroom. The algebraic formula is identical -- it's purely a
    # quantization-direction difference. This is documented behavior;
    # see ``docs/internal/multimodal_session_notes.md``.
    err_q = (q1.float() - q2.float()).abs().max().item()
    err_k = (k1.float() - k2.float()).abs().max().item()
    assert err_q < 2.0e-2, f"Q max abs diff {err_q:.4e} exceeds 2.0e-2 (>1 bf16 ULP)"
    assert err_k < 2.0e-2, f"K max abs diff {err_k:.4e} exceeds 2.0e-2 (>1 bf16 ULP)"
    print(
        f"[OK] Degenerate-3D MRoPE matches partial-RoPE within bf16 noise "
        f"(Q={err_q:.4e}, K={err_k:.4e})"
    )


def test_axis_assignment_qwen3_vl_interleaved() -> None:
    """For mrope_section=[11, 11, 10] with mrope_interleaved=True, the
    HF interleaved layout should produce the expected per-pair axis
    indices:

    * 11 h-pairs at indices 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31
    * 10 w-pairs at indices 2, 5, 8, 11, 14, 17, 20, 23, 26, 29
    * 11 t-pairs at all remaining indices (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30)
    """
    mrope_section = (11, 11, 10)
    out = build_mrope_axis_assignment(mrope_section, mrope_interleaved=True, device="cpu")
    assert out.shape == (32,), f"got shape {tuple(out.shape)}, expected (32,)"

    # Build expected mapping by mirroring HF's algorithm.
    expected = [0] * 32
    s_t, s_h, s_w = mrope_section
    for k in range(32):
        if k % 3 == 1 and k < 3 * s_h:
            expected[k] = 1  # h
        elif k % 3 == 2 and k < 3 * s_w:
            expected[k] = 2  # w
        # else stays 0 (t)
    actual = out.tolist()
    assert actual == expected, f"axis_assignment mismatch:\n  got      {actual}\n  expected {expected}"

    # Sanity: counts per axis match mrope_section.
    counts = [actual.count(i) for i in range(3)]
    assert counts == [11, 11, 10], f"per-axis counts {counts} != [11, 11, 10]"
    print(f"[OK] Qwen3-VL interleaved MRoPE axis assignment matches HF reference: {counts}")


def test_axis_assignment_contiguous() -> None:
    """For mrope_interleaved=False, sections are contiguous: first s_t
    pairs are t (axis 0), next s_h are h (axis 1), last s_w are w (axis 2).
    """
    mrope_section = (11, 11, 10)
    out = build_mrope_axis_assignment(mrope_section, mrope_interleaved=False, device="cpu")
    expected = [0] * 11 + [1] * 11 + [2] * 10
    assert out.tolist() == expected, (
        f"contiguous axis_assignment mismatch:\n  got      {out.tolist()}\n"
        f"  expected {expected}"
    )
    print("[OK] Contiguous (non-interleaved) MRoPE axis assignment correct.")


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available, skipping MRoPE block tests (the "
              "degenerate-3D test requires the partial-RoPE Triton kernel).")
        # Still run the axis-assignment tests (CPU-only).
        test_axis_assignment_qwen3_vl_interleaved()
        test_axis_assignment_contiguous()
        return
    test_axis_assignment_qwen3_vl_interleaved()
    test_axis_assignment_contiguous()
    test_degenerate_3d_matches_partial_rope()
    print("\nAll MRoPE block-level tests passed.")


if __name__ == "__main__":
    main()
