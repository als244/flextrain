"""Numeric parity for Gemma 4's "proportional" partial-rotary RoPE
(``flextrain.nn.blocks.rope.build_partial_rope_inv_freq`` with
``rope_type='proportional'``) against HF's reference implementation
(``transformers.modeling_rope_utils._compute_proportional_rope_parameters``).

The two formulas differ in their denominator:

* Default partial rope (Qwen3-Next/3.5/3.6): ``base ** (-2i / rot_dim)``
* Proportional rope (Gemma 4 global layers):   ``base ** (-2i / head_dim)``

For Gemma 4's 31B / 26B-A4B global-attention layers: ``head_dim=512``,
``rope_theta=1e6``, ``partial_rotary_factor=0.25``. The HF reference
zero-pads inv_freq to length ``head_dim/2``; we only return the rotated
prefix (length ``rot_dim/2``) because flextrain's
``apply_rope_partial_fwd/bwd`` kernel takes ``rot_dim`` as a separate
argument and never reads the zero tail.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.nn.blocks.rope import build_partial_rope_inv_freq


def _hf_proportional_inv_freq(
    head_dim: int, base: float, partial_rotary_factor: float,
) -> torch.Tensor:
    """Inline copy of HF's ``_compute_proportional_rope_parameters``
    (``modeling_rope_utils.py:187-251``) so this test stays standalone if
    transformers isn't installed locally.

    Returns the FULL ``head_dim/2``-length curve (rotated prefix + zero
    tail). The caller compares only the rotated prefix to flextrain's
    output.
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq_rotated = 1.0 / (
        base
        ** (
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64)
            .to(dtype=torch.float32)
            / head_dim
        )
    )
    nope_angles = head_dim // 2 - rope_angles
    if nope_angles > 0:
        return torch.cat(
            (inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32))
        )
    return inv_freq_rotated


@pytest.mark.parametrize(
    "head_dim, prf, theta",
    [
        # Gemma 4 31B + 26B-A4B global layers.
        (512, 0.25, 1_000_000.0),
        # Synthetic variants to exercise the math at other shapes.
        (256, 0.5, 10_000.0),
        (128, 1.0, 10_000.0),   # degenerate: rot_dim == head_dim, no zero tail.
        (64, 0.25, 500_000.0),
    ],
)
def test_proportional_matches_hf(head_dim: int, prf: float, theta: float) -> None:
    rope_angles = int(prf * head_dim // 2)
    rot_dim = 2 * rope_angles
    ft = build_partial_rope_inv_freq(
        rot_dim=rot_dim,
        rope_base=theta,
        rope_scaling={"rope_type": "proportional"},
        head_dim=head_dim,
    )
    hf_full = _hf_proportional_inv_freq(head_dim, theta, prf)
    # Compare only the rotated prefix; the zero tail is implicit in
    # flextrain (apply_rope_partial_fwd doesn't read past rot_dim/2).
    hf_prefix = hf_full[:rope_angles]
    assert ft.shape == hf_prefix.shape, (
        f"shape mismatch: ft={tuple(ft.shape)} hf_prefix={tuple(hf_prefix.shape)}"
    )
    assert torch.allclose(ft, hf_prefix, atol=0.0, rtol=0.0), (
        f"value mismatch (head_dim={head_dim}, prf={prf}, theta={theta}):\n"
        f"  ft[:5]={ft[:5].tolist()}\n"
        f"  hf[:5]={hf_prefix[:5].tolist()}\n"
        f"  max_abs_diff={(ft - hf_prefix).abs().max().item():.3e}"
    )


def test_proportional_differs_from_default_partial() -> None:
    """Sanity: at non-trivial PRF the proportional and default-partial
    curves diverge, so the new branch isn't a no-op alias of the old one."""
    head_dim = 512
    rot_dim = 128
    theta = 1_000_000.0
    proportional = build_partial_rope_inv_freq(
        rot_dim=rot_dim,
        rope_base=theta,
        rope_scaling={"rope_type": "proportional"},
        head_dim=head_dim,
    )
    default = build_partial_rope_inv_freq(
        rot_dim=rot_dim,
        rope_base=theta,
        rope_scaling=None,
    )
    assert not torch.allclose(proportional, default), (
        "proportional rope should NOT equal the default partial-rope curve "
        "when rot_dim != head_dim — denominators differ (head_dim vs rot_dim)"
    )
    # The first frequency (i=0) is base^0 = 1 in both cases. Beyond that
    # they should diverge by exactly the ratio (rot_dim/head_dim) in the
    # exponent.
    assert proportional[0].item() == pytest.approx(1.0)
    assert default[0].item() == pytest.approx(1.0)
    # Last entry: proportional = base^(-(rot_dim-2)/head_dim),
    #             default      = base^(-(rot_dim-2)/rot_dim).
    last_i = rot_dim // 2 - 1
    expected_prop_last = theta ** (-(2 * last_i) / head_dim)
    expected_def_last = theta ** (-(2 * last_i) / rot_dim)
    assert proportional[-1].item() == pytest.approx(expected_prop_last, rel=1e-5)
    assert default[-1].item() == pytest.approx(expected_def_last, rel=1e-5)


def test_proportional_requires_head_dim() -> None:
    """Caller must pass head_dim when rope_type='proportional'."""
    with pytest.raises(ValueError, match=r"requires the head_dim kwarg"):
        build_partial_rope_inv_freq(
            rot_dim=128,
            rope_base=1_000_000.0,
            rope_scaling={"rope_type": "proportional"},
        )


def test_default_partial_unchanged() -> None:
    """Adding the proportional branch must not change the default-partial
    behaviour (Qwen3-Next / 3.5 / 3.6 callers)."""
    inv_default = build_partial_rope_inv_freq(
        rot_dim=128, rope_base=10_000.0, rope_scaling=None,
    )
    # Recompute the inv_freq[i] = base^(-2i/rot_dim) reference.
    expected = 1.0 / (
        10_000.0
        ** (torch.arange(0, 64, dtype=torch.float32) * 2.0 / 128)
    )
    assert torch.allclose(inv_default, expected, atol=0.0, rtol=0.0)


def test_default_partial_with_linear_scaling_unchanged() -> None:
    """Linear scaling still divides the inv_freq curve by ``factor``."""
    inv = build_partial_rope_inv_freq(
        rot_dim=128,
        rope_base=10_000.0,
        rope_scaling={"rope_type": "linear", "factor": 8.0},
    )
    base = 1.0 / (
        10_000.0
        ** (torch.arange(0, 64, dtype=torch.float32) * 2.0 / 128)
    )
    assert torch.allclose(inv, base / 8.0, atol=0.0, rtol=0.0)
