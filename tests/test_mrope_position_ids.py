"""Unit tests for ``_compute_mrope_position_ids`` (Phase 1.5).

Verifies that per-token MRoPE position generation in
``flextrain.engine.schedule`` matches HF Qwen-VL's
``Qwen3VLForConditionalGeneration.get_rope_index`` algorithm for:

* Text-only sequences (modality_inputs empty/missing) -> returns None.
* A sequence with one image -> text-before / image-block / text-after
  positions match HF's reference layout.
* Counter advances by ``max(merged_h, merged_w)`` after each image
  (NOT by the number of placeholder tokens).

CPU-only.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.core.modality import ImageInputCPU
from flextrain.engine.schedule import _compute_mrope_position_ids


class _FakeSeq:
    """Minimal stand-in for :class:`Sequence` (the helper only reads
    ``.tokens`` and ``.modality_inputs``)."""
    def __init__(self, tokens, modality_inputs=None):
        self.tokens = tokens
        self.modality_inputs = modality_inputs or {}


def test_text_only_returns_none() -> None:
    seq = _FakeSeq(tokens=torch.arange(10, dtype=torch.int64))
    out = _compute_mrope_position_ids(seq)
    assert out is None, f"text-only seq should return None; got {out}"
    print("[OK] _compute_mrope_position_ids returns None for text-only seq.")


def test_single_image_layout_4x4_merge2() -> None:
    """Sequence: 3 text tokens, then 4 image placeholders (post-merge
    from a 4x4 grid with spatial_merge_size=2, so merged grid is 2x2),
    then 2 trailing text tokens. Total 9 tokens.

    Expected MRoPE positions (per HF):

        idx 0..2: (0, 0, 0), (1, 1, 1), (2, 2, 2)             -- text
        idx 3..6: image block, t_pos_start=3, merged_h=merged_w=2:
            (3, 3, 3),  # h=0, w=0
            (3, 3, 4),  # h=0, w=1
            (3, 4, 3),  # h=1, w=0
            (3, 4, 4),  # h=1, w=1
        After image: current_pos = 3 + max(2, 2) = 5.
        idx 7..8: (5, 5, 5), (6, 6, 6)                        -- text
    """
    img = ImageInputCPU(
        pixel_values=torch.zeros(1),  # unused by position generator
        grid_thw=torch.tensor([1, 4, 4], dtype=torch.int32),
        placeholder_positions=torch.tensor([3, 4, 5, 6], dtype=torch.int32),
    )
    seq = _FakeSeq(
        tokens=torch.arange(9, dtype=torch.int64),
        modality_inputs={"image": [img]},
    )
    out = _compute_mrope_position_ids(seq)
    assert out is not None
    assert out.shape == (9, 3), f"shape {tuple(out.shape)} != (9, 3)"
    expected = torch.tensor([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [3, 3, 4],
        [3, 4, 3],
        [3, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
    ], dtype=torch.int32)
    assert torch.equal(out, expected), (
        f"position mismatch:\n  got\n{out}\n  expected\n{expected}"
    )
    print("[OK] Single-image MRoPE layout matches HF reference (4x4 grid, merge=2).")


def test_single_image_no_text_before() -> None:
    """Image starts at position 0 (no text prefix). Sanity check that
    the cursor walk handles cursor=0 correctly."""
    img = ImageInputCPU(
        pixel_values=torch.zeros(1),
        grid_thw=torch.tensor([1, 4, 4], dtype=torch.int32),
        placeholder_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    seq = _FakeSeq(
        tokens=torch.arange(6, dtype=torch.int64),
        modality_inputs={"image": [img]},
    )
    out = _compute_mrope_position_ids(seq)
    expected = torch.tensor([
        [0, 0, 0],  # h=0, w=0
        [0, 0, 1],  # h=0, w=1
        [0, 1, 0],  # h=1, w=0
        [0, 1, 1],  # h=1, w=1
        [2, 2, 2],  # text, current_pos = 0 + max(2,2) = 2
        [3, 3, 3],
    ], dtype=torch.int32)
    assert torch.equal(out, expected), (
        f"position mismatch:\n  got\n{out}\n  expected\n{expected}"
    )
    print("[OK] MRoPE handles image at sequence start.")


def test_rectangular_image() -> None:
    """grid_thw = (1, 6, 4), spatial_merge_size = 2 -> merged grid 3x2.
    Per-frame post-merge count = 6. The fallback path solves
    merge**2 = 24/6 = 4, so merge=2.
    """
    img = ImageInputCPU(
        pixel_values=torch.zeros(1),
        grid_thw=torch.tensor([1, 6, 4], dtype=torch.int32),
        placeholder_positions=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
    )
    seq = _FakeSeq(
        tokens=torch.arange(7, dtype=torch.int64),
        modality_inputs={"image": [img]},
    )
    out = _compute_mrope_position_ids(seq)
    # merged_h=3, merged_w=2. Iteration: t outer (single t=0), h middle,
    # w inner. Expected:
    expected = torch.tensor([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 2, 0],
        [0, 2, 1],
        [3, 3, 3],  # text, current_pos = 0 + max(3, 2) = 3
    ], dtype=torch.int32)
    assert torch.equal(out, expected), (
        f"rectangular-image position mismatch:\n  got\n{out}\n  expected\n{expected}"
    )
    print("[OK] MRoPE handles rectangular grid (6x4 -> merged 3x2) correctly.")


def main() -> None:
    test_text_only_returns_none()
    test_single_image_layout_4x4_merge2()
    test_single_image_no_text_before()
    test_rectangular_image()
    print("\nAll MRoPE position-ID tests passed.")


if __name__ == "__main__":
    main()
