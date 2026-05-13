"""Unit tests for :mod:`flextrain.nn.splices.concat`.

Covers the only sharp edge in the Phase 1 multimodal input layer:
placeholder-row gradient routing. Verifies that
``concat_splice_fwd`` scatters encoder rows onto placeholder positions
and ``concat_splice_bwd`` (a) zeros placeholder rows in ``d_text_emb``
to protect the embed table and (b) routes placeholder-position dx
into the optional encoder grad accumulator.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.core.modality import ImageEmbeddings, ImageGradInputs
from flextrain.nn.splices import concat_splice_bwd, concat_splice_fwd


class _FakeChunkMeta:
    """Lightweight stand-in for ``ChunkMeta`` -- the splice only reads
    ``.extra``, so we don't need the full dataclass."""
    def __init__(self, extra):
        self.extra = extra


def _build_meta(positions: list[int], assignments: list[int]) -> _FakeChunkMeta:
    """Build a chunk meta carrying one image's worth of splice metadata."""
    return _FakeChunkMeta(extra={
        "mm_placeholder_positions": {
            "image": {0: torch.tensor(positions, dtype=torch.int64)},
        },
        "mm_image_assignment": {
            "image": {0: torch.tensor(assignments, dtype=torch.int64)},
        },
    })


def test_fwd_scatters_encoder_rows() -> None:
    """concat_splice_fwd should replace placeholder rows with encoder rows."""
    T_chunk, d_model = 8, 4
    text_emb = torch.arange(T_chunk * d_model, dtype=torch.float32).reshape(T_chunk, d_model)
    text_emb_ref = text_emb.clone()
    encoder_rows = torch.tensor(
        [[-1, -2, -3, -4], [-5, -6, -7, -8]], dtype=torch.float32,
    )
    image_embeddings = ImageEmbeddings(
        embeds=encoder_rows,
        token_offsets=torch.tensor([0, 2], dtype=torch.int32),
        grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.int32),
    )
    # Place encoder rows at chunk positions 3 and 5; image rows 0, 1.
    meta = _build_meta(positions=[3, 5], assignments=[0, 1])

    out = concat_splice_fwd(text_emb, image_embeddings, meta, encoder_id=0)
    # Out is the same tensor (in-place).
    assert out.data_ptr() == text_emb.data_ptr()
    assert torch.equal(out[3], encoder_rows[0]), "row 3 should be encoder row 0"
    assert torch.equal(out[5], encoder_rows[1]), "row 5 should be encoder row 1"
    # Untouched positions must match the original text_emb.
    for i in (0, 1, 2, 4, 6, 7):
        assert torch.equal(out[i], text_emb_ref[i]), f"row {i} corrupted"
    print("[OK] concat_splice_fwd scatters encoder rows onto placeholder positions.")


def test_bwd_zeros_placeholders_phase1_frozen() -> None:
    """Phase 1: with d_image_grad_accum=None (frozen encoder), the
    placeholder positions in d_text_emb must be zeroed so that
    TokenEmbed.backward doesn't poison the embed-table row."""
    T_chunk, d_model = 8, 4
    d_text_emb = torch.arange(T_chunk * d_model, dtype=torch.float32).reshape(T_chunk, d_model).clone()
    image_embeddings = ImageEmbeddings(
        embeds=torch.zeros(2, d_model),
        token_offsets=torch.tensor([0, 2], dtype=torch.int32),
        grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.int32),
    )
    meta = _build_meta(positions=[3, 5], assignments=[0, 1])

    out = concat_splice_bwd(
        d_text_emb,
        image_embeddings,
        d_image_grad_accum=None,
        chunk_meta=meta,
        encoder_id=0,
    )
    assert out.data_ptr() == d_text_emb.data_ptr()
    assert torch.equal(out[3], torch.zeros(d_model)), "placeholder row 3 must be zero"
    assert torch.equal(out[5], torch.zeros(d_model)), "placeholder row 5 must be zero"
    # Non-placeholder rows unchanged.
    for i in (0, 1, 2, 4, 6, 7):
        expected = torch.tensor([i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3], dtype=torch.float32)
        assert torch.equal(out[i], expected), f"non-placeholder row {i} corrupted"
    print("[OK] concat_splice_bwd zeros placeholder rows in d_text_emb (Phase 1 frozen).")


def test_bwd_accumulates_into_encoder_grad_phase3() -> None:
    """Phase 3: when d_image_grad_accum is non-None, placeholder-row dx
    must accumulate into it via index_add at the encoder-row index
    given by image_assignment."""
    T_chunk, d_model = 8, 4
    # Distinct values at placeholder positions so we can check the route.
    d_text_emb = torch.arange(T_chunk * d_model, dtype=torch.float32).reshape(T_chunk, d_model).clone()
    placeholder_dx_at_3 = d_text_emb[3].clone()
    placeholder_dx_at_5 = d_text_emb[5].clone()

    image_embeddings = ImageEmbeddings(
        embeds=torch.zeros(2, d_model),
        token_offsets=torch.tensor([0, 2], dtype=torch.int32),
        grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.int32),
    )
    meta = _build_meta(positions=[3, 5], assignments=[1, 0])  # swapped assignment

    accum = ImageGradInputs(d_embeds=torch.zeros(2, d_model))
    concat_splice_bwd(
        d_text_emb,
        image_embeddings,
        d_image_grad_accum=accum,
        chunk_meta=meta,
        encoder_id=0,
    )
    # position 3 -> encoder row 1; position 5 -> encoder row 0.
    assert torch.equal(accum.d_embeds[1], placeholder_dx_at_3), (
        f"encoder row 1 should equal d_text_emb[3], got {accum.d_embeds[1]} vs {placeholder_dx_at_3}"
    )
    assert torch.equal(accum.d_embeds[0], placeholder_dx_at_5), (
        f"encoder row 0 should equal d_text_emb[5], got {accum.d_embeds[0]} vs {placeholder_dx_at_5}"
    )
    # And d_text_emb still has zeros at placeholder positions afterward.
    assert torch.equal(d_text_emb[3], torch.zeros(d_model))
    assert torch.equal(d_text_emb[5], torch.zeros(d_model))
    print("[OK] concat_splice_bwd routes placeholder dx into encoder accumulator (Phase 3).")


def test_no_op_when_chunk_has_no_images() -> None:
    """Splice should be a no-op when the chunk's metadata has no
    placeholder positions for this encoder. Avoids penalizing
    text-only chunks in a multimodal batch."""
    T_chunk, d_model = 4, 4
    text_emb = torch.arange(T_chunk * d_model, dtype=torch.float32).reshape(T_chunk, d_model)
    text_emb_ref = text_emb.clone()
    empty_meta = _FakeChunkMeta(extra={})  # no mm_* entries at all
    image_embeddings = ImageEmbeddings(
        embeds=torch.zeros(0, d_model),
        token_offsets=torch.tensor([0], dtype=torch.int32),
        grid_thw=torch.tensor([], dtype=torch.int32).reshape(0, 3),
    )
    out = concat_splice_fwd(text_emb, image_embeddings, empty_meta, encoder_id=0)
    assert torch.equal(out, text_emb_ref), "no-op forward should not modify text_emb"
    print("[OK] concat_splice_fwd is a no-op when chunk has no placeholders.")


def main() -> None:
    test_fwd_scatters_encoder_rows()
    test_bwd_zeros_placeholders_phase1_frozen()
    test_bwd_accumulates_into_encoder_grad_phase3()
    test_no_op_when_chunk_has_no_images()
    print("\nAll ConcatSplice unit tests passed.")


if __name__ == "__main__":
    main()
