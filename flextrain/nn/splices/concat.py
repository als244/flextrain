"""ConcatSplice -- Qwen-VL pattern.

Vision tokens in ``input_ids`` are pre-expanded by the chat-template
tokenizer to occupy N slots per image, where N equals the
``spatial_merge_size**2``-reduced patch count derived from the
image's ``grid_thw``. The text embedder produces a per-token
embedding for *every* slot, including those vision-placeholder slots
(the placeholder ``image_token_id`` is a real row in the embed table);
the encoder produces a parallel stream of vision embeddings; this
splice replaces the placeholder rows with the encoder output.

The 3-D position metadata ``(t, h, w)`` per token is carried via
``chunk.seq_positions`` (shape ``(T, 3) int32``) and consumed by the
attention block's MRoPE dispatch -- the splice doesn't touch
positions.

Chunk-metadata contract
-----------------------
Chunk preparation populates the following entries in
``chunk.meta.extra``:

* ``"mm_placeholder_positions"["image"][encoder_id]`` -- ``(N_in_chunk,)
  int64`` chunk-local token indices that this encoder's output should
  be scattered onto.
* ``"mm_image_assignment"["image"][encoder_id]`` -- parallel
  ``(N_in_chunk,) int64`` row index into the encoder's per-round
  ``ImageEmbeddings.embeds`` for each placeholder.

If either entry is missing for the encoder, this splice is a no-op
(the chunk has no images of that modality).

Backward semantics
------------------
The vision-placeholder rows in the chunk's ``dx`` belong to the
encoder grad accumulator, NOT the embed table -- they were overwritten
by the encoder in forward, so the embed-table row had no influence on
downstream loss. Trying to scatter-add ``dx`` for those rows back to
the embed table would corrupt the placeholder-token's learned vector.

Phase 1 (frozen encoder): ``d_image_grad_accum`` is None; we just
zero the placeholder rows in ``d_text_emb`` to suppress the embed
table contribution. Phase 3 (trainable encoder): a non-None
accumulator collects the placeholder-position ``dx`` via
``index_add_`` keyed by the per-placeholder image-row assignment.
"""

from __future__ import annotations

from typing import Any

import torch

from flextrain.core.modality import ImageEmbeddings, ImageGradInputs

# Sentinels recognised in chunk.meta.extra.
_KEY_POSITIONS = "mm_placeholder_positions"
_KEY_ASSIGNMENT = "mm_image_assignment"
_MODALITY = "image"


def _gather_meta(chunk_meta: Any, encoder_id: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (placeholder_positions, image_assignment) for this
    encoder, or (None, None) if the chunk has no images of this
    modality."""
    extra = getattr(chunk_meta, "extra", None) or {}
    pos_map = extra.get(_KEY_POSITIONS)
    asn_map = extra.get(_KEY_ASSIGNMENT)
    if not pos_map or not asn_map:
        return None, None
    pos_per_modality = pos_map.get(_MODALITY)
    asn_per_modality = asn_map.get(_MODALITY)
    if not pos_per_modality or not asn_per_modality:
        return None, None
    return pos_per_modality.get(encoder_id), asn_per_modality.get(encoder_id)


def concat_splice_fwd(
    text_emb: torch.Tensor,            # (T_chunk, d_model)
    image_embeddings: ImageEmbeddings,  # round-level cache
    chunk_meta: Any,
    encoder_id: int,
) -> torch.Tensor:
    """Scatter the encoder's per-placeholder rows onto ``text_emb`` and
    return the resulting tensor.

    Modifies ``text_emb`` in place (no extra allocation).
    """
    placeholder_positions, image_assignment = _gather_meta(chunk_meta, encoder_id)
    if placeholder_positions is None or placeholder_positions.numel() == 0:
        return text_emb
    if image_assignment is None or image_assignment.shape != placeholder_positions.shape:
        raise ValueError(
            "ConcatSplice: image_assignment must be the same shape as "
            f"placeholder_positions; got {placeholder_positions.shape!r} vs "
            f"{None if image_assignment is None else image_assignment.shape!r}"
        )
    # Gather encoder rows then scatter to text_emb. Both index ops are
    # int64. The encoder's output dtype matches text_emb's dtype
    # (both bfloat16 in Phase 1) so no cast is needed.
    src_rows = image_embeddings.embeds.index_select(
        0, image_assignment.to(text_emb.device).long(),
    )
    if src_rows.dtype != text_emb.dtype:
        src_rows = src_rows.to(text_emb.dtype)
    text_emb.index_copy_(
        0, placeholder_positions.to(text_emb.device).long(), src_rows,
    )
    return text_emb


def concat_splice_bwd(
    d_text_emb: torch.Tensor,           # (T_chunk, d_model)
    image_embeddings: ImageEmbeddings,
    d_image_grad_accum: ImageGradInputs | None,
    chunk_meta: Any,
    encoder_id: int,
) -> torch.Tensor:
    """Route placeholder-position dx to the encoder grad accumulator
    (Phase 3) and zero those rows in ``d_text_emb`` so the embed-table
    scatter-add ignores them.

    Returns ``d_text_emb`` (same tensor, modified in place).
    """
    placeholder_positions, image_assignment = _gather_meta(chunk_meta, encoder_id)
    if placeholder_positions is None or placeholder_positions.numel() == 0:
        return d_text_emb
    placeholder_idx = placeholder_positions.to(d_text_emb.device).long()
    # Optional: accumulate into the encoder grad accumulator (Phase 3
    # trainable encoder). Phase 1 frozen -> ``d_image_grad_accum`` is
    # None and we just drop the placeholder-position dx.
    if d_image_grad_accum is not None:
        if image_assignment is None:
            raise ValueError(
                "ConcatSplice bwd: image_assignment is None but the encoder "
                "accumulator was provided; chunk meta is inconsistent."
            )
        assign_idx = image_assignment.to(d_text_emb.device).long()
        placeholder_dx = d_text_emb.index_select(0, placeholder_idx)
        d_image_grad_accum.d_embeds.index_add_(0, assign_idx, placeholder_dx)
    # Zero placeholder rows so TokenEmbed.backward scatter-add doesn't
    # add bogus gradient to the placeholder embed-table row.
    zero_rows = torch.zeros(
        (placeholder_idx.numel(), d_text_emb.shape[-1]),
        dtype=d_text_emb.dtype,
        device=d_text_emb.device,
    )
    d_text_emb.index_copy_(0, placeholder_idx, zero_rows)
    return d_text_emb
