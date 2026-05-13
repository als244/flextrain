"""Data-side schedule: sequences -> rounds -> chunks.

Two functions:

* :func:`split_sequences` — group sequences into gradient-accumulation
  rounds subject to a token-count and chunk-count budget.
* :func:`prepare_training_chunks` — pack one round's sequences into
  concrete :class:`TrainingChunk` objects and their :class:`ChunkMeta`.

Both match the observable contract of
``orig/active_model.py`` ``split_sequences`` (line 820) and
``prepare_training_chunks`` (line 897), but the internal structure is
cleaner: a small-sequence packing buffer as one dataclass with typed
slots, a big-sequence chunker as a generator, and a single fold over
the merged chunk stream to compute sequence groups. Also extended to
support non-causal attention (see :class:`ChunkPolicy`).

Causal vs. non-causal attention
-------------------------------
Causal attention (default for autoregressive LMs) lets us split a
long sequence across multiple consecutive chunks: each token only
attends to positions ≤ its own, so the KV rolling window carries the
earlier-chunk tokens forward and backward correctness is preserved.

Non-causal attention (bidirectional encoders, some embedding /
contrastive training objectives) lets every token attend to every
other token in the same sequence. Splitting a sequence across chunks
would drop the cross-chunk attention edges, so the packing policy
must REJECT any sequence longer than ``max_chunk_size`` when the
model is non-causal.

The :class:`ChunkPolicy` enum controls this. The engine reads it from
the model configuration (``ActiveModel.chunk_policy``) and passes it
through.

Observable contract (causal, default)
-------------------------------------
Packing rules:

* **Small sequence** (``len(s) <= max_chunk_size``): greedy first-fit
  into a running buffer. When a new small sequence wouldn't fit, flush
  the buffer into a chunk and start a fresh buffer.
* **Large sequence** (``len(s) > max_chunk_size``): flush any pending
  buffer, then produce ``ceil(len / max_chunk_size)`` dedicated chunks
  (each a single-sequence slice).

Sequence-group rule:

* A new group starts at every chunk whose first sequence begins at
  intra-sequence offset 0. Continuation chunks of a large sequence
  (prior_len > 0) attach to the same group as the start chunk.

The engine's backward pass walks groups in reverse to drive KV-context
refresh (``orig/active_model.py:1426-1462``). Breaking either rule
corrupts the KV window.

Observable contract (non-causal)
--------------------------------
Same small-sequence packing as above. **Large sequences are not
allowed** — :func:`prepare_training_chunks` raises ``ValueError``.
Callers must pre-filter or truncate oversize sequences before handing
them to the engine.

Since no sequence spans chunks, every chunk's first sequence starts
at offset 0, so every chunk begins a new sequence group. Equivalent:
``len(seq_groups) == len(chunks)`` in non-causal mode. This trivially
preserves the KV-context refresh contract (there is no cross-chunk KV
continuity to preserve).

What this module does NOT do
----------------------------
* Save-level DP — :mod:`flextrain.core.save_level`.
* Buffer allocation — :mod:`flextrain.engine.buffers`.
* Compute — :mod:`flextrain.nn`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterator, Sequence as _Seq

import torch

from flextrain.core.layer import ChunkMeta


class ChunkPolicy(enum.Enum):
    """How the scheduler is allowed to break sequences across chunks.

    * :attr:`CAUSAL` — a sequence longer than ``max_chunk_size`` is
      split into consecutive single-sequence chunks. Correct for
      autoregressive LMs where every token attends only to its
      predecessors.
    * :attr:`NON_CAUSAL` — splitting would break bidirectional
      attention edges. Oversize sequences are a hard error; callers
      must pre-filter / truncate.
    """

    CAUSAL = "causal"
    NON_CAUSAL = "non_causal"


# ---------------------------------------------------------------------------
# Dataclasses: chunk payloads + round structure.
# ---------------------------------------------------------------------------


@dataclass
class ChunkSeqRef:
    """One sequence's contribution to a chunk.

    Attributes
    ----------
    seq
        The :class:`Sequence` object (or anything with a
        ``per_token_loss`` attribute). The engine writes per-token loss
        back into ``seq.per_token_loss`` after head runs
        (``orig/active_model.py:1388-1390``).
    seq_range
        Half-open ``(start, end)`` range within ``seq.tokens`` that this
        chunk covers.
    chunk_range
        Half-open ``(start, end)`` range within the packed chunk where
        this sequence's tokens live.
    """

    seq: object
    seq_range: tuple[int, int]
    chunk_range: tuple[int, int]


@dataclass
class TrainingChunk:
    """One packed chunk of tokens ready to run.

    Attributes
    ----------
    id
        Zero-based chunk index within the round.
    meta
        :class:`ChunkMeta` with flash-attn varlen offsets / lengths.
    token_ids
        Int64 on the compute device, shape ``(total_q,)``.
    label_ids
        Int64 on the compute device, shape ``(total_q,)``; may be
        ``None`` for inference-only rounds (future extension).
        Positions where ``label_ids == -100`` are excluded from
        the loss + gradient (PyTorch / HF ``ignore_index`` convention).
    seqs
        :class:`ChunkSeqRef` for each contributing sequence.
    """

    id: int
    meta: ChunkMeta
    token_ids: torch.Tensor
    label_ids: torch.Tensor | None
    seqs: list[ChunkSeqRef]


@dataclass
class PreparedRound:
    """One gradient-accumulation round's prepared state.

    Attributes
    ----------
    chunks
        Flat list of all chunks in this round, indexed by ``chunk.id``.
        Equivalent to orig's ``chunk_mapping`` dict.
    seq_groups
        List-of-lists. A new group begins wherever a chunk's first
        sequence starts at intra-sequence offset 0.
    total_tokens
        Sum of ``chunk.meta.total_q`` across all chunks in the round.
    """

    chunks: list[TrainingChunk]
    seq_groups: list[list[TrainingChunk]]
    total_tokens: int


# ---------------------------------------------------------------------------
# Round splitting.
# ---------------------------------------------------------------------------


def _estimate_chunks(
    seqs: _Seq,
    max_chunk_size: int,
    policy: ChunkPolicy = ChunkPolicy.CAUSAL,
) -> int:
    """Dry-run chunk count: replay the packing without materializing
    anything. Needed because ``max_training_chunks`` is an engine-side
    hard limit we must not overshoot.

    Under :attr:`ChunkPolicy.NON_CAUSAL`, a sequence larger than
    ``max_chunk_size`` is treated as its own dedicated chunk (size =
    len). ``split_sequences`` will then reject it at its
    ``max_total_round_tokens`` check, or :func:`prepare_training_chunks`
    will raise at materialization time; we count it as 1 here so the
    chunk budget doesn't mask the real error location.
    """
    count = 0
    buf = 0
    for s in seqs:
        n = len(s)
        if n > max_chunk_size:
            if buf > 0:
                count += 1
                buf = 0
            if policy is ChunkPolicy.NON_CAUSAL:
                count += 1
            else:
                count += (n + max_chunk_size - 1) // max_chunk_size
        else:
            if buf + n > max_chunk_size:
                if buf > 0:
                    count += 1
                buf = 0
            buf += n
    if buf > 0:
        count += 1
    return count


def split_sequences(
    sequences: _Seq,
    *,
    target_round_tokens: int,
    max_total_round_tokens: int,
    max_chunk_size: int,
    max_training_chunks: int,
    policy: ChunkPolicy = ChunkPolicy.CAUSAL,
) -> tuple[list[list], int]:
    """Group sequences into gradient-accumulation rounds.

    A sequence is added to the current round iff both:

    * Adding its tokens would NOT exceed ``target_round_tokens``, AND
    * The resulting chunk count (via :func:`_estimate_chunks`) would
      NOT exceed ``max_training_chunks``.

    If either limit would be broken, the current round is finalized
    and the sequence starts a new one.

    Errors
    ------
    * Any sequence longer than ``max_total_round_tokens`` is a hard
      error regardless of policy.
    * Under :attr:`ChunkPolicy.NON_CAUSAL`, any sequence longer than
      ``max_chunk_size`` is also a hard error — splitting would break
      bidirectional attention. Caller must pre-filter or truncate.

    Returns
    -------
    ``(rounds, total_tokens)``.
    """
    rounds: list[list] = []
    current: list = []
    current_tokens = 0
    total_tokens = 0

    for seq in sequences:
        n = len(seq)
        total_tokens += n

        if n > max_total_round_tokens:
            raise ValueError(
                f"Sequence too long: {n} tokens exceeds "
                f"max_total_round_tokens={max_total_round_tokens}"
            )
        if policy is ChunkPolicy.NON_CAUSAL and n > max_chunk_size:
            raise ValueError(
                f"Non-causal attention cannot split a sequence across chunks, "
                f"but got a sequence of length {n} > max_chunk_size="
                f"{max_chunk_size}. Pre-filter or truncate before calling."
            )

        over_tokens = current_tokens + n > target_round_tokens
        over_chunks = (
            _estimate_chunks(current + [seq], max_chunk_size, policy)
            > max_training_chunks
        )
        if current and (over_tokens or over_chunks):
            rounds.append(current)
            current = []
            current_tokens = 0

        current.append(seq)
        current_tokens += n

    if current:
        rounds.append(current)
    return rounds, total_tokens


# ---------------------------------------------------------------------------
# Chunk preparation.
#
# We split the problem into three stages with no shared mutable state:
#
#   (1) ``_pack_sequences`` produces an iterator of ``_PendingChunk``
#       (CPU-only, no GPU transfer, no ChunkMeta). One pass over the
#       sequence list, two cases (small / large), readable packing
#       logic.
#   (2) For each ``_PendingChunk``, we materialize ``TrainingChunk``
#       (torch.cat + to(device) + ChunkMeta.build).
#   (3) Fold over the chunks to compute sequence groups.
# ---------------------------------------------------------------------------


@dataclass
class _SeqContribution:
    """One sequence's rows inside a pending chunk -- pure-Python view,
    no tensors yet."""

    seq: object
    seq_range: tuple[int, int]
    chunk_range: tuple[int, int]


@dataclass
class _PendingChunk:
    """Pre-materialization: lists + counts, no device tensors yet.

    Fields mirror the ``lens`` / ``pos`` / ``prior_lens`` /
    ``prior_offsets`` fed into :meth:`ChunkMeta.build`.
    """

    token_slices: list[torch.Tensor] = field(default_factory=list)
    label_slices: list[torch.Tensor] = field(default_factory=list)
    lens: list[int] = field(default_factory=list)
    pos: list[int] = field(default_factory=list)
    prior_lens: list[int] = field(default_factory=list)
    prior_offsets: list[int] = field(default_factory=list)
    # Per-contributing-seq flag: True iff later chunks of this seq exist
    # in fwd order. Small (packed) seqs are always False (they fit in
    # one chunk). Large-seq continuation chunks are True for every
    # emitted chunk except the seq's final chunk. Used by attention
    # bwd to decide whether prior reverse iters' cross-chunk dK/dV
    # contributions are sitting in the kv_cache.dk/dv window and must
    # be preserved across this chunk's flash_attn_bwd call.
    has_more_chunks: list[bool] = field(default_factory=list)
    contributions: list[_SeqContribution] = field(default_factory=list)
    size: int = 0

    def is_empty(self) -> bool:
        return self.size == 0

    def starts_a_new_group(self) -> bool:
        """``True`` iff the first contributing sequence starts at offset
        0 within its sequence. Small sequences always satisfy this (they
        enter at offset 0); large-sequence continuation chunks do not.
        """
        return self.prior_lens[0] == 0 if self.prior_lens else True


def _pack_small(seq, pc: _PendingChunk) -> None:
    """Append a small (fits-in-chunk) sequence to a pending chunk."""
    n = len(seq)
    start = pc.size
    pc.token_slices.append(seq.tokens)
    pc.label_slices.append(seq.targets)
    pc.lens.append(n)
    pc.pos.extend(range(n))
    pc.prior_lens.append(0)
    pc.prior_offsets.append(start)
    pc.has_more_chunks.append(False)
    pc.contributions.append(
        _SeqContribution(
            seq=seq, seq_range=(0, n), chunk_range=(start, start + n)
        )
    )
    pc.size += n


def _emit_large(seq, max_chunk_size: int) -> Iterator[_PendingChunk]:
    """Yield one dedicated :class:`_PendingChunk` per
    ``max_chunk_size``-sized slice of a large sequence. Last chunk may
    be partial."""
    n = len(seq)
    cursor = 0
    while cursor < n:
        take = min(max_chunk_size, n - cursor)
        pc = _PendingChunk()
        pc.token_slices.append(seq.tokens[cursor : cursor + take])
        pc.label_slices.append(seq.targets[cursor : cursor + take])
        pc.lens.append(take)
        pc.pos.extend(range(cursor, cursor + take))
        pc.prior_lens.append(cursor)  # <-- key: continuation chunks have prior_len>0
        pc.prior_offsets.append(0)
        # has_more_chunks: True iff this is NOT the final chunk of seq.
        pc.has_more_chunks.append(cursor + take < n)
        pc.contributions.append(
            _SeqContribution(
                seq=seq, seq_range=(cursor, cursor + take), chunk_range=(0, take)
            )
        )
        pc.size = take
        yield pc
        cursor += take


def _pack_sequences(
    seqs: _Seq,
    max_chunk_size: int,
    policy: ChunkPolicy = ChunkPolicy.CAUSAL,
) -> Iterator[_PendingChunk]:
    """Stream sequences into :class:`_PendingChunk` objects.

    Small-sequence packing keeps a single mutable buffer in flight;
    big-sequence chunking yields dedicated chunks directly (causal only).
    Under :attr:`ChunkPolicy.NON_CAUSAL`, oversize sequences raise.
    """
    buf = _PendingChunk()
    for s in seqs:
        if len(s) > max_chunk_size:
            if policy is ChunkPolicy.NON_CAUSAL:
                raise ValueError(
                    f"Non-causal attention cannot split a sequence across "
                    f"chunks, but got a sequence of length {len(s)} > "
                    f"max_chunk_size={max_chunk_size}."
                )
            if not buf.is_empty():
                yield buf
                buf = _PendingChunk()
            yield from _emit_large(s, max_chunk_size)
        else:
            if buf.size + len(s) > max_chunk_size and not buf.is_empty():
                yield buf
                buf = _PendingChunk()
            _pack_small(s, buf)
    if not buf.is_empty():
        yield buf


def _materialize(
    pc: _PendingChunk, chunk_id: int, device: torch.device | str
) -> TrainingChunk:
    """Turn a :class:`_PendingChunk` into a device-resident
    :class:`TrainingChunk` + :class:`ChunkMeta`.

    Each pending chunk's token/label slices are host tensors produced
    by the sequence source; we ``torch.cat`` and DMA once per chunk
    (not once per sequence contribution).
    """
    token_ids = (
        torch.cat(pc.token_slices).long().to(device, non_blocking=True)
    )
    label_ids = (
        torch.cat(pc.label_slices).long().to(device, non_blocking=True)
    )
    meta = ChunkMeta.build(
        seq_lens=pc.lens,
        seq_positions=pc.pos,
        prior_seq_lens=pc.prior_lens,
        prior_seq_offsets=pc.prior_offsets,
        has_more_chunks=pc.has_more_chunks,
        device=device,
    )
    seqs = [
        ChunkSeqRef(seq=c.seq, seq_range=c.seq_range, chunk_range=c.chunk_range)
        for c in pc.contributions
    ]
    return TrainingChunk(
        id=chunk_id,
        meta=meta,
        token_ids=token_ids,
        label_ids=label_ids,
        seqs=seqs,
    )


def prepare_training_chunks(
    round_seqs: _Seq,
    *,
    max_chunk_size: int,
    device: torch.device | str,
    policy: ChunkPolicy = ChunkPolicy.CAUSAL,
) -> PreparedRound:
    """Pack ``round_seqs`` into :class:`TrainingChunk` objects and group
    them into sequence groups.

    See module docstring for the observable packing contract.
    """
    pending: list[_PendingChunk] = list(
        _pack_sequences(round_seqs, max_chunk_size, policy)
    )
    chunks: list[TrainingChunk] = [
        _materialize(pc, i, device) for i, pc in enumerate(pending)
    ]

    seq_groups: list[list[TrainingChunk]] = []
    current: list[TrainingChunk] = []
    for pc, chunk in zip(pending, chunks):
        if pc.starts_a_new_group() and current:
            seq_groups.append(current)
            current = []
        current.append(chunk)
    if current:
        seq_groups.append(current)

    # Multimodal extras (Phase 1: image only). Walks chunks in order,
    # tracks the per-modality cumulative encoder-row offset across the
    # round, and populates ``chunk.meta.extra["mm_*"]`` for each chunk
    # that has any vision-placeholder positions. No-op for text-only
    # rounds (sequences with empty / missing ``modality_inputs``).
    _populate_mm_chunk_extras(pending, chunks, device=device)

    total_tokens = sum(c.meta.total_q for c in chunks)
    return PreparedRound(
        chunks=chunks, seq_groups=seq_groups, total_tokens=total_tokens
    )


def _compute_mrope_position_ids(seq) -> torch.Tensor | None:
    """Compute per-token 3-D MRoPE positions for one multimodal Sequence.

    Mirrors HF ``Qwen3VLForConditionalGeneration.get_rope_index`` (in
    ``transformers/models/qwen3_vl/modeling_qwen3_vl.py``). For each
    token in ``seq.tokens``:

    * **Text token** (NOT a placeholder for any image): position is
      ``(p, p, p)`` where ``p`` is the global LLM position counter.
      The counter advances by 1 per text token.
    * **Image-placeholder token**: position is
      ``(t_pos, t_pos + h_idx, t_pos + w_idx)`` where ``(h_idx, w_idx)``
      is the patch's coordinate in the image's POST-merge grid and
      ``t_pos`` is ``current_pos`` at the start of the image block.
      The counter advances by ``max(merged_h, merged_w)`` after the
      whole image block (not by the number of placeholders, which is
      typically much larger).

    Returns ``None`` if ``seq.modality_inputs`` is empty / missing (the
    seq is text-only; caller leaves chunk.meta.seq_positions at K=1).

    Returns ``(N, 3) int32`` otherwise where ``N == len(seq.tokens)``.

    Notes
    -----
    * Phase 1 supports only images. Audio / video are Phase 2/3 -- the
      walk over modality_inputs only consumes "image" entries.
    * The "post-merge" grid dimensions are derived from
      ``ImageInputCPU.placeholder_positions.numel()``: the number of
      placeholder rows IS exactly the image's post-merge token count.
      We pair them with the image's ``grid_thw`` (after dividing H/W by
      ``spatial_merge_size``) to spread positions across the grid.
    * The ``spatial_merge_size`` is NOT carried on the
      ``ImageInputCPU`` -- we derive ``merged_h`` and ``merged_w`` from
      ``sqrt(n_placeholders / T)`` when ``grid_thw[1] == grid_thw[2]``;
      otherwise we fall back to a per-image attribute the data
      adapter should set: ``ImageInputCPU.merged_grid_thw``. Phase 1
      pilot images are Qwen-VL ``patch_size=16`` square images so the
      sqrt branch is the common path.
    """
    mi = getattr(seq, "modality_inputs", None) or {}
    if not mi:
        return None
    images = mi.get("image", []) or []
    if not images:
        return None

    N = int(seq.tokens.numel())
    pos = torch.zeros((N, 3), dtype=torch.int32)

    # Sort images by the *first* placeholder position so we can walk
    # ``input_ids`` once and assign image blocks in order. Each image's
    # block in input_ids is contiguous (consecutive placeholders for
    # the same image's post-merge tokens).
    images_sorted = sorted(images, key=lambda it: int(it.placeholder_positions.min().item()))

    cursor = 0          # next position in input_ids to assign
    current_pos = 0     # the running LLM position counter

    for img in images_sorted:
        pp = img.placeholder_positions.to(torch.int64)
        n_placeholders = int(pp.numel())
        if n_placeholders == 0:
            continue
        img_start = int(pp.min().item())
        img_end_exclusive = int(pp.max().item()) + 1
        if img_end_exclusive - img_start != n_placeholders:
            raise ValueError(
                f"image's placeholder_positions are not contiguous "
                f"[{img_start}, {img_end_exclusive}) but has "
                f"{n_placeholders} entries -- chunk-prep MRoPE assumes "
                "each image's block is a single contiguous slice of input_ids."
            )

        # Fill text positions from ``cursor`` up to ``img_start``.
        for k in range(cursor, img_start):
            pos[k, 0] = current_pos
            pos[k, 1] = current_pos
            pos[k, 2] = current_pos
            current_pos += 1

        # Derive merged-grid dims from grid_thw + n_placeholders.
        # grid_thw stores PRE-merge (T, H, W). Post-merge token count
        # is T * (H // merge) * (W // merge) = n_placeholders.
        T = int(img.grid_thw[0].item())
        H_pre = int(img.grid_thw[1].item())
        W_pre = int(img.grid_thw[2].item())
        per_frame_post = n_placeholders // max(T, 1)
        # We need merged_h, merged_w. Try the square-image fast-path
        # (Qwen-VL pilot images are typically square), then fall back
        # to factoring from (H_pre / W_pre) ratio if rectangular.
        if H_pre == W_pre:
            import math
            side = int(round(math.sqrt(per_frame_post)))
            if side * side != per_frame_post:
                raise ValueError(
                    f"image is square (H=W={H_pre}) but per-frame post-merge "
                    f"count {per_frame_post} is not a perfect square; cannot "
                    "derive merged grid."
                )
            merged_h = merged_w = side
        else:
            # H_pre / merge : W_pre / merge with the same merge.
            # Per-frame count = merged_h * merged_w; ratio constrained
            # by H_pre : W_pre, so merge = gcd-like derivation:
            # merged_h = H_pre // merge, merged_w = W_pre // merge.
            # Solve: (H_pre * W_pre) / merge**2 == per_frame_post.
            import math
            denom = H_pre * W_pre
            if denom % per_frame_post != 0:
                raise ValueError(
                    f"rectangular image grid ({H_pre} x {W_pre}) does not "
                    f"divide evenly into per-frame post-merge count "
                    f"{per_frame_post}; cannot derive merged grid."
                )
            merge_sq = denom // per_frame_post
            merge = int(round(math.sqrt(merge_sq)))
            if merge * merge != merge_sq:
                raise ValueError(
                    f"derived merge**2 = {merge_sq} is not a perfect square "
                    f"for image grid ({H_pre} x {W_pre}) and per-frame post-"
                    f"merge count {per_frame_post}."
                )
            merged_h = H_pre // merge
            merged_w = W_pre // merge

        # Generate 3-D positions per HF ``get_vision_position_ids``:
        # post-merge tokens iterate t outer, h middle, w inner.
        # All share ``t_pos = current_pos`` (image temporal axis is a
        # single value at start_position).
        t_pos_start = current_pos
        ph_cursor = 0
        for t_frame in range(T):
            for h_idx in range(merged_h):
                for w_idx in range(merged_w):
                    abs_pos = img_start + ph_cursor
                    pos[abs_pos, 0] = t_pos_start
                    pos[abs_pos, 1] = t_pos_start + h_idx
                    pos[abs_pos, 2] = t_pos_start + w_idx
                    ph_cursor += 1
        # Advance the global counter by max(merged_h, merged_w).
        current_pos = t_pos_start + max(merged_h, merged_w)
        cursor = img_end_exclusive

    # Trailing text tokens.
    for k in range(cursor, N):
        pos[k, 0] = current_pos
        pos[k, 1] = current_pos
        pos[k, 2] = current_pos
        current_pos += 1
    return pos


def _populate_mm_chunk_extras(
    pending: list[_PendingChunk],
    chunks: list[TrainingChunk],
    *,
    device: torch.device | str,
) -> None:
    """Walk packed chunks in order; populate per-modality placeholder
    positions and round-level image-row assignments on each chunk's
    ``meta.extra``.

    Contract (consumed by :mod:`flextrain.nn.splices.concat`):

    * ``extra["mm_placeholder_positions"]["image"][encoder_id]`` --
      ``(N_in_chunk,) int64`` chunk-local token indices that the
      encoder output should be scattered onto.
    * ``extra["mm_image_assignment"]["image"][encoder_id]`` --
      ``(N_in_chunk,) int64`` index into the round's flat encoder
      output for each placeholder.

    The round's flat encoder output is the concatenation of every
    image's post-merge tokens in the order they appear across
    sequences in the round; we track ``image_row_offset_per_modality``
    as we walk the chunks. ``encoder_id`` is always 0 in Phase 1
    (single image encoder).

    For text-only rounds (no sequence carries ``modality_inputs``),
    this is a fast O(N_chunks) walk that detects emptiness early and
    writes nothing to ``extra`` -- the existing tests stay
    bit-identical.
    """
    # Fast bail: if no sequence in any chunk has modality_inputs, do
    # nothing. This keeps the text-only path byte-identical.
    has_any = False
    for chunk in chunks:
        for ref in chunk.seqs:
            mi = getattr(ref.seq, "modality_inputs", None)
            if mi:
                has_any = True
                break
        if has_any:
            break
    if not has_any:
        return

    # Per-modality round-level row offset into the encoder output.
    # Indexed by modality name; Phase 1 only uses "image".
    row_offset_per_modality: dict[str, int] = {}
    # Track which sequence-image pairs have already had their row
    # offsets assigned this round (a sequence's image may straddle
    # multiple chunks via large-seq chunking; the row offset is
    # consistent across chunks because it's tied to the seq+image
    # identity, not the chunk).
    image_row_by_id: dict[tuple[int, int, str], int] = {}

    # Per-sequence MRoPE 3-D positions, computed lazily on first
    # encounter of a multimodal sequence (a sequence may contribute
    # to multiple chunks via large-seq chunking, so we cache per id).
    mrope_pos_by_seq: dict[int, torch.Tensor] = {}

    for chunk in chunks:
        # Per-(modality, encoder_id) lists of (chunk_local_pos, image_row)
        # pairs we'll accumulate while walking this chunk's seqs.
        positions_per_enc: dict[tuple[str, int], list[int]] = {}
        assignment_per_enc: dict[tuple[str, int], list[int]] = {}

        for ref in chunk.seqs:
            mi = getattr(ref.seq, "modality_inputs", None) or {}
            if not mi:
                continue
            seq_start_in_chunk = ref.chunk_range[0]
            seq_lo_in_seq = ref.seq_range[0]
            seq_hi_in_seq = ref.seq_range[1]
            seq_id_key = id(ref.seq)
            for modality, items in mi.items():
                if not items:
                    continue
                # Phase 1: single encoder per modality.
                encoder_id = 0
                key = (modality, encoder_id)
                positions_per_enc.setdefault(key, [])
                assignment_per_enc.setdefault(key, [])
                for img_idx_in_seq, item in enumerate(items):
                    pp = getattr(item, "placeholder_positions", None)
                    if pp is None or pp.numel() == 0:
                        continue
                    pp_host = pp.to(torch.int64).tolist()
                    # Filter to placeholder positions that fall inside
                    # this chunk's [seq_lo_in_seq, seq_hi_in_seq) window
                    # of the underlying sequence, and translate to
                    # chunk-local indices via seq_start_in_chunk.
                    local_positions: list[int] = []
                    intra_image_offsets: list[int] = []
                    for intra_idx, abs_pos in enumerate(pp_host):
                        if seq_lo_in_seq <= abs_pos < seq_hi_in_seq:
                            local_positions.append(
                                seq_start_in_chunk + (abs_pos - seq_lo_in_seq)
                            )
                            intra_image_offsets.append(intra_idx)
                    if not local_positions:
                        continue
                    # Assign a round-level row offset for this image
                    # once (consistent across chunks that share it).
                    img_id_key = (seq_id_key, img_idx_in_seq, modality)
                    if img_id_key not in image_row_by_id:
                        image_row_by_id[img_id_key] = row_offset_per_modality.get(
                            modality, 0
                        )
                        row_offset_per_modality[modality] = (
                            image_row_by_id[img_id_key] + len(pp_host)
                        )
                    base = image_row_by_id[img_id_key]
                    positions_per_enc[key].extend(local_positions)
                    assignment_per_enc[key].extend(
                        base + off for off in intra_image_offsets
                    )

        if not positions_per_enc:
            continue
        # Materialize per-encoder int64 device tensors and stash in
        # ``chunk.meta.extra``. Use a fresh dict for ``extra`` to avoid
        # aliasing with other code that might mutate it.
        extra = dict(chunk.meta.extra)
        pos_map: dict[str, dict[int, torch.Tensor]] = {}
        asn_map: dict[str, dict[int, torch.Tensor]] = {}
        for (modality, encoder_id), positions in positions_per_enc.items():
            pos_map.setdefault(modality, {})[encoder_id] = torch.tensor(
                positions, dtype=torch.int64, device=device,
            )
            asn_map.setdefault(modality, {})[encoder_id] = torch.tensor(
                assignment_per_enc[(modality, encoder_id)],
                dtype=torch.int64,
                device=device,
            )
        extra["mm_placeholder_positions"] = pos_map
        extra["mm_image_assignment"] = asn_map

        # ---- Build 3-D MRoPE seq_positions for this chunk ----
        # For each contributing seq, slice its per-seq 3-D positions
        # at ``[seq_range[0]:seq_range[1])`` (seq-local), concat in
        # chunk-local order. Text-only contributing seqs (no
        # modality_inputs) get degenerate (p, p, p) computed on the
        # fly from their seq_positions slice. The final tensor is
        # ``(total_q, 3) int32`` on the chunk's device.
        chunk_3d_parts: list[torch.Tensor] = []
        for ref in chunk.seqs:
            seq = ref.seq
            seq_lo, seq_hi = ref.seq_range
            seq_id_key = id(seq)
            if getattr(seq, "modality_inputs", None):
                if seq_id_key not in mrope_pos_by_seq:
                    seq_pos = _compute_mrope_position_ids(seq)
                    # May still be None if modality_inputs map is empty
                    # for every modality; treat as text-only fallback.
                    if seq_pos is None:
                        seq_len = int(seq.tokens.numel())
                        seq_pos = torch.arange(
                            seq_len, dtype=torch.int32,
                        ).reshape(-1, 1).expand(seq_len, 3).contiguous()
                    mrope_pos_by_seq[seq_id_key] = seq_pos
                seq_pos = mrope_pos_by_seq[seq_id_key]
            else:
                # Text-only contributing seq under a multimodal chunk:
                # build degenerate (p, p, p) positions for this slice.
                # Use the same arange the engine would have produced.
                seq_len = int(seq.tokens.numel())
                seq_pos = (
                    torch.arange(seq_len, dtype=torch.int32)
                    .reshape(-1, 1)
                    .expand(seq_len, 3)
                    .contiguous()
                )
            # Slice and append.
            chunk_3d_parts.append(seq_pos[seq_lo:seq_hi].contiguous())
        new_seq_positions = torch.cat(chunk_3d_parts, dim=0).to(
            dtype=torch.int32, device=device,
        )
        if new_seq_positions.shape[0] != chunk.meta.total_q:
            raise RuntimeError(
                f"MRoPE chunk position concat produced "
                f"{new_seq_positions.shape[0]} rows but chunk.total_q = "
                f"{chunk.meta.total_q}; seq_range vs chunk_range mismatch."
            )

        # ChunkMeta is a dataclass with mutable ``extra`` Mapping;
        # replace via dataclasses.replace to keep frozen-ness happy.
        from dataclasses import replace as _replace
        chunk.meta = _replace(
            chunk.meta, extra=extra, seq_positions=new_seq_positions,
        )


__all__ = [
    "ChunkPolicy",
    "ChunkSeqRef",
    "PreparedRound",
    "TrainingChunk",
    "prepare_training_chunks",
    "split_sequences",
]
