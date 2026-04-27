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

    total_tokens = sum(c.meta.total_q for c in chunks)
    return PreparedRound(
        chunks=chunks, seq_groups=seq_groups, total_tokens=total_tokens
    )


__all__ = [
    "ChunkPolicy",
    "ChunkSeqRef",
    "PreparedRound",
    "TrainingChunk",
    "prepare_training_chunks",
    "split_sequences",
]
