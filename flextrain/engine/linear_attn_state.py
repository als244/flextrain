"""Cross-chunk linear-attention round plan.

Builds the per-(chunk, packed-seq) metadata the engine needs to drive
linear-attention's cross-chunk state plumbing during fwd and bwd.

The state itself lives in two engine-managed structures, NOT here:

* :class:`flextrain.engine.buffers.LinAttnStateWindow` — global per-
  layer buffer for FLA's ``initial_state`` (fwd) and ``dh0`` chain
  accumulator (bwd). Allocated once at engine init.

* The per-(layer, chunk) ``lin_final_state`` activation slot field —
  saved during fwd, prefetched during bwd via the engine's
  ``_refresh_lin_state_window`` helper (mirror of dense's
  ``_refresh_kv_window``).

This module just produces the small Python data structure that tells
the engine and the linear-attn block which packed-seqs in each chunk
participate in cross-chunk state. See
``docs/internal/multi_chunk_seq_handling.md`` for the full design.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiChunkSeqInfo:
    """Per-(chunk, packed_seq_index) metadata.

    Computed once per round at prepare time, then read by the engine
    inside fwd/bwd loops.

    Fields:

    * ``seq_id``: stable Sequence.seq_id used as bookkeeping reference.
    * ``chunk_in_seq_idx``: 0 if this packed-seq starts the sequence
      (``prior_seq_lens_host[i] == 0``), else > 0.
    * ``has_more_chunks``: True iff a later chunk continues this seq —
      i.e. the FLA fwd should request ``output_final_state``.
    * ``has_prior_chunks``: True iff an earlier chunk already produced
      a state that this fwd should consume as ``initial_state``.
      Equivalent to ``chunk_in_seq_idx > 0``.

    The pair (``has_prior_chunks``, ``has_more_chunks``) tells fwd/bwd
    how to drive FLA's recurrent state plumbing:

    * (False, False): single-chunk seq, fall through to today's
      ``initial_state=None, output_final_state=False`` path.
    * (False, True):  first chunk of multi-chunk seq — start fresh,
      save final state.
    * (True, True):   middle chunk — load prior, save final.
    * (True, False):  last chunk — load prior, no save.
    """

    seq_id: int
    chunk_in_seq_idx: int
    has_more_chunks: bool
    has_prior_chunks: bool


@dataclass
class LinearAttnRoundPlan:
    """Per-round metadata for cross-chunk linear-attention.

    Built once by :func:`build_linear_attn_round_plan` at the start
    of a round. The engine threads it into ``_forward_pass`` /
    ``_backward_pass`` so each chunk's fwd/bwd can decide whether to
    interact with the state windows and slot fields.

    Layout: ``per_chunk[chunk_id]`` is a list parallel to
    ``ChunkMeta.seq_lens_host`` (one entry per packed-seq inside the
    chunk).

    A round is "trivial" (no cross-chunk state needed) iff every
    entry has both ``has_prior_chunks=False`` and
    ``has_more_chunks=False`` — i.e. nothing actually spans chunks.
    """

    per_chunk: list[list[MultiChunkSeqInfo]]

    def is_trivial(self) -> bool:
        """True when no packed-seq in any chunk spans chunks."""
        for chunk_infos in self.per_chunk:
            for info in chunk_infos:
                if info.has_more_chunks or info.has_prior_chunks:
                    return False
        return True


def build_linear_attn_round_plan(prepared) -> LinearAttnRoundPlan:
    """Walk a :class:`PreparedRound` and build the cross-chunk plan.

    Per packed-seq in each chunk:

    * ``chunk_in_seq_idx`` is the count of EARLIER chunks of the same
      ``seq_id`` already seen in the round walk (0 for the first
      chunk where this seq appears).
    * ``has_prior_chunks = (chunk_in_seq_idx > 0)``.
    * ``has_more_chunks`` is computed by sweeping all chunks once: a
      packed-seq has more chunks iff some later chunk contains the
      same ``seq_id``.

    No GPU work, no allocations. Pure Python, runs in
    O(total_packed_seqs) time per round.
    """
    chunks = prepared.chunks
    n_chunks = len(chunks)

    # Per (chunk_id, packed_seq_idx_in_chunk), record seq_id. Order
    # matches ChunkMeta.seq_lens_host because TrainingChunk.seqs is
    # populated in pack order by ``prepare_training_chunks``.
    chunk_seq_ids: list[list[int]] = []
    for c in chunks:
        chunk_seq_ids.append([ref.seq.seq_id for ref in c.seqs])

    # Per seq_id, list of (chunk_id, packed_idx) where it appears.
    seq_appearances: dict[int, list[tuple[int, int]]] = {}
    for c_id, sids in enumerate(chunk_seq_ids):
        for p_idx, sid in enumerate(sids):
            seq_appearances.setdefault(sid, []).append((c_id, p_idx))

    # Collect per-chunk infos in (packed_idx, info) form, then sort by
    # packed_idx so the final order matches ``seq_lens_host``.
    per_chunk_pairs: list[list[tuple[int, MultiChunkSeqInfo]]] = [
        [] for _ in range(n_chunks)
    ]
    for sid, appearances in seq_appearances.items():
        n = len(appearances)
        for i, (c_id, p_idx) in enumerate(appearances):
            info = MultiChunkSeqInfo(
                seq_id=sid,
                chunk_in_seq_idx=i,
                has_prior_chunks=(i > 0),
                has_more_chunks=(i < n - 1),
            )
            per_chunk_pairs[c_id].append((p_idx, info))

    sorted_per_chunk: list[list[MultiChunkSeqInfo]] = []
    for items in per_chunk_pairs:
        items.sort(key=lambda t: t[0])
        sorted_per_chunk.append([info for _p_idx, info in items])

    return LinearAttnRoundPlan(per_chunk=sorted_per_chunk)


__all__ = [
    "MultiChunkSeqInfo",
    "LinearAttnRoundPlan",
    "build_linear_attn_round_plan",
]
