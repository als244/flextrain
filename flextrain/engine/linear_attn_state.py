"""Cross-chunk recurrent-state buffers for linear-attention layers.

Background
----------
Linear-attention (Gated DeltaNet) is a recurrent operator: the output
at token t depends on a per-(head_v, head_k, head_v_dim) state that
accumulates over all earlier tokens of the same sequence. FLA's
``chunk_gated_delta_rule_fwd`` computes this state internally over the
tokens it sees in one call.

When a sequence fits in a single FT chunk, all its tokens are passed
to one FLA call and the recurrent state is correct end-to-end. But
when one sequence spans multiple chunks (because its length exceeds
``max_chunk_size``), each chunk's FLA call sees ``initial_state=None``
and produces a state that effectively starts fresh — wrong for any
chunk after the first.

This module owns the GPU buffers that carry recurrent state between
chunks of the same sequence so cross-chunk linear-attn is bit-correct.

Two banks
---------
:class:`LinearAttnStateBank` -- forward path. Saves the FINAL state
produced by each chunk's fwd, indexed by ``(layer_id, seq_id,
chunk_in_seq_index)``. The next chunk of the same seq's fwd reads
slot ``chunk_in_seq_index - 1`` as its ``initial_state``.

:class:`LinearAttnDStateBank` -- backward path. Saves the ``dh0``
returned by chunk N+1's bwd; chunk N's bwd reads it back as ``dht``
(the gradient w.r.t. the final state ``h[N]`` that chunk N produced).

Lifecycle
---------
Both banks are allocated lazily at round-prepare time once the engine
knows which sequences span multiple chunks (``prior_seq_lens`` > 0
in any chunk's ``ChunkMeta``). Cleared at the start of the next round.

Memory cost
-----------
For each (layer, multi-chunk-seq) pair: one ``(HV, K, V) fp32`` tensor.
Qwen3.5-MoE-A3B (HV=32, K=128, V=128, ~30 linear-attn layers): 2 MB
per (layer, seq), so 8 multi-chunk seqs in flight at once = ~512 MB.
Smaller models proportionally less. Single-chunk seqs cost zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch


# Type alias for the (HV, K, V) fp32 state tensor a single sequence's
# linear-attn block produces at its tokens' end.
LinAttnStateShape = tuple[int, int, int]


@dataclass
class LinearAttnStateBank:
    """Forward-path state buffer.

    A single bank instance lives for the duration of one round (one
    fwd + bwd pass). Allocated by the engine when the round contains
    any multi-chunk sequence.

    Shape per slot: ``(num_v_heads, head_k_dim, head_v_dim)`` fp32.

    Indexing scheme
    ---------------
    Key ``(layer_id, seq_id, chunk_in_seq_idx)``:

    * ``layer_id``         -- the linear-attn layer's logical id
                              (engine ``Layer.layer_id``).
    * ``seq_id``           -- the originating ``Sequence.seq_id``.
    * ``chunk_in_seq_idx`` -- 0-based index of the chunk within this
                              seq's chunk run (chunk 0 is the first
                              chunk that contains this seq's tokens).

    Forward of chunk N of a multi-chunk seq:

    * Reads slot ``(layer_id, seq_id, N - 1)`` as ``initial_state``,
      OR uses zeros when ``N == 0``.
    * Writes slot ``(layer_id, seq_id, N)`` with FLA's
      ``output_final_state``.

    Single-chunk seqs do not interact with the bank — fwd passes
    ``initial_state=None, output_final_state=False`` and the kernel
    runs the same as today. The bank is therefore zero-cost (no
    allocations, no kernel work) when no seq spans chunks.
    """

    state_shape: LinAttnStateShape
    """``(num_v_heads, head_k_dim, head_v_dim)`` -- per-(layer, seq, chunk)
    state shape. Pinned to fp32 for accumulation stability across
    chunk boundaries (FLA returns its internal h state in fp32)."""

    device: torch.device
    """Where the slots live."""

    dtype: torch.dtype = torch.float32
    """fp32 keeps cross-chunk drift below bf16 noise. The caller gets
    a fp32 view; FLA accepts fp32 ``initial_state`` directly."""

    _slots: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)
    """Backing storage. Keys: ``(layer_id, seq_id, chunk_in_seq_idx)``."""

    def has(self, layer_id: int, seq_id: int, chunk_in_seq_idx: int) -> bool:
        return (layer_id, seq_id, chunk_in_seq_idx) in self._slots

    def get(
        self,
        layer_id: int,
        seq_id: int,
        chunk_in_seq_idx: int,
    ) -> torch.Tensor | None:
        """Return the saved state, or ``None`` if no such slot exists.

        Returning ``None`` rather than zeros lets the caller decide
        whether to allocate a zero tensor (the "fresh state" case for
        chunk 0 of a seq) or to error (for any other chunk_in_seq_idx
        that ought to have been written by an earlier fwd call)."""
        return self._slots.get((layer_id, seq_id, chunk_in_seq_idx))

    def put(
        self,
        layer_id: int,
        seq_id: int,
        chunk_in_seq_idx: int,
        state: torch.Tensor,
    ) -> None:
        """Save a per-seq final state. Caller passes the (HV, K, V) view
        FLA returned (``output_final_state=True``); we ``.detach().clone()``
        to take ownership and to ensure later backward pass sees a
        stable buffer.
        """
        if state.shape != self.state_shape:
            raise ValueError(
                f"LinearAttnStateBank.put: expected shape {self.state_shape}, "
                f"got {tuple(state.shape)}"
            )
        # ``.detach()`` to detach from any autograd graph (we run our
        # own bwd manually). ``.contiguous()`` so FLA can read it later.
        self._slots[(layer_id, seq_id, chunk_in_seq_idx)] = (
            state.detach().to(dtype=self.dtype, device=self.device).contiguous()
        )

    def clear(self) -> None:
        """Drop all slots. Called at end-of-round."""
        self._slots.clear()

    def num_slots(self) -> int:
        return len(self._slots)


@dataclass
class LinearAttnDStateBank:
    """Backward-path gradient-of-state buffer.

    Mirror of :class:`LinearAttnStateBank` but for the chain
    ``dht[N] = dh0[N+1]`` that bwd needs to thread cross-chunk.

    Bwd of chunk N of a multi-chunk seq:

    * Reads ``dht`` from slot ``(layer_id, seq_id, N)`` (the gradient
      w.r.t. the FINAL state ``h[N]`` produced by chunk N's fwd), OR
      uses None / zeros for the LAST chunk of the seq (no chunk after).
    * After bwd, writes its returned ``dh0`` into slot
      ``(layer_id, seq_id, N - 1)`` so chunk N-1's bwd can consume it
      as its ``dht``.
    """

    state_shape: LinAttnStateShape
    device: torch.device
    dtype: torch.dtype = torch.float32

    _slots: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)

    def has(self, layer_id: int, seq_id: int, chunk_in_seq_idx: int) -> bool:
        return (layer_id, seq_id, chunk_in_seq_idx) in self._slots

    def get(
        self,
        layer_id: int,
        seq_id: int,
        chunk_in_seq_idx: int,
    ) -> torch.Tensor | None:
        return self._slots.get((layer_id, seq_id, chunk_in_seq_idx))

    def put(
        self,
        layer_id: int,
        seq_id: int,
        chunk_in_seq_idx: int,
        dstate: torch.Tensor,
    ) -> None:
        if dstate.shape != self.state_shape:
            raise ValueError(
                f"LinearAttnDStateBank.put: expected shape {self.state_shape}, "
                f"got {tuple(dstate.shape)}"
            )
        self._slots[(layer_id, seq_id, chunk_in_seq_idx)] = (
            dstate.detach().to(dtype=self.dtype, device=self.device).contiguous()
        )

    def clear(self) -> None:
        self._slots.clear()

    def num_slots(self) -> int:
        return len(self._slots)


# ---------------------------------------------------------------------------
# Round-prepare helpers: compute the per-chunk per-packed-seq metadata
# the engine needs to drive the banks.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiChunkSeqInfo:
    """Per-(chunk, packed_seq_index) metadata: where this packed seq
    sits in its sequence's chunk run.

    Computed once per round at prepare time, then read by the engine
    inside fwd/bwd loops.

    Fields:

    * ``seq_id``: stable Sequence.seq_id used as bank key.
    * ``chunk_in_seq_idx``: 0 if this packed-seq starts the sequence
      (``prior_seq_lens_host[i] == 0``), else > 0.
    * ``has_more_chunks``: True iff a later chunk continues this seq —
      i.e. the FLA fwd should request ``output_final_state``.
    * ``has_prior_chunks``: True iff an earlier chunk already produced
      a state that this fwd should consume as ``initial_state``.
      Equivalent to ``chunk_in_seq_idx > 0``.

    The pair (``has_prior_chunks``, ``has_more_chunks``) tells the
    fwd how to drive FLA:

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
    """Per-round data that fwd/bwd consume to drive the state banks.

    Built once by :func:`build_linear_attn_round_plan` at the start
    of a round. The engine threads it into ``_forward_pass`` /
    ``_backward_pass`` so each chunk's fwd/bwd can decide whether to
    interact with the banks and where to look up state.

    Layout: ``per_chunk[chunk_id]`` is a list parallel to
    ``ChunkMeta.seq_lens_host`` (one entry per packed-seq inside the
    chunk). Each entry is a :class:`MultiChunkSeqInfo`.

    A round is "trivial" (no cross-chunk state needed) iff every
    entry has both ``has_prior_chunks=False`` and
    ``has_more_chunks=False`` — i.e. nothing actually spans chunks.
    The engine can skip bank allocation for trivial rounds.
    """

    per_chunk: list[list[MultiChunkSeqInfo]]

    def is_trivial(self) -> bool:
        """True when no packed-seq in any chunk spans chunks. The
        engine can avoid allocating the banks in that case."""
        for chunk_infos in self.per_chunk:
            for info in chunk_infos:
                if info.has_more_chunks or info.has_prior_chunks:
                    return False
        return True

    def multi_chunk_seq_ids(self) -> set[int]:
        """Stable set of all seq_ids that span chunks. Useful for
        sizing the bank up-front."""
        out: set[int] = set()
        for chunk_infos in self.per_chunk:
            for info in chunk_infos:
                if info.has_more_chunks or info.has_prior_chunks:
                    out.add(info.seq_id)
        return out


def build_linear_attn_round_plan(prepared) -> LinearAttnRoundPlan:
    """Walk a :class:`PreparedRound` and build the cross-chunk plan.

    Per packed-seq in each chunk:

    * ``chunk_in_seq_idx`` is the count of EARLIER chunks of the same
      ``seq_id`` already seen in the round walk (0 for the first
      chunk where this seq appears).
    * ``has_prior_chunks = (chunk_in_seq_idx > 0)``.
    * ``has_more_chunks`` is computed by a single forward sweep over
      all chunks: a packed-seq has more chunks iff some later chunk
      contains the same ``seq_id``.

    No GPU work, no allocations. Pure Python, runs in O(total_packed_seqs).
    """
    chunks = prepared.chunks
    n_chunks = len(chunks)

    # First pass: per (chunk_id, packed_seq_idx_in_chunk), record seq_id.
    # We rely on TrainingChunk.seqs[i].seq.seq_id — same ordering as
    # ChunkMeta.seq_lens_host.
    chunk_seq_ids: list[list[int]] = []
    for c in chunks:
        chunk_seq_ids.append([ref.seq.seq_id for ref in c.seqs])

    # Per seq_id, list of (chunk_id, packed_idx) where it appears.
    seq_appearances: dict[int, list[tuple[int, int]]] = {}
    for c_id, sids in enumerate(chunk_seq_ids):
        for p_idx, sid in enumerate(sids):
            seq_appearances.setdefault(sid, []).append((c_id, p_idx))

    # Build per_chunk infos in the same order as ChunkMeta.seq_lens_host.
    per_chunk: list[list[MultiChunkSeqInfo]] = [[] for _ in range(n_chunks)]
    for sid, appearances in seq_appearances.items():
        n = len(appearances)
        for i, (c_id, p_idx) in enumerate(appearances):
            info = MultiChunkSeqInfo(
                seq_id=sid,
                chunk_in_seq_idx=i,
                has_prior_chunks=(i > 0),
                has_more_chunks=(i < n - 1),
            )
            # Append in packed-seq order within chunk. Since we walked
            # chunks in order and per-chunk packed index is well-defined,
            # we just need to ensure we sort by p_idx after the loop.
            per_chunk[c_id].append((p_idx, info))

    # Sort each chunk's list by packed_idx so it parallels seq_lens_host.
    sorted_per_chunk: list[list[MultiChunkSeqInfo]] = []
    for c_id, items in enumerate(per_chunk):
        items.sort(key=lambda t: t[0])
        sorted_per_chunk.append([info for _p_idx, info in items])

    return LinearAttnRoundPlan(per_chunk=sorted_per_chunk)


__all__ = [
    "LinAttnStateShape",
    "LinearAttnStateBank",
    "LinearAttnDStateBank",
    "MultiChunkSeqInfo",
    "LinearAttnRoundPlan",
    "build_linear_attn_round_plan",
]
