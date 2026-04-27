"""Tests for :mod:`flextrain.engine.schedule`.

Exercises:
1. ``split_sequences`` round partitioning (token + chunk limits).
2. ``prepare_training_chunks`` packing behavior for mixed-size inputs,
   plus the observable invariants:
   - all input tokens appear in output chunks
   - each chunk is at most ``max_chunk_size``
   - seq_groups partition chunks into contiguous runs
   - each chunk's meta matches its token layout (total_q equals the
     sum of its ``lens``)
3. Non-causal policy (:class:`ChunkPolicy.NON_CAUSAL`): oversize
   sequences are rejected at both ``split_sequences`` and
   ``prepare_training_chunks`` entry points.
4. Equivalence with orig's packing for mixed causal inputs.

Uses CPU for these tests -- no kernels involved, just packing logic.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ORIG = os.path.join(ROOT, "orig")
if ORIG not in sys.path:
    sys.path.insert(0, ORIG)

from flextrain.engine.schedule import (  # noqa: E402
    ChunkPolicy,
    prepare_training_chunks,
    split_sequences,
)


# ---------------------------------------------------------------------------
# Minimal Sequence-like stand-in. Matches the duck-type shape the scheduler
# actually touches (``len``, ``.tokens``, ``.targets``).
# ---------------------------------------------------------------------------


class FakeSeq:
    def __init__(self, seq_id: int, n: int, start_val: int = 0) -> None:
        self.seq_id = seq_id
        self.tokens = torch.arange(
            start_val, start_val + n, dtype=torch.int64
        )
        self.targets = torch.arange(
            start_val + 1, start_val + n + 1, dtype=torch.int64
        )
        self.per_token_loss = torch.zeros(n, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.tokens)


# ---------------------------------------------------------------------------
# split_sequences
# ---------------------------------------------------------------------------


def test_split_sequences_respects_token_budget() -> None:
    seqs = [FakeSeq(i, 100) for i in range(10)]  # 1000 tokens total
    rounds, total = split_sequences(
        seqs,
        target_round_tokens=250,
        max_total_round_tokens=500,
        max_chunk_size=64,
        max_training_chunks=100,
    )
    assert total == 1000
    # target=250, so each round holds <=250 tokens of small seqs.
    # With seqs of len 100, rounds will have 2-3 each.
    for r in rounds:
        assert sum(len(s) for s in r) <= 300  # 2 seqs of 100 is 200; 3 is 300
    # All sequences accounted for.
    all_in_rounds = [s for r in rounds for s in r]
    assert [s.seq_id for s in all_in_rounds] == list(range(10))


def test_split_sequences_rejects_too_long() -> None:
    seqs = [FakeSeq(0, 2000)]
    try:
        split_sequences(
            seqs,
            target_round_tokens=500,
            max_total_round_tokens=1000,
            max_chunk_size=256,
            max_training_chunks=100,
        )
    except ValueError as e:
        assert "too long" in str(e).lower()
        return
    raise AssertionError("Expected ValueError for oversized sequence")


def test_split_sequences_noncausal_rejects_oversize() -> None:
    """Non-causal mode rejects any sequence longer than max_chunk_size."""
    seqs = [FakeSeq(0, 512)]
    # Room in max_total_round_tokens, but the chunk_size constraint applies.
    try:
        split_sequences(
            seqs,
            target_round_tokens=2048,
            max_total_round_tokens=2048,
            max_chunk_size=256,
            max_training_chunks=100,
            policy=ChunkPolicy.NON_CAUSAL,
        )
    except ValueError as e:
        assert "non-causal" in str(e).lower()
        return
    raise AssertionError("Expected ValueError for non-causal oversize")


def test_split_sequences_noncausal_accepts_fitting() -> None:
    """Non-causal mode accepts sequences up to max_chunk_size."""
    seqs = [FakeSeq(i, 200) for i in range(5)]
    rounds, total = split_sequences(
        seqs,
        target_round_tokens=500,
        max_total_round_tokens=500,
        max_chunk_size=256,
        max_training_chunks=100,
        policy=ChunkPolicy.NON_CAUSAL,
    )
    assert total == 5 * 200
    assert sum(len(r) for r in rounds) == 5


# ---------------------------------------------------------------------------
# prepare_training_chunks (causal)
# ---------------------------------------------------------------------------


def _chunk_invariants(prepared, *, expected_total_tokens: int,
                     max_chunk_size: int) -> None:
    """Shared invariant checks for a causal prepared round."""
    # total_tokens matches sum of chunk total_q.
    assert prepared.total_tokens == expected_total_tokens
    assert sum(c.meta.total_q for c in prepared.chunks) == expected_total_tokens
    # Every chunk at most max_chunk_size.
    for c in prepared.chunks:
        assert c.meta.total_q <= max_chunk_size
        # token_ids shape agrees.
        assert c.token_ids.shape == (c.meta.total_q,)
        assert c.label_ids.shape == (c.meta.total_q,)
        # sum of per-seq lens = total_q
        assert sum(c.meta.seq_lens_host) == c.meta.total_q
    # seq_groups partition chunks (each chunk appears exactly once,
    # in-order).
    flat = [c for g in prepared.seq_groups for c in g]
    assert [c.id for c in flat] == [c.id for c in prepared.chunks]


def test_pack_small_sequences_greedy_fit() -> None:
    """Small sequences pack into shared chunks, flushing on overflow."""
    seqs = [FakeSeq(i, 100) for i in range(5)]  # 500 tokens total
    prepared = prepare_training_chunks(
        seqs, max_chunk_size=256, device="cpu"
    )
    _chunk_invariants(prepared, expected_total_tokens=500, max_chunk_size=256)
    # 256-byte chunks fit floor(256/100) = 2 small seqs each -> 3 chunks total
    # (2 + 2 + 1 = 5 seqs).
    assert len(prepared.chunks) == 3
    # All chunks start a new group (small seqs always start at offset 0).
    assert len(prepared.seq_groups) == 3


def test_pack_large_sequence_across_chunks() -> None:
    """A sequence bigger than max_chunk_size takes dedicated chunks."""
    seqs = [FakeSeq(0, 700)]  # 700 tokens, max_chunk_size=256
    prepared = prepare_training_chunks(
        seqs, max_chunk_size=256, device="cpu"
    )
    _chunk_invariants(prepared, expected_total_tokens=700, max_chunk_size=256)
    # 700 / 256 = 2 full + 1 partial -> 3 chunks.
    assert len(prepared.chunks) == 3
    sizes = [c.meta.total_q for c in prepared.chunks]
    assert sizes == [256, 256, 700 - 512]
    # All continuation chunks belong to ONE group (the big seq).
    assert len(prepared.seq_groups) == 1
    # Verify prior_seq_lens_host on continuation chunks is non-zero.
    assert prepared.chunks[0].meta.prior_seq_lens_host == [0]
    assert prepared.chunks[1].meta.prior_seq_lens_host == [256]
    assert prepared.chunks[2].meta.prior_seq_lens_host == [512]


def test_pack_mixed_small_and_large() -> None:
    """Small seqs before a big seq get flushed; big seq's chunks don't
    try to share with small."""
    seqs = [
        FakeSeq(0, 50),
        FakeSeq(1, 50),
        FakeSeq(2, 600),  # big
        FakeSeq(3, 80),
        FakeSeq(4, 80),
    ]
    prepared = prepare_training_chunks(
        seqs, max_chunk_size=256, device="cpu"
    )
    total = 50 + 50 + 600 + 80 + 80
    _chunk_invariants(prepared, expected_total_tokens=total, max_chunk_size=256)
    # Chunks: [50+50], [256 of big], [256 of big], [600-512=88 of big], [80+80]
    sizes = [c.meta.total_q for c in prepared.chunks]
    assert sizes == [100, 256, 256, 88, 160]
    # Groups:
    # - chunk 0: small seqs (first prior=0) -> group 1
    # - chunks 1,2,3: big seq start + 2 continuations -> group 2
    # - chunk 4: small seqs (first prior=0) -> group 3
    group_sizes = [len(g) for g in prepared.seq_groups]
    assert group_sizes == [1, 3, 1]


def test_token_identity_after_packing() -> None:
    """Concatenating all chunks' token_ids must equal the concatenation
    of the input sequences' tokens (no re-ordering, no drops)."""
    seqs = [FakeSeq(i, n) for i, n in enumerate([50, 50, 600, 80, 80])]
    prepared = prepare_training_chunks(
        seqs, max_chunk_size=256, device="cpu"
    )
    recovered = torch.cat([c.token_ids for c in prepared.chunks])
    expected = torch.cat([s.tokens.long() for s in seqs])
    assert torch.equal(recovered, expected)


# ---------------------------------------------------------------------------
# prepare_training_chunks (non-causal)
# ---------------------------------------------------------------------------


def test_prepare_noncausal_rejects_oversize() -> None:
    seqs = [FakeSeq(0, 300)]
    try:
        prepare_training_chunks(
            seqs,
            max_chunk_size=256,
            device="cpu",
            policy=ChunkPolicy.NON_CAUSAL,
        )
    except ValueError as e:
        assert "non-causal" in str(e).lower()
        return
    raise AssertionError("Expected ValueError for non-causal oversize")


def test_prepare_noncausal_every_chunk_is_a_group() -> None:
    """In non-causal mode, no sequence spans chunks, so every chunk
    starts a fresh group. len(seq_groups) == len(chunks)."""
    # 5 seqs of len 100; max_chunk_size=256 packs 2 per chunk; 3 chunks.
    seqs = [FakeSeq(i, 100) for i in range(5)]
    prepared = prepare_training_chunks(
        seqs,
        max_chunk_size=256,
        device="cpu",
        policy=ChunkPolicy.NON_CAUSAL,
    )
    _chunk_invariants(prepared, expected_total_tokens=500, max_chunk_size=256)
    assert len(prepared.seq_groups) == len(prepared.chunks)


# ---------------------------------------------------------------------------
# Parity with orig on a representative mixed input.
# ---------------------------------------------------------------------------


def _orig_prepare(seqs, max_chunk_size: int):
    """Port of orig/active_model.py::prepare_training_chunks stripped
    down to the packing logic (no ChunkMeta.build). Returns the flat
    list of per-chunk ``lens`` lists and ``prior_lens`` lists so we
    can diff against our implementation shape-for-shape."""
    final_chunks_data = []
    cur_chunk_buf = {
        "lens": [], "pos": [], "prior_lens": [], "prior_offsets": [],
        "current_size": 0,
    }

    def flush():
        if cur_chunk_buf["current_size"] == 0:
            return
        final_chunks_data.append({
            "lens": list(cur_chunk_buf["lens"]),
            "prior_lens": list(cur_chunk_buf["prior_lens"]),
        })
        cur_chunk_buf["lens"].clear()
        cur_chunk_buf["prior_lens"].clear()
        cur_chunk_buf["pos"].clear()
        cur_chunk_buf["prior_offsets"].clear()
        cur_chunk_buf["current_size"] = 0

    for s in seqs:
        s_len = len(s)
        if s_len > max_chunk_size:
            flush()
            cursor = 0
            while cursor < s_len:
                take = min(max_chunk_size, s_len - cursor)
                final_chunks_data.append({
                    "lens": [take],
                    "prior_lens": [cursor],
                })
                cursor += take
        else:
            if cur_chunk_buf["current_size"] + s_len > max_chunk_size:
                flush()
            cur_chunk_buf["lens"].append(s_len)
            cur_chunk_buf["prior_lens"].append(0)
            cur_chunk_buf["current_size"] += s_len
    flush()
    return final_chunks_data


def test_parity_with_orig_packing() -> None:
    """Exercise a variety of size distributions and check that our
    packing (ignoring the ChunkMeta tensors, which require device work)
    produces the same per-chunk ``lens`` and ``prior_lens`` as the
    stripped-down orig implementation."""
    import random

    random.seed(42)

    for trial in range(20):
        num_seqs = random.randint(1, 20)
        seqs = []
        for i in range(num_seqs):
            # Mix small (5-250) and occasional large (300-900).
            if random.random() < 0.25:
                n = random.randint(300, 900)
            else:
                n = random.randint(5, 250)
            seqs.append(FakeSeq(i, n))
        max_chunk_size = random.choice([128, 256, 512])

        ft_prepared = prepare_training_chunks(
            seqs, max_chunk_size=max_chunk_size, device="cpu"
        )
        orig_chunks = _orig_prepare(seqs, max_chunk_size)

        assert len(ft_prepared.chunks) == len(orig_chunks), (
            f"trial {trial}: len(ft)={len(ft_prepared.chunks)} "
            f"len(orig)={len(orig_chunks)}"
        )
        for i, (fc, oc) in enumerate(zip(ft_prepared.chunks, orig_chunks)):
            assert list(fc.meta.seq_lens_host) == oc["lens"], (
                f"trial {trial} chunk {i}: lens mismatch "
                f"ft={list(fc.meta.seq_lens_host)} orig={oc['lens']}"
            )
            assert list(fc.meta.prior_seq_lens_host) == oc["prior_lens"], (
                f"trial {trial} chunk {i}: prior_lens mismatch "
                f"ft={list(fc.meta.prior_seq_lens_host)} "
                f"orig={oc['prior_lens']}"
            )
    print("    parity across 20 random trials: ok")


def _run_all() -> None:
    tests = [
        ("test_split_sequences_respects_token_budget",
         test_split_sequences_respects_token_budget),
        ("test_split_sequences_rejects_too_long",
         test_split_sequences_rejects_too_long),
        ("test_split_sequences_noncausal_rejects_oversize",
         test_split_sequences_noncausal_rejects_oversize),
        ("test_split_sequences_noncausal_accepts_fitting",
         test_split_sequences_noncausal_accepts_fitting),
        ("test_pack_small_sequences_greedy_fit",
         test_pack_small_sequences_greedy_fit),
        ("test_pack_large_sequence_across_chunks",
         test_pack_large_sequence_across_chunks),
        ("test_pack_mixed_small_and_large",
         test_pack_mixed_small_and_large),
        ("test_token_identity_after_packing",
         test_token_identity_after_packing),
        ("test_prepare_noncausal_rejects_oversize",
         test_prepare_noncausal_rejects_oversize),
        ("test_prepare_noncausal_every_chunk_is_a_group",
         test_prepare_noncausal_every_chunk_is_a_group),
        ("test_parity_with_orig_packing",
         test_parity_with_orig_packing),
    ]
    for name, fn in tests:
        print(f"  - {name}")
        fn()
    print(f"  OK ({len(tests)} tests)")


if __name__ == "__main__":
    _run_all()
