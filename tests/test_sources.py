"""Tests for :mod:`flextrain.io.sources` adapters.

Covers:
* :class:`RawTokenSource` — list and dict input shapes
* :class:`SyntheticTokenSource` — deterministic output across two runs
* :class:`ShardTokenSource` — parses orig's FineWeb shard format
* :class:`CustomSchemaTokenSource` — user-supplied extractor
* :class:`HFTokenSource` — import-only smoke (network dependency skipped)

Runs CPU-only; no CUDA required.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.io.sources import (  # noqa: E402
    CustomSchemaTokenSource,
    JsonSFTTokenSource,
    RawTokenSource,
    ShardTokenSource,
    SyntheticTokenSource,
    TokenSource,
)


# ---------------------------------------------------------------------------
# RawTokenSource
# ---------------------------------------------------------------------------


def test_raw_source_list_input() -> None:
    seqs = [torch.arange(0, 64, dtype=torch.int64) for _ in range(5)]
    source = RawTokenSource(seqs)
    assert isinstance(source, TokenSource)
    batch = source.get_sequences(max_token_count=200)
    assert len(batch) >= 3
    assert batch[0].tokens.dtype == torch.int64
    batch2 = source.get_sequences(max_token_count=1000)
    total = sum(len(b) for b in batch) + sum(len(b) for b in batch2)
    assert total == 64 * 5


def test_raw_source_dict_input_with_loss_mask() -> None:
    tokens_list = [torch.arange(i * 10, i * 10 + 32) for i in range(3)]
    loss_mask_list = [
        torch.tensor([True] * 32) if i != 0
        else torch.tensor([False] * 16 + [True] * 16)
        for i in range(3)
    ]
    source = RawTokenSource({
        "tokens": tokens_list,
        "loss_mask": loss_mask_list,
    })
    seqs = source.get_sequences(max_token_count=1000)
    assert len(seqs) == 3
    assert seqs[0].loss_mask is not None
    assert bool(seqs[0].loss_mask[0].item()) is False
    assert bool(seqs[0].loss_mask[16].item()) is True


def test_raw_source_iter() -> None:
    seqs = [torch.arange(0, 16, dtype=torch.int64) for _ in range(3)]
    source = RawTokenSource(seqs)
    got = list(source.iter_sequences())
    assert len(got) == 3
    assert all(len(s) == 16 for s in got)


# ---------------------------------------------------------------------------
# JsonSFTTokenSource
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [10 + (ord(ch) % 17) for ch in text]


def test_json_sft_source_builds_prompt_masked_sequences() -> None:
    import tempfile

    records = [
        {"instruction": "Add 2 and 2.", "output": "4"},
        {"instruction": "Name a primary color.", "input": "One word.", "output": "red"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.json")
        with open(path, "w") as f:
            json.dump(records, f)
        source = JsonSFTTokenSource(
            path,
            tokenizer=_FakeTokenizer(),
            min_seq_len=8,
            max_seq_len=128,
            loop=False,
        )
        batch = source.get_sequences(max_token_count=1024)
        assert len(batch) == 2
        assert batch[0].loss_mask is not None
        assert bool(batch[0].loss_mask[0].item()) is False
        assert bool(batch[0].loss_mask[-1].item()) is False
        assert bool(batch[0].loss_mask.any().item()) is True


def test_json_sft_source_loops_for_small_datasets() -> None:
    import tempfile

    records = [{"instruction": "Ping?", "output": "Pong."}]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.jsonl")
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        source = JsonSFTTokenSource(
            path,
            tokenizer=_FakeTokenizer(),
            min_seq_len=4,
            max_seq_len=64,
            loop=True,
        )
        batch = source.get_sequences(max_token_count=40)
        assert len(batch) >= 2


# ---------------------------------------------------------------------------
# SyntheticTokenSource
# ---------------------------------------------------------------------------


def test_synthetic_source_deterministic() -> None:
    s1 = SyntheticTokenSource(vocab_size=1000, seq_lens=128, seed=42)
    s2 = SyntheticTokenSource(vocab_size=1000, seq_lens=128, seed=42)
    b1 = s1.get_sequences(max_token_count=1000)
    b2 = s2.get_sequences(max_token_count=1000)
    assert len(b1) == len(b2)
    for a, b in zip(b1, b2):
        assert torch.equal(a.tokens, b.tokens)


def test_synthetic_source_cycles_lengths() -> None:
    s = SyntheticTokenSource(vocab_size=500, seq_lens=[64, 128, 256], seed=0)
    got = []
    it = s.iter_sequences()
    for _ in range(6):
        got.append(next(it))
    assert [len(g) for g in got] == [64, 128, 256, 64, 128, 256]


# ---------------------------------------------------------------------------
# ShardTokenSource
# ---------------------------------------------------------------------------


def _make_fake_shard(path: str, docs: list[list[int]], eot: int = 50256) -> None:
    tokens: list[int] = []
    for doc in docs:
        tokens.append(eot)
        tokens.extend(doc)
    tokens.append(eot)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    arr = np.array(tokens, dtype=np.uint16)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())


def test_shard_source_parses_fake_shard() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        shard_path = os.path.join(tmp, "fake_001.bin")
        docs = [
            list(range(100, 200)),  # 100 tokens
            list(range(200, 280)),  # 80 tokens
            list(range(300, 340)),  # 40 tokens -- below min_seq_len=50
            list(range(400, 600)),  # 200 tokens
        ]
        _make_fake_shard(shard_path, docs)
        source = ShardTokenSource(
            shard_path,
            num_shards=1,
            min_seq_len=50,
            max_seq_len=150,
            vocab_size=1000,
        )
        seqs = list(source.iter_sequences())
        assert len(seqs) == 3
        assert len(seqs[0]) == 100
        assert len(seqs[1]) == 80
        assert len(seqs[2]) == 150  # truncated


def test_shard_source_get_sequences_respects_budget() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        shard_path = os.path.join(tmp, "fake_001.bin")
        docs = [list(range(100, 200)) for _ in range(10)]
        _make_fake_shard(shard_path, docs)
        source = ShardTokenSource(
            shard_path, num_shards=1,
            min_seq_len=50, max_seq_len=200, vocab_size=1000,
        )
        batch = source.get_sequences(max_token_count=250)
        total = sum(len(s) for s in batch)
        assert 200 <= total <= 400


# ---------------------------------------------------------------------------
# CustomSchemaTokenSource
# ---------------------------------------------------------------------------


def test_custom_schema_source_filters_records() -> None:
    from flextrain.io.sequence import Sequence

    records = [
        {"id": 0, "text_tokens": [1, 2, 3, 4, 5]},
        {"id": 1, "text_tokens": []},
        {"id": 2, "text_tokens": [10, 20, 30]},
    ]

    def _extract(rec):
        toks = rec["text_tokens"]
        if not toks:
            return None
        return Sequence(tokens=torch.tensor(toks, dtype=torch.int64))

    source = CustomSchemaTokenSource(records=records, extract=_extract)
    got = list(source.iter_sequences())
    assert len(got) == 2
    assert got[0].tokens.tolist() == [1, 2, 3, 4, 5]
    assert got[1].tokens.tolist() == [10, 20, 30]


def test_custom_schema_source_get_sequences() -> None:
    from flextrain.io.sequence import Sequence

    records = [
        {"toks": list(range(100))} for _ in range(10)
    ]

    def _extract(rec):
        return Sequence(tokens=torch.tensor(rec["toks"], dtype=torch.int64))

    source = CustomSchemaTokenSource(records=records, extract=_extract)
    batch = source.get_sequences(max_token_count=250)
    assert len(batch) == 3


# ---------------------------------------------------------------------------
# HFTokenSource: import smoke only.
# ---------------------------------------------------------------------------


def test_hf_source_importable_smoke() -> None:
    from flextrain.io.sources import HFTokenSource
    import inspect
    sig = inspect.signature(HFTokenSource.__init__)
    params = sig.parameters
    assert "dataset" in params
    assert "tokenizer" in params
    assert "text_field" in params


def _run_all() -> None:
    tests = [
        ("test_raw_source_list_input", test_raw_source_list_input),
        ("test_raw_source_dict_input_with_loss_mask",
         test_raw_source_dict_input_with_loss_mask),
        ("test_raw_source_iter", test_raw_source_iter),
        ("test_synthetic_source_deterministic",
         test_synthetic_source_deterministic),
        ("test_json_sft_source_builds_prompt_masked_sequences",
         test_json_sft_source_builds_prompt_masked_sequences),
        ("test_json_sft_source_loops_for_small_datasets",
         test_json_sft_source_loops_for_small_datasets),
        ("test_synthetic_source_cycles_lengths",
         test_synthetic_source_cycles_lengths),
        ("test_shard_source_parses_fake_shard",
         test_shard_source_parses_fake_shard),
        ("test_shard_source_get_sequences_respects_budget",
         test_shard_source_get_sequences_respects_budget),
        ("test_custom_schema_source_filters_records",
         test_custom_schema_source_filters_records),
        ("test_custom_schema_source_get_sequences",
         test_custom_schema_source_get_sequences),
        ("test_hf_source_importable_smoke",
         test_hf_source_importable_smoke),
    ]
    for name, fn in tests:
        print(f"  - {name}")
        fn()
    print(f"  OK ({len(tests)} tests)")


if __name__ == "__main__":
    _run_all()
