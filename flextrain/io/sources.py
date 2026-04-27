"""Data ingestion: :class:`TokenSource` Protocol + built-in adapters.

The engine consumes any object that yields :class:`~flextrain.io.sequence.Sequence`
objects. This module provides several adapters that all satisfy the
same Protocol, so the training loop doesn't care where tokens come
from.

Built-in adapters
-----------------
* :class:`HFTokenSource` — HuggingFace ``datasets`` + ``AutoTokenizer``.
  Streams documents from any HF dataset (FineWeb, The Pile,
  open-orca, redpajama, chat datasets, ...). User-specified field
  name, split, and tokenizer.
* :class:`ShardTokenSource` — orig's ``.bin`` shard reader (ports
  ``orig/sequence_pool.py``). FineWeb-format shards (uint16 tokens
  delimited by an EOT id).
* :class:`RawTokenSource` — user hands us pre-tokenized tensors
  directly. Skip tokenization entirely. For benchmarking or custom
  pipelines.
* :class:`JsonSFTTokenSource` — local JSON / JSONL instruction-tuning
  records with prompt-masked loss.
* :class:`SyntheticTokenSource` — random token ids at caller-given
  sequence lengths. For benchmarking throughput / scheduling
  without real data.
* :class:`CustomSchemaTokenSource` — user passes a callable
  ``fn(record) -> Sequence`` that extracts tokens from any HF-style
  dataset record. For non-standard schemas (chat-templated
  datasets, DPO pairs with accepted/rejected, role-tagged SFT
  examples, ...).

Example
-------
::

    from flextrain.io.sources import HFTokenSource

    source = HFTokenSource(
        dataset="HuggingFaceFW/fineweb",
        subset="sample-10BT",
        split="train",
        tokenizer="meta-llama/Llama-3-8B",
        text_field="text",
        min_seq_len=32,
        max_seq_len=2048,
    )
    seqs = source.get_sequences(max_token_count=524288)  # one step's worth

All adapters expose ``get_sequences(max_token_count)`` (the surface
``orig/train.py`` uses) and ``iter_sequences()``.

Naming
------
This module uses "source" rather than "stream" because FlexTrain
already heavily uses "stream" for CUDA stream objects
(:class:`flextrain.engine.streams.StreamBundle` etc.). Keeping the
two concepts separate prevents ambiguity in type annotations and
error messages.
"""

from __future__ import annotations

import json
import glob
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

import torch

from .sequence import Sequence


# ---------------------------------------------------------------------------
# The protocol every adapter satisfies.
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenSource(Protocol):
    """Any source of :class:`Sequence` objects the engine consumes.

    The engine calls :meth:`get_sequences(max_token_count)` once per
    optimization step to pull a batch sized up to ~``max_token_count``
    tokens. Adapters may block on I/O (e.g. downloading next dataset
    shard), or cache + prefetch, as they see fit.

    Some adapters (e.g. :class:`SyntheticTokenSource`) are effectively
    infinite; callers should enforce their own stop criterion. Other
    adapters (e.g. :class:`ShardTokenSource` on a fixed-size shard)
    return an empty list when exhausted — the training loop should
    treat that as "end of epoch".
    """

    def get_sequences(self, max_token_count: int) -> list[Sequence]:
        """Return a batch of sequences whose total token count is up
        to ``max_token_count``. Empty list = stream exhausted."""
        ...

    def iter_sequences(self) -> Iterator[Sequence]:
        """Iterate all available sequences (one by one). Useful for
        offline scanning or caching."""
        ...


# ---------------------------------------------------------------------------
# RawTokenSource: user-supplied, pre-tokenized.
# ---------------------------------------------------------------------------


@dataclass
class _RawEntry:
    tokens: torch.Tensor
    targets: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None


class RawTokenSource:
    """Stream wrapping a user-provided list of pre-tokenized tensors.

    No tokenizer, no shard format, no HF dependency. Use for
    benchmarking, unit tests, or custom pipelines that already
    emitted int64 token tensors.

    Parameters
    ----------
    tokens
        Either:
          * a list of 1-D int64 tensors (one per sequence), OR
          * a dict with keys ``"tokens"``, optionally
            ``"targets"`` / ``"loss_mask"``, each a list of
            per-sequence tensors.
    """

    def __init__(
        self,
        tokens: (
            list[torch.Tensor]
            | list[dict[str, torch.Tensor]]
            | dict[str, list[torch.Tensor]]
        ),
    ) -> None:
        self._entries: list[_RawEntry] = []
        if isinstance(tokens, dict):
            t_list = tokens["tokens"]
            tgt_list = tokens.get("targets", [None] * len(t_list))
            lm_list = tokens.get("loss_mask", [None] * len(t_list))
            for t, tg, lm in zip(t_list, tgt_list, lm_list):
                self._entries.append(
                    _RawEntry(tokens=t, targets=tg, loss_mask=lm)
                )
        else:
            for item in tokens:
                if isinstance(item, dict):
                    self._entries.append(
                        _RawEntry(
                            tokens=item["tokens"],
                            targets=item.get("targets"),
                            loss_mask=item.get("loss_mask"),
                        )
                    )
                else:
                    self._entries.append(_RawEntry(tokens=item))
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def _make_seq(self, entry: _RawEntry) -> Sequence:
        return Sequence(
            tokens=entry.tokens.to(torch.int64),
            targets=None if entry.targets is None else entry.targets.to(torch.int64),
            loss_mask=entry.loss_mask,
        )

    def iter_sequences(self) -> Iterator[Sequence]:
        for e in self._entries:
            yield self._make_seq(e)

    def get_sequences(self, max_token_count: int) -> list[Sequence]:
        out: list[Sequence] = []
        total = 0
        while self._cursor < len(self._entries):
            e = self._entries[self._cursor]
            if out and total + len(e.tokens) > max_token_count:
                break
            out.append(self._make_seq(e))
            total += len(e.tokens)
            self._cursor += 1
        return out


# ---------------------------------------------------------------------------
# JsonSFTTokenSource: local instruction/output JSON with prompt masking.
# ---------------------------------------------------------------------------


class JsonSFTTokenSource:
    """Local JSON / JSONL supervised fine-tuning data."""

    def __init__(
        self,
        path: str,
        *,
        tokenizer: str | Any,
        prompt_field: str = "instruction",
        response_field: str = "output",
        input_field: str | None = "input",
        min_seq_len: int = 32,
        max_seq_len: int = 2048,
        loop: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        self.path = path
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.input_field = input_field
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.loop = loop
        if isinstance(tokenizer, str):
            try:
                from transformers import AutoTokenizer  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "JsonSFTTokenSource needs `transformers`. "
                    "Install via `pip install transformers`."
                ) from e
            self._tok = AutoTokenizer.from_pretrained(
                tokenizer, trust_remote_code=trust_remote_code,
            )
        else:
            self._tok = tokenizer
        self._records = self._load_records(path)
        self._cursor = 0

    @staticmethod
    def _load_records(path: str) -> list[dict[str, Any]]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"JSON SFT file not found: {path}")
        if path.endswith(".jsonl"):
            out = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
            return out
        with open(path) as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(
                f"expected a list of records in {path!r}, got {type(payload).__name__}"
            )
        return payload

    def reset(self) -> None:
        self._cursor = 0

    def _build_prompt(self, rec: dict[str, Any]) -> tuple[str, str] | None:
        prompt = str(rec.get(self.prompt_field, "") or "").strip()
        response = str(rec.get(self.response_field, "") or "").strip()
        extra = ""
        if self.input_field:
            extra = str(rec.get(self.input_field, "") or "").strip()
        if not prompt or not response:
            return None
        if extra:
            prompt_text = (
                f"Instruction:\n{prompt}\n\n"
                f"Input:\n{extra}\n\n"
                f"Response:\n"
            )
        else:
            prompt_text = f"Instruction:\n{prompt}\n\nResponse:\n"
        return prompt_text, response

    def _next_seq(self) -> Sequence | None:
        if not self._records:
            return None
        while True:
            if self._cursor >= len(self._records):
                if not self.loop:
                    return None
                self._cursor = 0
            idx = self._cursor
            rec = self._records[idx]
            self._cursor += 1
            prompt_and_response = self._build_prompt(rec)
            if prompt_and_response is None:
                continue
            prompt_text, response_text = prompt_and_response
            prompt_ids = self._tok.encode(prompt_text, add_special_tokens=False)
            response_ids = self._tok.encode(response_text, add_special_tokens=False)
            eos_id = getattr(self._tok, "eos_token_id", None)
            if eos_id is not None:
                response_ids = response_ids + [int(eos_id)]
            if not response_ids:
                continue
            if len(prompt_ids) >= self.max_seq_len:
                continue
            room = self.max_seq_len - len(prompt_ids)
            if room <= 1:
                continue
            response_ids = response_ids[:room]
            token_ids = prompt_ids + response_ids
            if len(token_ids) < self.min_seq_len:
                continue

            tokens = torch.tensor(token_ids, dtype=torch.int64)
            targets = torch.roll(tokens, -1)
            loss_mask = torch.ones(len(tokens), dtype=torch.bool)
            loss_mask[: len(prompt_ids)] = False
            loss_mask[-1] = False
            if not bool(loss_mask.any().item()):
                continue
            return Sequence(
                tokens=tokens,
                targets=targets,
                loss_mask=loss_mask,
                seq_id=idx,
            )

    def iter_sequences(self) -> Iterator[Sequence]:
        while True:
            seq = self._next_seq()
            if seq is None:
                return
            yield seq

    def get_sequences(self, max_token_count: int) -> list[Sequence]:
        out: list[Sequence] = []
        total = 0
        while True:
            seq = self._next_seq()
            if seq is None:
                break
            if out and total + len(seq) > max_token_count:
                out.append(seq)
                total += len(seq)
                break
            out.append(seq)
            total += len(seq)
            if total >= max_token_count:
                break
        return out


# ---------------------------------------------------------------------------
# SyntheticTokenSource: random tokens at caller-given seq lengths.
# ---------------------------------------------------------------------------


class SyntheticTokenSource:
    """Infinite stream of random-token sequences. Useful for
    benchmarking scheduling / throughput without touching real data.

    Parameters
    ----------
    vocab_size
        Token id range is ``[0, vocab_size)``.
    seq_lens
        Either a fixed int or a ``Sequence[int]``. If a list, the
        stream cycles through the lengths in order.
    seed
        RNG seed (deterministic output).
    """

    def __init__(
        self,
        vocab_size: int,
        seq_lens: int | list[int] | tuple[int, ...] = 512,
        seed: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        if isinstance(seq_lens, int):
            self._lens = (seq_lens,)
        else:
            self._lens = tuple(seq_lens)
            if not self._lens:
                raise ValueError("seq_lens must be non-empty")
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        self._cursor = 0  # cycles through _lens

    def _make_seq(self) -> Sequence:
        n = self._lens[self._cursor % len(self._lens)]
        self._cursor += 1
        tokens = torch.randint(
            0, self.vocab_size, (n,), generator=self._gen, dtype=torch.int64,
        )
        return Sequence(tokens=tokens)

    def iter_sequences(self) -> Iterator[Sequence]:
        while True:
            yield self._make_seq()

    def get_sequences(self, max_token_count: int) -> list[Sequence]:
        out: list[Sequence] = []
        total = 0
        while True:
            # Peek the next length without consuming.
            n = self._lens[self._cursor % len(self._lens)]
            if out and total + n > max_token_count:
                break
            out.append(self._make_seq())
            total += n
            # Safety: if we've produced one sequence that already exceeds
            # the budget (seq_len > max_token_count), stop at 1 to avoid
            # infinite loop.
            if n > max_token_count and len(out) >= 1:
                break
        return out


# ---------------------------------------------------------------------------
# ShardTokenSource: port of orig/sequence_pool.py's .bin shard reader.
# ---------------------------------------------------------------------------


class ShardTokenSource:
    """FineWeb-format (orig's ``.bin``) shard reader.

    Format (see ``orig/fineweb.py``): 256 int32 header (1024 bytes)
    followed by uint16 tokens, documents delimited by a configurable
    EOT id (default 50256 = GPT-2's ``<|endoftext|>``).

    Parameters
    ----------
    shard_pattern
        glob or format-string for shard paths. If the string contains
        ``{shard_index:06d}`` it is formatted per shard; otherwise the
        string is treated as a glob.
    num_shards
        Number of shards to iterate (1-indexed).
    min_seq_len, max_seq_len
        Skip documents shorter than ``min_seq_len``; truncate longer
        than ``max_seq_len``. Matches
        ``orig/sequence_pool.py:SequencePool``.
    vocab_size
        Clip tokens to ``[0, vocab_size)``. In practice FineWeb
        tokens fit in uint16 with GPT-2's 50257 vocab.
    token_dtype
        Raw on-disk dtype. Default ``uint16``.
    start_id, end_id
        Delimiter(s). If the same value is used for both, start-IDs
        are derived from end-IDs shifted by one (orig's convention).

    Thread-safety: this adapter is NOT thread-safe. For async
    prefetching use ``orig.sequence_pool.SequencePool`` directly or
    spin up a worker thread around ``get_sequences``.
    """

    def __init__(
        self,
        shard_pattern: str,
        num_shards: int | None = None,
        *,
        min_seq_len: int = 32,
        max_seq_len: int = 2048,
        vocab_size: int = 50432,
        token_dtype: Any = None,
        start_id: int = 50256,
        end_id: int = 50256,
    ) -> None:
        import numpy as np

        self.shard_pattern = shard_pattern
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.token_dtype = token_dtype or np.uint16
        self.start_id = start_id
        self.end_id = end_id

        # Resolve shard list.
        if "{shard_index" in shard_pattern:
            assert num_shards is not None, (
                "num_shards required when shard_pattern uses "
                "{shard_index} formatting"
            )
            self._shard_paths = [
                shard_pattern.format(shard_index=i)
                for i in range(1, num_shards + 1)
            ]
        else:
            self._shard_paths = sorted(glob.glob(shard_pattern))
            if num_shards is not None:
                self._shard_paths = self._shard_paths[:num_shards]
        if not self._shard_paths:
            raise FileNotFoundError(
                f"no shards found for pattern {shard_pattern!r}"
            )

        self._shard_idx = 0
        self._cursor_in_shard = 0
        self._current_arr = None  # lazy-load per shard

    def _load_shard(self, path: str):
        import numpy as np
        arr = np.fromfile(path, dtype=self.token_dtype, offset=256 * 4)
        return arr

    def _next_doc(self) -> torch.Tensor | None:
        """Return next document's tokens as int64 cpu tensor, or None
        if exhausted."""
        import numpy as np

        while self._shard_idx < len(self._shard_paths):
            if self._current_arr is None:
                self._current_arr = self._load_shard(
                    self._shard_paths[self._shard_idx]
                )
                self._cursor_in_shard = 0

            arr = self._current_arr
            while self._cursor_in_shard < len(arr):
                end = self._cursor_in_shard
                while end < len(arr) and arr[end] != self.end_id:
                    end += 1
                if end > self._cursor_in_shard + 1:
                    raw = arr[self._cursor_in_shard : end]
                    if len(raw) > 0 and raw[0] == self.start_id:
                        raw = raw[1:]
                    if len(raw) >= self.min_seq_len:
                        take = min(len(raw), self.max_seq_len)
                        doc = np.clip(
                            raw[:take].astype(np.int64),
                            0, self.vocab_size - 1,
                        )
                        self._cursor_in_shard = end + 1
                        return torch.from_numpy(doc.copy())
                self._cursor_in_shard = end + 1
            # Exhausted current shard; move to next.
            self._current_arr = None
            self._shard_idx += 1
        return None

    def reset(self) -> None:
        self._shard_idx = 0
        self._cursor_in_shard = 0
        self._current_arr = None

    def iter_sequences(self) -> Iterator[Sequence]:
        while True:
            tokens = self._next_doc()
            if tokens is None:
                return
            yield Sequence(tokens=tokens)

    def get_sequences(self, max_token_count: int) -> list[Sequence]:
        out: list[Sequence] = []
        total = 0
        while True:
            tokens = self._next_doc()
            if tokens is None:
                break
            if out and total + len(tokens) > max_token_count:
                # Put back? We'd need a peek buffer. For simplicity
                # include this seq even if slightly over budget.
                # That matches orig's behavior of packing greedy.
                out.append(Sequence(tokens=tokens))
                total += len(tokens)
                break
            out.append(Sequence(tokens=tokens))
            total += len(tokens)
            if total >= max_token_count:
                break
        return out


# ---------------------------------------------------------------------------
# CustomSchemaTokenSource: user-callable extractor over a record iterator.
# ---------------------------------------------------------------------------


class CustomSchemaTokenSource:
    """Adapter over an arbitrary record source with a user-provided
    extractor function.

    Parameters
    ----------
    records
        An iterable of records (dicts, dataclass instances, or
        anything else).
    extract
        ``fn(record) -> Sequence | None``. Returning ``None`` skips
        that record (e.g. filter by length, language, role).
    loop
        If ``True`` and ``records`` is finite, we loop over it
        repeatedly. For infinite datasets set to ``False`` (default).

    Use for non-standard schemas:

    * chat-template datasets where you assemble role-tagged tokens
      with a loss mask that excludes the prompt;
    * DPO / reward-model datasets that store accepted/rejected pairs;
    * multi-modal datasets where you concatenate text + vision
      token ids.
    """

    def __init__(
        self,
        records: Iterator[Any] | list[Any],
        extract: Callable[[Any], Sequence | None],
        *,
        loop: bool = False,
    ) -> None:
        self._records = records
        self._extract = extract
        self._loop = loop
        self._iter = iter(records)
        self._exhausted = False

    def _next_seq(self) -> Sequence | None:
        while not self._exhausted:
            try:
                rec = next(self._iter)
            except StopIteration:
                if self._loop and isinstance(
                    self._records, (list, tuple)
                ):
                    self._iter = iter(self._records)
                    continue
                self._exhausted = True
                return None
            seq = self._extract(rec)
            if seq is None:
                continue
            return seq
        return None

    def reset(self) -> None:
        if isinstance(self._records, (list, tuple)):
            self._iter = iter(self._records)
            self._exhausted = False

    def iter_sequences(self) -> Iterator[Sequence]:
        while True:
            s = self._next_seq()
            if s is None:
                return
            yield s

    def get_sequences(self, max_token_count: int) -> list[Sequence]:
        out: list[Sequence] = []
        total = 0
        while True:
            s = self._next_seq()
            if s is None:
                break
            if out and total + len(s) > max_token_count:
                out.append(s)
                total += len(s)
                break
            out.append(s)
            total += len(s)
            if total >= max_token_count:
                break
        return out


# ---------------------------------------------------------------------------
# HFTokenSource: HuggingFace datasets + AutoTokenizer.
# ---------------------------------------------------------------------------


class HFTokenSource:
    """HuggingFace ``datasets`` + ``AutoTokenizer`` adapter.

    Streams documents from any HF dataset (FineWeb, The Pile, OpenOrca,
    RedPajama, chat datasets, ...) with a user-specified tokenizer.

    Parameters
    ----------
    dataset
        HF dataset repo id (e.g. ``"HuggingFaceFW/fineweb"``).
    subset
        Optional dataset config name (``"sample-10BT"``).
    split
        Dataset split. Default ``"train"``.
    tokenizer
        HF tokenizer repo id (e.g. ``"meta-llama/Llama-3-8B"``) OR
        a pre-instantiated ``PreTrainedTokenizerFast``.
    text_field
        Which dataset column holds the document text. Default
        ``"text"``.
    min_seq_len, max_seq_len
        Skip too-short, truncate too-long.
    streaming
        If ``True`` use ``load_dataset(..., streaming=True)`` which
        iterates over the dataset without downloading the full shard
        tree. Good for big datasets like FineWeb.
    trust_remote_code
        Forwarded to both ``load_dataset`` and ``AutoTokenizer``.

    Dependencies
    ------------
    Requires ``datasets`` and ``transformers`` to be installed. They
    are imported lazily so this module is safely importable without
    them; the import error surfaces only when the HF stream is
    actually constructed.
    """

    def __init__(
        self,
        dataset: str,
        *,
        subset: str | None = None,
        split: str = "train",
        tokenizer: str | Any = "gpt2",
        text_field: str = "text",
        min_seq_len: int = 32,
        max_seq_len: int = 2048,
        streaming: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "HFTokenSource needs `datasets`. "
                "Install via `pip install datasets`."
            ) from e
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "HFTokenSource needs `transformers`. "
                "Install via `pip install transformers`."
            ) from e

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.text_field = text_field

        self._ds = load_dataset(
            dataset,
            name=subset,
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
        )
        self._iter = iter(self._ds)

        if isinstance(tokenizer, str):
            self._tok = AutoTokenizer.from_pretrained(
                tokenizer, trust_remote_code=trust_remote_code,
            )
        else:
            self._tok = tokenizer

    def _next_seq(self) -> Sequence | None:
        while True:
            try:
                rec = next(self._iter)
            except StopIteration:
                return None
            text = rec.get(self.text_field) if hasattr(rec, "get") else None
            if text is None:
                text = rec[self.text_field]
            if not text:
                continue
            ids = self._tok.encode(text, add_special_tokens=False)
            if len(ids) < self.min_seq_len:
                continue
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]
            tokens = torch.tensor(ids, dtype=torch.int64)
            return Sequence(tokens=tokens)

    def iter_sequences(self) -> Iterator[Sequence]:
        while True:
            s = self._next_seq()
            if s is None:
                return
            yield s

    def get_sequences(self, max_token_count: int) -> list[Sequence]:
        out: list[Sequence] = []
        total = 0
        while True:
            s = self._next_seq()
            if s is None:
                break
            if out and total + len(s) > max_token_count:
                out.append(s)
                total += len(s)
                break
            out.append(s)
            total += len(s)
            if total >= max_token_count:
                break
        return out


__all__ = [
    "CustomSchemaTokenSource",
    "HFTokenSource",
    "JsonSFTTokenSource",
    "RawTokenSource",
    "ShardTokenSource",
    "SyntheticTokenSource",
    "TokenSource",
]
