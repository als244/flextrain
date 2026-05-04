"""Synthetic token datasets shared by baseline trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset, IterableDataset

LabelMode = Literal["self", "shifted"]


@dataclass(frozen=True)
class SyntheticTokenConfig:
    vocab_size: int
    seq_length: int
    num_samples: int
    seed: int = 42
    label_mode: LabelMode = "self"
    include_position_ids: bool = True


def make_sample(
    *,
    vocab_size: int,
    seq_length: int,
    seed: int,
    label_mode: LabelMode = "self",
    include_position_ids: bool = True,
) -> dict[str, torch.Tensor]:
    """Create one deterministic random-token causal-LM sample."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Keep token 0 available for models that use it as a pad token, but randomize
    # over the whole declared vocabulary so the synthetic stream matches the model.
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(seq_length,),
        dtype=torch.long,
        generator=generator,
    )
    attention_mask = torch.ones(seq_length, dtype=torch.long)

    if label_mode == "shifted":
        labels = torch.roll(input_ids, shifts=-1, dims=0)
        labels[-1] = -100
    else:
        labels = input_ids.clone()

    sample = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if include_position_ids:
        sample["position_ids"] = torch.arange(seq_length, dtype=torch.long)
    return sample


class RandomTokenMapDataset(Dataset):
    """Finite deterministic map-style random token dataset."""

    def __init__(self, config: SyntheticTokenConfig):
        if config.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {config.vocab_size}")
        if config.seq_length <= 1:
            raise ValueError(f"seq_length must be > 1, got {config.seq_length}")
        if config.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {config.num_samples}")
        self.config = config

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.config.num_samples:
            raise IndexError(idx)
        return make_sample(
            vocab_size=self.config.vocab_size,
            seq_length=self.config.seq_length,
            seed=self.config.seed + idx,
            label_mode=self.config.label_mode,
            include_position_ids=self.config.include_position_ids,
        )


class RandomTokenIterableDataset(IterableDataset):
    """Infinite deterministic random token stream for trainer APIs that expect it."""

    def __init__(self, config: SyntheticTokenConfig):
        self.config = config

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        idx = worker_id
        while True:
            yield make_sample(
                vocab_size=self.config.vocab_size,
                seq_length=self.config.seq_length,
                seed=self.config.seed + idx,
                label_mode=self.config.label_mode,
                include_position_ids=self.config.include_position_ids,
            )
            idx += 1 if worker is None else worker.num_workers


def collate_token_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a batch of synthetic token samples."""
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch]) for key in keys}
