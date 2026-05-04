"""TorchTitan config registry using random token IDs instead of text datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig
from torchtitan.trainer import Trainer

# Loss config: post-v0.2.2 torchtitan main moved loss selection onto
# Trainer.Config.loss (a concrete subclass of the abstract BaseLoss).
# Without setting it, ``Trainer.__init__`` raises
# ``TypeError: Can't instantiate abstract class BaseLoss``. The
# pre-v0.2.2 orig path went through ``model_spec.build_loss_fn`` so
# this didn't surface there. We import the new concrete class lazily
# and only add the field when present, so the registry stays usable
# against both old and new torchtitan checkouts.
try:
    from torchtitan.components.loss import CrossEntropyLoss as _CrossEntropyLoss
except ImportError:  # orig API, no concrete CrossEntropyLoss class
    _CrossEntropyLoss = None


class _SyntheticTitanDataset(IterableDataset):
    def __init__(self, vocab_size: int, seq_len: int, seed: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        stride = 1 if worker is None else worker.num_workers
        idx = worker_id
        while True:
            generator = torch.Generator()
            generator.manual_seed(self.seed + idx)
            x = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long, generator=generator)
            yield {"input": x[:-1]}, x[1:]
            idx += stride


class SyntheticTokenDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        vocab_size: int = 2048
        seed: int = 42

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        dataset = _SyntheticTitanDataset(
            vocab_size=config.vocab_size,
            seq_len=seq_len,
            seed=config.seed + dp_rank * 1_000_000,
        )
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            batch_size=local_batch_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
        )


def _base_config(model_spec, *, batch_size: int = 1, seq_len: int = 2048) -> Trainer.Config:
    extra: dict = {}
    if _CrossEntropyLoss is not None:
        # post-v0.2.2 torchtitan main: must specify a concrete loss
        # (BaseLoss is abstract). CrossEntropyLoss is the standard
        # next-token loss the orig path used implicitly via
        # model_spec.build_loss_fn.
        extra["loss"] = _CrossEntropyLoss.Config()
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=OptimizersContainer.Config(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=batch_size,
            seq_len=seq_len,
            steps=10,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        dataloader=SyntheticTokenDataLoader.Config(),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(),
        checkpoint=CheckpointManager.Config(enable=False, interval=500),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        **extra,
    )


def llama3_debugmodel() -> Trainer.Config:
    from torchtitan.models.llama3 import model_registry

    return _base_config(model_registry("debugmodel"))


def llama3_8b() -> Trainer.Config:
    from torchtitan.models.llama3 import model_registry

    return _base_config(model_registry("8B"))


def qwen3_debugmodel() -> Trainer.Config:
    from torchtitan.models.qwen3 import model_registry

    return _base_config(model_registry("debugmodel"))


def qwen3_1_7b() -> Trainer.Config:
    from torchtitan.models.qwen3 import model_registry

    return _base_config(model_registry("1.7B"))


def qwen3_32b() -> Trainer.Config:
    from torchtitan.models.qwen3 import model_registry

    return _base_config(model_registry("32B"))
