"""Top-level dataclass configs for a FlexTrain training run.

A run is fully specified by four things:

1. **Model architecture** -- which :class:`nn/layers/*` class to instantiate,
   plus its dims (d_model, n_heads, etc.). Loadable from
   ``orig/model_dims.json`` or derived from a HF ``config.json`` via
   ``io/arch/<family>.hf_config_to_flextrain``.

2. **Training hyperparameters** -- global batch size, sequence length,
   optimizer + its hyperparams, LR schedule, total steps.

3. **Hardware budget** -- GPU memory cap, host memory cap, device id.
   Passed through to :func:`~flextrain.core.working_set.determine_working_set_config`.

4. **IO** -- HF checkpoint to load, data source (HF dataset id or
   pre-tokenized shard glob), tokenizer spec, output directory.

This module defines the dataclasses; loading from YAML / CLI happens in
:mod:`flextrain.cli`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class ModelConfig:
    """Matches the per-model dict in ``orig/model_dims.json`` plus a few
    additions that orig keeps separate (norm eps, RoPE base, window sizes).
    """

    arch: str  # registered architecture family name (e.g. "llama", "qwen", "olmoe")
    vocab_size: int
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int  # FFN intermediate dim (dense) or per-expert dim (MoE)
    num_shared_experts: int = 0
    num_routed_experts: int = 0
    top_k: int = 0
    is_causal: bool = True

    # Hyperparams the layer needs that aren't pure dims.
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Mapping[str, Any] | None = None
    window_size_left: int = -1
    window_size_right: int = 0

    # Per-tensor dtype selection -- default to bf16 everywhere, matching
    # orig/model_dims.json. Override via YAML for mixed-precision runs.
    dtypes: Mapping[str, str] = field(
        default_factory=lambda: {
            "embed": "bfloat16",
            "head_proj": "bfloat16",
            "attn_proj": "bfloat16",
            "expert_proj": "bfloat16",
            "router": "bfloat16",
            "norm": "bfloat16",
            "residual": "bfloat16",
            # Master storage can be fp32 even when compute is bf16. Defaults
            # follow compute dtype if unset.
            "master_attn_proj": "bfloat16",
            "master_expert_proj": "bfloat16",
        }
    )


@dataclass(frozen=True)
class OptimizerConfig:
    kind: Literal["adamw", "muon"] = "adamw"
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.001


@dataclass(frozen=True)
class TrainConfig:
    global_batch_tokens: int  # e.g. 524288 = 512K
    max_seq_len: int
    total_steps: int
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # LR schedule (cosine warmup + decay, matching orig/train.py defaults):
    warmup_pct: float = 0.05
    cooldown_pct: float = 0.2
    final_lr_fraction: float = 0.1
    # For fixed-seq-len runs, passed to working_set sizing:
    fixed_seq_len: bool = False


@dataclass(frozen=True)
class HardwareConfig:
    device_id: int = 0
    max_gpu_mem_gib: float | None = None  # None -> all available
    max_host_mem_gib: float | None = None  # None -> all available
    leeway_gpu_mem_gib: float = 2.0
    leeway_host_mem_gib: float = 10.0
    min_chunk_size: int | None = None
    max_chunk_size: int | None = None


@dataclass(frozen=True)
class DataConfig:
    """Token-source selection.

    Exactly one of ``hf_dataset`` / ``shard_pattern`` / ``synthetic``
    / ``raw`` / ``custom`` should be set. See
    :mod:`flextrain.io.sources` for the adapter classes.
    """

    # HuggingFace `datasets`. Requires `hf_dataset`, optional
    # `hf_subset`, `hf_split`, `hf_text_field`, `hf_streaming`,
    # and a `tokenizer`.
    hf_dataset: str | None = None
    hf_subset: str | None = None
    hf_split: str = "train"
    hf_text_field: str = "text"
    hf_streaming: bool = True

    # FineWeb-format .bin shard. `shard_pattern` accepts a glob or
    # a `{shard_index:06d}` format string. Requires `shard_num_shards`.
    shard_pattern: str | None = None
    shard_num_shards: int | None = None

    # Synthetic random-token source (benchmarking). Requires
    # `synthetic_seq_len`.
    synthetic: bool = False
    synthetic_seq_len: int | list[int] = 512

    # Pre-tokenized raw tensors. `raw_path` points at a .pt file
    # containing a list[Tensor] or a dict with tokens/targets/loss_mask.
    raw_path: str | None = None

    # Custom-schema extractor -- point at a Python module:callable pair
    # that returns (Iterable[record], extract_fn). Opt-in since it
    # loads arbitrary code.
    custom_factory: str | None = None  # e.g. "my_module:build"

    # Shared knobs.
    min_seq_len: int = 32
    max_seq_len: int = 2048


@dataclass(frozen=True)
class IOConfig:
    hf_checkpoint: str | None = None  # e.g. "meta-llama/Llama-3-8B" or local path
    tokenizer: str | None = None  # HF tokenizer id; defaults to hf_checkpoint
    output_dir: str = "runs/flextrain"
    # Native FlexTrain checkpoint to resume from (takes precedence over
    # hf_checkpoint when both set).
    resume_from: str | None = None
    # Data source -- detailed selection lives in DataConfig.
    data: DataConfig = field(default_factory=DataConfig)


@dataclass(frozen=True)
class RunConfig:
    """Top-level: one of these fully specifies a training run."""

    model: ModelConfig
    train: TrainConfig
    hardware: HardwareConfig
    io: IOConfig
