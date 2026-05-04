"""Shared configuration and model metadata helpers for baseline launchers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

BackendName = Literal[
    "megatrain",
    "torchtitan",
    "trl_deepspeed",
    "deepspeed_arctic",
    "megatron",
]


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    slug: str
    hf_config: dict[str, Any]
    vocab_size: int
    model_type: str
    num_layers: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    num_experts: int = 0
    num_experts_per_tok: int = 0
    num_shared_experts: int = 0


@dataclass
class HarnessConfig:
    backend: BackendName
    model_path: Path
    seq_length: int
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_steps: int = 3
    num_gpus: int = 1
    master_port: int = 29500
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    seed: int = 42
    output_dir: Path | None = None
    dry_run: bool = False
    attn_implementation: str = "flash_attention_2"
    moe_kernel_backend: Literal["hf", "auto", "sonic"] = "hf"
    activation_checkpointing: Literal["none", "selective", "full", "memory_budget"] = "full"
    activation_checkpoint_fraction: float | None = None
    activation_checkpoint_interval: int | None = None
    activation_checkpoint_selective_option: str = "op"
    activation_offload: Literal["none", "cpu"] = "none"
    optimizer_offload: Literal["none", "cpu"] = "none"
    param_offload: Literal["none", "cpu"] = "none"
    zero_stage: int = 3
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel_size: int = 1
    fsdp_shard_degree: int = -1
    fsdp_replicate_degree: int = 1
    recompute_granularity: Literal["selective", "full"] = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int | None = None
    recompute_modules: list[str] = field(default_factory=list)
    offload_modules: list[str] = field(default_factory=list)
    cpu_offloading_num_layers: int | None = None
    num_grad_slabs: int = 12
    tiled_loss_shards: int = 1
    tiled_mlp: bool = False
    liger_kernel: Literal["auto", "on", "off"] = "auto"
    torchtitan_module: str | None = None
    torchtitan_config: str | None = None
    megatron_script: Path | None = None
    backend_extra_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LaunchPlan:
    backend: BackendName
    command: list[str]
    cwd: Path
    env: dict[str, str]
    output_dir: Path
    generated_files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "model"


def _nested_config(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("text_config", "llm_config", "language_config", "model_config"):
        nested = config.get(key)
        if isinstance(nested, dict):
            return nested
    return config


def _first_int(config: dict[str, Any], *keys: str) -> int | None:
    nested = _nested_config(config)
    for source in (config, nested):
        for key in keys:
            value = source.get(key)
            if isinstance(value, int):
                return value
    return None


def _first_str(config: dict[str, Any], *keys: str) -> str:
    nested = _nested_config(config)
    for source in (config, nested):
        for key in keys:
            value = source.get(key)
            if isinstance(value, str):
                return value
    return ""


def load_model_info(model_path: Path) -> ModelInfo:
    model_path = model_path.expanduser().resolve()
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected HuggingFace config at {config_path}")
    config = json.loads(config_path.read_text())
    nested = _nested_config(config)

    vocab_size = _first_int(config, "vocab_size")
    if vocab_size is None:
        raise ValueError(
            f"Could not infer vocab_size from {config_path}. "
            "Add it to config.json or pass a model with a standard HF text config."
        )

    num_heads = _first_int(config, "num_attention_heads", "n_heads")
    hidden_size = _first_int(config, "hidden_size", "d_model", "dim")
    head_dim = _first_int(config, "head_dim")
    if head_dim is None and hidden_size and num_heads:
        head_dim = hidden_size // num_heads

    model_type = _first_str(config, "model_type") or _first_str(nested, "model_type")
    num_experts = _first_int(
        config,
        "num_experts",
        "num_routed_experts",
        "num_local_experts",
        "n_routed_experts",
    ) or 0

    return ModelInfo(
        path=model_path,
        slug=slugify(model_path.name),
        hf_config=config,
        vocab_size=vocab_size,
        model_type=model_type,
        num_layers=_first_int(config, "num_hidden_layers", "n_layers", "num_layers"),
        hidden_size=hidden_size,
        intermediate_size=_first_int(
            config,
            "intermediate_size",
            "ffn_hidden_size",
            "moe_intermediate_size",
        ),
        num_attention_heads=num_heads,
        num_key_value_heads=_first_int(
            config,
            "num_key_value_heads",
            "num_kv_heads",
            "n_kv_heads",
        )
        or num_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        num_experts_per_tok=_first_int(
            config,
            "num_experts_per_tok",
            "moe_router_topk",
            "top_k",
        )
        or 0,
        num_shared_experts=_first_int(config, "num_shared_experts") or 0,
    )


def infer_model_alias(model: ModelInfo) -> str:
    name = model.path.name.lower()
    if "llama-3.1-8b" in name or "llama3_8b" in name:
        return "llama3_8B"
    if "olmoe-7b" in name or "olmoe_7b" in name:
        return "olmoe_7Bx1B"
    if "qwen3-32b" in name:
        return "qwen3_32B"
    if "qwen3-30b" in name:
        return "qwen3_30Bx3B"
    return slugify(model.path.name)


def infer_torchtitan_target(model: ModelInfo) -> tuple[str, str] | None:
    name = model.path.name.lower()
    if model.model_type == "llama" and model.num_layers == 32 and model.hidden_size == 4096:
        return "baseline.backends.torchtitan.synthetic_registry", "llama3_8b"
    if model.model_type == "qwen3" and model.num_layers == 28 and model.hidden_size == 2048:
        return "baseline.backends.torchtitan.synthetic_registry", "qwen3_1_7b"
    if model.model_type == "qwen3" and model.num_layers == 64 and model.hidden_size == 5120:
        return "baseline.backends.torchtitan.synthetic_registry", "qwen3_32b"
    if "qwen3" in name:
        return "baseline.backends.torchtitan.synthetic_registry", "qwen3_debugmodel"
    if "llama" in name:
        return "baseline.backends.torchtitan.synthetic_registry", "llama3_debugmodel"
    return None


def model_dims_entry(model: ModelInfo) -> dict[str, Any]:
    if not all(
        [
            model.num_layers,
            model.hidden_size,
            model.num_attention_heads,
            model.num_key_value_heads,
            model.head_dim,
        ]
    ):
        raise ValueError(f"Insufficient architecture metadata in {model.path / 'config.json'}")

    expert_dim = model.intermediate_size
    if model.num_experts and model.hf_config.get("moe_intermediate_size"):
        expert_dim = int(model.hf_config["moe_intermediate_size"])
    if expert_dim is None:
        raise ValueError(f"Could not infer FFN/expert dimension for {model.path}")

    return {
        "vocab_size": model.vocab_size,
        "n_layers": model.num_layers,
        "d_model": model.hidden_size,
        "head_dim": model.head_dim,
        "n_heads": model.num_attention_heads,
        "n_kv_heads": model.num_key_value_heads,
        "expert_dim": expert_dim,
        "num_shared_experts": model.num_shared_experts,
        "num_routed_experts": model.num_experts,
        "top_k": model.num_experts_per_tok,
        "is_causal": True,
        "datatypes": {
            "embed": "bfloat16",
            "head_proj": "bfloat16",
            "attn_proj": "bfloat16",
            "expert_proj": "bfloat16",
            "router": "bfloat16",
            "norm": "bfloat16",
            "residual": "bfloat16",
        },
    }
