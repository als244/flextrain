"""Backend command builders for the unified baseline runner."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from .spec import (
    HarnessConfig,
    LaunchPlan,
    ModelInfo,
    infer_model_alias,
    infer_torchtitan_target,
    model_dims_entry,
    repo_root,
)


def _validate_fraction(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"activation checkpoint fraction must be in [0, 1], got {value}")
    return value


def _reject_true_fraction(config: HarnessConfig, backend: str) -> None:
    fraction = config.activation_checkpoint_fraction
    if fraction is not None:
        _validate_fraction(fraction)
        if 0.0 < fraction < 1.0:
            raise ValueError(
                f"{backend} does not expose supported fractional activation checkpointing. "
                "Use --activation-checkpointing none/full, or use a backend with layer/block selection."
            )


def _binary_checkpointing_mode(config: HarnessConfig) -> str:
    fraction = config.activation_checkpoint_fraction
    if fraction is None:
        return config.activation_checkpointing
    _validate_fraction(fraction)
    return "full" if fraction >= 1.0 else "none"


def _selective_interval_from_fraction(fraction: float) -> int:
    _validate_fraction(fraction)
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return 1
    return max(1, round(1.0 / fraction))


def _fractional_mode(config: HarnessConfig) -> tuple[str, str]:
    fraction = config.activation_checkpoint_fraction
    if fraction is None:
        return config.activation_checkpointing, config.activation_checkpoint_selective_option
    if fraction <= 0.0:
        return "none", config.activation_checkpoint_selective_option
    if fraction >= 1.0:
        return "full", config.activation_checkpoint_selective_option
    return "selective", str(_selective_interval_from_fraction(fraction))


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _output_dir(config: HarnessConfig, model: ModelInfo) -> Path:
    if config.output_dir is not None:
        return config.output_dir.expanduser().resolve()
    return (
        repo_root()
        / "baseline"
        / "runs"
        / f"{config.backend}_{model.slug}_seq{config.seq_length}_{_timestamp()}"
    )


def _base_env(*paths: Path) -> dict[str, str]:
    root = repo_root()
    pythonpath_parts = [str(root), *(str(path) for path in paths)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    alloc_conf = os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        os.environ.get(
            "PYTORCH_ALLOC_CONF",
            "pinned_use_cuda_host_register:True,expandable_segments:True",
        ),
    )
    return {
        "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        "PYTORCH_CUDA_ALLOC_CONF": alloc_conf,
        "PYTORCH_ALLOC_CONF": alloc_conf,
        "DS_SKIP_CUDA_CHECK": os.environ.get("DS_SKIP_CUDA_CHECK", "1"),
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _write_yaml(path: Path, payload: dict) -> Path:
    """Tiny in-tree YAML dumper — accelerate launch reads its config as YAML.

    Limited to the value types we actually emit (str, int, float, bool, dict,
    list of scalars, None). Avoids pulling PyYAML into the harness env just
    for one config file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(value, indent: int) -> str:
        pad = "  " * indent
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            # Quote strings that contain YAML metacharacters, look like
            # other types (number/bool/null), or match YAML 1.1 reserved
            # boolean spellings. ``downcast_bf16: no`` would otherwise be
            # decoded as Python ``False`` by accelerate's YAML 1.1 loader.
            yaml11_reserved = {
                "y", "Y", "yes", "Yes", "YES", "n", "N", "no", "No", "NO",
                "true", "True", "TRUE", "false", "False", "FALSE",
                "on", "On", "ON", "off", "Off", "OFF",
                "null", "Null", "NULL", "~",
            }
            try:
                float(value)
                looks_numeric = True
            except ValueError:
                looks_numeric = False
            needs_quote = (
                value == ""
                or value in yaml11_reserved
                or looks_numeric
                or any(c in value for c in ":#{}[]&*!|>'\"%@`,")
            )
            if needs_quote:
                escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
                return f"\"{escaped}\""
            return value
        if isinstance(value, list):
            if not value:
                return "[]"
            return "\n" + "\n".join(f"{pad}- {_emit(item, indent + 1)}" for item in value)
        if isinstance(value, dict):
            if not value:
                return "{}"
            lines = []
            for k, v in value.items():
                rendered = _emit(v, indent + 1)
                if isinstance(v, (dict, list)) and rendered and rendered[0] == "\n":
                    lines.append(f"{pad}{k}:{rendered}")
                else:
                    lines.append(f"{pad}{k}: {rendered}")
            return "\n" + "\n".join(lines)
        raise TypeError(f"Unsupported YAML value type: {type(value).__name__}")

    body = _emit(payload, 0)
    if body.startswith("\n"):
        body = body[1:]
    path.write_text(body + "\n")
    return path


def _accelerate_fsdp_config(config: HarnessConfig) -> dict:
    """Generate an accelerate launch config that wraps the model with FSDP2.

    bf16 master/grads/opt: ``mixed_precision: bf16`` plus FSDP2's
    MixedPrecisionPolicy (param_dtype=bf16, reduce_dtype=bf16) keep params
    sharded in bf16 with no fp32 master copy. This matches the bf16-master
    semantics the DeepSpeed config establishes for trl_deepspeed.

    Offloading: FSDP2 ties param/grad/opt offload together (they share the
    sharded ``DTensor`` storage). Setting either ``--param-offload cpu`` or
    ``--optimizer-offload cpu`` flips ``fsdp_offload_params: true`` — both
    move with the params. We surface this in the launch plan ``notes``.
    Activation offload is handled by SFTConfig's ``activation_offloading``.
    """
    offload_params = (
        config.param_offload == "cpu" or config.optimizer_offload == "cpu"
    )
    fsdp_block = {
        "fsdp_version": 2,
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_sharding_strategy": "FULL_SHARD"
        if config.fsdp_replicate_degree <= 1
        else "HYBRID_SHARD",
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        "fsdp_use_orig_params": True,
        "fsdp_offload_params": offload_params,
        "fsdp_cpu_ram_efficient_loading": True,
        "fsdp_sync_module_states": True,
        "fsdp_forward_prefetch": False,
        "fsdp_backward_prefetch": "BACKWARD_PRE",
        "fsdp_activation_checkpointing": config.activation_checkpointing != "none",
    }
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "downcast_bf16": "no",
        "fsdp_config": fsdp_block,
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": "bf16",
        "num_machines": 1,
        "num_processes": config.num_gpus,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
    }


def _deepspeed_config(config: HarnessConfig, *, include_sequence_parallel: bool) -> dict:
    ds_config = {
        "train_micro_batch_size_per_gpu": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": config.weight_decay,
            },
        },
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
        "zero_allow_untested_optimizer": True,
    }
    if config.zero_stage > 0:
        ds_config["bf16"]["bf16_master_weights_and_grads"] = True
        ds_config["bf16"]["bf16_optimizer_states"] = True
        ds_config["optimizer"]["fp32_optimizer_states"] = False
        zero = {
            "stage": config.zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
        }
        if config.optimizer_offload == "cpu":
            zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        if config.param_offload == "cpu":
            zero["offload_param"] = {"device": "cpu", "pin_memory": True}
        ds_config["zero_optimization"] = zero
    if include_sequence_parallel and config.sequence_parallel_size > 1:
        ds_config["sequence_parallel_size"] = config.sequence_parallel_size
    return ds_config


def _bool_flag(command: list[str], condition: bool, flag: str) -> None:
    if condition:
        command.append(flag)


def build_trl_deepspeed(config: HarnessConfig, model: ModelInfo) -> LaunchPlan:
    _reject_true_fraction(config, "trl_deepspeed/HF/TRL")
    out = _output_dir(config, model)
    ds_path = _write_json(out / "deepspeed_bf16.json", _deepspeed_config(config, include_sequence_parallel=False))
    script = repo_root() / "baseline" / "backends" / "trl_deepspeed" / "train_synthetic.py"
    checkpointing_mode = _binary_checkpointing_mode(config)
    command = [
        "deepspeed",
        f"--num_gpus={config.num_gpus}",
        f"--master_port={config.master_port}",
        str(script),
        "--model-path",
        str(model.path),
        "--seq-length",
        str(config.seq_length),
        "--vocab-size",
        str(model.vocab_size),
        "--micro-batch-size",
        str(config.micro_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--num-steps",
        str(config.num_steps),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--seed",
        str(config.seed),
        "--output-dir",
        str(out),
        "--deepspeed-config",
        str(ds_path),
        "--attn-implementation",
        config.attn_implementation,
        "--moe-kernel-backend",
        config.moe_kernel_backend,
        "--activation-checkpointing",
        checkpointing_mode,
        "--activation-offload",
        config.activation_offload,
        "--liger-kernel",
        config.liger_kernel,
    ]
    command.extend(config.backend_extra_args)
    return LaunchPlan(
        backend="trl_deepspeed",
        command=command,
        cwd=repo_root(),
        env=_base_env(),
        output_dir=out,
        generated_files=[ds_path],
    )


def build_deepspeed_arctic(config: HarnessConfig, model: ModelInfo) -> LaunchPlan:
    _reject_true_fraction(config, "deepspeed_arctic/HF")
    out = _output_dir(config, model)
    ds_path = _write_json(out / "deepspeed_arctic_bf16.json", _deepspeed_config(config, include_sequence_parallel=True))
    script = repo_root() / "baseline" / "backends" / "deepspeed_arctic" / "train_synthetic.py"
    checkpointing_mode = _binary_checkpointing_mode(config)
    command = [
        "deepspeed",
        f"--num_gpus={config.num_gpus}",
        f"--master_port={config.master_port}",
        str(script),
        "--model-path",
        str(model.path),
        "--seq-length",
        str(config.seq_length),
        "--vocab-size",
        str(model.vocab_size),
        "--micro-batch-size",
        str(config.micro_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--num-steps",
        str(config.num_steps),
        "--seed",
        str(config.seed),
        "--output-dir",
        str(out),
        "--deepspeed-config",
        str(ds_path),
        "--attn-implementation",
        config.attn_implementation,
        "--moe-kernel-backend",
        config.moe_kernel_backend,
        "--activation-checkpointing",
        checkpointing_mode,
        "--activation-offload",
        config.activation_offload,
        "--sequence-parallel-size",
        str(config.sequence_parallel_size),
        "--tiled-loss-shards",
        str(config.tiled_loss_shards),
    ]
    _bool_flag(command, config.tiled_mlp, "--tiled-mlp")
    command.extend(config.backend_extra_args)
    return LaunchPlan(
        backend="deepspeed_arctic",
        command=command,
        cwd=repo_root(),
        env=_base_env(),
        output_dir=out,
        generated_files=[ds_path],
        notes=[
            "DeepSpeed ALST/Ulysses requires deepspeed with sequence_parallel. "
            "The script falls back to normal DeepSpeed when sequence_parallel_size=1."
        ],
    )


def build_megatrain(config: HarnessConfig, model: ModelInfo) -> LaunchPlan:
    _reject_true_fraction(config, "megatrain")
    if config.attn_implementation == "flash_attention_3":
        raise ValueError(
            "MegaTrain does not expose FlashAttention 3. "
            "Use --attn-implementation flash_attention_2, sdpa, eager, or auto."
        )
    checkpoint_interval = config.activation_checkpoint_interval
    if checkpoint_interval is None:
        checkpoint_interval = 4
    if checkpoint_interval < 1:
        raise ValueError(f"MegaTrain checkpoint interval must be >= 1, got {checkpoint_interval}")
    out = _output_dir(config, model)
    megatrain_candidates = [
        repo_root() / "baseline" / "MegaTrain",
        repo_root() / "baseline" / "vendor" / "MegaTrain",
    ]
    megatrain_root = next(
        (candidate for candidate in megatrain_candidates if (candidate / "infinity").exists()),
        megatrain_candidates[0],
    )
    script = repo_root() / "baseline" / "backends" / "megatrain" / "train_synthetic.py"
    devices = ",".join(str(i) for i in range(config.num_gpus))
    command = [
        "python",
        str(script),
        "--model-path",
        str(model.path),
        "--seq-length",
        str(config.seq_length),
        "--vocab-size",
        str(model.vocab_size),
        "--batch-size",
        str(config.micro_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--num-steps",
        str(config.num_steps),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--seed",
        str(config.seed),
        "--output-dir",
        str(out),
        "--devices",
        devices,
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--num-grad-slabs",
        str(config.num_grad_slabs),
        "--attn-implementation",
        config.attn_implementation,
        "--moe-kernel-backend",
        config.moe_kernel_backend,
    ]
    command.extend(config.backend_extra_args)
    return LaunchPlan(
        backend="megatrain",
        command=command,
        cwd=repo_root(),
        env=_base_env(megatrain_root),
        output_dir=out,
        notes=["MegaTrain stores and computes in bf16 through its CPUMasterConfig dtype."],
    )


def build_megatron(config: HarnessConfig, model: ModelInfo) -> LaunchPlan:
    out = _output_dir(config, model)
    model_alias = infer_model_alias(model)
    dims_path = _write_json(out / "model_dims.json", {model_alias: model_dims_entry(model)})
    script = config.megatron_script
    if script is None:
        local_script = repo_root() / "baseline" / "backends" / "megatron" / "train.py"
        orig_script = repo_root() / "orig" / "baseline" / "megatron" / "train.py"
        script = local_script if local_script.exists() else orig_script
    command = [
        "torchrun",
        f"--nproc_per_node={config.num_gpus}",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        str(script),
        "--model",
        model_alias,
        "--model-dims",
        str(dims_path),
        "--seq-length",
        str(config.seq_length),
        "--micro-batch-size",
        str(config.micro_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--num-iters",
        str(config.num_steps),
        "--lr",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
    ]
    fraction = config.activation_checkpoint_fraction
    if fraction is not None:
        _validate_fraction(fraction)
        if fraction <= 0.0:
            command.extend(["--recompute-granularity", "none"])
        elif fraction >= 1.0:
            command.extend(["--recompute-granularity", "full"])
            command.extend(["--recompute-method", "uniform"])
            command.extend(["--recompute-num-layers", "1"])
        else:
            num_layers = model.num_layers or 1
            num_recompute_layers = max(1, int(num_layers * fraction))
            command.extend(["--recompute-granularity", "full"])
            command.extend(["--recompute-method", "block"])
            command.extend(["--recompute-num-layers", str(num_recompute_layers)])
    elif config.activation_checkpointing == "none":
        command.extend(["--recompute-granularity", "none"])
    else:
        command.extend(["--recompute-granularity", config.recompute_granularity])
        if config.recompute_granularity == "full":
            recompute_num_layers = config.recompute_num_layers
            if recompute_num_layers is None:
                recompute_num_layers = config.activation_checkpoint_interval or 1
            command.extend(["--recompute-method", config.recompute_method])
            command.extend(["--recompute-num-layers", str(recompute_num_layers)])
        elif config.recompute_modules:
            command.extend(["--recompute-modules", *config.recompute_modules])

    if config.activation_offload == "cpu":
        command.append("--cpu-offloading")
        command.append("--cpu-offloading-activations")
        command.append("--cpu-offloading-weights")
        if config.cpu_offloading_num_layers is not None:
            command.extend(["--cpu-offloading-num-layers", str(config.cpu_offloading_num_layers)])
    elif config.offload_modules:
        command.append("--fine-grained-activation-offloading")
        command.extend(["--offload-modules", *config.offload_modules])

    if config.optimizer_offload != "cpu":
        command.append("--no-optimizer-cpu-offload")
    command.extend(["--optimizer-offload-fraction", "1.0" if config.optimizer_offload == "cpu" else "0.0"])
    command.extend(config.backend_extra_args)
    return LaunchPlan(
        backend="megatron",
        command=command,
        cwd=repo_root(),
        env=_base_env(),
        output_dir=out,
        generated_files=[dims_path],
    )


def build_torchtitan(config: HarnessConfig, model: ModelInfo) -> LaunchPlan:
    out = _output_dir(config, model)
    inferred = infer_torchtitan_target(model)
    module = config.torchtitan_module or (inferred[0] if inferred else None)
    registry_config = config.torchtitan_config or (inferred[1] if inferred else None)
    if module is None or registry_config is None:
        raise ValueError(
            "Could not infer TorchTitan module/config for this model. "
            "Pass --torchtitan-module and --torchtitan-config."
        )
    torchtitan_candidates = [
        repo_root() / "baseline" / "TorchTitan",
        repo_root() / "baseline" / "vendor" / "TorchTitan",
        repo_root() / "baseline" / "torchtitan",
        repo_root() / "orig" / "baseline" / "torchtitan",
    ]
    torchtitan_root = next(
        (candidate for candidate in torchtitan_candidates if (candidate / "torchtitan").exists()),
        torchtitan_candidates[0],
    )

    ac_mode, ac_option = _fractional_mode(config)
    command = [
        "torchrun",
        f"--nproc_per_node={config.num_gpus}",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "--tee",
        "3",
        "-m",
        "torchtitan.train",
        "--module",
        module,
        "--config",
        registry_config,
        "--hf_assets_path",
        str(model.path),
        "--dump_folder",
        str(out),
        "--training.local_batch_size",
        str(config.micro_batch_size),
        "--training.seq_len",
        str(config.seq_length),
        "--training.steps",
        str(config.num_steps),
        "--training.dtype",
        "bfloat16",
        "--training.mixed_precision_param",
        "bfloat16",
        "--training.mixed_precision_reduce",
        "float32",
        "--activation_checkpoint.mode",
        ac_mode,
        "--activation_checkpoint.selective_ac_option",
        ac_option,
        "--parallelism.data_parallel_replicate_degree",
        str(config.fsdp_replicate_degree),
        "--parallelism.data_parallel_shard_degree",
        str(config.fsdp_shard_degree),
        "--parallelism.tensor_parallel_degree",
        str(config.tensor_parallel_size),
        "--parallelism.pipeline_parallel_degree",
        str(config.pipeline_parallel_size),
        "--parallelism.context_parallel_degree",
        str(config.context_parallel_size),
        "--dataloader.vocab_size",
        str(model.vocab_size),
        "--dataloader.seed",
        str(config.seed),
        "--metrics.log_freq",
        "1",
    ]
    _bool_flag(command, config.optimizer_offload == "cpu" or config.param_offload == "cpu", "--training.enable_cpu_offload")
    _bool_flag(command, config.activation_offload == "cpu", "--training.enable_activation_offload")
    command.extend(config.backend_extra_args)
    return LaunchPlan(
        backend="torchtitan",
        command=command,
        cwd=torchtitan_root,
        env=_base_env(torchtitan_root),
        output_dir=out,
        notes=[
            "TorchTitan synthetic registry currently covers built-in TorchTitan model specs; "
            "pass --torchtitan-module/--torchtitan-config for custom specs."
        ],
    )


def build_trl_fsdp(config: HarnessConfig, model: ModelInfo) -> LaunchPlan:
    _reject_true_fraction(config, "trl_fsdp/HF/TRL")
    out = _output_dir(config, model)
    accel_path = _write_yaml(
        out / "accelerate_fsdp.yaml", _accelerate_fsdp_config(config)
    )
    script = repo_root() / "baseline" / "backends" / "trl_fsdp" / "train_synthetic.py"
    checkpointing_mode = _binary_checkpointing_mode(config)
    command = [
        "accelerate",
        "launch",
        "--config_file",
        str(accel_path),
        "--main_process_port",
        str(config.master_port),
        str(script),
        "--model-path",
        str(model.path),
        "--seq-length",
        str(config.seq_length),
        "--vocab-size",
        str(model.vocab_size),
        "--micro-batch-size",
        str(config.micro_batch_size),
        "--gradient-accumulation-steps",
        str(config.gradient_accumulation_steps),
        "--num-steps",
        str(config.num_steps),
        "--learning-rate",
        str(config.learning_rate),
        "--weight-decay",
        str(config.weight_decay),
        "--seed",
        str(config.seed),
        "--output-dir",
        str(out),
        "--attn-implementation",
        config.attn_implementation,
        "--moe-kernel-backend",
        config.moe_kernel_backend,
        "--activation-checkpointing",
        checkpointing_mode,
        "--activation-offload",
        config.activation_offload,
        "--liger-kernel",
        config.liger_kernel,
    ]
    command.extend(config.backend_extra_args)
    notes = [
        "FSDP2 ties param + grad + optimizer-state offload together; "
        "--param-offload cpu and --optimizer-offload cpu both map to "
        "fsdp_offload_params=true.",
    ]
    return LaunchPlan(
        backend="trl_fsdp",
        command=command,
        cwd=repo_root(),
        env=_base_env(),
        output_dir=out,
        generated_files=[accel_path],
        notes=notes,
    )


BUILDERS = {
    "megatrain": build_megatrain,
    "torchtitan": build_torchtitan,
    "trl_deepspeed": build_trl_deepspeed,
    "deepspeed_arctic": build_deepspeed_arctic,
    "megatron": build_megatron,
    "trl_fsdp": build_trl_fsdp,
}


def build_launch(config: HarnessConfig, model: ModelInfo) -> LaunchPlan:
    return BUILDERS[config.backend](config, model)


def run_launch(plan: LaunchPlan, *, dry_run: bool) -> int:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": plan.backend,
        "command": plan.command,
        "cwd": str(plan.cwd),
        "env": plan.env,
        "output_dir": str(plan.output_dir),
        "generated_files": [str(path) for path in plan.generated_files],
        "notes": plan.notes,
    }
    _write_json(plan.output_dir / "launch_plan.json", manifest)

    print(shell_join(plan.command), flush=True)
    if dry_run:
        return 0

    env = os.environ.copy()
    env.update(plan.env)
    log_path = plan.output_dir / "run.log"
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            plan.command,
            cwd=plan.cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.wait()
