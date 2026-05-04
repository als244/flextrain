#!/usr/bin/env python3
"""Unified baseline launcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.baseline_harness.backends import build_launch, run_launch
from baseline.baseline_harness.spec import BackendName, HarnessConfig, load_model_info

BACKENDS: tuple[BackendName, ...] = (
    "megatrain",
    "torchtitan",
    "trl_deepspeed",
    "deepspeed_arctic",
    "megatron",
)


def _comma_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(",") if item]


def _validate_fraction(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise SystemExit(f"{name} must be in [0, 1], got {value}")
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch synthetic-token training across baseline backends."
    )
    parser.add_argument("--backend", choices=[*BACKENDS, "all"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_3", "flash_attention_2", "sdpa", "eager"],
        default="flash_attention_2",
        help="Attention backend for HF-based trainers. Default is strict FlashAttention 2; pass sdpa/eager explicitly to compare fallbacks.",
    )
    parser.add_argument(
        "--moe-kernel-backend",
        choices=["hf", "auto", "sonic"],
        default="hf",
        help="HF sparse-MoE backend. sonic replaces compatible HF MoE blocks with kernels-community/sonic-moe.",
    )

    parser.add_argument(
        "--activation-checkpointing",
        choices=["none", "selective", "full", "memory_budget"],
        default="full",
    )
    parser.add_argument(
        "--activation-checkpoint-interval",
        type=int,
        default=None,
        help=(
            "MegaTrain checkpoint segment size / Megatron fallback recompute layer count. "
            "If omitted, MegaTrain uses its vendor default 4."
        ),
    )
    parser.add_argument(
        "--activation-checkpoint-fraction",
        type=float,
        default=None,
        help="Fraction of decoder layers to activation-checkpoint/recompute where supported.",
    )
    parser.add_argument(
        "--save-activation-layer-fraction",
        type=float,
        default=None,
        help=(
            "Compatibility with orig/baseline DeepSpeed: fraction of layer activations "
            "to save instead of recompute. Converted to 1 - fraction."
        ),
    )
    parser.add_argument("--activation-checkpoint-selective-option", default="op")
    parser.add_argument("--activation-offload", choices=["none", "cpu"], default="none")
    parser.add_argument("--optimizer-offload", choices=["none", "cpu"], default="none")
    parser.add_argument("--param-offload", choices=["none", "cpu"], default="none")
    parser.add_argument("--zero-stage", type=int, choices=[0, 1, 2, 3], default=3)

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel-size", type=int, default=1)
    parser.add_argument("--fsdp-shard-degree", type=int, default=-1)
    parser.add_argument("--fsdp-replicate-degree", type=int, default=1)

    parser.add_argument("--recompute-granularity", choices=["selective", "full"], default="full")
    parser.add_argument("--recompute-method", default="uniform")
    parser.add_argument("--recompute-num-layers", type=int, default=None)
    parser.add_argument("--recompute-modules", default="")
    parser.add_argument("--offload-modules", default="")
    parser.add_argument("--cpu-offloading-num-layers", type=int, default=None)
    parser.add_argument("--num-grad-slabs", type=int, default=12)

    parser.add_argument("--tiled-loss-shards", type=int, default=1)
    parser.add_argument("--tiled-mlp", action="store_true")
    parser.add_argument(
        "--liger-kernel",
        choices=["auto", "on", "off"],
        default="auto",
        help="TRL Liger kernel mode. auto enables it when liger-kernel is installed.",
    )
    parser.add_argument("--use-liger-kernel", action="store_true", help="Legacy alias for --liger-kernel on")
    parser.add_argument("--no-use-liger-kernel", action="store_true", help="Legacy alias for --liger-kernel off")
    parser.add_argument("--torchtitan-module", default=None)
    parser.add_argument("--torchtitan-config", default=None)
    parser.add_argument("--megatron-script", type=Path, default=None)
    parser.add_argument(
        "--backend-extra-arg",
        action="append",
        default=[],
        help="Append one raw argument to the backend command. Repeat for multiple args.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.activation_checkpoint_fraction is not None and args.save_activation_layer_fraction is not None:
        raise SystemExit("Use only one of --activation-checkpoint-fraction or --save-activation-layer-fraction")
    activation_checkpoint_fraction = args.activation_checkpoint_fraction
    if args.save_activation_layer_fraction is not None:
        save_fraction = _validate_fraction("--save-activation-layer-fraction", args.save_activation_layer_fraction)
        activation_checkpoint_fraction = 1.0 - save_fraction
    if activation_checkpoint_fraction is not None:
        activation_checkpoint_fraction = _validate_fraction(
            "--activation-checkpoint-fraction", activation_checkpoint_fraction
        )
    if args.use_liger_kernel and args.no_use_liger_kernel:
        raise SystemExit("Use only one of --use-liger-kernel or --no-use-liger-kernel")
    liger_kernel = args.liger_kernel
    if args.use_liger_kernel:
        liger_kernel = "on"
    if args.no_use_liger_kernel:
        liger_kernel = "off"

    model = load_model_info(args.model_path)
    selected = BACKENDS if args.backend == "all" else (args.backend,)
    rc = 0
    for idx, backend in enumerate(selected):
        output_dir = args.output_dir
        if output_dir is not None and args.backend == "all":
            output_dir = output_dir / backend
        config = HarnessConfig(
            backend=backend,
            model_path=args.model_path,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_steps=args.num_steps,
            num_gpus=args.num_gpus,
            master_port=args.master_port + idx,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            output_dir=output_dir,
            dry_run=args.dry_run,
            attn_implementation=args.attn_implementation,
            moe_kernel_backend=args.moe_kernel_backend,
            activation_checkpointing=args.activation_checkpointing,
            activation_checkpoint_fraction=activation_checkpoint_fraction,
            activation_checkpoint_interval=args.activation_checkpoint_interval,
            activation_checkpoint_selective_option=args.activation_checkpoint_selective_option,
            activation_offload=args.activation_offload,
            optimizer_offload=args.optimizer_offload,
            param_offload=args.param_offload,
            zero_stage=args.zero_stage,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            context_parallel_size=args.context_parallel_size,
            sequence_parallel_size=args.sequence_parallel_size,
            fsdp_shard_degree=args.fsdp_shard_degree,
            fsdp_replicate_degree=args.fsdp_replicate_degree,
            recompute_granularity=args.recompute_granularity,
            recompute_method=args.recompute_method,
            recompute_num_layers=args.recompute_num_layers,
            recompute_modules=_comma_list(args.recompute_modules),
            offload_modules=_comma_list(args.offload_modules),
            cpu_offloading_num_layers=args.cpu_offloading_num_layers,
            num_grad_slabs=args.num_grad_slabs,
            tiled_loss_shards=args.tiled_loss_shards,
            tiled_mlp=args.tiled_mlp,
            liger_kernel=liger_kernel,
            torchtitan_module=args.torchtitan_module,
            torchtitan_config=args.torchtitan_config,
            megatron_script=args.megatron_script,
            backend_extra_args=args.backend_extra_arg,
        )
        try:
            plan = build_launch(config, model)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        step_rc = run_launch(plan, dry_run=args.dry_run)
        if step_rc != 0:
            rc = step_rc
            break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
