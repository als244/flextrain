#!/usr/bin/env python3
"""TRL SFTTrainer over synthetic random token IDs with FSDP2 (via accelerate)."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer, TrainerCallback

from baseline.backends.common.attention import load_causal_lm_with_attention
from baseline.backends.common.moe_kernel import apply_moe_kernel_backend
from baseline.backends.common.synthetic import (
    RandomTokenMapDataset,
    SyntheticTokenConfig,
    collate_token_batch,
)

_ALLOC_CONF = "pinned_use_cuda_host_register:True,expandable_segments:True"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("PYTORCH_ALLOC_CONF", _ALLOC_CONF)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_3", "flash_attention_2", "sdpa", "eager"],
        default="auto",
    )
    parser.add_argument("--moe-kernel-backend", choices=["hf", "auto", "sonic"], default="hf")
    parser.add_argument(
        "--activation-checkpointing",
        choices=["none", "selective", "full", "memory_budget"],
        default="full",
    )
    parser.add_argument("--activation-checkpoint-fraction", type=float, default=None)
    parser.add_argument("--activation-offload", choices=["none", "cpu"], default="none")
    parser.add_argument("--liger-kernel", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--use-liger-kernel", action="store_true", help="Legacy alias for --liger-kernel on")
    parser.add_argument("--no-use-liger-kernel", action="store_true", help="Legacy alias for --liger-kernel off")
    return parser.parse_args()


def _filtered_kwargs(cls, kwargs: dict) -> dict:
    accepted = set(inspect.signature(cls).parameters)
    return {key: value for key, value in kwargs.items() if key in accepted}


def _use_gradient_checkpointing(args: argparse.Namespace) -> bool:
    fraction = args.activation_checkpoint_fraction
    if fraction is None:
        return args.activation_checkpointing != "none"
    if not 0.0 <= fraction <= 1.0:
        raise SystemExit(f"--activation-checkpoint-fraction must be in [0, 1], got {fraction}")
    if 0.0 < fraction < 1.0:
        raise SystemExit(
            "TRL/HuggingFace does not expose supported fractional activation checkpointing. "
            "Use --activation-checkpointing none/full instead."
        )
    return fraction >= 1.0


def _resolve_liger_kernel(args: argparse.Namespace, sft_config_cls) -> bool:
    if args.use_liger_kernel and args.no_use_liger_kernel:
        raise SystemExit("Use only one of --use-liger-kernel or --no-use-liger-kernel")

    mode = args.liger_kernel
    if args.use_liger_kernel:
        mode = "on"
    if args.no_use_liger_kernel:
        mode = "off"

    if mode == "off":
        print("resolved_liger_kernel=False mode=off", flush=True)
        return False

    available = importlib.util.find_spec("liger_kernel") is not None
    sft_supports_liger = "use_liger_kernel" in inspect.signature(sft_config_cls).parameters
    if mode == "on" and not available:
        raise SystemExit("--liger-kernel on requested, but liger-kernel is not installed")
    if mode == "on" and not sft_supports_liger:
        raise SystemExit("--liger-kernel on requested, but this TRL SFTConfig lacks use_liger_kernel")

    enabled = available and sft_supports_liger
    print(
        f"resolved_liger_kernel={enabled} mode={mode} "
        f"available={available} trl_support={sft_supports_liger}",
        flush=True,
    )
    return enabled


class _ThroughputCallback(TrainerCallback):
    def __init__(self, *, tokens_per_step: int):
        self.tokens_per_step = tokens_per_step
        self._last_step = 0
        self._step_start = None
        self._rank = int(os.environ.get("RANK", "0"))

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._rank != 0 or state.global_step <= self._last_step:
            return
        elapsed = max(time.perf_counter() - (self._step_start or time.perf_counter()), 1e-6)
        print(
            f"step={state.global_step} step_time_s={elapsed:.3f} "
            f"tokens={self.tokens_per_step} tokens_per_s={self.tokens_per_step / elapsed:.1f}",
            flush=True,
        )
        self._last_step = state.global_step


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "TRL is not installed. Install the FSDP baseline environment first, "
            "then rerun this launcher."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _ = load_causal_lm_with_attention(
        args.model_path,
        args.attn_implementation,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    apply_moe_kernel_backend(model, args.moe_kernel_backend)
    # NOTE: under accelerate's FSDP2 plugin we let
    # ``fsdp_activation_checkpointing`` (set in
    # baseline_harness.backends._accelerate_fsdp_config) own
    # checkpointing — HF Trainer raises
    # "The activation_checkpointing in FSDP config and the
    # gradient_checkpointing in training arg can't be set to True
    # simultaneously" when both are on. We therefore do NOT call
    # model.gradient_checkpointing_enable() and pass
    # gradient_checkpointing=False to SFTConfig below. The FSDP plugin
    # handles activation checkpointing on the wrapped layers directly.
    # ``--activation-checkpointing none`` skips both paths uniformly.
    # Validate fractional-ckpt args even though we don't use the
    # return value — _use_gradient_checkpointing raises SystemExit on
    # unsupported 0 < f < 1 values, surfacing the "use full/none for
    # FSDP" error early rather than mid-training.
    _use_gradient_checkpointing(args)

    num_samples = max(args.num_steps * args.gradient_accumulation_steps * args.micro_batch_size * 2, 8)
    dataset = RandomTokenMapDataset(
        SyntheticTokenConfig(
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            num_samples=num_samples,
            seed=args.seed,
            label_mode="self",
            include_position_ids=False,
        )
    )

    use_liger_kernel = _resolve_liger_kernel(args, SFTConfig)
    # FSDP plugin is configured externally by ``accelerate launch
    # --config_file`` (see baseline_harness.backends._accelerate_fsdp_config).
    # SFTConfig itself needs ``bf16=False, fp16=False`` here:
    # SFTConfig(bf16=True) makes HF Trainer construct its own
    # Accelerator with mixed_precision="bf16", which OVERRIDES the YAML's
    # ``mixed_precision: "no"`` and re-enables the FSDP fp32 upcast we
    # are explicitly disabling. With both mixed_precision knobs set to
    # "no" *and* the model loaded with ``torch_dtype=torch.bfloat16``,
    # FSDP2 keeps the shards bf16 throughout — params, grads, opt
    # state — without any fp32 master copy.
    sft_kwargs = _filtered_kwargs(
        SFTConfig,
        {
            "output_dir": str(args.output_dir),
            "overwrite_output_dir": True,
            "per_device_train_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_steps": args.num_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            # Intentional: see comment above. ``bf16=True`` here would
            # cause HF Trainer to override the YAML's
            # ``mixed_precision: "no"`` and re-trigger the fp32 upcast
            # warning ("FSDP upcast of low precision parameters to fp32
            # ...") that blew the 80 GiB budget on Llama3-8B @ 128K.
            "bf16": False,
            "fp16": False,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "remove_unused_columns": False,
            # FSDP plugin owns activation checkpointing — see comment
            # above. Setting gradient_checkpointing=True here would
            # collide with fsdp_activation_checkpointing.
            "gradient_checkpointing": False,
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "max_length": args.seq_length,
            "packing": False,
            "use_liger_kernel": use_liger_kernel,
            "activation_offloading": args.activation_offload == "cpu",
        },
    )

    trainer_kwargs = {
        "model": model,
        "args": SFTConfig(**sft_kwargs),
        "train_dataset": dataset,
        "data_collator": collate_token_batch,
    }
    trainer_params = set(inspect.signature(SFTTrainer).parameters)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    trainer.add_callback(
        _ThroughputCallback(
            tokens_per_step=args.micro_batch_size
            * args.gradient_accumulation_steps
            * args.seq_length
            * max(1, world_size)
        )
    )
    trainer.train()


if __name__ == "__main__":
    main()
