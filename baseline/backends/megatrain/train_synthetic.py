#!/usr/bin/env python3
"""MegaTrain CPUMaster synthetic-token trainer."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import psutil
import torch
from torch.utils.data import DataLoader

from baseline.backends.common.attention import (
    attention_candidates,
    load_causal_lm_with_attention,
)
from baseline.backends.common.moe_kernel import apply_moe_kernel_backend
from baseline.backends.common.synthetic import (
    RandomTokenMapDataset,
    SyntheticTokenConfig,
    collate_token_batch,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("megatrain_synthetic")

MEGATRAIN_ATTENTION_CHOICES = ("flash_attention_2", "sdpa", "eager")


def pick_megatrain_attention_implementation(requested: str) -> str:
    """Resolve attention without selecting implementations MegaTrain rejects."""
    if requested == "auto":
        for candidate in attention_candidates(requested, allow_flash=True):
            if candidate in MEGATRAIN_ATTENTION_CHOICES:
                return candidate
        raise SystemExit("MegaTrain could not resolve a supported attention implementation")
    if requested not in MEGATRAIN_ATTENTION_CHOICES:
        raise SystemExit(
            "MegaTrain supports attn_implementation values "
            f"{MEGATRAIN_ATTENTION_CHOICES}; got {requested!r}"
        )
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--num-grad-slabs", type=int, default=12)
    parser.add_argument("--attn-implementation", choices=["auto", *MEGATRAIN_ATTENTION_CHOICES], default="auto")
    parser.add_argument(
        "--moe-kernel-backend",
        choices=["hf", "auto", "sonic"],
        default="hf",
        help=(
            "MoE experts kernel: 'hf' uses the model's default; 'auto' tries "
            "sonic-moe (HF native via model.set_experts_implementation) and "
            "falls back to default; 'sonic' is strict."
        ),
    )
    # Default to deepspeed_cpu_adam: MegaTrain is a CPU-master/offload
    # backend, so a CPU-resident optimizer is the right default to keep
    # the memory invariant (params, grads, opt state all in host RAM).
    # The plain ``adamw`` choice stays available for parity testing.
    parser.add_argument("--optimizer", choices=["adamw", "deepspeed_cpu_adam"], default="deepspeed_cpu_adam")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from infinity import CPUMasterModel
    from infinity.config import CPUMasterConfig

    devices = [int(item) for item in args.devices.split(",") if item != ""]
    attn_implementation = pick_megatrain_attention_implementation(args.attn_implementation)
    print(f"selected_attn_implementation={attn_implementation}", flush=True)
    config = CPUMasterConfig(
        model_name=args.model_path,
        dataset_path="__synthetic__",
        max_seq_len=args.seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_interval=args.checkpoint_interval,
        num_grad_slabs=args.num_grad_slabs,
        devices=devices,
        dtype=torch.bfloat16,
        seed=args.seed,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        log_interval=1,
    )

    logger.info("Loading HF model on CPU in bf16")
    hf_model, _ = load_causal_lm_with_attention(
        args.model_path,
        attn_implementation,
        allow_flash=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    hf_model.config.use_cache = False
    apply_moe_kernel_backend(hf_model, args.moe_kernel_backend)
    model = CPUMasterModel(hf_model, config)
    del hf_model

    if args.optimizer == "deepspeed_cpu_adam":
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except ImportError as exc:
            raise SystemExit("DeepSpeedCPUAdam requested but deepspeed is not installed") from exc
        optimizer = DeepSpeedCPUAdam(
            model.get_parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            adamw_mode=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.get_parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    num_samples = max(args.num_steps * args.batch_size * 2, 8)
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_token_batch)
    data_iter = iter(dataloader)

    process = psutil.Process()
    step_times = []
    tflops_values = []
    cpu_mems = []

    for dev_id in devices:
        torch.cuda.reset_peak_memory_stats(dev_id)

    for step in range(args.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        start = time.perf_counter()
        loss_val, n_tokens, timing = model.forward_and_backward(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.get_parameters(), config.max_grad_norm)
            optimizer.step()
            model._sync_params_to_gpu()
            model.zero_grad()
            optimizer.zero_grad()

        step_time = time.perf_counter() - start
        num_params = sum(p.numel() for p in model.get_parameters())
        flops = 6 * num_params * n_tokens
        tflops = (flops / 1e12) / max(step_time, 1e-9)
        tokens_per_s = n_tokens / max(step_time, 1e-9)
        cpu_mem = process.memory_info().rss / 1024**3
        step_times.append(step_time)
        tflops_values.append(tflops)
        cpu_mems.append(cpu_mem)
        logger.info(
            "step=%d/%d loss=%.6f time=%.3fs fwd=%.3fs bwd=%.3fs tokens_per_s=%.1f tflops=%.2f cpu=%.2fGB",
            step + 1,
            args.num_steps,
            loss_val,
            step_time,
            timing["forward"],
            timing["backward"],
            tokens_per_s,
            tflops,
            cpu_mem,
        )

    peak_gpu = max(torch.cuda.max_memory_allocated(d) for d in devices) / 1024**3
    logger.info("avg_step_time=%.3fs avg_tflops=%.2f peak_gpu=%.2fGB peak_cpu=%.2fGB",
                sum(step_times) / len(step_times), sum(tflops_values) / len(tflops_values), peak_gpu, max(cpu_mems))
    model.cleanup()


if __name__ == "__main__":
    main()
