#!/usr/bin/env python3
"""DeepSpeed HF/ALST synthetic-token trainer."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baseline.backends.common.attention import (
    load_causal_lm_with_attention,
    pick_attention_implementation,
)
from baseline.backends.common.moe_kernel import apply_moe_kernel_backend
from baseline.backends.common.synthetic import (
    RandomTokenMapDataset,
    SyntheticTokenConfig,
    collate_token_batch,
)

_ALLOC_CONF = "pinned_use_cuda_host_register:True,expandable_segments:True"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("PYTORCH_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepspeed-config", type=Path, required=True)
    parser.add_argument("--attn-implementation", choices=["auto", "flash_attention_3", "flash_attention_2", "sdpa", "eager"], default="flash_attention_2")
    parser.add_argument("--moe-kernel-backend", choices=["hf", "auto", "sonic"], default="hf")
    parser.add_argument("--activation-checkpointing", choices=["none", "selective", "full", "memory_budget"], default="full")
    parser.add_argument("--activation-checkpoint-fraction", type=float, default=None)
    parser.add_argument("--activation-offload", choices=["none", "cpu"], default="none")
    parser.add_argument("--sequence-parallel-size", type=int, default=1)
    parser.add_argument("--tiled-loss-shards", type=int, default=1)
    parser.add_argument("--tiled-mlp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def _maybe_patch_activation_offload(enabled: bool) -> None:
    if not enabled:
        return
    try:
        from arctic_training.monkey_patches import monkey_patch_checkpoint_function_with_cpu_offload
    except ImportError:
        print("activation checkpoint CPU offload requested, but arctic_training is not installed", flush=True)
        return
    monkey_patch_checkpoint_function_with_cpu_offload()


def _maybe_patch_tiled_mlp(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import deepspeed.comm as dist
        from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP
        from transformers.models.llama import modeling_llama
    except ImportError:
        print("tiled MLP requested, but DeepSpeed ALST/Llama hooks are unavailable", flush=True)
        return

    def tiled_mlp_forward_common(self, x):
        bs, seqlen, hidden = x.shape
        num_shards = math.ceil(seqlen / hidden)
        tensor = torch.tensor(num_shards, device=x.device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        num_shards = int(tensor.item())
        compute_params = [self.down_proj.weight, self.gate_proj.weight, self.up_proj.weight]

        def mlp_forward(module, hidden_states):
            return module.down_proj(module.act_fn(module.gate_proj(hidden_states)) * module.up_proj(hidden_states))

        return TiledMLP.apply(mlp_forward, self, x, num_shards, compute_params)

    modeling_llama.LlamaMLP.forward = tiled_mlp_forward_common


def _use_gradient_checkpointing(args: argparse.Namespace) -> bool:
    fraction = args.activation_checkpoint_fraction
    if fraction is None:
        return args.activation_checkpointing != "none"
    if not 0.0 <= fraction <= 1.0:
        raise SystemExit(f"--activation-checkpoint-fraction must be in [0, 1], got {fraction}")
    if 0.0 < fraction < 1.0:
        raise SystemExit(
            "The HuggingFace/DeepSpeed Arctic path does not expose supported fractional "
            "activation checkpointing. Use --activation-checkpointing none/full instead."
        )
    return fraction >= 1.0


def _weighted_sp_loss(model, logits, shift_labels, sp_group, sp_world_size):
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )
    good_tokens = (shift_labels != -100).view(-1).sum()
    losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
    good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
    total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
    total_good_tokens = sum(good_tokens_per_rank)
    return total_loss / torch.clamp(total_good_tokens, min=1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    import deepspeed
    import deepspeed.comm as dist

    _maybe_patch_activation_offload(args.activation_offload == "cpu")
    _maybe_patch_tiled_mlp(args.tiled_mlp)

    with args.deepspeed_config.open() as f:
        ds_config = json.load(f)

    use_sp = args.sequence_parallel_size > 1
    mpu = None
    attn_implementation = pick_attention_implementation(args.attn_implementation)
    print(f"selected_attn_implementation={attn_implementation}", flush=True)
    if use_sp:
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF

        dist.init_distributed(dist_backend="nccl", dist_init_required=True)
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=args.model_path,
            core_attn_implementation=attn_implementation,
            sequence_parallel_size=args.sequence_parallel_size,
            micro_batch_size=args.micro_batch_size,
            seq_length=args.seq_length,
            seq_length_is_variable=False,
        )

    # Critical for ZeRO-3 to work as advertised: instantiate
    # HfDeepSpeedConfig BEFORE from_pretrained. transformers detects
    # this on the global stack and partitions the model across
    # ZeRO-3 ranks (and offloads to CPU when configured) during
    # construction. Without it, from_pretrained materializes the full
    # model on a single rank's GPU first, then deepspeed.initialize
    # tries to partition it after the fact — works for small models,
    # OOMs at 128K context for 8B+.
    # See: https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    zero_stage = int(ds_config.get("zero_optimization", {}).get("stage", 0))
    hf_ds_handle = None
    if zero_stage == 3:
        from transformers.integrations import HfDeepSpeedConfig

        # Bind to a name so it stays alive on the stack until
        # from_pretrained returns; transformers checks
        # is_deepspeed_zero3_enabled() which inspects the most-recent
        # live HfDeepSpeedConfig.
        hf_ds_handle = HfDeepSpeedConfig(ds_config)

    model, _ = load_causal_lm_with_attention(
        args.model_path,
        attn_implementation,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Keep the handle live until after the model is fully constructed.
    del hf_ds_handle
    model.config.use_cache = False
    apply_moe_kernel_backend(model, args.moe_kernel_backend)
    if _use_gradient_checkpointing(args) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    num_samples = max(args.num_steps * args.gradient_accumulation_steps * args.micro_batch_size * 2, 8)
    dataset = RandomTokenMapDataset(
        SyntheticTokenConfig(
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            num_samples=num_samples,
            seed=args.seed,
            label_mode="self",
            include_position_ids=True,
        )
    )
    dataloader = DataLoader(dataset, batch_size=args.micro_batch_size, collate_fn=collate_token_batch)

    model_engine, _, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters(),
        mpu=mpu,
    )

    if use_sp:
        from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPDataLoaderAdapter
        from deepspeed.runtime.utils import move_to_device
        from deepspeed.utils import groups

        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()
        sp_rank = groups._get_sequence_parallel_rank()
        dataloader = UlyssesSPDataLoaderAdapter(
            dataloader,
            sp_rank=sp_rank,
            sp_group=sp_group,
            sp_world_size=sp_world_size,
            device=model_engine.device,
        )
    else:
        move_to_device = None
        sp_group = None
        sp_world_size = 1

    for step, batch in enumerate(dataloader):
        if step >= args.num_steps:
            break
        step_start = time.perf_counter()
        if use_sp:
            batch = move_to_device(batch, model_engine.device)
            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                position_ids=batch.get("position_ids"),
                use_cache=False,
            )
            loss = _weighted_sp_loss(model_engine, outputs.logits, batch["shift_labels"], sp_group, sp_world_size)
        else:
            batch = {key: value.to(model_engine.device) for key, value in batch.items()}
            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(model_engine.device)
        is_initialized = getattr(dist, "is_initialized", lambda: True)
        rank = dist.get_rank() if is_initialized() else 0
        if rank == 0:
            world_size = dist.get_world_size() if is_initialized() else 1
            dp_world_size = max(1, world_size // max(1, args.sequence_parallel_size))
            tokens_this_step = args.micro_batch_size * args.seq_length * dp_world_size
            step_time = max(time.perf_counter() - step_start, 1e-6)
            print(
                f"step={step + 1} loss={float(loss.detach()):.6f} "
                f"step_time_s={step_time:.3f} tokens={tokens_this_step} tokens_per_s={tokens_this_step / step_time:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
