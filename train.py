from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from flextrain import from_pretrained
from flextrain.config import IOConfig
from flextrain.io.download import (
    download_dataset,
    download_model,
    hf_checkpoint_is_complete,
)
from flextrain.io.hf_weights import select_arch
from flextrain.io.sources import JsonSFTTokenSource, SyntheticTokenSource
from flextrain.optim.adamw import AdamW, AdamWHyperparams
from flextrain.optim.hybrid import HybridMuonAdamW, HybridMuonAdamWHyperparams
from flextrain.optim.muon import MuonHyperparams


def _maybe_download_hf_snapshot(io_cfg: IOConfig) -> None:
    """If ``io_cfg`` points at a missing/incomplete local snapshot AND we
    know the source repo, download it. Compute-node-only deployments
    should pre-fetch via ``download.py model ...`` so this path is a
    no-op."""
    local_path = io_cfg.hf_checkpoint
    repo_id = io_cfg.hf_repo_id
    if not local_path or not repo_id:
        return
    if hf_checkpoint_is_complete(local_path):
        return
    download_model(repo_id, local_path, revision=io_cfg.hf_revision)


def _resolve_model(model_arg: str) -> tuple[str, str | None]:
    if os.path.exists(model_arg):
        return model_arg, None
    if "/" in model_arg:
        repo_tail = model_arg.rstrip("/").split("/")[-1]
        return os.path.join("models", repo_tail), model_arg
    return os.path.join("models", model_arg), None


def _resolve_dataset(dataset_arg: str) -> str:
    """Mirror the air-gap-friendly path: if ``dataset_arg`` is already a
    local file just use it; otherwise route through the shared download
    helper, which writes to ``datasets/<name>.jsonl``. To pre-stage on a
    login node use ``download.py dataset <spec> --target ...``.
    """
    if os.path.isfile(dataset_arg):
        return os.path.abspath(dataset_arg)
    cache_dir = Path("datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dataset_arg.startswith(("http://", "https://")):
        filename = dataset_arg.rstrip("/").split("/")[-1] or "dataset.jsonl"
    else:
        dataset_name = dataset_arg.rstrip("/").split("/")[-1] or "dataset"
        filename = f"{dataset_name}.jsonl"
    return download_dataset(dataset_arg, cache_dir / filename)


def _load_hf_config_json(model_dir: str) -> dict[str, Any]:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"missing {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def _arch_module_for(hf_config: dict[str, Any]):
    archs = hf_config.get("architectures") or []
    arch_id = archs[0] if archs else ""
    overrides = {
        "LlamaForCausalLM": "llama",
        "MistralForCausalLM": "mistral",
        "Qwen2ForCausalLM": "qwen2",
        "Qwen3ForCausalLM": "qwen3",
        "Qwen3_5ForCausalLM": "qwen3_5",
        "Qwen3_5ForConditionalGeneration": "qwen3_5",
        "Qwen3MoeForCausalLM": "qwen3_moe",
        "Qwen3NextForCausalLM": "qwen3_next",
        "Qwen3_5MoeForCausalLM": "qwen3_5_moe",
        "Qwen3_5MoeForConditionalGeneration": "qwen3_5_moe",
        "OlmoeForCausalLM": "olmoe",
        "Gemma2ForCausalLM": "gemma2",
        "Gemma3ForConditionalGeneration": "gemma3",
    }
    if arch_id in overrides:
        mod_name = overrides[arch_id]
    else:
        s = arch_id.replace("ForCausalLM", "").replace("Model", "").lower()
        out = []
        for i, c in enumerate(s):
            if c.isupper() and i > 0:
                out.append("_")
            out.append(c.lower())
        mod_name = "".join(out)
    return importlib.import_module(f"flextrain.io.arch.{mod_name}")


def _get_model_flops_per_token(model_cfg, seq_len: int, *, using_lora: bool = False) -> int:
    d_model = model_cfg.d_model
    n_heads = model_cfg.n_heads
    head_dim = model_cfg.head_dim
    n_kv_heads = model_cfg.n_kv_heads
    expert_dim = model_cfg.expert_dim
    vocab = model_cfg.vocab_size
    n_layers = model_cfg.n_layers
    top_k = getattr(model_cfg, "top_k", 0)
    num_shared = getattr(model_cfg, "num_shared_experts", 0)
    is_causal = getattr(model_cfg, "is_causal", True)
    matmul_factor = 4 if using_lora else 6

    ctx_dim = n_kv_heads * head_dim
    attn_dim = n_heads * head_dim
    # MoE is gated on top_k > 0 (number of routed experts per token). The
    # arch registry sets num_shared_experts=1 for plain dense MLPs — that
    # is one shared MLP, not a shared expert that stacks on top of routed
    # ones. Conflating the two double-counts the dense MLP FLOPs.
    is_moe = top_k > 0
    mlp_experts = num_shared + top_k if is_moe else 1
    active_params_per_layer = (
        2 * d_model * attn_dim
        + 2 * d_model * ctx_dim
        + 3 * mlp_experts * d_model * expert_dim
    )
    matmul_flops_per_layer = matmul_factor * seq_len * active_params_per_layer
    attn_factor = 0.5 if is_causal else 1.0
    attn_flops_per_layer = 12 * attn_factor * seq_len * seq_len * attn_dim
    backbone_flops = n_layers * (matmul_flops_per_layer + attn_flops_per_layer)
    head_flops = matmul_factor * seq_len * d_model * vocab
    return backbone_flops + head_flops


def _lr_schedule(
    step: int, *,
    max_lr: float, final_lr: float,
    warmup_steps: int, cooldown_start: int, total_steps: int,
) -> float:
    import math
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step < cooldown_start:
        return max_lr
    pct = (step - cooldown_start) / max(1, total_steps - cooldown_start)
    pct = min(1.0, max(0.0, pct))
    return final_lr + 0.5 * (max_lr - final_lr) * (1.0 + math.cos(math.pi * pct))


def _run_training_loop(am, source, *, model_cfg, args, output_dir: str, save_arch=None) -> int:
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Starting training loop: steps={args.steps} "
        f"max_seq_len={args.seq_len} "
        f"global_batch_tokens={args.global_batch_tokens}",
        flush=True,
    )
    print(
        "First step may take a while due to weight staging, kernel warmup, "
        "and the initial forward/backward pass.",
        flush=True,
    )

    total_steps = args.steps
    warmup_steps = int(total_steps * args.lr_warmup_pct)
    cooldown_start = int(total_steps * args.lr_cooldown_start_pct)
    max_lr = am.optimizer.hp.lr
    final_lr = max_lr * args.lr_final_pct

    # Profiler boundaries: nsys' --capture-range=cudaProfilerApi waits
    # for cudaProfilerStart and stops on cudaProfilerStop, so we wrap a
    # caller-chosen window of steps. Default window = 3 steps.
    profile_start_step = args.profile_start_step
    profile_stop_step = (
        args.profile_stop_step
        if args.profile_stop_step is not None
        else (profile_start_step + 2 if profile_start_step is not None else None)
    )
    profiler_running = False

    train_start = time.time()
    smoothed_loss = None
    smooth_decay = 0.95
    total_tokens_processed = 0
    total_flops_processed = 0
    for step in range(1, total_steps + 1):
        lr = _lr_schedule(
            step, max_lr=max_lr, final_lr=final_lr,
            warmup_steps=warmup_steps, cooldown_start=cooldown_start,
            total_steps=total_steps,
        )
        am.optimizer.hp = type(am.optimizer.hp)(
            **{**am.optimizer.hp.__dict__, "lr": lr}
        )

        seqs = source.get_sequences(max_token_count=args.global_batch_tokens)
        if not seqs:
            print(f"[step {step}] data source exhausted")
            break

        if profile_start_step is not None and step == profile_start_step:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            profiler_running = True
            print(f"[profile] cudaProfilerStart at step {step}", flush=True)

        if profiler_running:
            torch.cuda.nvtx.range_push(f"step {step}")

        step_tokens = sum(len(s) for s in seqs)
        step_start = time.time()
        stats = am.fwd_bwd(
            seqs,
            loss_scale_factor=1.0 / step_tokens,
            total_tokens_per_step=step_tokens,
        )
        am.step()
        torch.cuda.synchronize()
        step_time = time.time() - step_start

        if profiler_running:
            torch.cuda.nvtx.range_pop()
        if profile_stop_step is not None and step == profile_stop_step and profiler_running:
            torch.cuda.profiler.stop()
            profiler_running = False
            print(f"[profile] cudaProfilerStop after step {step}", flush=True)

        step_flops = sum(
            _get_model_flops_per_token(
                model_cfg,
                len(s),
                using_lora=(args.mode == "lora"),
            )
            for s in seqs
        )
        total_tokens_processed += step_tokens
        total_flops_processed += step_flops

        avg_loss = stats.total_loss / stats.total_tokens
        smoothed_loss = (
            avg_loss if smoothed_loss is None
            else smooth_decay * smoothed_loss + (1 - smooth_decay) * avg_loss
        )
        tokens_per_sec = step_tokens / step_time
        tflops_per_sec = (step_flops / step_time) / 1e12
        max_alloc = torch.cuda.max_memory_allocated() / (1 << 30)
        max_reserve = torch.cuda.max_memory_reserved() / (1 << 30)

        print(
            f"[step {step:5d}/{total_steps}]  "
            f"lr={lr:.2e}  loss={avg_loss:.4f}  "
            f"smoothed={smoothed_loss:.4f}  "
            f"tok/step={step_tokens}  "
            f"tok/s={tokens_per_sec:,.0f}  "
            f"TFLOPS={tflops_per_sec:.2f}  "
            f"max_alloc={max_alloc:.1f}GiB  "
            f"max_reserve={max_reserve:.1f}GiB  "
            f"step={step_time*1000:.0f}ms  "
            f"elapsed={(time.time()-train_start):.1f}s",
            flush=True,
        )

    # Close the profiler if the loop exited (data exhausted, KeyboardInterrupt
    # on next iter, etc.) before reaching the stop step.
    if profiler_running:
        torch.cuda.profiler.stop()
        profiler_running = False
        print("[profile] cudaProfilerStop at training end", flush=True)

    total_time = time.time() - train_start
    overall_tok_per_s = total_tokens_processed / total_time if total_time > 0 else 0
    overall_tflops = (total_flops_processed / total_time) / 1e12 if total_time > 0 else 0
    print(
        f"\nTraining complete.\n"
        f"  total time:          {total_time / 60:.1f} min\n"
        f"  total tokens:        {total_tokens_processed / 1e6:.2f} M\n"
        f"  overall tok/s:       {overall_tok_per_s:,.0f}\n"
        f"  overall TFLOPS:      {overall_tflops:.2f}\n"
        f"  max alloc / reserve: "
        f"{torch.cuda.max_memory_allocated() / (1 << 30):.2f}GiB / "
        f"{torch.cuda.max_memory_reserved() / (1 << 30):.2f}GiB"
    )

    if args.save:
        try:
            save_kwargs = {"out_filename": "final.safetensors"}
            if save_arch is not None:
                save_kwargs["arch"] = save_arch
            path = am.save_hf(output_dir, **save_kwargs)
            print(f"Saved final weights to {path}")
        except Exception as e:
            print(f"[warn] save_hf failed: {e}")
    else:
        print("Skipped final checkpoint export (--save not set)")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train.py")
    p.add_argument(
        "--model",
        required=True,
        help="Local model dir or HF repo id. Missing local repo ids download into models/.",
    )
    p.add_argument(
        "--mode",
        choices=("full", "lora"),
        required=True,
        help="Full fine-tuning or LoRA fine-tuning.",
    )
    p.add_argument("--seq-len", type=int, required=True, help="Maximum training sequence length.")
    p.add_argument("--global-batch-tokens", type=int, required=True, help="Tokens per optimizer step.")
    p.add_argument("--steps", type=int, default=20, help="Number of training steps to run. Default: 20.")
    p.add_argument(
        "--data-source",
        choices=("json_sft", "synthetic"),
        default="json_sft",
        help="Training data source. Default: json_sft.",
    )
    p.add_argument(
        "--dataset",
        default="flextrain/configs/examples/data/tiny_math_sft.json",
        help=(
            "Local JSON/JSONL SFT dataset. If the file is missing, FlexTrain "
            "tries to download/materialize it first from the dataset spec."
        ),
    )
    p.add_argument(
        "--synthetic-seq-len",
        type=int,
        default=None,
        help="Sequence length for synthetic-token training. Defaults to --seq-len.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Run output directory. Default: runs/<model>_<mode>_sl<seq_len>",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override peak (max) learning rate. Defaults: 3e-5 for full, 1e-4 for lora, 1e-3 for --use-muon.",
    )
    p.add_argument(
        "--lr-warmup-pct",
        type=float,
        default=0.1,
        help="Fraction of total steps used for linear warmup from 0 to peak LR. Default: 0.1.",
    )
    p.add_argument(
        "--lr-cooldown-start-pct",
        type=float,
        default=0.8,
        help="Fraction of total steps at which cosine cooldown begins. Default: 0.8.",
    )
    p.add_argument(
        "--lr-final-pct",
        type=float,
        default=0.1,
        help="Final LR as a fraction of peak LR after cooldown. Default: 0.1.",
    )
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank when --mode lora. Default: 16.")
    p.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha when --mode lora. Default: 16.0.")
    p.add_argument(
        "--use-muon",
        action="store_true",
        help=(
            "Use HybridMuonAdamW for --mode full: Muon updates 2-D dense "
            "projections (Q/K/V/O, MLP up/gate/down), AdamW updates 1-D / "
            "norms / routers / embeddings / head. Incompatible with --mode lora."
        ),
    )
    p.add_argument("--device-id", type=int, default=0, help="CUDA device id. Default: 0.")
    p.add_argument(
        "--max-gpu-mem-gib", type=float, default=None,
        help=(
            "GPU memory budget in GiB. Default: auto-discovered via "
            "torch.cuda.mem_get_info (with nvidia-smi / ROCm fallbacks)."
        ),
    )
    p.add_argument(
        "--max-host-mem-gib", type=float, default=None,
        help=(
            "Host memory budget in GiB. Default: auto-discovered, "
            "respecting Slurm allocation / cgroup v1+v2 limits / psutil "
            "available."
        ),
    )
    p.add_argument("--leeway-gpu-mem-gib", type=float, default=2.0, help="Reserved GPU slack in GiB. Default: 2.0.")
    p.add_argument("--leeway-host-mem-gib", type=float, default=10.0, help="Reserved host-memory slack in GiB. Default: 10.0.")
    p.add_argument(
        "--profile-start-step",
        type=int,
        default=None,
        help=(
            "Call cudaProfilerStart() right before this 1-indexed step "
            "begins. Pair with `nsys profile --capture-range=cudaProfilerApi "
            "--capture-range-end=stop` to skip warmup and capture only the "
            "steps you care about. Each step is also wrapped in an NVTX "
            "range so the timeline groups cleanly."
        ),
    )
    p.add_argument(
        "--profile-stop-step",
        type=int,
        default=None,
        help=(
            "1-indexed step after which to call cudaProfilerStop(). "
            "Default: --profile-start-step + 2 (i.e. capture 3 steps). "
            "Ignored if --profile-start-step isn't set."
        ),
    )
    p.add_argument(
        "--force-save-level",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help=(
            "Debug: force every host-resident (layer, chunk) pair to this "
            "save tier instead of letting the DP solver choose. Tier is "
            "clamped to each layer's max. The on-device tail keeps -1."
        ),
    )
    p.add_argument(
        "--min-chunk-size",
        type=int,
        default=None,
        help=(
            "Lower bound (in tokens) on the chunk size the working-set "
            "solver may pick. Overrides the arithmetic-intensity floor "
            "the solver computes from sustained TFLOPS / mem-bw. Useful "
            "for short-input forwards / debugging."
        ),
    )
    p.add_argument(
        "--max-chunk-size",
        type=int,
        default=None,
        help=(
            "Upper bound (in tokens) on the chunk size the working-set "
            "solver may pick. Useful when the auto-pick is too large for "
            "the host activation buffer or workspace, or when you want to "
            "force a specific tile shape for profiling."
        ),
    )
    p.add_argument("--save", action="store_true", help="Export final.safetensors at the end of the run.")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    local_model_dir, hf_repo_id = _resolve_model(args.model)
    output_dir = args.output_dir or (
        f"runs/{Path(local_model_dir).name}_{args.mode}_sl{args.seq_len}"
    )
    io_cfg = IOConfig(
        hf_checkpoint=local_model_dir,
        hf_repo_id=hf_repo_id,
        tokenizer=local_model_dir,
        output_dir=output_dir,
        save_final_checkpoint=args.save,
    )
    _maybe_download_hf_snapshot(io_cfg)

    hf_config = _load_hf_config_json(local_model_dir)
    arch = select_arch(hf_config)
    arch_module = _arch_module_for(hf_config)
    dims = dict(arch_module.hf_config_to_flextrain(hf_config))
    model_cfg = SimpleNamespace(**dims)

    if args.mode == "lora":
        if args.use_muon:
            raise SystemExit(
                "--use-muon is for full fine-tuning only; LoRA always uses AdamW "
                "on the small adapter parameters."
            )
        optimizer = AdamW(
            AdamWHyperparams(
                lr=args.lr or 1e-4,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
                weight_decay=0.0,
            ),
            state_dtype=torch.float32,
        )
        lora_targets = "all"
    elif args.use_muon:
        # HybridMuonAdamW dispatches per-tensor: Muon for 2-D dense
        # projections (Q/K/V/O, MLP up/gate/down), AdamW for 1-D / norms /
        # routers / embeddings / head. Both share LR by default.
        lr = args.lr or 1e-3
        optimizer = HybridMuonAdamW(
            HybridMuonAdamWHyperparams(
                lr=lr,
                adamw=AdamWHyperparams(
                    lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
                ),
                muon=MuonHyperparams(lr=lr),
            ),
        )
        lora_targets = None
    else:
        optimizer = AdamW(
            AdamWHyperparams(
                lr=args.lr or 3e-5,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
                weight_decay=0.0,
            ),
            state_dtype=torch.bfloat16,
        )
        lora_targets = None

    # Auto-discover memory budgets when the user doesn't override them.
    # Falls back to a conservative 24 / 110 GiB if discovery fails (no
    # CUDA, no psutil, etc.) so train.py still launches in dev sandboxes.
    from flextrain.core._memory import (
        get_available_gpu_memory,
        get_available_host_memory,
    )
    if args.max_gpu_mem_gib is not None:
        max_gpu_mem_bytes = int(args.max_gpu_mem_gib * (1 << 30))
        gpu_src = "user"
    else:
        discovered = get_available_gpu_memory(args.device_id)
        max_gpu_mem_bytes = discovered if discovered > 0 else int(24 * (1 << 30))
        gpu_src = "auto" if discovered > 0 else "fallback"
    if args.max_host_mem_gib is not None:
        max_host_mem_bytes = int(args.max_host_mem_gib * (1 << 30))
        host_src = "user"
    else:
        discovered = get_available_host_memory()
        max_host_mem_bytes = discovered if discovered > 0 else int(110 * (1 << 30))
        host_src = "auto" if discovered > 0 else "fallback"
    print(
        f"Memory budgets: gpu={max_gpu_mem_bytes / (1 << 30):.1f}GiB ({gpu_src}), "
        f"host={max_host_mem_bytes / (1 << 30):.1f}GiB ({host_src})",
        flush=True,
    )
    print(
        f"Train config: model={local_model_dir} mode={args.mode} "
        f"seq_len={args.seq_len} steps={args.steps} "
        f"batch_tokens={args.global_batch_tokens}",
        flush=True,
    )
    device_str = f"cuda:{args.device_id}"
    print(
        "Probing hardware (sustained matmul + memory bandwidth + PCIe; "
        "~14s)...",
        flush=True,
    )
    from flextrain.core.hw_probe import probe_hardware
    probe = probe_hardware(device=device_str)
    throttle_drop = (
        probe.achieved_tflops_first_half - probe.achieved_tflops_second_half
    )
    throttle_pct = (
        100.0 * throttle_drop / probe.achieved_tflops_first_half
        if probe.achieved_tflops_first_half > 0 else 0.0
    )
    print(
        f"[HW Probe] sustained_tflops={probe.hw_cost.peak_tflops:.1f} "
        f"(first half {probe.achieved_tflops_first_half:.1f} -> "
        f"second half {probe.achieved_tflops_second_half:.1f}, "
        f"throttle drop {throttle_pct:.1f}%), "
        f"pcie_bw_gbps={probe.hw_cost.pcie_bw_gbps:.1f}, "
        f"mem_bw_gbps={probe.mem_bw_gbps:.1f} "
        f"[matmul {probe.matmul_n}^2 ran {probe.matmul_total_seconds:.1f}s; "
        f"PCIe {probe.transfer_bytes/(1<<20):.0f}MiB per call "
        f"{probe.transfer_per_call_ms:.2f}ms]",
        flush=True,
    )
    print(
        f"Preparing model from {local_model_dir}. "
        "This includes the working-set solve, engine construction, and HF weight load.",
        flush=True,
    )
    am = from_pretrained(
        local_model_dir,
        optimizer=optimizer,
        max_seq_len=args.seq_len,
        max_global_batch_tokens=args.global_batch_tokens,
        max_gpu_mem_bytes=max_gpu_mem_bytes,
        max_host_mem_bytes=max_host_mem_bytes,
        device=device_str,
        leeway_gpu_mem_bytes=int(args.leeway_gpu_mem_gib * (1 << 30)),
        leeway_host_mem_bytes=int(args.leeway_host_mem_gib * (1 << 30)),
        lora_targets=lora_targets,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        hw_cost=probe.hw_cost,
        mem_bw_gbps=probe.mem_bw_gbps,
        force_saved_act_level=args.force_save_level,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
        strict=False,
        verbose=True,
    )
    if args.data_source == "synthetic":
        synthetic_seq_len = args.synthetic_seq_len or args.seq_len
        print(
            f"Model is ready. Building synthetic token source "
            f"(seq_len={synthetic_seq_len})...",
            flush=True,
        )
        source = SyntheticTokenSource(
            vocab_size=model_cfg.vocab_size,
            seq_lens=synthetic_seq_len,
        )
        print(
            f"Synthetic token source ready. Output dir: {output_dir}",
            flush=True,
        )
    else:
        dataset_path = _resolve_dataset(args.dataset)
        print("Model is ready. Building tokenizer-backed SFT data source...", flush=True)
        source = JsonSFTTokenSource(
            path=dataset_path,
            tokenizer=local_model_dir,
            min_seq_len=32,
            max_seq_len=args.seq_len,
            loop=True,
        )
        print(
            f"Dataset ready from {dataset_path}. Output dir: {output_dir}",
            flush=True,
        )
    return _run_training_loop(
        am,
        source,
        model_cfg=model_cfg,
        args=args,
        output_dir=output_dir,
        save_arch=arch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
