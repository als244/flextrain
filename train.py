from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from flextrain import from_pretrained
from flextrain.config import IOConfig
from flextrain.io.hf_weights import select_arch
from flextrain.io.sources import JsonSFTTokenSource, SyntheticTokenSource
from flextrain.optim.adamw import AdamW, AdamWHyperparams


def _hf_checkpoint_is_complete(local_path: str) -> bool:
    if not os.path.isdir(local_path):
        return False
    cfg_path = os.path.join(local_path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    single_path = os.path.join(local_path, "model.safetensors")
    if os.path.isfile(single_path):
        return True
    index_path = os.path.join(local_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return False
    with open(index_path) as f:
        index_payload = json.load(f)
    weight_map = index_payload.get("weight_map", {})
    if not weight_map:
        return False
    shard_files = {str(name) for name in weight_map.values()}
    return all(
        os.path.isfile(os.path.join(local_path, shard_name))
        for shard_name in shard_files
    )


def _maybe_download_hf_snapshot(io_cfg: IOConfig) -> None:
    local_path = io_cfg.hf_checkpoint
    repo_id = io_cfg.hf_repo_id
    if not local_path or not repo_id:
        return
    if _hf_checkpoint_is_complete(local_path):
        return

    target = Path(local_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    reason = "not found" if not os.path.exists(local_path) else "incomplete"
    print(
        f"Local HF checkpoint {reason} at {local_path}. "
        f"Downloading {repo_id}...",
        flush=True,
    )
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Auto-download needs `huggingface_hub`. "
            "Install via `pip install huggingface_hub`."
        ) from e

    snapshot_download(
        repo_id=repo_id,
        revision=io_cfg.hf_revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded {repo_id} to {local_path}", flush=True)


def _resolve_model(model_arg: str) -> tuple[str, str | None]:
    if os.path.exists(model_arg):
        return model_arg, None
    if "/" in model_arg:
        repo_tail = model_arg.rstrip("/").split("/")[-1]
        return os.path.join("models", repo_tail), model_arg
    return os.path.join("models", model_arg), None


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Iterable) and not isinstance(content, (bytes, bytearray, dict)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    text = str(item.get("text", "")).strip()
                else:
                    text = ""
            else:
                text = ""
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _normalize_chat_record(rec: dict[str, Any]) -> dict[str, str] | None:
    messages = rec.get("messages") or rec.get("conversations")
    if not isinstance(messages, list):
        return None

    normalized: list[tuple[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "") or msg.get("from", "")).strip().lower()
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        text = _flatten_message_content(msg.get("content", msg.get("value")))
        if role and text:
            normalized.append((role, text))

    last_assistant = None
    for i in range(len(normalized) - 1, -1, -1):
        if normalized[i][0] == "assistant":
            last_assistant = i
            break
    if last_assistant is None or last_assistant == 0:
        return None

    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    prompt_lines = [
        f"{role_names.get(role, role.title())}:\n{text}"
        for role, text in normalized[:last_assistant]
    ]
    response = normalized[last_assistant][1].strip()
    if not prompt_lines or not response:
        return None
    return {
        "instruction": "\n\n".join(prompt_lines),
        "output": response,
        "input": "",
    }


def _normalize_sft_record(rec: Any) -> dict[str, str] | None:
    if not isinstance(rec, dict):
        return None
    candidates = [
        ("instruction", "output", "input"),
        ("prompt", "completion", "input"),
        ("prompt", "response", "input"),
        ("question", "answer", "context"),
        ("query", "response", "context"),
    ]
    for prompt_key, response_key, input_key in candidates:
        prompt = str(rec.get(prompt_key, "") or "").strip()
        response = str(rec.get(response_key, "") or "").strip()
        if prompt and response:
            return {
                "instruction": prompt,
                "output": response,
                "input": str(rec.get(input_key, "") or "").strip(),
            }
    return _normalize_chat_record(rec)


def _materialize_hf_dataset(dataset_spec: str) -> str:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Auto-downloading datasets needs `datasets`. "
            "Install via `pip install datasets`."
        ) from e

    cache_dir = Path("datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_spec.rstrip("/").split("/")[-1] or "dataset"
    output_path = cache_dir / f"{dataset_name}.jsonl"
    if output_path.is_file():
        return str(output_path.resolve())

    print(
        f"Dataset not found locally at {dataset_spec}. "
        f"Downloading Hugging Face dataset {dataset_spec}...",
        flush=True,
    )
    ds = load_dataset(dataset_spec, split="train")

    kept = 0
    skipped = 0
    with output_path.open("w") as f:
        for rec in ds:
            normalized = _normalize_sft_record(rec)
            if normalized is None:
                skipped += 1
                continue
            f.write(json.dumps(normalized) + "\n")
            kept += 1

    if kept == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"Could not build SFT examples from {dataset_spec!r}. "
            "Expected records like instruction/output, prompt/completion, "
            "question/answer, or chat-style messages."
        )
    print(
        f"Materialized {kept} records from {dataset_spec} to {output_path} "
        f"(skipped {skipped} unsupported rows).",
        flush=True,
    )
    return str(output_path.resolve())


def _resolve_dataset(dataset_arg: str) -> str:
    if os.path.isfile(dataset_arg):
        return os.path.abspath(dataset_arg)
    if dataset_arg.startswith(("http://", "https://")):
        target_dir = Path("datasets")
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = dataset_arg.rstrip("/").split("/")[-1] or "dataset.jsonl"
        target_path = target_dir / filename
        if not target_path.is_file():
            try:
                from urllib.request import urlretrieve
            except ImportError as e:  # pragma: no cover
                raise ImportError("Could not import urllib.request") from e
            print(
                f"Dataset not found locally. Downloading {dataset_arg} to {target_path}...",
                flush=True,
            )
            urlretrieve(dataset_arg, target_path)
        return str(target_path.resolve())
    return _materialize_hf_dataset(dataset_arg)


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
        "Qwen3MoeForCausalLM": "qwen3_moe",
        "Qwen3NextForCausalLM": "qwen3_next",
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
    warmup_steps = int(total_steps * 0.1)
    cooldown_start = int(total_steps * 0.8)
    max_lr = am.optimizer.hp.lr
    final_lr = max_lr * 0.1

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
        help="Override learning rate. Defaults: 3e-5 for full, 1e-4 for lora.",
    )
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank when --mode lora. Default: 16.")
    p.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha when --mode lora. Default: 16.0.")
    p.add_argument("--device-id", type=int, default=0, help="CUDA device id. Default: 0.")
    p.add_argument("--max-gpu-mem-gib", type=float, default=None, help="GPU memory budget in GiB. Default: 24.")
    p.add_argument("--max-host-mem-gib", type=float, default=None, help="Host memory budget in GiB. Default: 110.")
    p.add_argument("--leeway-gpu-mem-gib", type=float, default=2.0, help="Reserved GPU slack in GiB. Default: 2.0.")
    p.add_argument("--leeway-host-mem-gib", type=float, default=10.0, help="Reserved host-memory slack in GiB. Default: 10.0.")
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

    max_gpu_mem_bytes = (
        int(args.max_gpu_mem_gib * (1 << 30))
        if args.max_gpu_mem_gib is not None else int(24 * (1 << 30))
    )
    max_host_mem_bytes = (
        int(args.max_host_mem_gib * (1 << 30))
        if args.max_host_mem_gib is not None else int(110 * (1 << 30))
    )
    print(
        f"Train config: model={local_model_dir} mode={args.mode} "
        f"seq_len={args.seq_len} steps={args.steps} "
        f"batch_tokens={args.global_batch_tokens}",
        flush=True,
    )
    device_str = f"cuda:{args.device_id}"
    print("Probing hardware (one matmul + one PCIe transfer)...", flush=True)
    from flextrain.core.hw_probe import probe_hardware
    probe = probe_hardware(device=device_str)
    print(
        f"[HW Probe] peak_tflops={probe.hw_cost.peak_tflops:.1f}, "
        f"pcie_bw_gbps={probe.hw_cost.pcie_bw_gbps:.1f}, "
        f"mem_bw_gbps={probe.mem_bw_gbps:.1f} "
        f"(matmul {probe.matmul_n}^2 bf16 = {probe.matmul_per_call_ms:.2f}ms; "
        f"PCIe {probe.transfer_bytes/(1<<20):.0f}MiB = "
        f"{probe.transfer_per_call_ms:.2f}ms)",
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
