"""Command-line entrypoint.

Subcommands:

    python -m flextrain info                  # package status
    python -m flextrain train <config.yaml>   # run one training job
    python -m flextrain load-hf <hf_path> --out <native_dir>   # future
    python -m flextrain export-hf <native_dir> --out <hf_path> # future

The training config is a YAML matching :class:`flextrain.config.RunConfig`.
See ``flextrain/bench/`` for parity-test examples and
``flextrain/configs/`` (when it lands) for a starter YAML.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from typing import Any

import torch


# ---------------------------------------------------------------------------
# YAML parsing.
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "flextrain train needs `pyyaml`. Install via `pip install pyyaml`."
        ) from e
    with open(path) as f:
        return yaml.safe_load(f)


def _to_dataclass(cls, data: dict):
    """Instantiate a frozen dataclass from a dict, dropping unknown keys."""
    from dataclasses import fields

    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for k, v in (data or {}).items():
        if k not in known:
            print(f"[warn] unknown {cls.__name__} key: {k!r}", file=sys.stderr)
            continue
        # Recurse into nested dataclass fields.
        field = known[k]
        from dataclasses import is_dataclass
        if (
            isinstance(v, dict)
            and hasattr(field.type, "__origin__") is False
            and isinstance(field.type, type)
            and is_dataclass(field.type)
        ):
            kwargs[k] = _to_dataclass(field.type, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def _build_run_config(path: str):
    from flextrain.config import (
        DataConfig, HardwareConfig, IOConfig, ModelConfig,
        OptimizerConfig, RunConfig, TrainConfig,
    )
    data = _load_yaml(path)

    model = _to_dataclass(ModelConfig, data.get("model", {}))
    opt_cfg = _to_dataclass(OptimizerConfig, data.get("train", {}).get("optimizer", {}))
    train_raw = dict(data.get("train", {}))
    train_raw["optimizer"] = opt_cfg
    train = _to_dataclass(TrainConfig, train_raw)
    hw = _to_dataclass(HardwareConfig, data.get("hardware", {}))

    io_raw = dict(data.get("io", {}))
    data_cfg = _to_dataclass(DataConfig, io_raw.pop("data", {}) or {})
    io_raw["data"] = data_cfg
    io = _to_dataclass(IOConfig, io_raw)

    return RunConfig(model=model, train=train, hardware=hw, io=io)


# ---------------------------------------------------------------------------
# Building the engine from a RunConfig.
# ---------------------------------------------------------------------------


def _build_model_for_arch(model_cfg, compute_dtype):
    """Instantiate backbone + embed + head for ``model_cfg.arch``.

    Currently supported: ``"llama"``. Other architectures slot in as
    we port them (nn/layers/<arch>.py).
    """
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig

    if model_cfg.arch == "llama":
        from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig

        cfg = LlamaBlockConfig(
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_kv_heads=model_cfg.n_kv_heads,
            head_dim=model_cfg.head_dim,
            expert_dim=model_cfg.expert_dim,
            rms_norm_eps=model_cfg.rms_norm_eps,
            rope_base=model_cfg.rope_theta,
            is_causal=model_cfg.is_causal,
            compute_dtype=compute_dtype,
            master_dtype=compute_dtype,
            grad_dtype=compute_dtype,
            norm_grad_dtype=torch.float32,
        )
        backbone = [
            LlamaBlock(layer_id=i, cfg=cfg) for i in range(model_cfg.n_layers)
        ]
    else:
        raise ValueError(
            f"architecture {model_cfg.arch!r} not yet wired into cli.train. "
            f"Supported: ['llama']. Add a branch here when the arch lands."
        )

    embed = TokenEmbedLayer(
        TokenEmbedConfig(
            vocab_size=model_cfg.vocab_size,
            d_model=model_cfg.d_model,
            compute_dtype=compute_dtype,
            master_dtype=compute_dtype,
            grad_dtype=compute_dtype,
        )
    )
    head = LMHead(
        LMHeadConfig(
            d_model=model_cfg.d_model,
            vocab_size=model_cfg.vocab_size,
            rms_norm_eps=model_cfg.rms_norm_eps,
            head_chunk_size=1024,
            compute_dtype=compute_dtype,
            master_dtype=compute_dtype,
            grad_dtype=compute_dtype,
            norm_grad_dtype=torch.float32,
        )
    )
    dims = {
        "d_model": model_cfg.d_model,
        "n_heads": model_cfg.n_heads,
        "n_kv_heads": model_cfg.n_kv_heads,
        "head_dim": model_cfg.head_dim,
        "expert_dim": model_cfg.expert_dim,
        "vocab_size": model_cfg.vocab_size,
    }
    return embed, backbone, head, dims


def _build_optimizer(opt_cfg):
    if opt_cfg.kind == "adamw":
        from flextrain.optim.adamw import AdamW, AdamWHyperparams
        return AdamW(AdamWHyperparams(
            lr=opt_cfg.lr, beta1=opt_cfg.beta1, beta2=opt_cfg.beta2,
            eps=opt_cfg.eps, weight_decay=opt_cfg.weight_decay,
        ))
    elif opt_cfg.kind == "muon":
        from flextrain.optim.muon import Muon, MuonHyperparams
        return Muon(MuonHyperparams(
            lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay,
        ))
    else:
        raise ValueError(f"unknown optimizer kind {opt_cfg.kind!r}")


def _build_token_source(data_cfg, tokenizer_spec, vocab_size):
    """Return a :class:`TokenSource` matching the data config.

    Exactly one of the data-config fields should be set.
    """
    from flextrain.io.sources import (
        CustomSchemaTokenSource, HFTokenSource, RawTokenSource,
        ShardTokenSource, SyntheticTokenSource,
    )

    picks = [
        ("hf", bool(data_cfg.hf_dataset)),
        ("shard", bool(data_cfg.shard_pattern)),
        ("synthetic", bool(data_cfg.synthetic)),
        ("raw", bool(data_cfg.raw_path)),
        ("custom", bool(data_cfg.custom_factory)),
    ]
    active = [name for name, on in picks if on]
    if len(active) != 1:
        raise ValueError(
            f"exactly one data source must be set, got: {active}"
        )
    kind = active[0]

    if kind == "hf":
        return HFTokenSource(
            dataset=data_cfg.hf_dataset,
            subset=data_cfg.hf_subset,
            split=data_cfg.hf_split,
            tokenizer=tokenizer_spec or data_cfg.hf_dataset,
            text_field=data_cfg.hf_text_field,
            min_seq_len=data_cfg.min_seq_len,
            max_seq_len=data_cfg.max_seq_len,
            streaming=data_cfg.hf_streaming,
        )
    if kind == "shard":
        return ShardTokenSource(
            shard_pattern=data_cfg.shard_pattern,
            num_shards=data_cfg.shard_num_shards,
            min_seq_len=data_cfg.min_seq_len,
            max_seq_len=data_cfg.max_seq_len,
            vocab_size=vocab_size,
        )
    if kind == "synthetic":
        return SyntheticTokenSource(
            vocab_size=vocab_size,
            seq_lens=data_cfg.synthetic_seq_len,
        )
    if kind == "raw":
        payload = torch.load(data_cfg.raw_path, map_location="cpu")
        return RawTokenSource(payload)
    if kind == "custom":
        mod_name, _, attr = data_cfg.custom_factory.partition(":")
        if not attr:
            raise ValueError(
                f"custom_factory must be 'module:callable', got "
                f"{data_cfg.custom_factory!r}"
            )
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, attr)
        records, extract = factory()
        return CustomSchemaTokenSource(records=records, extract=extract)
    raise AssertionError("unreachable")


def _get_model_flops_per_token(model_cfg, seq_len: int) -> int:
    """Approximate FLOPs per sequence for a forward+backward pass.

    Matches orig's ``get_model_flops_per_sequence`` in
    ``orig/awsm_transformer/utils.py:196`` — 6× per-token-param for
    fwd+bwd matmuls, 12× for causal attention, plus the head.
    """
    d_model = model_cfg.d_model
    n_heads = model_cfg.n_heads
    head_dim = model_cfg.head_dim
    n_kv_heads = model_cfg.n_kv_heads
    expert_dim = model_cfg.expert_dim
    vocab = model_cfg.vocab_size
    n_layers = model_cfg.n_layers
    top_k = model_cfg.top_k
    num_shared = model_cfg.num_shared_experts
    is_causal = model_cfg.is_causal

    ctx_dim = n_kv_heads * head_dim
    attn_dim = n_heads * head_dim
    # Active params per layer (attention + ffn routed).
    active_params_per_layer = (
        2 * d_model * attn_dim
        + 2 * d_model * ctx_dim
        + 3 * (num_shared + max(top_k, 1 if num_shared else 0)) * d_model * expert_dim
        if top_k > 0 or num_shared > 0
        else (
            2 * d_model * attn_dim
            + 2 * d_model * ctx_dim
            + 3 * d_model * expert_dim
        )
    )
    matmul_flops_per_layer = 6 * seq_len * active_params_per_layer
    # Attention: causal halves the work (only T²/2 pairs). Per-pair
    # fwd+bwd is 12 MACs over attn_dim — so causal = 6 × T² × attn_dim,
    # non-causal = 12 × T² × attn_dim. Expressed here as
    # ``12 × attn_factor × T² × attn_dim`` to match orig's formula
    # (orig/awsm_transformer/utils.py:220). Equivalent to the user's
    # "6× for causal" note.
    attn_factor = 0.5 if is_causal else 1.0
    attn_flops_per_layer = 12 * attn_factor * seq_len * seq_len * attn_dim
    backbone_flops = n_layers * (matmul_flops_per_layer + attn_flops_per_layer)
    head_flops = 6 * seq_len * d_model * vocab
    return backbone_flops + head_flops


def _lr_schedule(
    step: int, *,
    max_lr: float, final_lr: float,
    warmup_steps: int, cooldown_start: int, total_steps: int,
) -> float:
    """Linear warmup + cosine cooldown schedule (orig/train.py style)."""
    import math
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step < cooldown_start:
        return max_lr
    # Cosine decay between cooldown_start and total_steps.
    pct = (step - cooldown_start) / max(1, total_steps - cooldown_start)
    pct = min(1.0, max(0.0, pct))
    return final_lr + 0.5 * (max_lr - final_lr) * (1.0 + math.cos(math.pi * pct))


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------


def _train(run_cfg) -> int:
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import determine_working_set_config
    from flextrain.engine.active_model import ActiveModel

    compute_dtype = torch.bfloat16
    device = f"cuda:{run_cfg.hardware.device_id}"

    # Build model.
    embed, backbone, head, dims = _build_model_for_arch(
        run_cfg.model, compute_dtype
    )
    optimizer = _build_optimizer(run_cfg.train.optimizer)

    # Compute working-set config (heuristics in orig/working_set.py).
    max_gpu = (
        int(run_cfg.hardware.max_gpu_mem_gib * (1 << 30))
        if run_cfg.hardware.max_gpu_mem_gib is not None else None
    )
    max_host = (
        int(run_cfg.hardware.max_host_mem_gib * (1 << 30))
        if run_cfg.hardware.max_host_mem_gib is not None else None
    )
    training_config = {
        "master_weight_dtype": "bfloat16",
        "grad_dtype": "bfloat16",
        "opt_choice": "AdamW" if run_cfg.train.optimizer.kind == "adamw" else "Muon",
        "opt_dtype": "bfloat16",
    }
    working_set = determine_working_set_config(
        model_dims={
            "d_model": run_cfg.model.d_model,
            "n_layers": run_cfg.model.n_layers,
            "n_heads": run_cfg.model.n_heads,
            "n_kv_heads": run_cfg.model.n_kv_heads,
            "head_dim": run_cfg.model.head_dim,
            "expert_dim": run_cfg.model.expert_dim,
            "vocab_size": run_cfg.model.vocab_size,
            "num_shared_experts": run_cfg.model.num_shared_experts,
            "num_routed_experts": run_cfg.model.num_routed_experts,
            "top_k": run_cfg.model.top_k,
            "is_causal": run_cfg.model.is_causal,
            "datatypes": dict(run_cfg.model.dtypes),
        },
        max_seq_len=run_cfg.train.max_seq_len,
        max_global_batch_tokens=run_cfg.train.global_batch_tokens,
        training_config=training_config,
        max_gpu_mem_bytes=max_gpu,
        max_host_mem_bytes=max_host,
        leeway_gpu_mem_bytes=int(run_cfg.hardware.leeway_gpu_mem_gib * (1 << 30)),
        leeway_host_mem_bytes=int(run_cfg.hardware.leeway_host_mem_gib * (1 << 30)),
        device_id=run_cfg.hardware.device_id,
        min_chunk_size=run_cfg.hardware.min_chunk_size,
        max_chunk_size=run_cfg.hardware.max_chunk_size,
        verbose=True,
    )
    hw_env = working_set.hardware_env
    hw_cost = HardwareCost(
        peak_tflops=hw_env["matmul_report"][
            "overall_layer_matmul_throughput_tflops_per_sec"
        ],
        pcie_bw_gbps=hw_env["transfer_report"][
            "overall_unidirectional_concurrent_bandwidth_gb_per_sec"
        ],
    )

    # Build engine.
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head,
        optimizer=optimizer, working_set=working_set, hw_cost=hw_cost,
        dims=dims, device=device,
    )

    # Optional: load HF weights.
    if run_cfg.io.hf_checkpoint:
        print(f"Loading HF weights from {run_cfg.io.hf_checkpoint}")
        am.load_hf(run_cfg.io.hf_checkpoint)

    # Build data source.
    source = _build_token_source(
        run_cfg.io.data,
        tokenizer_spec=run_cfg.io.tokenizer,
        vocab_size=run_cfg.model.vocab_size,
    )

    # Output dir.
    os.makedirs(run_cfg.io.output_dir, exist_ok=True)

    # LR schedule.
    total_steps = run_cfg.train.total_steps
    warmup_steps = int(total_steps * run_cfg.train.warmup_pct)
    cooldown_start = int(total_steps * (1.0 - run_cfg.train.cooldown_pct))
    max_lr = run_cfg.train.optimizer.lr
    final_lr = max_lr * run_cfg.train.final_lr_fraction

    # Training loop with full throughput / memory reporting.
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
        if hasattr(am.optimizer, "hp"):
            am.optimizer.hp = type(am.optimizer.hp)(
                **{**am.optimizer.hp.__dict__, "lr": lr}
            )

        seqs = source.get_sequences(
            max_token_count=run_cfg.train.global_batch_tokens
        )
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
        torch.cuda.synchronize()  # accurate wall-clock for tok/sec
        step_time = time.time() - step_start

        # FLOPs: sum per-sequence fwd+bwd flops. Matches orig's
        # get_model_flops_per_sequence reporting.
        step_flops = sum(
            _get_model_flops_per_token(run_cfg.model, len(s)) for s in seqs
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
    overall_tok_per_s = (
        total_tokens_processed / total_time if total_time > 0 else 0
    )
    overall_tflops = (
        (total_flops_processed / total_time) / 1e12 if total_time > 0 else 0
    )
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

    # Save final weights.
    final_out = os.path.join(run_cfg.io.output_dir, "final.safetensors")
    try:
        path = am.save_hf(run_cfg.io.output_dir, out_filename="final.safetensors")
        print(f"Saved final weights to {path}")
    except Exception as e:
        print(f"[warn] save_hf failed: {e}")

    return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _cmd_info(_args: argparse.Namespace) -> int:
    import flextrain

    print("FlexTrain package at", flextrain.__file__)
    print("Phase 3 complete: engine + 8-config loss-curve parity against")
    print("naive PyTorch on real FineWeb data passes.")
    print("Run a training job with:")
    print("    python -m flextrain train <config.yaml>")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    run_cfg = _build_run_config(args.config)
    return _train(run_cfg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="flextrain")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="print package layout / status").set_defaults(
        func=_cmd_info
    )

    train = sub.add_parser("train", help="run one training job from a YAML")
    train.add_argument("config", help="path to RunConfig YAML")
    train.set_defaults(func=_cmd_train)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
