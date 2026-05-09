"""Pretrain CLI — random-init a model from dims and train on FineWeb-format
binary shards.

Usage
-----

Dims-by-flags::

    python -m flextrain pretrain \\
        --arch llama --d-model 768 --n-layers 12 \\
        --vocab-size 50304 --expert-dim 2048 \\
        --total-tokens 1B --batch-tokens 16384 \\
        --data-dir /data/fineweb10B \\
        --train-shards 1-99 --val-shard 0 \\
        --out runs/llama_124M_1B

Dims-from-JSON (matching ``orig/model_dims.json`` schema)::

    python -m flextrain pretrain \\
        --arch llama \\
        --dims-json orig/model_dims.json --model-name llama3_8B \\
        --total-tokens 5B \\
        --data-dir /data/fineweb10B \\
        --out runs/llama_8B_5B

Token-budget options are mutually exclusive:

* ``--total-tokens 1B`` — fixed token count (suffixes K/M/B accepted).
* ``--total-flops 1e18`` — derived as ``F / (6 * N * batch_tokens)`` rounded
  up; matches the Chinchilla 6N·T accounting used by
  ``experiments/scaling_law.py``.

The output directory holds:

* ``meta.json``    dims, hyperparams, derived totals.
* ``train.jsonl``  per-step (every ``--log-every`` steps): step, loss, lr,
                   tok/s, tokens_seen, wall_time.
* ``eval.jsonl``   per-eval (every ``--eval-every`` steps): step, val_loss,
                   val_ppl, eval_time_s.
* ``final.json``   final-window train/val loss, total time, achieved tokens.

Re-running with the same ``--out`` is a no-op when ``final.json`` already
exists (matches the resume convention used by the scaling-law grid).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# RunConfig — single source of truth for one training run's knobs.
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Inputs for :func:`run_one`. Mirrors what ``flextrain pretrain`` accepts
    on the command line, plus a few internals."""

    # Model
    arch: str = "llama"
    dims: dict = field(default_factory=dict)

    # Compute budget — exactly one of these drives the loop length. The
    # CLI converts ``--total-flops`` to ``total_steps`` via 6N·T before
    # constructing the config; ``run_one`` always reads ``total_steps``.
    total_steps: int = 0

    # Data
    data_dir: str = ""
    train_shards: tuple[str, ...] = ()
    val_shard: str = ""

    # Output
    out_dir: str = ""

    # Training
    batch_tokens: int = 16384
    max_seq_len: int = 1024
    peak_lr: float = 3.0e-4
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    init_seed: int = 42
    init_std: float = 0.02

    # Evaluation + logging cadence
    eval_every: int = 200
    eval_tokens: int = 200_000
    log_every: int = 20

    # Memory caps for the working-set solver
    max_gpu_mem_gib: int = 22
    max_host_mem_gib: int = 80


# ---------------------------------------------------------------------------
# Dims helpers — accept either flag-set or JSON file.
# ---------------------------------------------------------------------------


def _parse_dims_flags(args: argparse.Namespace) -> dict:
    """Build a dims dict from individual --d-model / --n-layers / etc flags.

    Hard-required from the user: ``--d-model`` and ``--n-layers``.

    Convention-derived when omitted (Llama-3 SwiGLU shape):
    * ``n_heads = d_model / head_dim``
    * ``n_kv_heads = n_heads`` (MHA; GQA still requires explicit override)
    * ``expert_dim = round_up(8/3 * d_model, head_dim)``

    Per-arch ``expand_dims`` is the canonical validator and raises if any
    required field is missing — we just pre-fill the obvious shapes here
    so casual flag-driven runs don't need to specify everything.
    """
    dims: dict = {}
    for src, dst in [
        ("d_model", "d_model"), ("n_layers", "n_layers"),
        ("vocab_size", "vocab_size"), ("n_heads", "n_heads"),
        ("n_kv_heads", "n_kv_heads"), ("head_dim", "head_dim"),
        ("expert_dim", "expert_dim"),
        ("num_routed_experts", "num_routed_experts"),
        ("top_k", "top_k"),
    ]:
        v = getattr(args, src, None)
        if v is not None:
            dims[dst] = v

    # Convention defaults (only when both d_model + head_dim are known).
    if "d_model" in dims and "head_dim" in dims:
        d = int(dims["d_model"]); hd = int(dims["head_dim"])
        if d % hd != 0:
            raise ValueError(
                f"--d-model ({d}) must be divisible by --head-dim ({hd})"
            )
        dims.setdefault("n_heads", d // hd)
        dims.setdefault("n_kv_heads", dims["n_heads"])
        # Llama SwiGLU: 8/3 * d_model rounded up to a multiple of head_dim.
        e_target = (8 * d) // 3
        dims.setdefault("expert_dim",
                        ((e_target + hd - 1) // hd) * hd)
    return dims


def _load_dims_from_json(path: str, model_name: str) -> dict:
    """Load one entry from a JSON file shaped like ``orig/model_dims.json``
    (a top-level dict mapping name → dims sub-dict)."""
    with open(path) as f:
        all_dims = json.load(f)
    if model_name not in all_dims:
        names = ", ".join(sorted(all_dims))
        raise KeyError(
            f"--model-name {model_name!r} not in {path}. Available: {names}"
        )
    return dict(all_dims[model_name])


def _coerce_token_count(s: str) -> int:
    """Accept ``1B`` / ``2.5M`` / ``500K`` / ``16384`` and return an int."""
    s = s.strip().upper()
    if not s:
        raise ValueError("empty token count")
    mult = 1
    if s.endswith("K"): mult = 1_000; s = s[:-1]
    elif s.endswith("M"): mult = 1_000_000; s = s[:-1]
    elif s.endswith("B"): mult = 1_000_000_000; s = s[:-1]
    return int(float(s) * mult)


# ---------------------------------------------------------------------------
# Stream + evaluation. Code shape mirrors experiments/scaling_law.py
# (same FineWeb .bin format = 256 int32 header then uint16 tokens, EOT=50256).
# ---------------------------------------------------------------------------


class _MultiShardStream:
    """Document iterator over a list of FineWeb-format shards. Wraps to the
    first shard on exhaustion so caller-side budget loops never hit
    StopIteration."""
    EOT = 50256

    def __init__(self, shard_paths: list[str], min_len: int = 128,
                 max_len: int = 1024):
        import numpy as np
        if not shard_paths:
            raise ValueError("no shard paths supplied")
        self.shard_paths = list(shard_paths)
        self._np = np
        self._min_len = min_len
        self._max_len = max_len
        self._shard_idx = 0
        self._arr = None
        self._cursor = 0
        self._load_shard(0)

    def _load_shard(self, idx: int) -> None:
        path = self.shard_paths[idx]
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self._arr = self._np.fromfile(path, dtype=self._np.uint16,
                                       offset=256 * 4)
        self._cursor = 0
        self._shard_idx = idx

    def next_doc(self):
        import torch
        np = self._np
        for _ in range(len(self.shard_paths) + 1):
            arr = self._arr
            while self._cursor < len(arr):
                end = self._cursor
                while end < len(arr) and arr[end] != self.EOT:
                    end += 1
                if end > self._cursor + 1:
                    raw = arr[self._cursor:end]
                    if len(raw) > 0 and raw[0] == self.EOT:
                        raw = raw[1:]
                    if len(raw) >= self._min_len:
                        take = min(len(raw), self._max_len)
                        doc = raw[:take].astype(np.int64)
                        self._cursor = end + 1
                        return torch.from_numpy(doc.copy())
                self._cursor = end + 1
            self._load_shard((self._shard_idx + 1) % len(self.shard_paths))
        raise RuntimeError("no doc yielded across all shards — corrupt data?")


# ---------------------------------------------------------------------------
# LR schedule — linear warmup then cosine decay to ``min_lr_frac * peak``.
# ---------------------------------------------------------------------------


def cosine_lr(step: int, *, peak_lr: float, total_steps: int,
              warmup_steps: int, min_lr_frac: float = 0.1) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * cos_factor)


# ---------------------------------------------------------------------------
# Hardware probe cache — shared with experiments/scaling_law.py via the
# same on-disk path so back-to-back grid + CLI runs don't re-probe.
# ---------------------------------------------------------------------------


def _hw_cost_cache_path() -> Path:
    return Path("runs/_hw_cost_cache.json")


def _get_or_probe_hw_cost():
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.hw_probe import probe_hardware
    p = _hw_cost_cache_path()
    if p.is_file():
        d = json.loads(p.read_text())
        return (HardwareCost(peak_tflops=d["peak_tflops"],
                             pcie_bw_gbps=d["pcie_bw_gbps"]),
                d["mem_bw_gbps"])
    res = probe_hardware()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "peak_tflops": res.hw_cost.peak_tflops,
        "pcie_bw_gbps": res.hw_cost.pcie_bw_gbps,
        "mem_bw_gbps": res.mem_bw_gbps,
    }, indent=2))
    return res.hw_cost, res.mem_bw_gbps


# ---------------------------------------------------------------------------
# The training run itself.
# ---------------------------------------------------------------------------


def run_one(cfg: RunConfig) -> dict:
    """Build a model from ``cfg.dims``/``cfg.arch``, train for
    ``cfg.total_steps`` on ``cfg.train_shards``, evaluate on ``cfg.val_shard``,
    and write JSONL/JSON logs to ``cfg.out_dir``. Returns the final stats."""
    import torch
    import flextrain
    from flextrain.io.arch import get_arch_module
    from flextrain.io.sequence import Sequence
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resume short-circuit (matches scaling_law.run_grid).
    if (out / "final.json").is_file():
        print(f"[pretrain] {cfg.out_dir} already has final.json — skipping",
              flush=True)
        return json.loads((out / "final.json").read_text())

    # Validate dims via the arch module (catches missing required fields
    # with a clear KeyError before any GPU work).
    arch_mod = get_arch_module(cfg.arch)
    expanded_dims = arch_mod.expand_dims(cfg.dims)
    n_params = _estimate_params(expanded_dims)

    total_steps = max(1, int(cfg.total_steps))
    warmup_steps = max(10, int(cfg.warmup_frac * total_steps))

    print(
        f"[pretrain] arch={cfg.arch} d_model={expanded_dims['d_model']} "
        f"L={expanded_dims['n_layers']} params={n_params/1e6:.1f}M "
        f"steps={total_steps} warmup={warmup_steps} "
        f"target_tokens={total_steps * cfg.batch_tokens / 1e6:.1f}M",
        flush=True,
    )

    hw_cost, mem_bw = _get_or_probe_hw_cost()

    am = flextrain.from_dims(
        cfg.dims, arch=cfg.arch,
        optimizer=AdamW(AdamWHyperparams(
            lr=cfg.peak_lr, beta1=0.9, beta2=0.95,
            weight_decay=cfg.weight_decay,
        )),
        max_seq_len=cfg.max_seq_len,
        max_global_batch_tokens=cfg.batch_tokens,
        max_gpu_mem_bytes=int(cfg.max_gpu_mem_gib * (1 << 30)),
        max_host_mem_bytes=int(cfg.max_host_mem_gib * (1 << 30)),
        init_seed=cfg.init_seed, init_std=cfg.init_std,
        hw_cost=hw_cost, mem_bw_gbps=mem_bw,
    )

    train_stream = _MultiShardStream(
        list(cfg.train_shards), min_len=128, max_len=cfg.max_seq_len,
    )
    val_stream = _MultiShardStream(
        [cfg.val_shard], min_len=128, max_len=cfg.max_seq_len,
    )

    def gather_batch(stream, target_tokens):
        seqs, total = [], 0
        while total < target_tokens:
            tok = stream.next_doc()
            seqs.append(Sequence(tokens=tok))
            total += len(tok)
        return seqs, total

    def evaluate():
        # No-op step (no am.step() call) — gradients land in host buffers
        # and get overwritten on the next training fwd_bwd. Optimizer
        # state untouched.
        torch.cuda.synchronize()
        t0 = time.time()
        total_loss, total_tok, seen = 0.0, 0, 0
        while seen < cfg.eval_tokens:
            target = min(cfg.batch_tokens, cfg.eval_tokens - seen)
            seqs, tok = gather_batch(val_stream, target)
            stats = am.fwd_bwd(
                seqs, loss_scale_factor=1.0/tok, total_tokens_per_step=tok,
            )
            total_loss += float(stats.total_loss)
            total_tok += int(stats.total_tokens)
            seen += tok
        torch.cuda.synchronize()
        return total_loss / total_tok, time.time() - t0

    # Persist run metadata before starting (so a crash mid-training still
    # leaves a debuggable record).
    meta = {
        "arch": cfg.arch,
        "dims": expanded_dims,
        "params": n_params,
        "config": {
            **{k: v for k, v in asdict(cfg).items() if k not in ("dims",)},
            "expanded_dims": expanded_dims,
        },
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    train_log = open(out / "train.jsonl", "w")
    eval_log = open(out / "eval.jsonl", "w")

    losses, eval_losses = [], []
    t_start = last_log_t = time.time()
    last_log_step = 0

    for step in range(1, total_steps + 1):
        lr = cosine_lr(step - 1, peak_lr=cfg.peak_lr,
                       total_steps=total_steps, warmup_steps=warmup_steps)
        # AdamW reads self.hp.lr at step time — reassign .hp to swap in
        # the new schedule value before am.step() runs.
        am.optimizer.hp = AdamWHyperparams(
            lr=lr, beta1=0.9, beta2=0.95, weight_decay=cfg.weight_decay,
        )

        seqs, tok = gather_batch(train_stream, cfg.batch_tokens)
        stats = am.fwd_bwd(
            seqs, loss_scale_factor=1.0/tok, total_tokens_per_step=tok,
        )
        am.step()

        loss = float(stats.total_loss / stats.total_tokens)
        losses.append(loss)

        if step % cfg.log_every == 0 or step == 1 or step == total_steps:
            now = time.time()
            tok_s = ((step - last_log_step) * cfg.batch_tokens
                     / max(1e-6, now - last_log_t))
            train_log.write(json.dumps({
                "step": step, "loss": loss, "lr": lr,
                "tok_per_s": tok_s,
                "tokens_seen": step * cfg.batch_tokens,
                "wall_time": now - t_start,
            }) + "\n")
            train_log.flush()
            print(
                f"  step {step:5d}/{total_steps}  loss={loss:.4f}  "
                f"lr={lr:.2e}  tok/s={tok_s:7.0f}  "
                f"elapsed={(now-t_start)/60:5.1f}min",
                flush=True,
            )
            last_log_t, last_log_step = now, step

        if step % cfg.eval_every == 0 or step == total_steps:
            val_loss, val_dt = evaluate()
            eval_losses.append((step, val_loss))
            eval_log.write(json.dumps({
                "step": step, "val_loss": val_loss,
                "val_ppl": math.exp(min(30.0, val_loss)),
                "eval_time_s": val_dt,
            }) + "\n")
            eval_log.flush()
            print(
                f"    [eval] step {step}: val_loss={val_loss:.4f}  "
                f"ppl={math.exp(min(30.0, val_loss)):.1f}  "
                f"eval_dt={val_dt:.1f}s",
                flush=True,
            )

    train_log.close()
    eval_log.close()
    total_time = time.time() - t_start

    final_train_loss = sum(losses[-50:]) / max(1, len(losses[-50:]))
    final_val_loss = eval_losses[-1][1] if eval_losses else float("nan")

    final = {
        "arch": cfg.arch,
        "params": n_params,
        "actual_steps": total_steps,
        "actual_tokens": total_steps * cfg.batch_tokens,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "total_time_s": total_time,
        "d_model": expanded_dims["d_model"],
        "n_layers": expanded_dims["n_layers"],
    }
    (out / "final.json").write_text(json.dumps(final, indent=2))

    print(
        f"[pretrain] {cfg.out_dir} done  "
        f"final_train={final_train_loss:.4f}  "
        f"final_val={final_val_loss:.4f}  "
        f"time={total_time/60:.1f}min",
        flush=True,
    )
    return final


def _estimate_params(dims: dict) -> int:
    """Llama-style param estimator. Matches the block + embed + head
    TensorSpecs in flextrain.nn (ignoring small biases)."""
    v = dims["vocab_size"]; d = dims["d_model"]; L = dims["n_layers"]
    nh = dims["n_heads"]; nkh = dims["n_kv_heads"]; hd = dims["head_dim"]
    e = dims["expert_dim"]
    n_routed = dims.get("num_routed_experts", 0)
    n_shared = dims.get("num_shared_experts", 0)
    attn_dim = nh * hd
    kv_dim = nkh * hd
    embed = v * d
    head = d + v * d
    per_layer_attn = 2 * d + 2 * d * attn_dim + 2 * d * kv_dim
    if n_routed > 0:
        # Routed-MoE FFN: w_router (d, E) + w_up (E, 2*F, d) + w_down (E, d, F)
        per_layer_ffn = d + d * n_routed + 3 * n_routed * d * e
    else:
        # Dense Llama FFN with SwiGLU: w_1, w_3 (d,e) + w_2 (e,d).
        per_layer_ffn = d + 3 * d * e
    if n_shared > 0:
        # Optional shared expert (Qwen3-Next, Qwen3.5-MoE).
        sd = dims.get("shared_expert_dim", e)
        per_layer_ffn += 3 * n_shared * d * sd
    return embed + head + L * (per_layer_attn + per_layer_ffn)


# ---------------------------------------------------------------------------
# CLI surface.
# ---------------------------------------------------------------------------


def add_argparse_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``pretrain`` subcommand into ``flextrain.cli``."""
    p = sub.add_parser(
        "pretrain",
        help="random-init a model from dims and pretrain on FineWeb shards",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- arch + dims (mutually exclusive groups; either flags or json) ---
    p.add_argument("--arch", required=True,
                   help="short arch name (llama, mistral, qwen2, qwen3, "
                        "olmoe, qwen3_moe, gemma2)")
    p.add_argument("--dims-json", help="path to a JSON file shaped like "
                                       "orig/model_dims.json")
    p.add_argument("--model-name", help="key into --dims-json")
    # Individual dims-flags (used when --dims-json absent).
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--n-kv-heads", type=int)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--expert-dim", type=int)
    p.add_argument("--num-routed-experts", type=int)
    p.add_argument("--top-k", type=int)
    # --- compute budget (one of) ---
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--total-tokens",
                   help="train for this many tokens (suffixes K/M/B ok)")
    g.add_argument("--total-flops", type=float,
                   help="train for this many FLOPs (6N*T accounting)")
    # --- data ---
    p.add_argument("--data-dir", required=True,
                   help="directory holding fineweb_train_*.bin and "
                        "fineweb_val_*.bin")
    p.add_argument("--train-shards", default="1-99",
                   help="train shard range, e.g. '1-99' or '1,2,3'. "
                        "Resolves to fineweb_train_NNNNNN.bin paths.")
    p.add_argument("--val-shard", default="0",
                   help="val shard index (default 0 = fineweb_val_000000.bin)")
    # --- output ---
    p.add_argument("--out", required=True, help="output directory")
    # --- training knobs ---
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--peak-lr", type=float, default=3.0e-4)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--init-seed", type=int, default=42)
    p.add_argument("--init-std", type=float, default=0.02)
    # --- cadence ---
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-tokens", type=int, default=200_000)
    p.add_argument("--log-every", type=int, default=20)
    # --- memory caps ---
    p.add_argument("--max-gpu-gib", type=int, default=22)
    p.add_argument("--max-host-gib", type=int, default=80)
    p.set_defaults(func=_cmd_pretrain)


def _resolve_train_shards(spec: str, data_dir: str) -> tuple[str, ...]:
    """Resolve '1-99' or '1,2,3' or '1-3,7,10-12' into shard paths."""
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            indices.extend(range(int(a), int(b) + 1))
        else:
            indices.append(int(part))
    paths = [os.path.join(data_dir, f"fineweb_train_{i:06d}.bin")
             for i in indices]
    return tuple(paths)


def _cmd_pretrain(args: argparse.Namespace) -> int:
    """Subcommand wiring: build a RunConfig from flags and call run_one."""
    # Resolve dims.
    if args.dims_json:
        if not args.model_name:
            print("[pretrain] --dims-json also needs --model-name",
                  file=sys.stderr)
            return 2
        dims = _load_dims_from_json(args.dims_json, args.model_name)
    else:
        dims = _parse_dims_flags(args)
        if "d_model" not in dims or "n_layers" not in dims:
            print(
                "[pretrain] need --d-model and --n-layers (or --dims-json + "
                "--model-name) when --dims-json absent",
                file=sys.stderr,
            )
            return 2

    # Validate dims early — the arch-side expand_dims has the canonical
    # required-fields list and gives a clear error message.
    from flextrain.io.arch import get_arch_module
    arch_mod = get_arch_module(args.arch)
    arch_mod.expand_dims(dims)  # raises KeyError on missing fields

    # Resolve compute budget into total_steps via 6N*T if --total-flops.
    expanded = arch_mod.expand_dims(dims)
    n_params = _estimate_params(expanded)
    if args.total_tokens:
        total_tokens = _coerce_token_count(args.total_tokens)
        total_steps = max(1, total_tokens // args.batch_tokens)
    else:
        flops_per_step = 6.0 * n_params * args.batch_tokens
        total_steps = max(1, int(args.total_flops / flops_per_step))

    # Resolve data shards.
    train_shards = _resolve_train_shards(args.train_shards, args.data_dir)
    val_shard = os.path.join(args.data_dir,
                              f"fineweb_val_{int(args.val_shard):06d}.bin")
    if not os.path.isfile(val_shard):
        print(f"[pretrain] val shard not found: {val_shard}", file=sys.stderr)
        return 2

    cfg = RunConfig(
        arch=args.arch, dims=dims,
        total_steps=total_steps,
        data_dir=args.data_dir,
        train_shards=train_shards,
        val_shard=val_shard,
        out_dir=args.out,
        batch_tokens=args.batch_tokens, max_seq_len=args.max_seq_len,
        peak_lr=args.peak_lr, warmup_frac=args.warmup_frac,
        weight_decay=args.weight_decay,
        init_seed=args.init_seed, init_std=args.init_std,
        eval_every=args.eval_every, eval_tokens=args.eval_tokens,
        log_every=args.log_every,
        max_gpu_mem_gib=args.max_gpu_gib,
        max_host_mem_gib=args.max_host_gib,
    )
    run_one(cfg)
    return 0


def main(argv: Optional[list] = None) -> int:
    """Standalone entry point — useful when invoked as
    ``python -m flextrain.scripts.pretrain ...`` rather than via
    ``python -m flextrain pretrain ...``."""
    p = argparse.ArgumentParser(prog="flextrain pretrain")
    sub = p.add_subparsers(dest="cmd", required=True)
    add_argparse_subparser(sub)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
