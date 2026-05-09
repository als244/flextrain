"""Scaling-law experiment for flextrain `from_dims` random init.

IsoFLOP design (Hoffmann/Chinchilla-style): pick K compute budgets, sweep
M model sizes per budget, train each to its FLOP target on FineWeb, eval
on a held-out FineWeb shard. The grid of (compute, params, val_loss)
points fits the standard scaling formula.

Usage:
    # full grid (overnight)
    python experiments/scaling_law.py run-grid --out runs/scaling_law

    # quick smoke (~5 min, 2 budgets x 2 sizes)
    python experiments/scaling_law.py run-grid --smoke --out runs/sl_smoke

    # one run
    python experiments/scaling_law.py run-one --d-model 384 --n-layers 5 \\
        --total-flops 3e16 --out runs/one

    # re-plot from saved logs (no GPU needed)
    python experiments/scaling_law.py plot --runs-dir runs/scaling_law

Outputs per run (each subdir of --out):
    meta.json        dims, hyperparams, FLOP target, achieved params
    train.jsonl      per-step (step, loss, lr, throughput, tokens_seen)
    eval.jsonl       per-eval (step, val_loss, val_ppl)
    final.json       last-step train/val loss + total time

Outputs at grid level (--out root):
    grid.json        list of completed (params, flops, val_loss) tuples
    scaling_law.png  IsoFLOP curves + power-law fit
    fit.json         fitted Chinchilla coefficients

Data: FineWeb-10B GPT-2 BPE shards (vocab=50304). Train on shards 1-99,
validate on shards 0 (val).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# --- Data path config -------------------------------------------------------
FINEWEB_DIR = "/home/shein/Documents/grad_school/research/awsm_dataflow/fineweb10B"
TRAIN_SHARDS = [f"{FINEWEB_DIR}/fineweb_train_{i:06d}.bin" for i in range(1, 100)]
VAL_SHARD = f"{FINEWEB_DIR}/fineweb_val_000000.bin"
VOCAB_SIZE = 50304


# --- Shape ladder -----------------------------------------------------------
# Hand-picked Llama-style shapes covering ~16M to ~400M params. Each tuple:
# (d_model, n_layers). expert_dim, n_heads derived. head_dim=64.

SHAPE_LADDER = [
    (192, 3),    # ~21M
    (256, 4),    # ~28M
    (320, 4),    # ~37M
    (384, 5),    # ~48M
    (448, 6),    # ~60M
    (512, 7),    # ~74M
    (576, 8),    # ~90M
    (640, 9),    # ~109M
    (768, 10),   # ~148M
    (896, 12),   # ~205M
    (1024, 14),  # ~290M
]


def make_dims(d_model: int, n_layers: int, *, vocab: int = VOCAB_SIZE,
              head_dim: int = 64) -> dict:
    """Llama-style dims dict for `from_dims(arch='llama', ...)`.

    expert_dim chosen as round_to(8/3 * d_model, head_dim) — the
    Llama-3 SwiGLU FFN convention. n_heads = d_model / head_dim.
    """
    assert d_model % head_dim == 0, f"d_model={d_model} % head_dim={head_dim}"
    n_heads = d_model // head_dim
    e_target = (8 * d_model) // 3
    expert_dim = ((e_target + head_dim - 1) // head_dim) * head_dim
    return {
        "vocab_size": vocab,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_heads,
        "head_dim": head_dim,
        "expert_dim": expert_dim,
    }


def estimate_params(dims: dict) -> int:
    """Estimate total Llama param count from a dims dict (matches the
    block + embed + head TensorSpecs in flextrain)."""
    v = dims["vocab_size"]; d = dims["d_model"]; L = dims["n_layers"]
    nh = dims["n_heads"]; nkh = dims["n_kv_heads"]; hd = dims["head_dim"]
    e = dims["expert_dim"]
    attn_dim = nh * hd
    kv_dim = nkh * hd
    embed = v * d
    head = d + v * d                     # final_norm + head_proj
    per_layer_attn = 2 * d + 2 * d * attn_dim + 2 * d * kv_dim   # norms + qkvo (ignoring tiny qkv biases)
    per_layer_ffn = d + 3 * d * e
    layer_total = L * (per_layer_attn + per_layer_ffn)
    return embed + head + layer_total


# --- Sequence reader (port of flextrain.bench.parity.FineWebDocStream
#     but supporting multiple shards transparently). -----------------------
class FineWebMultiShardStream:
    """Yields documents across a list of FineWeb .bin shards. EOT=50256.

    On exhaustion of the last shard, wraps back to the first."""
    EOT = 50256

    def __init__(self, shard_paths: list[str], min_len: int = 128,
                 max_len: int = 1024):
        import numpy as np
        self.shard_paths = list(shard_paths)
        self._np = np
        self._min_len = min_len
        self._max_len = max_len
        self._shard_idx = 0
        self._arr = None
        self._cursor = 0
        self._load_shard(0)

    def _load_shard(self, idx: int) -> None:
        np = self._np
        path = self.shard_paths[idx]
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        # FineWeb format: 256 int32 header (1024 bytes) then uint16 tokens.
        self._arr = np.fromfile(path, dtype=np.uint16, offset=256 * 4)
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
            # Move to next shard (with wraparound).
            self._load_shard((self._shard_idx + 1) % len(self.shard_paths))
        raise RuntimeError("could not yield a single doc — corrupt shards?")


# --- LR schedule ------------------------------------------------------------
def cosine_lr(step: int, *, peak_lr: float, total_steps: int,
              warmup_steps: int, min_lr_frac: float = 0.1) -> float:
    """Linear warmup + cosine decay to ``min_lr_frac * peak_lr``."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * cos_factor)


# --- Single training run ----------------------------------------------------
@dataclass
class RunConfig:
    d_model: int
    n_layers: int
    total_flops: float
    out_dir: str
    batch_tokens: int = 16384
    max_seq_len: int = 1024
    peak_lr: float = 3.0e-4
    warmup_frac: float = 0.02
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    init_seed: int = 42
    eval_every: int = 200
    eval_tokens: int = 200_000
    log_every: int = 20
    max_gpu_mem_gib: int = 22
    max_host_mem_gib: int = 80


def _hw_cost_cache_path() -> Path:
    return Path("runs/_hw_cost_cache.json")


def _get_or_probe_hw_cost():
    """Probe hardware once per machine; cache to JSON so subsequent runs
    don't pay the ~10s probe time."""
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.hw_probe import probe_hardware
    p = _hw_cost_cache_path()
    if p.is_file():
        d = json.loads(p.read_text())
        return HardwareCost(peak_tflops=d["peak_tflops"],
                            pcie_bw_gbps=d["pcie_bw_gbps"]), d["mem_bw_gbps"]
    res = probe_hardware()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "peak_tflops": res.hw_cost.peak_tflops,
        "pcie_bw_gbps": res.hw_cost.pcie_bw_gbps,
        "mem_bw_gbps": res.mem_bw_gbps,
    }, indent=2))
    return res.hw_cost, res.mem_bw_gbps


def run_one(cfg: RunConfig) -> dict:
    """Build a model from cfg, train for cfg.total_flops, eval, save logs.
    Returns a dict with the final stats; also writes JSONL/JSON files to
    cfg.out_dir."""
    import torch
    import flextrain
    from flextrain.io.sequence import Sequence
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build dims, estimate params, derive total_steps from FLOP budget.
    dims = make_dims(cfg.d_model, cfg.n_layers)
    n_params = estimate_params(dims)
    # FLOPs per step ≈ 6 * N * tokens_per_step (forward + backward, no
    # activation-recompute tax — the 6N approximation is the standard
    # Chinchilla / Kaplan FLOP accounting).
    flops_per_step = 6.0 * n_params * cfg.batch_tokens
    total_steps = max(1, int(cfg.total_flops / flops_per_step))
    warmup_steps = max(10, int(cfg.warmup_frac * total_steps))
    total_tokens_target = total_steps * cfg.batch_tokens

    print(
        f"[run_one] d_model={cfg.d_model} L={cfg.n_layers} "
        f"params={n_params/1e6:.1f}M target_flops={cfg.total_flops:.2e} "
        f"steps={total_steps} warmup={warmup_steps} "
        f"target_tokens={total_tokens_target/1e6:.1f}M",
        flush=True,
    )

    hw_cost, mem_bw = _get_or_probe_hw_cost()

    am = flextrain.from_dims(
        dims, arch="llama",
        optimizer=AdamW(AdamWHyperparams(
            lr=cfg.peak_lr, beta1=0.9, beta2=0.95,
            weight_decay=cfg.weight_decay,
        )),
        max_seq_len=cfg.max_seq_len,
        max_global_batch_tokens=cfg.batch_tokens,
        max_gpu_mem_bytes=int(cfg.max_gpu_mem_gib * (1 << 30)),
        max_host_mem_bytes=int(cfg.max_host_mem_gib * (1 << 30)),
        init_seed=cfg.init_seed, init_std=0.02,
        hw_cost=hw_cost, mem_bw_gbps=mem_bw,
    )

    train_stream = FineWebMultiShardStream(
        TRAIN_SHARDS, min_len=128, max_len=cfg.max_seq_len,
    )
    val_stream = FineWebMultiShardStream(
        [VAL_SHARD], min_len=128, max_len=cfg.max_seq_len,
    )

    def gather_batch(stream, target_tokens):
        seqs, total = [], 0
        while total < target_tokens:
            tok = stream.next_doc()
            seqs.append(Sequence(tokens=tok))
            total += len(tok)
        return seqs, total

    def evaluate():
        # Snapshot val loss over `eval_tokens` from the val shard. Use
        # fwd_bwd with a no-op step (we don't call am.step()) — the
        # gradients land in host buffers but get overwritten by the
        # next training fwd_bwd, so this is safe. Optimizer state is
        # untouched. (Pure-fwd APIs aren't exposed; this is the cheap
        # path.)
        torch.cuda.synchronize()
        t0 = time.time()
        total_loss = 0.0
        total_tok = 0
        seen = 0
        while seen < cfg.eval_tokens:
            batch_target = min(cfg.batch_tokens, cfg.eval_tokens - seen)
            seqs, tok = gather_batch(val_stream, batch_target)
            stats = am.fwd_bwd(
                seqs, loss_scale_factor=1.0/tok, total_tokens_per_step=tok,
            )
            total_loss += float(stats.total_loss)
            total_tok += int(stats.total_tokens)
            seen += tok
        torch.cuda.synchronize()
        return total_loss / total_tok, time.time() - t0

    # Save run metadata.
    meta = {
        "dims": dims,
        "params": n_params,
        "config": asdict(cfg),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "flops_per_step": flops_per_step,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    train_log = open(out / "train.jsonl", "w")
    eval_log = open(out / "eval.jsonl", "w")

    losses = []
    eval_losses = []
    t_start = time.time()
    last_log_t = t_start
    last_log_step = 0

    for step in range(1, total_steps + 1):
        lr = cosine_lr(step - 1, peak_lr=cfg.peak_lr,
                       total_steps=total_steps,
                       warmup_steps=warmup_steps)
        # Per-step LR override. AdamW reads self.hp.lr at step time
        # (flextrain/optim/adamw.py:103); reassigning .hp swaps in the
        # new schedule value before the next am.step() runs.
        am.optimizer.hp = AdamWHyperparams(
            lr=lr, beta1=0.9, beta2=0.95, weight_decay=cfg.weight_decay,
        )

        seqs, tok = gather_batch(train_stream, cfg.batch_tokens)
        stats = am.fwd_bwd(
            seqs, loss_scale_factor=1.0/tok, total_tokens_per_step=tok,
        )
        # Gradient clipping (flextrain exposes a clip on the engine).
        if hasattr(am, "clip_grad_norm_"):
            am.clip_grad_norm_(cfg.grad_clip)
        am.step()

        loss = float(stats.total_loss / stats.total_tokens)
        losses.append(loss)

        if step % cfg.log_every == 0 or step == 1 or step == total_steps:
            now = time.time()
            tok_s = ((step - last_log_step) * cfg.batch_tokens) / max(1e-6, now - last_log_t)
            train_log.write(json.dumps({
                "step": step, "loss": loss, "lr": lr,
                "tok_per_s": tok_s, "tokens_seen": step * cfg.batch_tokens,
                "wall_time": now - t_start,
            }) + "\n")
            train_log.flush()
            print(
                f"  step {step:5d}/{total_steps}  loss={loss:.4f}  "
                f"lr={lr:.2e}  tok/s={tok_s:7.0f}  "
                f"elapsed={(now-t_start)/60:5.1f}min",
                flush=True,
            )
            last_log_t = now
            last_log_step = step

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
        "params": n_params,
        "total_flops": cfg.total_flops,
        "actual_steps": total_steps,
        "actual_tokens": total_steps * cfg.batch_tokens,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "total_time_s": total_time,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
    }
    (out / "final.json").write_text(json.dumps(final, indent=2))

    print(
        f"[run_one done] {cfg.out_dir}  "
        f"final_train={final_train_loss:.4f}  "
        f"final_val={final_val_loss:.4f}  "
        f"time={total_time/60:.1f}min",
        flush=True,
    )
    return final


# --- IsoFLOP grid -----------------------------------------------------------
DEFAULT_BUDGETS = [3e16, 1e17, 3e17, 1e18]
DEFAULT_SIZES_PER_BUDGET = {
    3e16: [(192, 3), (256, 4), (320, 4), (384, 5)],
    1e17: [(256, 4), (320, 4), (384, 5), (448, 6)],
    3e17: [(320, 4), (384, 5), (448, 6), (576, 8)],
    1e18: [(384, 5), (448, 6), (576, 8), (768, 10)],
}

SMOKE_BUDGETS = [1e16, 3e16]
SMOKE_SIZES = {
    1e16: [(192, 3), (256, 4)],
    3e16: [(256, 4), (320, 4)],
}


def run_grid(out_root: str, smoke: bool = False,
             budgets: Optional[list] = None,
             sizes_map: Optional[dict] = None) -> None:
    """Run the full IsoFLOP grid. Each run launches as its own
    subprocess so GPU/host pinned memory gets a fresh process every
    time — flextrain's BufferManager calls ``cudaHostRegister`` on
    large buffers, and back-to-back builds in the same process leave
    those mappings live until the python process exits.

    Resumable: skips completed runs by checking for ``final.json``."""
    import subprocess
    if smoke:
        budgets = SMOKE_BUDGETS
        sizes_map = SMOKE_SIZES
    else:
        budgets = budgets or DEFAULT_BUDGETS
        sizes_map = sizes_map or DEFAULT_SIZES_PER_BUDGET

    Path(out_root).mkdir(parents=True, exist_ok=True)
    grid = []
    n_total = sum(len(sizes_map[b]) for b in budgets)
    n_done = 0
    t_grid_start = time.time()
    self_path = os.path.abspath(__file__)

    for budget in budgets:
        for d_model, n_layers in sizes_map[budget]:
            n_done += 1
            run_dir = Path(out_root) / f"F{budget:.0e}_d{d_model}_L{n_layers}"
            final_path = run_dir / "final.json"

            if final_path.is_file():
                print(f"[grid {n_done}/{n_total}] skip {run_dir.name} (already done)", flush=True)
                grid.append(json.loads(final_path.read_text()))
                _save_grid(out_root, grid)
                continue

            print(
                f"\n=== [grid {n_done}/{n_total}] {run_dir.name} === "
                f"({(time.time()-t_grid_start)/60:.1f} min into grid)",
                flush=True,
            )
            cmd = [
                sys.executable, "-u", self_path, "run-one",
                "--d-model", str(d_model),
                "--n-layers", str(n_layers),
                "--total-flops", str(budget),
                "--out", str(run_dir),
            ]
            try:
                # Inherit stdout/stderr so the parent log captures the
                # child's per-step output. Inherit env (PYTHONPATH etc).
                proc = subprocess.run(cmd, check=False)
                if proc.returncode != 0:
                    print(f"[grid] run exited with code {proc.returncode}", flush=True)
                if final_path.is_file():
                    grid.append(json.loads(final_path.read_text()))
                    _save_grid(out_root, grid)
                else:
                    print(f"[grid] {run_dir.name}: no final.json produced", flush=True)
            except Exception as e:
                print(f"[grid] orchestrator error: {e}", flush=True)
                import traceback; traceback.print_exc()

    print(
        f"\n=== grid complete: {n_done} runs in "
        f"{(time.time()-t_grid_start)/60:.1f} min ===",
        flush=True,
    )
    _save_grid(out_root, grid)


def _save_grid(out_root: str, grid: list) -> None:
    Path(out_root, "grid.json").write_text(json.dumps(grid, indent=2))


# --- Plot + fit -------------------------------------------------------------
def plot_grid(runs_dir: str) -> None:
    """Read every final.json under runs_dir, plot IsoFLOP curves, fit a
    Chinchilla-style power law, save figure + fit.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs_dir = Path(runs_dir)
    finals = []
    for sub in sorted(runs_dir.iterdir()):
        f = sub / "final.json"
        if f.is_file():
            d = json.loads(f.read_text())
            finals.append(d)

    if not finals:
        print(f"no final.json files under {runs_dir}", flush=True)
        return

    print(f"loaded {len(finals)} runs", flush=True)

    # Group by FLOP budget for IsoFLOP curves.
    by_budget = {}
    for f in finals:
        by_budget.setdefault(f["total_flops"], []).append(f)
    for runs in by_budget.values():
        runs.sort(key=lambda r: r["params"])

    fig, (ax_iso, ax_law) = plt.subplots(1, 2, figsize=(12, 5))

    # IsoFLOP curves.
    cmap = plt.get_cmap("viridis")
    budgets = sorted(by_budget.keys())
    for i, b in enumerate(budgets):
        runs = by_budget[b]
        params = np.array([r["params"] / 1e6 for r in runs])
        vlosses = np.array([r["final_val_loss"] for r in runs])
        color = cmap(i / max(1, len(budgets) - 1))
        ax_iso.plot(params, vlosses, "o-", color=color,
                    label=f"{b:.1e} FLOPs")
        # Mark the compute-optimal point (lowest loss in this curve).
        opt = int(np.argmin(vlosses))
        ax_iso.plot(params[opt], vlosses[opt], "*", color=color,
                    markersize=14, markeredgecolor="black", markeredgewidth=0.8)
    ax_iso.set_xscale("log")
    ax_iso.set_xlabel("Model size (M params)")
    ax_iso.set_ylabel("Held-out FineWeb val loss")
    ax_iso.set_title("IsoFLOP curves\n(★ = compute-optimal model at each budget)")
    ax_iso.legend(fontsize=8, loc="best")
    ax_iso.grid(alpha=0.3)

    # Compute-optimal scaling law: pick the optimal point at each budget,
    # fit loss vs flops on log-log.
    opt_F, opt_L, opt_N = [], [], []
    for b in budgets:
        runs = by_budget[b]
        i = int(np.argmin([r["final_val_loss"] for r in runs]))
        opt_F.append(b)
        opt_L.append(runs[i]["final_val_loss"])
        opt_N.append(runs[i]["params"] / 1e6)
    opt_F = np.array(opt_F); opt_L = np.array(opt_L); opt_N = np.array(opt_N)

    # Two-parameter fit: loss = E + A * F^(-alpha)  on the optimal frontier.
    # Equivalent to log(loss - E) ~ -alpha * log F + log A. Sweep E.
    best = None
    for E_try in np.linspace(0.5, max(0.5, opt_L.min() - 0.05), 100):
        y = np.log(np.clip(opt_L - E_try, 1e-8, None))
        x = np.log(opt_F)
        if not np.isfinite(y).all():
            continue
        slope, intercept = np.polyfit(x, y, 1)
        resid = (y - (slope * x + intercept))
        sse = float((resid ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {"E": float(E_try), "alpha": float(-slope),
                    "logA": float(intercept), "sse": sse}

    fit_summary = {"compute_optimal_fit": best, "n_runs": len(finals)}

    # Plot the fit.
    ax_law.plot(opt_F, opt_L, "o", markersize=10,
                label="Compute-optimal points")
    if best is not None:
        F_dense = np.linspace(opt_F.min() * 0.5, opt_F.max() * 2, 200)
        E = best["E"]; alpha = best["alpha"]; A = math.exp(best["logA"])
        ax_law.plot(F_dense, E + A * F_dense ** (-alpha),
                    "-", color="C1",
                    label=f"loss = {E:.2f} + {A:.2e} · F^(-{alpha:.3f})")
    ax_law.set_xscale("log")
    ax_law.set_xlabel("Compute (FLOPs)")
    ax_law.set_ylabel("Compute-optimal val loss")
    ax_law.set_title("Scaling-law fit on the compute-optimal frontier")
    ax_law.legend(fontsize=8, loc="best")
    ax_law.grid(alpha=0.3)

    fig.suptitle(f"flextrain `from_dims` scaling-law experiment "
                 f"({len(finals)} runs, {sum(r['actual_tokens'] for r in finals)/1e9:.1f}B tokens total)")
    fig.tight_layout()
    out_png = runs_dir / "scaling_law.png"
    fig.savefig(out_png, dpi=140)
    print(f"saved {out_png}", flush=True)

    fit_summary["all_runs"] = [
        {k: r[k] for k in ("d_model", "n_layers", "params",
                            "total_flops", "final_val_loss",
                            "final_train_loss", "actual_tokens",
                            "total_time_s")}
        for r in finals
    ]
    Path(runs_dir, "fit.json").write_text(json.dumps(fit_summary, indent=2))
    print(f"saved {runs_dir / 'fit.json'}", flush=True)
    print(f"\nfit:  {best}", flush=True)


# --- CLI --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("run-one", help="single training run")
    p1.add_argument("--d-model", type=int, required=True)
    p1.add_argument("--n-layers", type=int, required=True)
    p1.add_argument("--total-flops", type=float, required=True)
    p1.add_argument("--out", type=str, required=True)
    p1.add_argument("--batch-tokens", type=int, default=16384)
    p1.add_argument("--peak-lr", type=float, default=3e-4)
    p1.add_argument("--init-seed", type=int, default=42)

    p2 = sub.add_parser("run-grid", help="full IsoFLOP grid")
    p2.add_argument("--out", type=str, required=True)
    p2.add_argument("--smoke", action="store_true",
                    help="2 budgets x 2 sizes (~5 min total)")

    p3 = sub.add_parser("plot", help="plot + fit from saved logs")
    p3.add_argument("--runs-dir", type=str, required=True)

    args = ap.parse_args()
    if args.cmd == "run-one":
        run_one(RunConfig(
            d_model=args.d_model, n_layers=args.n_layers,
            total_flops=args.total_flops, out_dir=args.out,
            batch_tokens=args.batch_tokens, peak_lr=args.peak_lr,
            init_seed=args.init_seed,
        ))
    elif args.cmd == "run-grid":
        run_grid(args.out, smoke=args.smoke)
        plot_grid(args.out)
    elif args.cmd == "plot":
        plot_grid(args.runs_dir)


if __name__ == "__main__":
    main()
