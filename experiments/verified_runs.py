"""Reproduce ``docs/verified_runs.md`` table on the local machine.

Each row of the verified-runs table runs as its own subprocess by
shelling out to the project's ``train.py`` CLI (so the numbers match
what a normal training run prints). Per-step we capture train.py's
stdout — which already logs loss, tok/s, eff/hw TFLOPS, max_alloc,
max_reserve in a stable format — and then parse step-3 from the saved
log to populate the markdown table.

Usage::

    # full grid (subprocess per row, ~10-20 min on RTX 5090)
    python experiments/verified_runs.py run-grid \\
        --out runs/verified_runs

    # one row
    python experiments/verified_runs.py run-one --row qwen3_5_9b_lora \\
        --out runs/verified_runs/qwen3_5_9b_lora

    # rebuild the markdown table from saved logs (no GPU)
    python experiments/verified_runs.py report \\
        --runs-dir runs/verified_runs

Paths assumed (override with env vars when not at the standard layout):

    FLEXTRAIN_MODELS_DIR        default: ``<repo>/models``
                                each row's HF snapshot lives at
                                ``$FLEXTRAIN_MODELS_DIR/<model_name>``
    FLEXTRAIN_VERIFIED_DATASET  default: ``<repo>/datasets/mathinstruct.jsonl``
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Row registry. Each entry maps to one verified-table row.
# ---------------------------------------------------------------------------


@dataclass
class Row:
    key: str            # short identifier used as subdir name + --row arg
    label: str          # human-readable model name (table column)
    params: str         # "1B", "9B / 1B-active", etc
    arch: str           # short arch description for the table
    model_path: str     # local HF checkpoint dir
    mode: str           # "lora" or "full"
    batch_tokens: int = 65_536
    max_seq_len: int = 2048
    expected_curve: str = ""   # original table value, for side-by-side


_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_TRAIN_PY = f"{_REPO_ROOT}/train.py"
# Default to repo-relative ``models/`` and ``datasets/`` so a clone with
# the standard layout reproduces the table out of the box. Override
# either via env var when the model snapshots / dataset live elsewhere.
_MODELS_DIR = os.environ.get("FLEXTRAIN_MODELS_DIR", f"{_REPO_ROOT}/models")
_DATASET = os.environ.get(
    "FLEXTRAIN_VERIFIED_DATASET",
    f"{_REPO_ROOT}/datasets/mathinstruct.jsonl",
)


ROWS = {r.key: r for r in [
    Row(
        key="llama_3_2_1b_lora",
        label="Llama-3.2-1B", params="1B", arch="dense",
        model_path=f"{_MODELS_DIR}/Llama-3.2-1B", mode="lora",
        expected_curve="not re-verified",
    ),
    Row(
        key="llama_3_2_1b_full",
        label="Llama-3.2-1B", params="1B", arch="dense",
        model_path=f"{_MODELS_DIR}/Llama-3.2-1B", mode="full",
        expected_curve="not re-verified",
    ),
    Row(
        key="llama_3_1_8b_lora",
        label="Llama-3.1-8B-Instruct", params="8B", arch="dense",
        model_path=f"{_MODELS_DIR}/Llama-3.1-8B-Instruct", mode="lora",
        expected_curve="not re-verified",
    ),
    Row(
        key="llama_3_1_8b_full",
        label="Llama-3.1-8B-Instruct", params="8B", arch="dense",
        model_path=f"{_MODELS_DIR}/Llama-3.1-8B-Instruct", mode="full",
        expected_curve="not re-verified",
    ),
    Row(
        key="olmoe_7b_a1b_lora",
        label="OLMoE-1B-7B", params="7B / 1B-active",
        arch="MoE (64 experts)",
        model_path=f"{_MODELS_DIR}/OLMoE-7B-A1B", mode="lora",
        expected_curve="not re-verified",
    ),
    Row(
        key="olmoe_7b_a1b_full",
        label="OLMoE-1B-7B", params="7B / 1B-active",
        arch="MoE (64 experts)",
        model_path=f"{_MODELS_DIR}/OLMoE-7B-A1B", mode="full",
        expected_curve="not re-verified",
    ),
    Row(
        key="qwen3_8b_lora",
        label="Qwen3-8B", params="8B", arch="dense, QK-norm",
        model_path=f"{_MODELS_DIR}/Qwen3-8B", mode="lora",
        expected_curve="not re-verified",
    ),
    Row(
        key="qwen3_8b_full",
        label="Qwen3-8B", params="8B", arch="dense, QK-norm",
        model_path=f"{_MODELS_DIR}/Qwen3-8B", mode="full",
        expected_curve="not re-verified",
    ),
    Row(
        key="qwen3_5_9b_lora",
        label="Qwen3.5-9B", params="9B",
        arch="hybrid linear+full attn, dense MLP",
        model_path=f"{_MODELS_DIR}/Qwen3.5-9B", mode="lora",
        expected_curve="0.797 → 0.620",
    ),
    Row(
        key="qwen3_5_9b_full",
        label="Qwen3.5-9B", params="9B",
        arch="hybrid linear+full attn, dense MLP",
        model_path=f"{_MODELS_DIR}/Qwen3.5-9B", mode="full",
        expected_curve="0.744 → 0.455",
    ),
    Row(
        key="qwen3_6_27b_lora",
        label="Qwen3.6-27B", params="27B",
        arch="hybrid linear+full attn, dense MLP",
        model_path=f"{_MODELS_DIR}/Qwen3.6-27B", mode="lora",
        expected_curve="not re-verified",
    ),
    Row(
        key="qwen3_30b_a3b_lora",
        label="Qwen3-30B-A3B", params="30B / 3B-active",
        arch="MoE (128 experts)",
        model_path=f"{_MODELS_DIR}/Qwen3-30B-A3B", mode="lora",
        expected_curve="not re-verified",
    ),
    Row(
        key="qwen3_5_moe_35b_a3b_lora",
        label="Qwen3.5-MoE-35B-A3B", params="35B / 3B-active",
        arch="hybrid + MoE (256+1 experts)",
        model_path=f"{_MODELS_DIR}/Qwen3.5-35B-A3B", mode="lora",
        expected_curve="0.743 → 0.541",
    ),
]}


# ---------------------------------------------------------------------------
# train.py log parser. Extracts per-step records from the stdout that
# train.py prints in `_run_training_loop` (one line per step).
# ---------------------------------------------------------------------------


_STEP_RE = re.compile(
    r"\[step\s+(?P<step>\d+)/(?P<total>\d+)\]"
    r".*?lr=(?P<lr>[\d.eE+-]+)"
    r".*?loss=(?P<loss>[\d.]+)"
    r".*?tok/step=(?P<tok_step>\d+)"
    r".*?tok/s=(?P<tok_s>[\d,]+)"
    r".*?TFLOPS_eff=(?P<eff>[\d.]+)"
    r".*?TFLOPS_hw=(?P<hw>[\d.]+)"
    r".*?max_alloc=(?P<alloc>[\d.]+)GiB"
    r".*?max_reserve=(?P<resv>[\d.]+)GiB"
    r".*?step=(?P<step_ms>[\d.]+)ms"
)


def parse_train_log(log_path: Path) -> list[dict]:
    """Parse train.py stdout and return one dict per logged step."""
    if not log_path.is_file():
        return []
    out: list[dict] = []
    for line in log_path.read_text().splitlines():
        m = _STEP_RE.search(line)
        if not m:
            continue
        out.append({
            "step": int(m.group("step")),
            "lr": float(m.group("lr")),
            "loss": float(m.group("loss")),
            "tok_step": int(m.group("tok_step")),
            "tok_per_s": float(m.group("tok_s").replace(",", "")),
            "eff_tflops": float(m.group("eff")),
            "hw_tflops": float(m.group("hw")),
            "peak_alloc_gib": float(m.group("alloc")),
            "peak_reserved_gib": float(m.group("resv")),
            "wall_time_s": float(m.group("step_ms")) / 1000.0,
        })
    return out


# ---------------------------------------------------------------------------
# One-row driver. Shells out to train.py, captures stdout to train.log,
# parses, writes final.json compatible with report_table.
# ---------------------------------------------------------------------------


N_STEPS = 5
LEEWAY_GPU_GIB = 3.0


def _build_train_cmd(row: Row, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-u", _TRAIN_PY,
        "--model", row.model_path,
        "--mode", row.mode,
        "--max-seq-len", str(row.max_seq_len),
        "--max-global-batch-tokens", str(row.batch_tokens),
        "--steps", str(N_STEPS),
        "--data-source", "json_sft",
        "--dataset", _DATASET,
        "--output-dir", str(out_dir / "train_out"),
        "--leeway-gpu-mem-gib", str(LEEWAY_GPU_GIB),
        # Match the LR the verified-table runs were originally measured
        # at (3e-5 across both modes). train.py defaults differ per
        # mode — overriding here keeps the curves comparable.
        "--lr", "3.0e-5",
    ]
    if row.mode == "lora":
        cmd += ["--lora-rank", "16", "--lora-alpha", "16.0"]
    return cmd


def run_one(row: Row, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(row.model_path):
        raise FileNotFoundError(
            f"model dir not found: {row.model_path}. Skip this row in "
            f"the grid by removing it from ROWS."
        )
    if not os.path.isfile(_DATASET):
        raise FileNotFoundError(f"dataset missing: {_DATASET}")

    print(f"[verified] === {row.key}: {row.label} ({row.mode}) ===",
          flush=True)

    cmd = _build_train_cmd(row, out)
    log_path = out / "train.log"
    t0 = time.time()
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    print(f"  log: {log_path}", flush=True)
    env = {**os.environ, "PYTHONPATH": _REPO_ROOT}
    with log_path.open("w") as f:
        proc = subprocess.run(
            cmd, cwd=_REPO_ROOT, env=env,
            stdout=f, stderr=subprocess.STDOUT, check=False,
        )
    dt_run = time.time() - t0
    rc = proc.returncode
    print(f"  rc={rc} dt={dt_run:.1f}s", flush=True)

    steps = parse_train_log(log_path)
    if not steps:
        print(f"  no per-step records parsed from {log_path}", flush=True)

    losses = [s["loss"] for s in steps]
    tok_rates = [s["tok_per_s"] for s in steps]
    eff_tflops_list = [s["eff_tflops"] for s in steps]
    hw_tflops_list = [s["hw_tflops"] for s in steps]
    peak_alloc_list = [s["peak_alloc_gib"] for s in steps]
    peak_reserved_list = [s["peak_reserved_gib"] for s in steps]

    def _mean_steady(xs: list[float]) -> float:
        if len(xs) > 1:
            return sum(xs[1:]) / len(xs[1:])
        return xs[0] if xs else 0.0

    final = {
        "key": row.key,
        "label": row.label, "params": row.params, "arch": row.arch,
        "mode": row.mode,
        "batch_tokens": row.batch_tokens,
        "max_seq_len": row.max_seq_len,
        "model_path": row.model_path,
        "expected_curve": row.expected_curve,
        "rc": rc,
        "wall_time_s": dt_run,
        "n_steps": len(steps),
        "steps": steps,
        "losses": losses,
        "tok_per_s": tok_rates,
        "eff_tflops": eff_tflops_list,
        "hw_tflops": hw_tflops_list,
        "peak_alloc_gib": peak_alloc_list,
        "peak_reserved_gib": peak_reserved_list,
        "mean_tok_per_s_steady": _mean_steady(tok_rates),
        "mean_eff_tflops_steady": _mean_steady(eff_tflops_list),
        "mean_hw_tflops_steady": _mean_steady(hw_tflops_list),
        "max_peak_alloc_gib": max(peak_alloc_list) if peak_alloc_list else 0.0,
        "max_peak_reserved_gib": (
            max(peak_reserved_list) if peak_reserved_list else 0.0
        ),
    }
    (out / "final.json").write_text(json.dumps(final, indent=2))
    return final


# ---------------------------------------------------------------------------
# Grid orchestrator — subprocess per row, with mem-drain wait between.
# ---------------------------------------------------------------------------


def run_grid(out_root: str, only: Optional[list] = None) -> None:
    self_path = os.path.abspath(__file__)
    Path(out_root).mkdir(parents=True, exist_ok=True)
    rows = list(ROWS.values())
    if only:
        rows = [r for r in rows if r.key in only]

    grid = []
    n_total = len(rows)
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        run_dir = Path(out_root) / row.key
        if (run_dir / "final.json").is_file():
            print(f"[grid {i}/{n_total}] skip {row.key} (already done)",
                  flush=True)
            grid.append(json.loads((run_dir / "final.json").read_text()))
            _save_grid(out_root, grid)
            continue

        if not os.path.isdir(row.model_path):
            print(f"[grid {i}/{n_total}] skip {row.key}: model dir missing "
                  f"({row.model_path})", flush=True)
            continue

        # Wait for host + GPU memory to fully drain from the prior
        # row's subprocess — pinned-host pages take a moment to be
        # reclaimed by the kernel after a process exits.
        _wait_for_memory_drain(verbose=True)

        print(f"\n=== [grid {i}/{n_total}] {row.key} === "
              f"({(time.time()-t0)/60:.1f} min into grid)", flush=True)
        # Run the row in *another* subprocess (run-one) so train.py's
        # crash on one row doesn't take the orchestrator down with it.
        cmd = [sys.executable, "-u", self_path, "run-one",
               "--row", row.key, "--out", str(run_dir)]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"[grid] {row.key} exited with code {proc.returncode}",
                  flush=True)
        if (run_dir / "final.json").is_file():
            grid.append(json.loads((run_dir / "final.json").read_text()))
            _save_grid(out_root, grid)
        else:
            print(f"[grid] {row.key}: no final.json produced", flush=True)

    print(f"\n=== grid done in {(time.time()-t0)/60:.1f} min ===",
          flush=True)
    _save_grid(out_root, grid)
    report_table(out_root)


def _save_grid(out_root: str, grid: list) -> None:
    Path(out_root, "grid.json").write_text(json.dumps(grid, indent=2))


def _wait_for_memory_drain(
    *,
    host_free_floor_gib: float = 100.0,
    gpu_free_floor_gib: float = 28.0,
    poll_interval_s: float = 1.0,
    timeout_s: float = 60.0,
    verbose: bool = False,
) -> None:
    """Block until host RAM and GPU mem free are above floors. Run
    between subprocesses so the next row starts with a fully-drained
    state — pinned-host (cudaHostRegister) pages take a moment for
    the kernel to reclaim after a process exits, and racing
    cudaHostRegister into the next row before the prior pin is freed
    can OOM or thrash. After ``timeout_s`` we proceed anyway with a
    warning."""
    deadline = time.time() + timeout_s
    while True:
        try:
            with open("/proc/meminfo") as f:
                meminfo = dict(
                    line.split(":") for line in f.read().splitlines()
                )
            host_free_kib = int(meminfo["MemAvailable"].strip().split()[0])
            host_free_gib = host_free_kib / (1024 * 1024)
        except Exception:
            host_free_gib = float("inf")

        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            gpu_free_mib = int(out.stdout.strip().split("\n")[0].strip())
            gpu_free_gib = gpu_free_mib / 1024
        except Exception:
            gpu_free_gib = float("inf")

        ok = (host_free_gib >= host_free_floor_gib
              and gpu_free_gib >= gpu_free_floor_gib)
        if verbose:
            print(f"[grid] mem-drain check: host_free={host_free_gib:.1f} GiB "
                  f"(floor {host_free_floor_gib:.0f}) "
                  f"gpu_free={gpu_free_gib:.1f} "
                  f"GiB (floor {gpu_free_floor_gib:.0f})  ok={ok}",
                  flush=True)
        if ok:
            return
        if time.time() > deadline:
            print(f"[grid] WARN: memory drain timeout after {timeout_s:.0f}s; "
                  f"proceeding with host_free={host_free_gib:.1f} GiB "
                  f"gpu_free={gpu_free_gib:.1f} GiB", flush=True)
            return
        time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# Markdown report generator.
# ---------------------------------------------------------------------------


def _arrow_curve(losses: list) -> str:
    """Format ``[a, b, c, d, e]`` as ``"a → e"`` (start → end)."""
    if not losses:
        return ""
    if len(losses) == 1:
        return f"{losses[0]:.3f}"
    return f"{losses[0]:.3f} → {losses[-1]:.3f}"


def report_table(runs_dir: str) -> None:
    """Walk runs_dir/<row>/final.json, write ``new_table.md``.

    Per-step numbers (tok/sec, eff TFLOPS, hw TFLOPS, peak alloc, peak
    resv) are pulled from the parsed train.py log at step 3 — a
    steady-state mid-run sample that's past step-1 warmup but is a
    real logged value (not a derived average), so users can
    cross-reference against the train.py stdout directly.
    """
    rd = Path(runs_dir)
    finals = []
    for row in ROWS.values():
        f = rd / row.key / "final.json"
        if f.is_file():
            d = json.loads(f.read_text())
            step3 = next(
                (s for s in d.get("steps", []) if s.get("step") == 3),
                None,
            )
            d["_step3"] = step3
            finals.append(d)

    if not finals:
        print("no completed rows under", runs_dir, flush=True)
        return

    lines = []
    lines.append("# Re-verified runs (regenerated)\n")
    lines.append(
        f"Generated by `experiments/verified_runs.py report --runs-dir "
        f"{runs_dir}`. Each row shells out to project root `train.py` "
        f"with `--mode {{lora|full}} --max-seq-len 2048 "
        f"--max-global-batch-tokens 65536 --steps 5 --lr 3e-5 "
        f"--leeway-gpu-mem-gib 3 --dataset "
        f"datasets/mathinstruct.jsonl`. All runs at **auto memory "
        f"budget** (no manual `--max-gpu-mem-gib` / `--max-host-mem-gib`). "
        f"Per-step columns (`tok/sec`, `eff TFLOPS`, `hw TFLOPS`, "
        f"`peak alloc`, `peak resv`) are read **directly from train.py's "
        f"stdout at step 3** — a mid-run logged data point past step-1 "
        f"warmup. `peak alloc` is `torch.cuda.max_memory_allocated()` "
        f"(GiB; live peak); `peak resv` is "
        f"`torch.cuda.max_memory_reserved()` (GiB; caching-pool peak — "
        f"what determines OOM). Effective TFLOPS uses the canonical "
        f"formula in `flextrain/cli.py:_get_model_flops_per_token` "
        f"(`matmul_factor = 4 if LoRA else 6` — LoRA skips frozen-weight "
        f"wgrad — plus the causal attention term). Hardware TFLOPS adds "
        f"`recompute_flops / dt`; the gap reflects the working-set "
        f"solver's recompute trade-off.\n"
    )
    header = (
        "| Model | Params | Arch | Mode | Loss curve (5 steps) | "
        "tok/sec | eff TFLOPS | hw TFLOPS | peak alloc | peak resv | "
        "Original |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for f in finals:
        row = ROWS[f["key"]]
        loss_curve = _arrow_curve(f["losses"])
        s3 = f.get("_step3") or {}
        tps = s3.get("tok_per_s", 0.0)
        eff = s3.get("eff_tflops", 0.0)
        hw = s3.get("hw_tflops", 0.0)
        peak_a = s3.get("peak_alloc_gib", 0.0)
        peak_r = s3.get("peak_reserved_gib", 0.0)
        mode_label = row.mode.upper() if row.mode == "lora" else row.mode
        lines.append(
            f"| {row.label} | {row.params} | {row.arch} | "
            f"{mode_label} | "
            f"{loss_curve} | "
            f"{tps:,.0f} | {eff:.1f} | {hw:.1f} | "
            f"{peak_a:.2f} | {peak_r:.2f} | "
            f"{row.expected_curve} |"
        )
    out_path = rd / "new_table.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}", flush=True)
    print()
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("run-one")
    p1.add_argument("--row", required=True, choices=sorted(ROWS))
    p1.add_argument("--out", required=True)

    p2 = sub.add_parser("run-grid")
    p2.add_argument("--out", required=True)
    p2.add_argument("--only", nargs="+", choices=sorted(ROWS),
                    help="run only these rows (default: all)")

    p3 = sub.add_parser("report")
    p3.add_argument("--runs-dir", required=True)

    args = p.parse_args()
    if args.cmd == "run-one":
        run_one(ROWS[args.row], args.out)
    elif args.cmd == "run-grid":
        run_grid(args.out, only=args.only)
    elif args.cmd == "report":
        report_table(args.runs_dir)


if __name__ == "__main__":
    main()
