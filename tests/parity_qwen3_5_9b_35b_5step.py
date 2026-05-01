"""Parity check: Qwen3.5-9B full finetune + Qwen3.5-35B-A3B LoRA, 5 steps.

Two reference loss curves recorded at master cf71e81 (pre-C8/C9) in
``tests/parity_baselines/baseline_qwen3_5_9b_full.log`` and
``baseline_qwen3_5_35b_a3b_lora.log``. Run this script after engine
changes that might affect either model's bwd path; expected per-step
losses must match within bf16 noise (~5e-3 abs / 1% rel).

Both runs use the standard ``train.py`` CLI with mathinstruct.jsonl
and the default LRs (3e-5 for full, 1e-4 for lora). Sequences run
back-to-back so we can refer back to one log per session.

Hardware budget (matches the recorded baselines): 22.5 GiB GPU,
110.0 GiB host, 2.0 GiB GPU leeway. Designed for an RTX 3090
(24 GiB). The 2 GiB leeway (vs train.py's 5 GiB default) is REQUIRED
to fit these workloads — the script sets this automatically.

Usage:
  python tests/parity_qwen3_5_9b_35b_5step.py
  python tests/parity_qwen3_5_9b_35b_5step.py --skip-9b   # only 35B
  python tests/parity_qwen3_5_9b_35b_5step.py --skip-35b  # only 9B
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Reference loss curves from the recorded baselines. If a future change
# legitimately shifts these (e.g. you change the LR schedule), update
# the references AND note why in the commit message.
REFERENCE = {
    "9b_full": {
        "model": "models/Qwen3.5-9B",
        "mode": "full",
        "expected_losses": [0.7442, 0.5176, 0.4892, 0.4714, 0.4547],
        "log_path": "tests/parity_baselines/baseline_qwen3_5_9b_full.log",
    },
    "35b_lora": {
        "model": "models/Qwen3.5-35B-A3B",
        "mode": "lora",
        "expected_losses": [0.7432, 0.6866, 0.6452, 0.5855, 0.5407],
        "log_path": "tests/parity_baselines/baseline_qwen3_5_35b_a3b_lora.log",
    },
}

# Tolerance for per-step loss match. bf16 noise is ~1e-3 per kernel,
# but kernels involving atomic_add accumulation over tens of thousands
# of tokens (flash-attn bwd, MoE scatter combine, fla chunk32 fwd/bwd)
# compound across 5 steps with optimizer momentum. Empirically two
# back-to-back 9B runs land within ~3e-4 of each other but ~1e-2 from
# the recorded baseline at the worst step — likely the run that
# produced the baseline picked up slightly different atomic-add
# orderings. 1.5e-2 gives clean headroom without masking real bugs
# (those produce drift that grows step-over-step or differs an order
# of magnitude more).
DEFAULT_TOL = 1.5e-2


def _build_env() -> dict:
    env = dict(os.environ)
    env.setdefault("CONDA_PREFIX", "/home/shein/miniconda3/envs/flextrain")
    cu12 = os.path.join(
        env["CONDA_PREFIX"],
        "lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
    )
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{cu12}:{existing}" if existing else cu12
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return env


def _parse_losses(stdout: str) -> list[float]:
    """Pull per-step losses from train.py's ``[step N/M] ... loss=X`` lines."""
    losses: list[float] = []
    for m in re.finditer(r"\[step\s+\d+/\d+\]\s+lr=\S+\s+loss=(\S+)", stdout):
        try:
            losses.append(float(m.group(1)))
        except ValueError:
            pass
    return losses


def _run(label: str, model: str, mode: str, *, dataset: str,
         gpu_gib: float, host_gib: float, leeway_gpu_gib: float,
         log_path: str) -> tuple[int, list[float]]:
    print(f"\n{'='*72}")
    print(f"=== {label}: model={model} mode={mode} ===")
    print(f"{'='*72}")
    cmd = [
        sys.executable, os.path.join(ROOT, "train.py"),
        "--model", model,
        "--mode", mode,
        "--max-seq-len", "2048",
        "--max-global-batch-tokens", "65536",
        "--steps", "5",
        "--dataset", dataset,
        "--max-gpu-mem-gib", str(gpu_gib),
        "--max-host-mem-gib", str(host_gib),
        "--leeway-gpu-mem-gib", str(leeway_gpu_gib),
    ]
    print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    t0 = time.time()
    proc = subprocess.run(
        cmd, env=_build_env(), cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - t0
    with open(log_path, "w") as f:
        f.write(proc.stdout)
    print(f"  exit={proc.returncode}  elapsed={elapsed:.1f}s  log={log_path}")
    losses = _parse_losses(proc.stdout)
    return proc.returncode, losses


def _compare(label: str, observed: list[float], expected: list[float],
             tol: float) -> bool:
    print(f"\n--- {label} loss curve ---")
    print(f"  {'step':<6}{'observed':<14}{'expected':<14}{'|Δ|':<14}{'within tol':<10}")
    print(f"  {'-'*60}")
    all_ok = True
    if len(observed) < len(expected):
        print(f"  ! observed has {len(observed)} losses, expected {len(expected)}")
        all_ok = False
    for i, exp in enumerate(expected):
        if i >= len(observed):
            break
        obs = observed[i]
        d = abs(obs - exp)
        ok = d < tol
        all_ok = all_ok and ok
        print(f"  {i+1:<6}{obs:<14.4f}{exp:<14.4f}{d:<14.3e}{'OK' if ok else 'FAIL':<10}")
    print(f"  -> {'PASS' if all_ok else 'FAIL'} (tol={tol:.0e})")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/mathinstruct.jsonl")
    ap.add_argument("--gpu-gib", type=float, default=22.5)
    ap.add_argument("--host-gib", type=float, default=110.0)
    ap.add_argument("--leeway-gpu-gib", type=float, default=2.0,
                    help="GPU memory leeway. Baselines used 2.0; the "
                         "default 5.0 in train.py is too tight to fit "
                         "the 9B/35B parity workloads on a 24 GiB card.")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"Per-step abs loss tolerance. Default {DEFAULT_TOL}.")
    ap.add_argument("--skip-9b", action="store_true")
    ap.add_argument("--skip-35b", action="store_true")
    ap.add_argument(
        "--out-dir", default="tests/chunk_variance_logs",
        help="Directory to write per-run subprocess logs. Default lives "
             "alongside the recorded baseline logs.",
    )
    args = ap.parse_args()

    results: list[tuple[str, bool]] = []

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_9b:
        spec = REFERENCE["9b_full"]
        rc, losses = _run(
            "Qwen3.5-9B full finetune",
            spec["model"], spec["mode"],
            dataset=args.dataset,
            gpu_gib=args.gpu_gib, host_gib=args.host_gib,
            leeway_gpu_gib=args.leeway_gpu_gib,
            log_path=os.path.join(out_dir, "parity_9b_full.log"),
        )
        if rc != 0:
            print(f"  ! 9B run exited non-zero ({rc}); skipping comparison")
            results.append(("Qwen3.5-9B full", False))
        else:
            ok = _compare(
                "Qwen3.5-9B full", losses, spec["expected_losses"], args.tol,
            )
            results.append(("Qwen3.5-9B full", ok))

    if not args.skip_35b:
        spec = REFERENCE["35b_lora"]
        rc, losses = _run(
            "Qwen3.5-35B-A3B LoRA",
            spec["model"], spec["mode"],
            dataset=args.dataset,
            gpu_gib=args.gpu_gib, host_gib=args.host_gib,
            leeway_gpu_gib=args.leeway_gpu_gib,
            log_path=os.path.join(out_dir, "parity_35b_lora.log"),
        )
        if rc != 0:
            print(f"  ! 35B run exited non-zero ({rc}); skipping comparison")
            results.append(("Qwen3.5-35B-A3B LoRA", False))
        else:
            ok = _compare(
                "Qwen3.5-35B-A3B LoRA", losses, spec["expected_losses"], args.tol,
            )
            results.append(("Qwen3.5-35B-A3B LoRA", ok))

    print("\n" + "=" * 72)
    print("=== Summary ===")
    print("=" * 72)
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
