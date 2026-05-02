"""End-to-end numeric comparison: flextrain vs sonic MoE backends.

Runs ``train.py`` twice on the same config (same seed, same dataset)
with two different ``--moe-backend`` values, dumps per-layer
``dy / g_up / g_down / g_router`` tensors at every bwd call into two
separate dirs, then walks the file pairs and reports cos / mean_diff /
std / max_abs / rel-to-ref-scale per (call, layer, name).

Run from the repo root on the Hopper machine:

  python tests/scratch/compare_moe_backends_e2e.py

Adjust ``CONFIG`` at the top to change model / steps / mem budget.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch


# -- Config you might want to edit --
ROOT = Path(__file__).resolve().parents[2]
CONFIG = dict(
    model="models/Qwen3.5-35B-A3B",
    mode="full",
    max_seq_len=2048,
    max_global_batch_tokens=65536,
    steps=1,
    dataset="datasets/mathinstruct.jsonl",
    max_gpu_mem_gib=70.0,
    max_host_mem_gib=380.0,
    leeway_gpu_mem_gib=2.0,
)
_DUMP_ROOT = os.environ.get(
    "FLEXTRAIN_MOE_DUMP_ROOT", "/home/as1669/storage/flextrain/moe_dump",
)
DUMP_A = f"{_DUMP_ROOT}/flextrain"
DUMP_B = f"{_DUMP_ROOT}/sonicmoe"
COS_TOL = 0.999  # parity test convention
# -----------------------------------


def _build_train_cmd(backend: str) -> list[str]:
    return [
        sys.executable, str(ROOT / "train.py"),
        "--model", CONFIG["model"],
        "--mode", CONFIG["mode"],
        "--moe-backend", backend,
        "--max-seq-len", str(CONFIG["max_seq_len"]),
        "--max-global-batch-tokens", str(CONFIG["max_global_batch_tokens"]),
        "--steps", str(CONFIG["steps"]),
        "--dataset", CONFIG["dataset"],
        "--max-gpu-mem-gib", str(CONFIG["max_gpu_mem_gib"]),
        "--max-host-mem-gib", str(CONFIG["max_host_mem_gib"]),
        "--leeway-gpu-mem-gib", str(CONFIG["leeway_gpu_mem_gib"]),
    ]


def _env_for_backend(dump_dir: str) -> dict:
    env = dict(os.environ)
    env["FLEXTRAIN_MOE_DUMP_DIR"] = dump_dir
    # Match the parity script's libcudart shim.
    cu12 = (
        f"{env.get('CONDA_PREFIX', '')}"
        "/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    )
    if cu12:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cu12}:{existing}" if existing else cu12
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return env


def _wipe(path: str) -> None:
    p = Path(path)
    if p.exists():
        for f in p.iterdir():
            f.unlink()
    else:
        p.mkdir(parents=True, exist_ok=True)


def _run_one(backend: str, dump_dir: str, log_path: str) -> int:
    print(f"\n{'='*72}\n=== Running --moe-backend {backend} (dump={dump_dir}) ===\n{'='*72}")
    _wipe(dump_dir)
    cmd = _build_train_cmd(backend)
    print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    t0 = time.time()
    proc = subprocess.run(
        cmd, env=_env_for_backend(dump_dir), cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.time() - t0
    Path(log_path).write_text(proc.stdout)
    last = proc.stdout.strip().splitlines()[-25:]
    print("\n".join(last))
    print(f"\n  exit={proc.returncode}  elapsed={elapsed:.1f}s  log={log_path}")
    return proc.returncode


def _diffstats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    diff = a_f - b_f
    abs_diff = diff.abs()
    cos = torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0), dim=-1,
    ).item()
    return {
        "cos": cos,
        "mean_diff": diff.mean().item(),
        "std_diff": diff.std().item(),
        "max_abs": abs_diff.max().item(),
        "ref_scale": b_f.abs().mean().item(),
    }


def _compare(dir_a: str, dir_b: str) -> tuple[int, int]:
    pa = sorted(Path(dir_a).glob("*.pt"))
    pb_set = {p.name for p in Path(dir_b).glob("*.pt")}

    print(f"\n{'='*72}\n=== Per-tensor comparison (A=flextrain, B=sonicmoe) ===\n{'='*72}")
    header = (
        f"{'file':<48s} {'cos':>10s} {'mean_diff':>12s} {'std_diff':>12s} "
        f"{'max_abs':>12s} {'ref_scale':>12s}  status"
    )
    print(header)
    print("-" * len(header))

    n_total, n_fail = 0, 0
    failed_files: list[str] = []
    missing_in_b: list[str] = []

    for pa_file in pa:
        if pa_file.name not in pb_set:
            missing_in_b.append(pa_file.name)
            continue
        a = torch.load(pa_file, map_location="cpu", weights_only=True)
        b = torch.load(Path(dir_b) / pa_file.name, map_location="cpu", weights_only=True)
        if a.shape != b.shape:
            print(f"{pa_file.name:<48s}  SHAPE MISMATCH  a={tuple(a.shape)} b={tuple(b.shape)}")
            n_fail += 1
            n_total += 1
            failed_files.append(pa_file.name)
            continue
        s = _diffstats(a, b)
        n_total += 1
        ok = s["cos"] >= COS_TOL
        status = "OK  " if ok else "FAIL"
        if not ok:
            n_fail += 1
            failed_files.append(pa_file.name)
        print(
            f"{pa_file.name:<48s} {s['cos']:>10.6f} {s['mean_diff']:>+12.3e} "
            f"{s['std_diff']:>12.3e} {s['max_abs']:>12.3e} {s['ref_scale']:>12.3e}  {status}"
        )

    print(f"\n--- Summary ---")
    print(f"  total tensors compared: {n_total}")
    print(f"  failures (cos < {COS_TOL}): {n_fail}")
    if missing_in_b:
        print(f"  files in A missing from B: {len(missing_in_b)} (e.g. {missing_in_b[:3]})")
    if failed_files:
        print(f"  failing files (first 10): {failed_files[:10]}")
    return n_total, n_fail


def main() -> int:
    log_a = f"{_DUMP_ROOT}/flextrain.log"
    log_b = f"{_DUMP_ROOT}/sonicmoe.log"
    Path(_DUMP_ROOT).mkdir(parents=True, exist_ok=True)

    rc_a = _run_one("flextrain", DUMP_A, log_a)
    if rc_a != 0:
        print(f"\nFLEXTRAIN run failed (exit {rc_a}); aborting comparison.")
        return rc_a

    rc_b = _run_one("sonicmoe", DUMP_B, log_b)
    if rc_b != 0:
        print(f"\nSONICMOE run failed (exit {rc_b}); aborting comparison.")
        return rc_b

    n_total, n_fail = _compare(DUMP_A, DUMP_B)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
