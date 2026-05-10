"""Systematic memory-prediction-vs-reality matrix across arch types.

Runs a 1-step LoRA fwd_bwd on representative models from each arch class
and reports for each:
   * Solver `Expected GPU Memory Usage`  (its est)
   * Actual `max_allocated`              (PyTorch live peak)
   * Actual `max_reserved`               (PyTorch caching peak incl. slop)
   * Gap                                 (actual_alloc - solver_est)
   * Pass criterion: gap within ±1.5 GiB and no OOM

Arch classes covered:
   * dense (no MoE, full-attn only):          Llama-3.2-1B, Qwen3-8B
   * MoE + full-attn (LoRA-all):              OLMoE-7B-A1B, Qwen3-30B-A3B
   * hybrid (linear+full attn, dense MLP):    Qwen3.5-9B
   * hybrid + MoE (Qwen3.5-MoE):              Qwen3.5-MoE-35B-A3B (later)

Each row runs as its own subprocess so GPU/host memory state is fresh.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = "/home/shein/Documents/grad_school/research/flextrain/models"

CONFIGS = [
    # (label, arch_class, model_path, mode, gpu_cap_gib, host_cap_gib)
    ("Llama-3.2-1B          (dense, LoRA)", "dense",
     f"{MODELS}/Llama-3.2-1B", "lora", 28, 80),
    ("Qwen3-8B              (dense+QK-norm, LoRA)", "dense",
     f"{MODELS}/Qwen3-8B", "lora", 28, 100),
    ("OLMoE-7B-A1B          (MoE, LoRA)", "moe",
     f"{MODELS}/OLMoE-7B-A1B", "lora", 28, 100),
    ("OLMoE-7B-A1B          (MoE, full FT)", "moe",
     f"{MODELS}/OLMoE-7B-A1B", "full", 28, 100),
    ("Qwen3.5-9B            (hybrid lin+full, LoRA)", "hybrid_dense",
     f"{MODELS}/Qwen3.5-9B", "lora", 28, 120),
    # Larger hybrid-dense (27B class) — same arch family as Qwen3.5-9B
    # but ~3x params; tests how the persistence terms scale with d_model
    # and num_layers. (Qwen3.5-27B is not on local disk; using Qwen3.6-27B
    # which is the same hybrid-linear+full / dense-MLP architecture.)
    ("Qwen3.6-27B           (hybrid lin+full, LoRA)", "hybrid_dense",
     f"{MODELS}/Qwen3.6-27B", "lora", 28, 100),
    ("Qwen3-30B-A3B         (MoE, LoRA)", "moe",
     f"{MODELS}/Qwen3-30B-A3B", "lora", 28, 100),
    # Hybrid + MoE LoRA combo: heaviest config, needs more host headroom
    # (~70 GiB base pinned, the matrix runs as separate processes so 100
    # GiB host cap fits comfortably).
    ("Qwen3.5-MoE-35B-A3B   (hybrid+MoE, LoRA)", "hybrid_moe",
     f"{MODELS}/Qwen3.5-35B-A3B", "lora", 28, 100),
]


def run_one(cfg):
    label, arch_class, mp, mode, gpu_cap, host_cap = cfg
    print(f"\n{'='*70}\n  {label}\n{'='*70}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-u", "experiments/_mem_diag.py",
         "--cap", str(gpu_cap)],
        env={
            **os.environ,
            "FLEXTRAIN_DBG_MEM": "0",
            # override the diag's hardcoded model path
            "MEM_DIAG_MODEL": mp,
            "MEM_DIAG_MODE": mode,
            "MEM_DIAG_HOST_CAP": str(host_cap),
        },
        capture_output=True, text=True, timeout=900,
    )
    log = proc.stdout
    summary = {
        "label": label, "arch_class": arch_class,
        "model_path": mp, "mode": mode,
        "gpu_cap_gib": gpu_cap, "host_cap_gib": host_cap,
        "rc": proc.returncode,
    }
    # Pull solver est + per-step peaks from the log.
    per_step = []
    for line in log.splitlines():
        if "Expected GPU Memory Usage:" in line:
            try:
                est = float(line.split("Expected GPU Memory Usage:")[1]
                            .split("GiB")[0].strip())
                summary["solver_est_gib"] = est
            except Exception:
                pass
        if line.strip().startswith("[mem] after_last_step_or_oom") or \
           line.strip().startswith("[mem] after_step1_or_oom"):  # back-compat
            parts = line.replace("=", " ").split()
            try:
                summary["after_step_max_alloc"] = float(parts[parts.index("max_alloc") + 1])
                summary["after_step_max_reserved"] = float(parts[parts.index("max_reserved") + 1])
            except Exception:
                pass
        if "OutOfMemoryError" in line:
            summary["oom"] = True
        # Per-step peaks: "[step] step=N loss=X dt=Ys max_alloc=A max_reserved=B"
        if line.strip().startswith("[step] step="):
            try:
                step_n = int(line.split("step=")[1].split(" ")[0])
                loss = float(line.split("loss=")[1].split(" ")[0])
                ma = float(line.split("max_alloc=")[1].split(" ")[0])
                mr = float(line.split("max_reserved=")[1].split(" ")[0])
                dt = float(line.split("dt=")[1].split("s")[0])
                per_step.append({"step": step_n, "loss": loss,
                                 "max_alloc": ma, "max_reserved": mr,
                                 "dt_s": dt})
            except Exception:
                pass
    summary["per_step"] = per_step
    if per_step:
        summary["peak_max_alloc"] = max(p["max_alloc"] for p in per_step)
        summary["peak_max_reserved"] = max(p["max_reserved"] for p in per_step)
        summary["last_loss"] = per_step[-1]["loss"]
        summary["last_dt_s"] = per_step[-1]["dt_s"]
    summary["log_tail"] = log.splitlines()[-30:]
    return summary


def main():
    rows = []
    for cfg in CONFIGS:
        s = run_one(cfg)
        rows.append(s)
        # short status line
        oom = s.get("oom", False)
        est = s.get("solver_est_gib", float("nan"))
        peak = s.get("after_step_max_alloc", float("nan"))
        gap = (peak - est) if not oom else float("nan")
        loss = s.get("step_loss", "—")
        dt = s.get("step_dt_s", "—")
        print(f"\n  >> est={est:.2f}  peak_alloc={peak:.2f}  gap={gap:+.2f}  "
              f"loss={loss}  dt={dt}s  oom={oom}", flush=True)

    out = Path("runs/_mem_diag_matrix.json")
    out.write_text(json.dumps(rows, indent=2))
    print("\n\nFINAL TABLE  (peaks across all 3 steps; OOM check uses peak_reserved)\n" + "=" * 90)
    print(f"{'config':45s} {'est':>6s} {'p_alloc':>8s} {'p_resv':>8s} {'gap_alloc':>10s} {'gap_resv':>9s} {'OK?':>4s}")
    print("-" * 90)
    for r in rows:
        est = r.get("solver_est_gib", float("nan"))
        p_alloc = r.get("peak_max_alloc", float("nan"))
        p_resv = r.get("peak_max_reserved", float("nan"))
        oom = r.get("oom", False)
        gap_a = (p_alloc - est) if not oom else float("nan")
        gap_r = (p_resv - est) if not oom else float("nan")
        # Pass: no OOM AND peak reserved within ~1.5 GiB of est.
        ok = (not oom) and abs(gap_r) < 1.5
        marker = "PASS" if ok else "FAIL"
        print(f"{r['label']:45s} {est:6.2f} {p_alloc:8.2f} {p_resv:8.2f} "
              f"{gap_a:+10.2f} {gap_r:+9.2f} {marker:>4s}")
    print(f"\nFull data written to {out}")


if __name__ == "__main__":
    main()
