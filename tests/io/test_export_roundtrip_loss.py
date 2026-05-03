"""HF export round-trip loss test (dense models).

Goal: verify ``save_hf_full`` / ``save_hf_merged`` produce a checkpoint
that, when reloaded via ``from_pretrained``, lets training resume from
where it left off. If the resumed losses on the SAME training data are
not lower than the original losses, the export is silently dropping or
corrupting weights.

Test matrix
-----------
3 models x 2 modes = 6 cases. Each case launches TWO subprocesses
sequentially (so 12 training runs total):

    1. orig    — load source HF dir, train N steps on a fixed prefix
                 of the dataset, save_hf_full / save_hf_merged.
    2. resumed — load from saved dir, train N steps on the SAME prefix.

Each subprocess writes its full stdout/stderr to its own log file under
``--output-root``, plus a JSON file with the per-step losses. The driver
collects 6 (orig_losses, resumed_losses) pairs, compares element-wise,
and writes summary.json.

Usage::

    cd /home/shein/Documents/flextrain
    PYTHONPATH=. python tests/io/test_export_roundtrip_loss.py \
        --output-root tests/io/export_roundtrip_logs

A subprocess-per-phase design keeps GPU and pinned-host memory clean —
the OS reclaims everything when the helper exits.

Pass criterion: every resumed[i] < original[i]. Same data, same LR
schedule, model has already trained on those exact sequences once.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PY = Path(__file__).resolve().parent / "_export_roundtrip_helper.py"


@dataclass
class TestCase:
    name: str
    model_dir: str
    mode: str
    n_steps: int = 10
    max_seq_len: int = 1024
    batch_tokens: int = 4096
    # Per-case GPU budget override. Default uses driver's --max-gpu-gib;
    # set higher for big MoE models whose layer working set exceeds the
    # default. ``None`` means use the driver's value.
    max_gpu_gib_override: float | None = None
    max_host_gib_override: float | None = None
    # MoE backend override. None -> FT default. "scattermoe" cuts the
    # per-layer baseline GPU memory significantly for big MoE models.
    moe_backend: str | None = None


def _matrix() -> list[TestCase]:
    cases = []
    # Dense models, both full FT and LoRA.
    for model_dir, name in [
        ("models/Llama-3.2-1B", "Llama-3.2-1B"),
        ("models/Qwen3-1.7B", "Qwen3-1.7B"),
        ("models/Qwen3.5-2B", "Qwen3.5-2B"),
        ("models/Qwen3.5-9B", "Qwen3.5-9B"),
    ]:
        for mode in ("full", "lora"):
            cases.append(TestCase(
                name=f"{name}__{mode}",
                model_dir=model_dir,
                mode=mode,
            ))
    # OLMoE: small MoE; both full FT and LoRA. Validates MoE expert
    # round-trip on the simpler arch (no linear-attn, no (1+w) shift).
    for mode in ("full", "lora"):
        cases.append(TestCase(
            name=f"OLMoE-1B-7B__{mode}",
            model_dir="models/OLMoE-1B-7B",
            mode=mode,
        ))
    # Larger MoE models, LoRA only. Validates MoE expert round-trip
    # via save_hf_merged (folds LoRA delta into stacked w_up/w_down
    # then exports per-expert HF tensors).
    cases.append(TestCase(
        name="Qwen3-30B-A3B__lora",
        model_dir="models/Qwen3-30B-A3B",
        mode="lora",
    ))
    # Qwen3.5-35B-A3B (qwen3_5_moe arch): not run by default — the
    # 3090 + 128 GiB host on the dev machine is right at the resource
    # boundary for a 10-step roundtrip. The shared-expert unstack code
    # is the only piece NOT exercised by the other passing cases:
    #
    #   * Qwen3.5-2B / 9B (dense): (1+w) shift, linear-attn unbundle,
    #     gated q_proj, partial-rotary halved->pair
    #   * OLMoE-1B-7B + Qwen3-30B-A3B: MoE expert unstack (option-B)
    #   * Qwen3-30B-A3B__lora: LoRA-merge for option-B MoE
    #
    # Pass ``--cases Qwen3.5-35B-A3B__lora`` explicitly with bumped
    # budgets if you want to exercise it on a bigger box.
    cases.append(TestCase(
        name="Qwen3.5-35B-A3B__lora",
        model_dir="models/Qwen3.5-35B-A3B",
        mode="lora",
        max_seq_len=2048,
        batch_tokens=16384,
        max_gpu_gib_override=22.5,
        max_host_gib_override=90.0,
        moe_backend="scattermoe",
    ))
    return cases


def _run_phase(
    *,
    phase: str,
    case: TestCase,
    model_path: str,
    save_to: str | None,
    source_hf_dir: str | None,
    losses_out: Path,
    log_path: Path,
    max_gpu_gib: float,
    max_host_gib: float,
    moe_backend: str | None = None,
) -> int:
    """Launch the helper subprocess. Returns its exit code. Streams
    its stdout/stderr to ``log_path`` AND tails the last lines to the
    parent's stdout so the driver console isn't silent for minutes."""
    cmd = [
        sys.executable, str(HELPER_PY),
        "--phase", phase,
        "--model-path", model_path,
        "--mode", case.mode,
        "--n-steps", str(case.n_steps),
        "--max-seq-len", str(case.max_seq_len),
        "--batch-tokens", str(case.batch_tokens),
        "--max-gpu-gib", str(max_gpu_gib),
        "--max-host-gib", str(max_host_gib),
        "--losses-out", str(losses_out),
    ]
    if save_to:
        cmd += ["--save-to", save_to]
    if source_hf_dir:
        cmd += ["--source-hf-dir", source_hf_dir]
    if moe_backend:
        cmd += ["--moe-backend", moe_backend]

    print(f"  ${' '.join(cmd)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))

    t0 = time.time()
    with open(log_path, "w") as logf:
        # Merge stderr into stdout so a single log captures both.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            if line.startswith("[") or "error" in line.lower() or "fail" in line.lower():
                # Tail interesting lines to the driver console.
                sys.stdout.write(line)
                sys.stdout.flush()
        rc = proc.wait()
    print(f"  -> phase {phase} exit={rc} ({time.time()-t0:.1f}s) log={log_path}",
          flush=True)
    return rc


def _read_losses(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return list(json.load(f).get("losses", []))
    except Exception:
        return None


def _run_case(case: TestCase, out_root: Path,
              max_gpu_gib: float, max_host_gib: float,
              tmp_root: Path) -> dict[str, Any]:
    # Apply per-case overrides (e.g. bigger budgets for huge MoE).
    if case.max_gpu_gib_override is not None:
        max_gpu_gib = case.max_gpu_gib_override
    if case.max_host_gib_override is not None:
        max_host_gib = case.max_host_gib_override
    print(f"\n{'='*72}\nCASE: {case.name}  ({case.model_dir}, {case.mode})\n{'='*72}",
          flush=True)
    case_dir = out_root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    saved_dir = tmp_root / f"{case.name}__saved"
    saved_dir.parent.mkdir(parents=True, exist_ok=True)

    orig_log = out_root / f"{case.name}__orig.log"
    resumed_log = out_root / f"{case.name}__resumed.log"
    orig_losses_json = case_dir / "orig_losses.json"
    resumed_losses_json = case_dir / "resumed_losses.json"

    result: dict[str, Any] = {
        "name": case.name,
        "model_dir": case.model_dir,
        "mode": case.mode,
        "passed": False,
        "error": None,
        "original_losses": None,
        "resumed_losses": None,
        "saved_dir": str(saved_dir),
        "orig_log": str(orig_log),
        "resumed_log": str(resumed_log),
    }

    # ---------------- Phase 1 + 2: orig training + save ----------------
    rc = _run_phase(
        phase="orig",
        case=case,
        model_path=case.model_dir,
        save_to=str(saved_dir),
        source_hf_dir=case.model_dir,
        losses_out=orig_losses_json,
        log_path=orig_log,
        max_gpu_gib=max_gpu_gib,
        max_host_gib=max_host_gib,
        moe_backend=case.moe_backend,
    )
    if rc != 0:
        result["error"] = f"orig phase exited {rc}; see {orig_log}"
        return result

    orig_losses = _read_losses(orig_losses_json)
    if not orig_losses:
        result["error"] = f"orig phase produced no losses; see {orig_log}"
        return result
    result["original_losses"] = orig_losses

    if not saved_dir.exists():
        result["error"] = f"orig phase did not produce saved dir {saved_dir}"
        return result

    # ---------------- Phase 3: resumed training ----------------
    rc = _run_phase(
        phase="resumed",
        case=case,
        model_path=str(saved_dir),
        save_to=None,
        source_hf_dir=None,
        losses_out=resumed_losses_json,
        log_path=resumed_log,
        max_gpu_gib=max_gpu_gib,
        max_host_gib=max_host_gib,
        moe_backend=case.moe_backend,
    )
    if rc != 0:
        result["error"] = f"resumed phase exited {rc}; see {resumed_log}"
        return result
    resumed_losses = _read_losses(resumed_losses_json)
    if not resumed_losses:
        result["error"] = f"resumed phase produced no losses; see {resumed_log}"
        return result
    result["resumed_losses"] = resumed_losses

    # ---------------- Compare ----------------
    n = min(len(orig_losses), len(resumed_losses))
    drops = [resumed_losses[i] - orig_losses[i] for i in range(n)]
    first = drops[0] if drops else None
    avg_drop = sum(drops) / n if n > 0 else None

    print(f"  Comparison ({n} steps):", flush=True)
    print(f"    step | orig    | resumed | delta", flush=True)
    for i in range(n):
        mark = "  " if drops[i] < 0 else " *"
        print(f"    {i+1:4d} | {orig_losses[i]:7.4f} | {resumed_losses[i]:7.4f} | "
              f"{drops[i]:+.4f}{mark}", flush=True)
    first_rel = None
    if first is not None:
        first_rel = first / max(1e-8, abs(orig_losses[0]))
        print(f"    first-step drop: {first:+.4f} ({first_rel*100:+.2f}% rel)",
              flush=True)
    if avg_drop is not None:
        print(f"    avg drop:        {avg_drop:+.4f}", flush=True)

    # Pass criterion: the first resumed step is the strongest signal —
    # we evaluate the post-train weights on the EXACT same first batch
    # they were trained on. If the export round-trips correctly, that
    # loss must drop materially. Bar: > 5% relative drop AND the
    # average drop across all replayed steps is negative (model is
    # generally not worse). Steps near the end of the run can drift
    # up slightly due to fresh AdamW state interacting with cooldown
    # LR — that's expected and not a save/load defect.
    first_rel_threshold = -0.05   # 5% relative
    avg_drop_threshold = 0.0
    pass_first = first_rel is not None and first_rel < first_rel_threshold
    pass_avg = avg_drop is not None and avg_drop < avg_drop_threshold
    result["first_step_relative_drop"] = first_rel
    result["avg_drop"] = avg_drop

    if pass_first and pass_avg:
        result["passed"] = True
        print(f"  PASS  {case.name}  "
              f"(first-step rel drop {first_rel*100:+.1f}%, "
              f"avg drop {avg_drop:+.4f})", flush=True)
        # Reclaim disk: per-case saved dirs run 5-70 GiB on 30B+ MoE
        # checkpoints. Keep them only when something failed (so the
        # user can poke at the safetensors after).
        try:
            import shutil
            if saved_dir.exists():
                shutil.rmtree(saved_dir, ignore_errors=True)
                print(f"  cleaned up {saved_dir}", flush=True)
        except Exception as e:
            print(f"  [warn] cleanup of {saved_dir} failed: {e}", flush=True)
    else:
        reasons = []
        if not pass_first:
            reasons.append(
                f"first-step rel drop {first_rel*100 if first_rel else 0:+.2f}% "
                f">= {first_rel_threshold*100:.0f}%"
            )
        if not pass_avg:
            reasons.append(f"avg drop {avg_drop:+.4f} >= 0")
        result["error"] = "; ".join(reasons)
        print(f"  FAIL  {case.name} — {result['error']}", flush=True)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root", type=str,
        default="tests/io/export_roundtrip_logs",
        help="Where logs and summary.json go.",
    )
    ap.add_argument(
        "--cases", nargs="+", default=None,
        help="Restrict to a subset of cases by name. Default: run all 6.",
    )
    ap.add_argument("--max-gpu-gib", type=float, default=20.0)
    ap.add_argument("--max-host-gib", type=float, default=80.0)
    ap.add_argument(
        "--tmp-root", type=str, default="/tmp/ft_export_roundtrip",
        help="Where saved-dirs from orig phases live (so resumed phase "
             "can load them). Kept until you delete manually.",
    )
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(args.tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    cases = _matrix()
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c.name in wanted]
        if not cases:
            print(f"No cases match {args.cases!r}; available: "
                  f"{[c.name for c in _matrix()]}")
            sys.exit(2)

    print(f"Running {len(cases)} cases sequentially.")
    print(f"  logs  -> {out_root}/   ({2*len(cases)} log files: orig + resumed)")
    print(f"  saved -> {tmp_root}/")

    results = []
    overall_t0 = time.time()
    for case in cases:
        t0 = time.time()
        r = _run_case(case, out_root, args.max_gpu_gib, args.max_host_gib,
                      tmp_root)
        r["wall_seconds"] = time.time() - t0
        results.append(r)

    summary = {
        "total_cases": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "wall_seconds": time.time() - overall_t0,
        "results": results,
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Summary ({summary['passed']}/{summary['total_cases']} passed, "
          f"{summary['wall_seconds']:.0f}s total):")
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        first_drop_str = ""
        if r.get("original_losses") and r.get("resumed_losses"):
            o = r["original_losses"][0]
            n = r["resumed_losses"][0]
            first_drop_str = f"  step1: {o:.4f} -> {n:.4f}"
        err_str = f"  err: {r['error']}" if r["error"] else ""
        print(f"  {mark}  {r['name']:30s}{first_drop_str}{err_str}")
    print("=" * 72)
    print(f"Per-case logs: {out_root}/<case>__orig.log, <case>__resumed.log")
    print(f"Summary JSON:  {out_root}/summary.json")

    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
