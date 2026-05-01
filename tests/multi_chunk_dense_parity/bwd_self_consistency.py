"""Backward-pass self-consistency for multi-chunk linear-attn (Item 3c).

Runs FT fwd+bwd on the SAME sequence at TWO chunk sizes:

* Reference: chunk_size = full seq length → single chunk → no
  cross-chunk plumbing exercised. This is the known-correct path
  (validated by ``tests/test_arch_lora_e2e.py`` against HF-PEFT).

* Test: chunk_size < seq length → multiple chunks → cross-chunk
  plumbing exercised in fwd AND bwd.

Compares LoRA A/B grads tensor-by-tensor. If multi-chunk grads
match single-chunk grads within bf16 noise, the cross-chunk bwd
machinery is correct relative to the reference.

This is a self-consistency test, not a cross-implementation parity
test. Its validity rests on:
  (a) The single-chunk path being correct (verified by 50-step
      LoRA parity vs HF-PEFT in test_arch_lora_e2e).
  (b) FLA's chunk_gated_delta_rule_fwd/bwd producing identical
      results when given the right initial_state/dht — which is
      what we're testing.

Usage:
    python tests/multi_chunk_dense_parity/bwd_self_consistency.py \\
        --model models/Qwen3.5-2B \\
        --target-tokens 8000 \\
        --ref-chunk-size 8192 \\
        --test-chunk-size 2000

Defaults are tuned for a 24GB card running Qwen3.5-2B.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shlex
import subprocess
import sys
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIXTURE = os.path.join(ROOT, "tests/fixtures/long_real_sample.txt")


def _run_one_pass(
    *,
    py: str,
    env: dict,
    model: str,
    fixture: str,
    target_tokens: int,
    chunk_size: int,
    max_gpu_gib: float,
    out_path: str,
    label: str,
    lora_init_path: str | None,
) -> int:
    """Spawn a subprocess running ``_inner.py`` to do one fwd+bwd
    pass at ``chunk_size`` and pickle out the LoRA grads.

    Subprocesses to keep GPU memory bounded — each pass gets a
    clean engine alloc.
    """
    cmd = [
        py, os.path.join(HARNESS_DIR, "bwd_self_consistency_inner.py"),
        "--model", model,
        "--fixture", fixture,
        "--target-tokens", str(target_tokens),
        "--chunk-size", str(chunk_size),
        "--max-gpu-gib", str(max_gpu_gib),
        "--out", out_path,
    ]
    if lora_init_path is not None:
        cmd += ["--lora-init", lora_init_path]
    print(f"\n[{label}] {' '.join(shlex.quote(s) for s in cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=ROOT)
    print(f"[{label}] exit={proc.returncode}  in {time.time()-t0:.1f}s", flush=True)
    return proc.returncode


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


def _compare_grads(ref_path: str, test_path: str) -> tuple[bool, dict]:
    """Load two grad bundles and compare element-wise."""
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)
    with open(test_path, "rb") as f:
        test = pickle.load(f)

    # Both bundles: dict[str, torch.Tensor].
    print(f"\n=== Compare ===")
    print(f"  ref keys: {len(ref)}  test keys: {len(test)}")
    if set(ref.keys()) != set(test.keys()):
        only_ref = set(ref.keys()) - set(test.keys())
        only_test = set(test.keys()) - set(ref.keys())
        print(f"  KEY MISMATCH: only in ref={only_ref}  only in test={only_test}")
        return False, {"key_mismatch": True}

    rows: list[dict] = []
    overall_max_abs = 0.0
    overall_max_rel = 0.0
    n_zero_ref = 0
    n_zero_test = 0
    sum_abs = 0.0
    sum_count = 0
    for name in sorted(ref.keys()):
        a = ref[name].float()
        b = test[name].float()
        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            return False, {"shape_mismatch": name}
        diff = (a - b).abs()
        max_abs = float(diff.max().item())
        norm_a = float(a.norm().item())
        rel = max_abs / max(norm_a, 1e-12)
        mean_abs = float(diff.mean().item())
        # Track sparsity of grads — a grad of all zeros means LoRA
        # B init was zero and bwd produced zero (or LoRA wasn't
        # touched). Useful sanity.
        if norm_a == 0.0:
            n_zero_ref += 1
        if float(b.norm().item()) == 0.0:
            n_zero_test += 1
        overall_max_abs = max(overall_max_abs, max_abs)
        overall_max_rel = max(overall_max_rel, rel)
        sum_abs += float(diff.sum().item())
        sum_count += int(diff.numel())
        rows.append({
            "name": name,
            "shape": list(a.shape),
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rel": rel,
            "ref_norm": norm_a,
        })

    overall_mean_abs = sum_abs / max(sum_count, 1)
    print(f"  overall max|Δ|        = {overall_max_abs:.5f}")
    print(f"  overall mean|Δ|       = {overall_mean_abs:.5e}")
    print(f"  overall max rel       = {overall_max_rel:.5e}")
    print(f"  ref grads with norm=0 = {n_zero_ref}/{len(ref)}")
    print(f"  test grads with norm=0= {n_zero_test}/{len(test)}")

    print(f"\n  worst (by max_abs):")
    rows_sorted = sorted(rows, key=lambda r: r["max_abs"], reverse=True)
    for r in rows_sorted[:8]:
        print(
            f"    {r['name']:60s}  shape={tuple(r['shape'])!s:25s}  "
            f"max|Δ|={r['max_abs']:.5e}  rel={r['rel']:.3e}  "
            f"||ref||={r['ref_norm']:.3e}"
        )
    print(f"\n  worst (by rel):")
    rows_sorted = sorted(rows, key=lambda r: r["rel"], reverse=True)
    for r in rows_sorted[:8]:
        print(
            f"    {r['name']:60s}  shape={tuple(r['shape'])!s:25s}  "
            f"max|Δ|={r['max_abs']:.5e}  rel={r['rel']:.3e}  "
            f"||ref||={r['ref_norm']:.3e}"
        )

    return True, {
        "max_abs": overall_max_abs,
        "mean_abs": overall_mean_abs,
        "max_rel": overall_max_rel,
        "n_grads": len(ref),
        "n_zero_ref": n_zero_ref,
        "n_zero_test": n_zero_test,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-2B")
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE)
    ap.add_argument("--target-tokens", type=int, default=8000)
    ap.add_argument("--ref-chunk-size", type=int, default=8192,
                    help="Chunk size for reference run (>= target_tokens to "
                         "force single chunk).")
    ap.add_argument("--test-chunk-size", type=int, default=2000,
                    help="Chunk size for test run (multi-chunk path).")
    ap.add_argument("--max-gpu-gib", type=float, default=18.0)
    ap.add_argument("--rel-thresh", type=float, default=5e-2,
                    help="Acceptance bar: overall max rel < this.")
    args = ap.parse_args()

    env = _build_env()
    py = sys.executable

    print(f"=== Backward self-consistency: {args.model} ===")
    print(f"  target_tokens={args.target_tokens}")
    print(f"  ref_chunk_size={args.ref_chunk_size}  test_chunk_size={args.test_chunk_size}")
    print(f"  rel_thresh={args.rel_thresh}")

    with tempfile.TemporaryDirectory() as tmp:
        ref_grads = os.path.join(tmp, "ref_grads.pkl")
        test_grads = os.path.join(tmp, "test_grads.pkl")
        lora_init = os.path.join(tmp, "lora_init.pkl")

        rc = _run_one_pass(
            py=py, env=env,
            model=args.model, fixture=args.fixture,
            target_tokens=args.target_tokens,
            chunk_size=args.ref_chunk_size,
            max_gpu_gib=args.max_gpu_gib,
            out_path=ref_grads,
            label=f"ref chunk={args.ref_chunk_size}",
            lora_init_path=None,  # ref pass writes the init
        )
        if rc != 0:
            print("  ref pass failed")
            return 1
        # Move the lora_init bundle written alongside ref_grads.
        ref_lora_init = ref_grads + ".lora_init"
        if not os.path.exists(ref_lora_init):
            print(f"  ref pass did not write lora_init at {ref_lora_init}")
            return 1
        os.rename(ref_lora_init, lora_init)

        rc = _run_one_pass(
            py=py, env=env,
            model=args.model, fixture=args.fixture,
            target_tokens=args.target_tokens,
            chunk_size=args.test_chunk_size,
            max_gpu_gib=args.max_gpu_gib,
            out_path=test_grads,
            label=f"test chunk={args.test_chunk_size}",
            lora_init_path=lora_init,
        )
        if rc != 0:
            print("  test pass failed")
            return 1

        ok, summary = _compare_grads(ref_grads, test_grads)
        if not ok:
            print("  FAIL: incompatible bundles")
            return 1
        passed = summary["max_rel"] < args.rel_thresh
        print(f"\n  {'PASS' if passed else 'FAIL'} (rel {summary['max_rel']:.3e} vs threshold {args.rel_thresh:.0e})")
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
