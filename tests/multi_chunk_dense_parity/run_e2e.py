"""End-to-end driver for multi-chunk dense-attn parity.

Runs hf_capture.py, ft_replay.py, compare.py as separate subprocesses
in sequence so each phase has the full GPU to itself. (Both HF logits
and FT logits at 32k tokens × 248k vocab don't fit alongside model
state on a 24GB card; isolating the runs is the only way.)

Defaults reproduce the Stage 3a / Stage 3b numbers from the agenda:

    python tests/multi_chunk_dense_parity/run_e2e.py

That runs Llama-3.2-1B, Qwen3-1.7B, Qwen3.5-2B at chunk_size=8192,
max_tokens=32000, against the LongBench-v2 fixture. Outputs JSON stats
under tests/multi_chunk_logs/<model>__chunk<N>.json.

To target one specific model:

    python tests/multi_chunk_dense_parity/run_e2e.py --model models/Llama-3.2-1B
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARNESS_DIR = os.path.join(ROOT, "tests/multi_chunk_dense_parity")
LOGS_DIR = os.path.join(ROOT, "tests/multi_chunk_logs")
DEFAULT_FIXTURE = os.path.join(ROOT, "tests/fixtures/long_real_sample.txt")

DEFAULT_MODELS = [
    "models/Llama-3.2-1B",
    "models/Qwen3-1.7B",
    "models/Qwen3.5-2B",
]

# CUDA shim: flash_attn 2.8.x in this env is built against cu12 but
# torch is cu13, so libcudart.so.12 must be on LD_LIBRARY_PATH. The
# conda activation script handles it automatically; subprocesses we
# spawn here don't activate, so we replicate the shim once at the top.
def _build_env() -> dict:
    env = dict(os.environ)
    conda_prefix = env.get("CONDA_PREFIX")
    if not conda_prefix:
        conda_prefix = "/home/shein/miniconda3/envs/flextrain"
        env["CONDA_PREFIX"] = conda_prefix
    cu12 = os.path.join(
        conda_prefix, "lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    )
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{cu12}:{existing}" if existing else cu12
    )
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return env


def _python_bin() -> str:
    """Use the same python the driver was invoked with — robust whether
    you call us via /home/shein/miniconda3/envs/flextrain/bin/python or
    via `conda run -n flextrain python`."""
    return sys.executable


def _run(cmd: list[str], env: dict, dry_run: bool, label: str) -> int:
    print(f"\n[{label}] {' '.join(shlex.quote(s) for s in cmd)}", flush=True)
    if dry_run:
        return 0
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=ROOT)
    print(f"[{label}] exit={proc.returncode}  in {time.time()-t0:.1f}s", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--model", action="append", default=None,
                    help="HF model dir to test. Repeat for multiple. "
                         "Default: Llama-3.2-1B, Qwen3-1.7B, Qwen3.5-2B.")
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE,
                    help="Long-text fixture (default: tests/fixtures/long_real_sample.txt)")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--chunk-size", type=int, default=8192,
                    help="FT max_chunk_size (default 8192 → ~4 chunks at 32k tokens)")
    ap.add_argument("--max-gpu-gib", type=float, default=14.0,
                    help="GPU budget for FT engine (default 14 GiB on 24GB card)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--logs-dir", default=LOGS_DIR)
    ap.add_argument("--hf-attn-impl", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--skip-hf-capture", action="store_true",
                    help="Skip hf_capture if bundle exists")
    ap.add_argument("--skip-ft-replay", action="store_true",
                    help="Skip ft_replay if bundle exists")
    ap.add_argument("--keep-bundles", action="store_true",
                    help="Don't delete .pt bundles after compare (default: clean up)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = args.model if args.model else DEFAULT_MODELS
    os.makedirs(args.logs_dir, exist_ok=True)
    env = _build_env()
    py = _python_bin()

    summary: list[tuple[str, int, str]] = []
    for model in models:
        model_short = os.path.basename(model.rstrip("/"))
        hf_bundle = os.path.join(args.logs_dir, f"{model_short}__hf.pt")
        ft_bundle = os.path.join(args.logs_dir, f"{model_short}__ft_chunk{args.chunk_size}.pt")
        out_json = os.path.join(args.logs_dir, f"{model_short}__chunk{args.chunk_size}.json")

        print(f"\n========================================")
        print(f"= Model: {model}  (chunk={args.chunk_size})")
        print(f"========================================", flush=True)

        # 1. HF capture
        if args.skip_hf_capture and os.path.exists(hf_bundle):
            print(f"  HF bundle exists at {hf_bundle}; skipping capture")
        else:
            cmd = [
                py, os.path.join(HARNESS_DIR, "hf_capture.py"),
                "--model", model,
                "--fixture", args.fixture,
                "--max-tokens", str(args.max_tokens),
                "--out", hf_bundle,
                "--device", args.device,
                "--attn-impl", args.hf_attn_impl,
            ]
            rc = _run(cmd, env, args.dry_run, f"hf_capture {model_short}")
            if rc != 0:
                summary.append((model_short, rc, "hf_capture failed"))
                continue

        # 2. FT replay
        if args.skip_ft_replay and os.path.exists(ft_bundle):
            print(f"  FT bundle exists at {ft_bundle}; skipping replay")
        else:
            cmd = [
                py, os.path.join(HARNESS_DIR, "ft_replay.py"),
                "--hf-capture", hf_bundle,
                "--chunk-size", str(args.chunk_size),
                "--max-gpu-gib", str(args.max_gpu_gib),
                "--out", ft_bundle,
                "--device", args.device,
            ]
            rc = _run(cmd, env, args.dry_run, f"ft_replay {model_short}")
            if rc != 0:
                summary.append((model_short, rc, "ft_replay failed"))
                continue

        # 3. Compare (no GPU needed; small CPU job)
        cmd = [
            py, os.path.join(HARNESS_DIR, "compare.py"),
            "--hf", hf_bundle,
            "--ft", ft_bundle,
            "--out", out_json,
        ]
        rc = _run(cmd, env, args.dry_run, f"compare {model_short}")
        summary.append((model_short, rc, out_json))

        # Clean up bundles unless --keep-bundles. Each bundle is 7-15 GiB
        # on a 24GB-vocab model, so default-cleanup keeps disk sane when
        # running multiple models back-to-back.
        if not args.keep_bundles and not args.dry_run:
            for path in (hf_bundle, ft_bundle):
                if os.path.exists(path):
                    sz_mib = os.path.getsize(path) / (1 << 20)
                    os.remove(path)
                    print(f"  cleaned {path} ({sz_mib:.0f} MiB)")

    print("\n========================================")
    print("= Summary")
    print("========================================")
    for name, rc, info in summary:
        status = "PASS" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  {name:30s}  {status}  {info}")

    return 0 if all(rc == 0 for _, rc, _ in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
