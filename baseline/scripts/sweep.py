#!/usr/bin/env python3
"""Run a baseline sweep from a TOML config file.

Usage:
  python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml
  python baseline/scripts/sweep.py CONFIG --backends trl_fsdp,trl_deepspeed
  python baseline/scripts/sweep.py CONFIG --num-steps 1 --dry-run

The config has a ``[common]`` section with run-wide settings (model path,
seq length, num steps, etc.) and one section per backend that adds the
backend's memory-feature knobs (offloading, checkpointing, etc.). See
``baseline/configs/llama3_128k_maxmem.toml`` for the canonical example.

Each backend section produces one run under
``baseline/runs/<config-name>_<timestamp>/<backend>/``. If a backend
fails (non-zero exit, exception, OOM), the launcher records the failure
and continues to the next backend; a summary of pass/fail is printed at
the end. The intent: when running a sweep across 5 backends, you don't
want a single OOM to abort the comparison.

Each backend's stdout streams to ``run.log`` and stderr to ``run.err``
(in the same per-backend dir) so the user can find traceback / pip /
conda activation errors quickly without grepping training stdout. On
failure the launcher prints the tail of ``run.err`` to the sweep
console with the full paths.

After the sweep, the launcher invokes
``baseline/scripts/extract_step_throughput.py`` over the per-backend
``run.log`` files to produce ``throughput.csv`` in the sweep root.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "baseline"
RUN_IN_BACKEND_ENV = BASELINE_DIR / "scripts" / "run_in_backend_env.sh"
EXTRACT_THROUGHPUT = BASELINE_DIR / "scripts" / "extract_step_throughput.py"

# Backends the sweep can dispatch. Must stay in sync with run_baseline.py's
# BACKENDS tuple. We keep a local copy here so sweep.py can be invoked from
# the system Python (without the flextrain package on PYTHONPATH).
KNOWN_BACKENDS = (
    "megatrain",
    "torchtitan",
    "trl_deepspeed",
    "deepspeed_arctic",
    "megatron",
    "trl_fsdp",
)


def _detect_num_gpus() -> int:
    """Fall back to ``nvidia-smi -L | wc -l`` when --num-gpus is not given."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 1
    return max(1, sum(1 for line in out.stdout.splitlines() if line.strip()))


def _kebab(snake: str) -> str:
    return snake.replace("_", "-")


def _render_string(value: str, ctx: dict[str, object]) -> str:
    out = value
    for key, replacement in ctx.items():
        out = out.replace("${" + key + "}", str(replacement))
    return out


def _render(value, ctx: dict[str, object]):
    if isinstance(value, str):
        return _render_string(value, ctx)
    if isinstance(value, list):
        return [_render(item, ctx) for item in value]
    return value


def build_args(
    common: dict, backend_section: dict, ctx: dict[str, object]
) -> list[str]:
    """Convert a TOML section dict into a CLI arg list for run_baseline.py.

    Conventions:
      - Each key maps to its kebab-case CLI flag
        (``activation_checkpointing`` -> ``--activation-checkpointing``).
      - Bool true -> emit the flag (no value); bool false -> omit.
      - The special key ``extra_args`` is a list of *backend-script*
        flags (passed through to e.g. megatrain's train_synthetic.py
        via ``--backend-extra-arg``). Each list element becomes one
        ``--backend-extra-arg X`` pair, so
        ``extra_args = ["--optimizer", "deepspeed_cpu_adam"]`` becomes
        ``--backend-extra-arg --optimizer --backend-extra-arg deepspeed_cpu_adam``
        on the run_baseline.py command line. Use this when you need a
        backend-specific flag that the harness doesn't model as a
        first-class TOML key.
      - Strings containing ``${NUM_GPUS}`` etc. are templated from ``ctx``.
    """
    merged: dict = {**common, **backend_section}
    args: list[str] = []
    extra: list[str] = []
    for key, value in merged.items():
        rendered = _render(value, ctx)
        if key == "extra_args":
            if not isinstance(rendered, list):
                raise SystemExit(
                    f"config key 'extra_args' must be a list of strings, got {type(rendered).__name__}"
                )
            for item in rendered:
                extra.extend(["--backend-extra-arg", str(item)])
            continue
        flag = f"--{_kebab(key)}"
        if isinstance(rendered, bool):
            if rendered:
                args.append(flag)
            continue
        if isinstance(rendered, list):
            for item in rendered:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(rendered)])
    args.extend(extra)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline sweep from a TOML config.",
    )
    parser.add_argument("config", type=Path, help="Path to a sweep TOML config.")
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help=(
            "Comma-separated backend subset to run (default: every backend "
            "section in the config)."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Override [common].num_gpus / templated ${NUM_GPUS} in the config.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override [common].num_steps. Useful for fast smoke-tests.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Parent dir for per-sweep run dirs. Default: "
            "baseline/runs/<config-name>_<timestamp>/"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands the launcher would run; do not execute.",
    )
    return parser.parse_args()


def _resolve_backends(
    arg_backends: str | None, config_sections: list[str]
) -> list[str]:
    if arg_backends:
        requested = [b.strip() for b in arg_backends.split(",") if b.strip()]
    else:
        requested = list(config_sections)
    unknown = [b for b in requested if b not in KNOWN_BACKENDS]
    if unknown:
        raise SystemExit(
            f"unknown backend(s) in --backends/config: {', '.join(unknown)}. "
            f"Known: {', '.join(KNOWN_BACKENDS)}"
        )
    missing = [b for b in requested if b not in config_sections]
    if missing:
        raise SystemExit(
            f"backend(s) requested but missing from config: {', '.join(missing)}. "
            f"Add a [{missing[0]}] section to the TOML, or drop them from --backends."
        )
    return requested


def _fmt_duration(seconds: float) -> str:
    """Render seconds as e.g. ``3s``, ``1m 24s``, or ``2h 13m 05s``."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _run_one_backend(
    backend: str,
    cli_args: list[str],
    backend_run_dir: Path,
    *,
    dry_run: bool,
    run_idx: int,
    total_runs: int,
) -> tuple[str, int, str, float]:
    """Run a single backend. Returns (backend, returncode, status, duration_s).

    Status is one of:
      - 'ok'           : exited 0
      - 'failed'       : exited non-zero (incl. missing conda env — the
                          wrapper script reports that with rc=2)
      - 'skipped_dry'  : dry-run mode, command emitted but not executed
    """
    backend_run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(RUN_IN_BACKEND_ENV),
        backend,
        "--output-dir",
        str(backend_run_dir),
        *cli_args,
    ]

    log_path = backend_run_dir / "run.log"
    err_path = backend_run_dir / "run.err"
    progress = f"[{run_idx}/{total_runs}]"

    start_dt = datetime.datetime.now()
    print(f"\n[sweep] {progress} === {backend} ===", flush=True)
    print(f"[sweep]   start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[sweep]   log:   {log_path}", flush=True)
    print(f"[sweep]   err:   {err_path}", flush=True)
    print(f"[sweep]   cmd:   {' '.join(cmd)}", flush=True)
    if dry_run:
        return backend, 0, "skipped_dry", 0.0

    # Stream stdout and stderr into separate files so the user can
    # find traceback / pip resolver / conda activation errors in
    # run.err without grepping through training stdout. Both files
    # carry a ``# command / cwd / start`` preamble so each is
    # self-describing when read in isolation.
    #
    # Pin cwd=REPO_ROOT so any relative paths in the config (the
    # canonical example: ``model_path = "models/Llama-3.1-8B"``)
    # resolve consistently regardless of where the user invoked
    # sweep.py from. Without this, running ``cd baseline && python
    # scripts/sweep.py ...`` looks for ``baseline/models/...``.
    with log_path.open("w") as log_file, err_path.open("w") as err_file:
        for f in (log_file, err_file):
            f.write(f"# command: {' '.join(cmd)}\n")
            f.write(f"# cwd:     {REPO_ROOT}\n")
            f.write(f"# start:   {start_dt.isoformat(timespec='seconds')}\n")
            f.flush()
        try:
            proc = subprocess.run(
                cmd, stdout=log_file, stderr=err_file, check=False,
                cwd=REPO_ROOT,
            )
        except Exception as exc:  # noqa: BLE001 — surface any launcher failure
            duration = (datetime.datetime.now() - start_dt).total_seconds()
            err_file.write(f"\n# launcher exception: {exc!r}\n")
            print(f"[sweep] {backend} launcher raised: {exc!r}", flush=True)
            return backend, 1, "failed", duration

    end_dt = datetime.datetime.now()
    duration = (end_dt - start_dt).total_seconds()
    rc = proc.returncode
    status = "ok" if rc == 0 else "failed"
    # Append the stop timestamp + duration to both log and err so the
    # files are still self-describing after the run.
    for path in (log_path, err_path):
        try:
            with path.open("a") as f:
                f.write(f"# stop:    {end_dt.isoformat(timespec='seconds')}\n")
                f.write(f"# rc:      {rc}\n")
                f.write(f"# elapsed: {_fmt_duration(duration)}\n")
        except OSError:
            pass
    print(f"[sweep] {progress} {backend} done  rc={rc} ({status})", flush=True)
    print(f"[sweep]   stop:     {end_dt.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[sweep]   duration: {_fmt_duration(duration)}", flush=True)
    if status == "failed":
        # Tail of run.err is usually where the real error lives — point
        # the user at the file directly so they don't have to fish.
        try:
            err_tail = err_path.read_text().splitlines()[-5:]
            if err_tail:
                print(f"[sweep]   run.err tail:", flush=True)
                for line in err_tail:
                    print(f"             {line}", flush=True)
            print(f"[sweep]   full logs: {log_path} / {err_path}", flush=True)
        except OSError:
            pass
    return backend, rc, status, duration


def _print_summary(
    results: list[tuple[str, int, str, float]], total_duration: float
) -> None:
    print("\n" + "=" * 72, flush=True)
    print("[sweep] summary", flush=True)
    print("=" * 72, flush=True)
    width = max((len(b) for b, _, _, _ in results), default=8)
    for backend, rc, status, duration in results:
        dur_str = _fmt_duration(duration) if duration > 0 else "-"
        print(
            f"  {backend.ljust(width)}  rc={rc:<3} {status:<11} elapsed={dur_str}",
            flush=True,
        )
    print("-" * 72, flush=True)
    print(
        f"  {'total'.ljust(width)}  ({len(results)} runs)         "
        f"elapsed={_fmt_duration(total_duration)}",
        flush=True,
    )


def _run_throughput_extraction(sweep_root: Path) -> None:
    """Run extract_step_throughput.py over the per-backend run.log files."""
    if not EXTRACT_THROUGHPUT.is_file():
        return
    log_paths = sorted(sweep_root.glob("*/run.log"))
    if not log_paths:
        return
    csv_path = sweep_root / "throughput.csv"
    extractor = sys.executable
    cmd = [extractor, str(EXTRACT_THROUGHPUT), *(str(p) for p in log_paths)]
    print(f"\n[sweep] extracting throughput -> {csv_path}", flush=True)
    try:
        with csv_path.open("w") as out:
            subprocess.run(cmd, stdout=out, check=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[sweep] throughput extraction failed: {exc!r}", flush=True)
        return
    if csv_path.stat().st_size > 0:
        print(f"[sweep] wrote {csv_path}", flush=True)


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        raise SystemExit(f"config file not found: {args.config}")

    payload = tomllib.loads(args.config.read_text())
    common: dict = dict(payload.get("common", {}))
    backend_sections = [k for k in payload.keys() if k != "common"]

    if args.num_steps is not None:
        common["num_steps"] = args.num_steps
    if args.num_gpus is not None:
        common["num_gpus"] = args.num_gpus
    if "num_gpus" not in common:
        common["num_gpus"] = _detect_num_gpus()
    num_gpus = int(common["num_gpus"])
    ctx = {"NUM_GPUS": num_gpus}

    backends_to_run = _resolve_backends(args.backends, backend_sections)

    if args.output_root is not None:
        sweep_root = args.output_root
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_root = (
            REPO_ROOT
            / "baseline"
            / "runs"
            / f"{args.config.stem}_{timestamp}"
        )
    sweep_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, sweep_root / args.config.name)
    sweep_start = datetime.datetime.now()
    print(f"[sweep] config:     {args.config}", flush=True)
    print(f"[sweep] sweep_root: {sweep_root}", flush=True)
    print(f"[sweep] num_gpus:   {num_gpus}", flush=True)
    print(
        f"[sweep] backends:   {', '.join(backends_to_run)} "
        f"({len(backends_to_run)} run{'s' if len(backends_to_run) != 1 else ''})",
        flush=True,
    )
    print(
        f"[sweep] start:      {sweep_start.strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )

    results: list[tuple[str, int, str, float]] = []
    total_runs = len(backends_to_run)
    for idx, backend in enumerate(backends_to_run, start=1):
        section = dict(payload.get(backend, {}))
        cli_args = build_args(common, section, ctx)
        backend_run_dir = sweep_root / backend
        result = _run_one_backend(
            backend, cli_args, backend_run_dir,
            dry_run=args.dry_run,
            run_idx=idx,
            total_runs=total_runs,
        )
        results.append(result)

    sweep_end = datetime.datetime.now()
    total_duration = (sweep_end - sweep_start).total_seconds()
    _print_summary(results, total_duration)

    if not args.dry_run:
        _run_throughput_extraction(sweep_root)

    # Exit non-zero if any backend failed. Matches CI-friendly semantics.
    bad = [b for b, _, status, _ in results if status == "failed"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
