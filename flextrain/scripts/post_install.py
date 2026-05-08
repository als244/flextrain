"""Post-install: fetch prebuilt flash-attn and causal-conv1d wheels.

Modern pip default-isolates the build environment for editable installs,
which means anything ``setup.py`` tries to ``pip install`` lands in the
build env and is discarded with it — never reaching the user's env. The
clean fix is to do the wheel fetch as a separate step the user runs after
``pip install -e .``.

This script:

* always tries to install ``flash_attn`` (FA2);
* additionally tries to install ``flash_attn_3`` (FA3) when the local GPU
  reports compute-capability >= 9 (Hopper / Blackwell etc.);
* always tries to install ``causal_conv1d`` from Dao-AILab's prebuilt
  wheels.

Each call uses ``--optional`` on the underlying helper, so when no
matching wheel exists for the local (python, torch, cuda) combo the
helper prints a warning and exits 0 — the install does not fail.

Opt-out env vars:
  FLEXTRAIN_SKIP_FLASH_ATTN=1
  FLEXTRAIN_SKIP_CAUSAL_CONV1D=1

CLI:
  flextrain-post-install                # auto-detect everything
  flextrain-post-install --no-flash-attn --no-causal-conv1d   # selective skip
  flextrain-post-install --dry-run      # log what would be installed
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Locate the helpers/ dir.

    Editable installs install a .pth pointing at the source tree so we
    can resolve the helpers via the package's own location:
    ``flextrain/scripts/post_install.py`` -> repo root is two up.
    For wheel installs (no source tree) the helpers/ dir won't exist
    and we fall back to the cwd, which fails with a clear message.
    """
    here = Path(__file__).resolve()
    candidate = here.parents[2] / "helpers"
    if candidate.exists():
        return candidate.parent
    # wheel-install fallback: try cwd.
    cwd_helpers = Path.cwd() / "helpers"
    if cwd_helpers.exists():
        return cwd_helpers.parent
    raise SystemExit(
        f"[flextrain-post-install] could not locate helpers/ "
        f"(checked {candidate} and {cwd_helpers}). "
        f"Run from the FlexTrain repo root or use an editable install."
    )


def _check_torch_cuda() -> tuple[int, int] | None:
    """Return (major, minor) compute-capability, or None if no CUDA GPU.

    Failures (no torch installed, no driver, no GPU) all return None so
    the caller can skip with a clear message rather than crash.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "[flextrain-post-install] torch not installed; "
            "run `pip install torch` first.",
            flush=True,
        )
        return None
    import torch  # type: ignore[no-redef]

    if not torch.cuda.is_available():
        print(
            "[flextrain-post-install] no CUDA GPU detected; nothing to "
            "install. (Set FLEXTRAIN_SKIP_FLASH_ATTN=1 / "
            "FLEXTRAIN_SKIP_CAUSAL_CONV1D=1 to silence.)",
            flush=True,
        )
        return None
    try:
        return torch.cuda.get_device_capability(0)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[flextrain-post-install] could not query GPU capability "
            f"({exc!r}); proceeding with FA2 only.",
            flush=True,
        )
        return (0, 0)


def _run_helper(
    helper: Path, *, package: str | None, dry_run: bool, label: str,
) -> int:
    cmd = [sys.executable, str(helper)]
    if package is not None:
        cmd += ["--package", package]
    cmd += ["--optional"]
    print(f"[flextrain-post-install] {label}: {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    try:
        subprocess.check_call(cmd)
        return 0
    except subprocess.CalledProcessError as exc:
        # The helper itself raises on hard errors; --optional turns the
        # "no matching wheel" case into a warning + exit 0. Reaching here
        # means something else went wrong (e.g., GitHub unreachable).
        print(
            f"[flextrain-post-install] {label} failed ({exc}); "
            f"continuing. Try running the helper manually if you need "
            f"this dependency.",
            flush=True,
        )
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch prebuilt flash-attn and causal-conv1d wheels for FlexTrain."
        ),
    )
    parser.add_argument(
        "--no-flash-attn", action="store_true",
        help="Skip flash-attn (FA2 + FA3 if Hopper).",
    )
    parser.add_argument(
        "--no-causal-conv1d", action="store_true",
        help="Skip causal-conv1d.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log helper invocations without executing.",
    )
    args = parser.parse_args(argv)

    skip_fa = args.no_flash_attn or os.environ.get(
        "FLEXTRAIN_SKIP_FLASH_ATTN") == "1"
    skip_cc = args.no_causal_conv1d or os.environ.get(
        "FLEXTRAIN_SKIP_CAUSAL_CONV1D") == "1"

    if skip_fa and skip_cc:
        print(
            "[flextrain-post-install] both flash-attn and causal-conv1d "
            "skipped via flag/env; nothing to do.",
            flush=True,
        )
        return 0

    cap = _check_torch_cuda()
    if cap is None:
        return 0

    repo_root = _repo_root()
    helpers_dir = repo_root / "helpers"
    fa_helper = helpers_dir / "install_flash_attn_wheels.py"
    cc_helper = helpers_dir / "install_causal_conv1d_wheel.py"

    if not skip_fa:
        if fa_helper.exists():
            _run_helper(
                fa_helper, package="flash_attn",
                dry_run=args.dry_run, label="flash-attn 2",
            )
            # FA3 prebuilt wheels only ship Hopper (sm_90) kernels — they
            # do not yet cover Ada (sm_89), Blackwell (sm_120), or
            # earlier sm_80/86. Restrict to exactly Hopper to avoid the
            # "no kernel image is available for execution on the device"
            # runtime error on other GPUs.
            is_hopper = cap == (9, 0)
            print(
                f"[flextrain-post-install] detected sm_{cap[0]}{cap[1]}; "
                f"flash-attn 3 = "
                f"{'enabled' if is_hopper else 'skipped (FA3 requires Hopper sm_90)'}",
                flush=True,
            )
            if is_hopper:
                _run_helper(
                    fa_helper, package="flash_attn_3",
                    dry_run=args.dry_run, label="flash-attn 3",
                )
        else:
            print(
                f"[flextrain-post-install] {fa_helper} missing; "
                "skipping flash-attn",
                flush=True,
            )

    if not skip_cc:
        if cc_helper.exists():
            _run_helper(
                cc_helper, package=None,
                dry_run=args.dry_run, label="causal-conv1d",
            )
        else:
            print(
                f"[flextrain-post-install] {cc_helper} missing; "
                "skipping causal-conv1d",
                flush=True,
            )

    print("[flextrain-post-install] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
