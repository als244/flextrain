#!/usr/bin/env python3
"""Check that the active env's torch + CUDA stack is compatible with the host driver.

Reports three numbers:
  * host_max_cuda  — the maximum CUDA runtime version the installed NVIDIA
                     driver can satisfy. Read from ``nvidia-smi`` (it prints
                     ``CUDA Version: X.Y`` at the top of its output for this
                     reason; the field reflects driver-supported CUDA, not
                     the toolkit installed under /usr/local/cuda).
  * torch_cuda     — the CUDA version the installed torch wheel was built
                     against (``torch.version.cuda``).
  * torch_version  — the torch wheel's own version (informational).

CUDA is forward-compatible at the driver-->runtime boundary. A torch wheel
built for CUDA X works on any driver that supports CUDA X or newer. So:

  torch_cuda <= host_max_cuda  ->  OK
  torch_cuda  > host_max_cuda  ->  driver can't satisfy this torch; runtime
                                   calls will fail with "forward compatibility
                                   was attempted on non supported HW" or
                                   similar. We exit 1 (or warn with --warn-only).

Used as a pre-flight in ``run_in_backend_env.sh`` so users get a clear message
*before* the backend allocates a model — far better than a cryptic CUDA error
30 seconds into the run.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _version_tuple(version: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\.(\d+)", version)
    if match is None:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


def host_max_cuda_version() -> str | None:
    """Maximum CUDA runtime version the installed driver supports.

    nvidia-smi prints this as ``CUDA Version: X.Y`` regardless of which
    toolkit (if any) is installed system-wide. ``None`` means we couldn't
    detect the driver — usually because there's no GPU on the box.
    """
    output = _run(["nvidia-smi"])
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def torch_cuda_version() -> tuple[str | None, str | None]:
    """``(torch.version.cuda, torch.__version__)`` or ``(None, None)`` if torch isn't installed."""
    try:
        import torch
    except ImportError:
        return None, None
    return torch.version.cuda, torch.__version__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print mismatches to stderr but exit 0 (default: exit 1 on incompatibility).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip the OK status line (still prints mismatches).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    host_max = host_max_cuda_version()
    torch_cuda, torch_ver = torch_cuda_version()

    if torch_cuda is None and torch_ver is None:
        # No torch in this env — nothing to check. The downstream backend
        # will fail with its own clear error if it needs torch.
        if not args.quiet:
            print("[cuda-preflight] no torch in env; skipping check", flush=True)
        return 0

    if torch_cuda is None:
        msg = (
            f"[cuda-preflight] torch={torch_ver} is a CPU-only build (no CUDA). "
            f"Install a CUDA torch wheel before running GPU backends."
        )
        print(msg, file=sys.stderr, flush=True)
        return 0 if args.warn_only else 1

    if host_max is None:
        msg = (
            f"[cuda-preflight] no NVIDIA driver detected (nvidia-smi missing or "
            f"failing); cannot verify torch={torch_ver} cuda={torch_cuda}."
        )
        print(msg, file=sys.stderr, flush=True)
        return 0 if args.warn_only else 1

    host_tup = _version_tuple(host_max)
    torch_tup = _version_tuple(torch_cuda)

    if torch_tup > host_tup:
        msg = (
            f"[cuda-preflight] INCOMPATIBLE: torch was built for CUDA {torch_cuda} "
            f"but driver supports up to CUDA {host_max}. "
            f"Install a torch wheel built for CUDA {host_max} or older "
            f"(e.g. https://download.pytorch.org/whl/cu{host_tup[0]}{host_tup[1]}), "
            f"or upgrade the NVIDIA driver. torch={torch_ver}."
        )
        print(msg, file=sys.stderr, flush=True)
        return 0 if args.warn_only else 1

    if not args.quiet:
        print(
            f"[cuda-preflight] OK: torch={torch_ver} (CUDA {torch_cuda}) "
            f"<= driver max CUDA {host_max}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
