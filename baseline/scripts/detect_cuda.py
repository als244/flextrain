#!/usr/bin/env python3
"""Detect local CUDA and derive PyTorch/FlashAttention CUDA tags."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
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


def _detect_from_nvidia_smi() -> str | None:
    output = _run(["nvidia-smi"])
    if not output:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def _detect_from_nvcc() -> str | None:
    output = _run(["nvcc", "--version"])
    if not output:
        return None
    match = re.search(r"release\s+([0-9.]+)", output)
    return match.group(1) if match else None


def _detect_from_version_json() -> str | None:
    version_path = Path("/usr/local/cuda/version.json")
    if not version_path.exists():
        return None
    try:
        payload = json.loads(version_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    version = payload.get("cuda", {}).get("version")
    return str(version) if version else None


def detect_cuda_version() -> str | None:
    versions = [
        _detect_from_nvidia_smi(),
        _detect_from_nvcc(),
        _detect_from_version_json(),
    ]
    versions = [version for version in versions if version]
    if not versions:
        return None
    return sorted(versions, key=_version_tuple, reverse=True)[0]


def torch_cuda_tag(cuda_version: str) -> str:
    major, minor = _version_tuple(cuda_version)
    if major >= 13:
        return "cu130"
    if major == 12 and minor >= 8:
        return "cu128"
    if major == 12 and minor >= 6:
        return "cu126"
    if major == 12 and minor >= 4:
        return "cu124"
    if major == 12 and minor >= 1:
        return "cu121"
    raise SystemExit(f"No known PyTorch CUDA wheel tag for local CUDA version {cuda_version}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["version", "tag", "index-url", "summary"], default="summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = detect_cuda_version()
    if version is None:
        raise SystemExit("Could not detect a local CUDA version from nvidia-smi, nvcc, or /usr/local/cuda/version.json")
    tag = torch_cuda_tag(version)
    if args.format == "version":
        print(version)
    elif args.format == "tag":
        print(tag)
    elif args.format == "index-url":
        print(f"https://download.pytorch.org/whl/{tag}")
    else:
        print(f"cuda_version={version}")
        print(f"torch_cuda_tag={tag}")
        print(f"torch_index_url=https://download.pytorch.org/whl/{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
