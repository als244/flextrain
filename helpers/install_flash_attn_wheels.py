#!/usr/bin/env python3
"""Install matching prebuilt FlashAttention wheels from GitHub releases."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

REPO = "mjun0812/flash-attention-prebuild-wheels"


@dataclass(frozen=True)
class Asset:
    release: str
    name: str
    url: str


def _torch_tags() -> tuple[str, str]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install torch before installing FlashAttention wheels.") from exc

    torch_version = torch.__version__.split("+", 1)[0]
    match = re.match(r"(\d+)\.(\d+)", torch_version)
    if match is None:
        raise SystemExit(f"Could not parse torch version from {torch.__version__!r}")

    cuda_version = torch.version.cuda
    if not cuda_version:
        raise SystemExit("The installed torch build does not report CUDA; no CUDA FlashAttention wheel can be selected.")
    cuda_parts = cuda_version.split(".")
    if len(cuda_parts) < 2:
        raise SystemExit(f"Could not parse CUDA version from {cuda_version!r}")

    return f"torch{match.group(1)}.{match.group(2)}", f"cu{cuda_parts[0]}{cuda_parts[1]}"


def _python_tags() -> list[str]:
    major = sys.version_info.major
    minor = sys.version_info.minor
    tags = [f"cp{major}{minor}-cp{major}{minor}"]
    if major == 3 and minor >= 9:
        tags.append("cp39-abi3")
    return tags


def _arch_tag() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    raise SystemExit(f"Unsupported machine architecture for prebuilt wheels: {machine}")


def _load_releases(repo: str) -> list[dict]:
    releases: list[dict] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            chunk = json.load(response)
        if not chunk:
            break
        releases.extend(chunk)
        page += 1
    return releases


def _asset_version(asset: Asset) -> Version:
    match = re.match(r"[^-]+-([^+]+)", asset.name)
    if match is None:
        return Version("0")
    try:
        return Version(match.group(1))
    except InvalidVersion:
        return Version("0")


def _asset_score(asset: Asset) -> tuple[Version, int, int, str]:
    plain_platform = 1 if re.search(r"-(linux_x86_64|linux_aarch64|win_amd64)\.whl$", asset.name) else 0
    manylinux = 1 if "manylinux" in asset.name else 0
    return _asset_version(asset), plain_platform, manylinux, asset.name


def _find_asset(
    package: str,
    *,
    repo: str,
    torch_tag: str,
    cuda_tag: str,
    python_tags: list[str],
    arch_tag: str,
    version: str | None,
) -> Asset | None:
    prefix = f"{package}-"
    matches: list[Asset] = []
    for release in _load_releases(repo):
        for raw_asset in release.get("assets", []):
            name = raw_asset.get("name", "")
            if not name.startswith(prefix):
                continue
            if version is not None and not name.startswith(f"{prefix}{version}+"):
                continue
            if cuda_tag not in name or torch_tag not in name or arch_tag not in name:
                continue
            if not any(tag in name for tag in python_tags):
                continue
            matches.append(
                Asset(
                    release=release.get("tag_name", ""),
                    name=name,
                    url=raw_asset.get("browser_download_url", ""),
                )
            )
    if matches:
        return sorted(matches, key=_asset_score, reverse=True)[0]
    return None


def _install(asset: Asset, *, dry_run: bool) -> None:
    print(f"selected {asset.name} from {asset.release}", flush=True)
    print(asset.url, flush=True)
    if dry_run:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", asset.url])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", choices=["flash_attn", "flash_attn_3", "both"], default="both")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--torch-tag", default=None, help="Override detected torch tag, for example torch2.8")
    parser.add_argument("--cuda-tag", default=None, help="Override detected CUDA tag, for example cu128")
    parser.add_argument("--python-tag", action="append", default=None, help="Override Python ABI tag. Repeatable.")
    parser.add_argument("--arch-tag", default=None, help="Override architecture tag, for example x86_64")
    parser.add_argument("--version", default=None, help="Require an exact FlashAttention package version, for example 2.8.3")
    parser.add_argument("--optional", action="store_true", help="Skip packages with no matching wheel instead of failing.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.torch_tag is not None and args.cuda_tag is not None:
        torch_tag = args.torch_tag
        cuda_tag = args.cuda_tag
    else:
        try:
            detected_torch_tag, detected_cuda_tag = _torch_tags()
        except SystemExit as exc:
            if args.optional:
                print(f"warning: {exc}", flush=True)
                return 0
            raise
        torch_tag = args.torch_tag or detected_torch_tag
        cuda_tag = args.cuda_tag or detected_cuda_tag
    python_tags = args.python_tag or _python_tags()
    arch_tag = args.arch_tag or _arch_tag()

    packages = ["flash_attn", "flash_attn_3"] if args.package == "both" else [args.package]
    print(
        "flash_attention_wheel_query "
        f"repo={args.repo} torch={torch_tag} cuda={cuda_tag} python={python_tags} arch={arch_tag}",
        flush=True,
    )
    for package in packages:
        asset = _find_asset(
            package,
            repo=args.repo,
            torch_tag=torch_tag,
            cuda_tag=cuda_tag,
            python_tags=python_tags,
            arch_tag=arch_tag,
            version=args.version,
        )
        if asset is None:
            version_label = f", version={args.version}" if args.version else ""
            message = (
                f"No prebuilt {package} wheel found for torch={torch_tag}, cuda={cuda_tag}, "
                f"python={python_tags}, arch={arch_tag}{version_label}"
            )
            if args.optional:
                print(f"warning: {message}", flush=True)
                continue
            raise SystemExit(message)
        _install(asset, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
