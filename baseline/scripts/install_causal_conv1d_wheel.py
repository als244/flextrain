#!/usr/bin/env python3
"""Install a matching prebuilt causal-conv1d wheel from GitHub releases."""

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

REPO = "Dao-AILab/causal-conv1d"


@dataclass(frozen=True)
class Asset:
    release: str
    name: str
    url: str


def _torch_tags() -> tuple[str, str, str]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install torch before installing causal-conv1d wheels.") from exc

    torch_match = re.match(r"(\d+)\.(\d+)", torch.__version__.split("+", 1)[0])
    if torch_match is None:
        raise SystemExit(f"Could not parse torch version from {torch.__version__!r}")

    cuda_version = torch.version.cuda
    if not cuda_version:
        raise SystemExit("The installed torch build does not report CUDA; no CUDA causal-conv1d wheel can be selected.")
    cuda_match = re.match(r"(\d+)", cuda_version)
    if cuda_match is None:
        raise SystemExit(f"Could not parse CUDA version from {cuda_version!r}")

    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    return f"torch{torch_match.group(1)}.{torch_match.group(2)}", f"cu{cuda_match.group(1)}", cxx11_abi


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}-cp{sys.version_info.major}{sys.version_info.minor}"


def _platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        if machine in {"x86_64", "amd64"}:
            return "linux_x86_64"
        if machine in {"aarch64", "arm64"}:
            return "linux_aarch64"
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return "win_amd64"
    raise SystemExit(f"Unsupported platform for prebuilt causal-conv1d wheels: {platform.system()} {platform.machine()}")


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
    match = re.match(r"causal_conv1d-([^+]+)", asset.name)
    if match is None:
        return Version("0")
    try:
        return Version(match.group(1))
    except InvalidVersion:
        return Version("0")


def _asset_score(asset: Asset) -> tuple[Version, str]:
    return _asset_version(asset), asset.name


def _find_asset(
    *,
    repo: str,
    torch_tag: str,
    cuda_tag: str,
    cxx11_abi: str,
    python_tag: str,
    platform_tag: str,
    version: str | None,
) -> Asset | None:
    prefix = "causal_conv1d-"
    matches: list[Asset] = []
    for release in _load_releases(repo):
        for raw_asset in release.get("assets", []):
            name = raw_asset.get("name", "")
            if not name.startswith(prefix) or not name.endswith(".whl"):
                continue
            if version is not None and not name.startswith(f"{prefix}{version}+"):
                continue
            required_parts = [
                cuda_tag,
                torch_tag,
                f"cxx11abi{cxx11_abi}",
                python_tag,
                platform_tag,
            ]
            if not all(part in name for part in required_parts):
                continue
            matches.append(
                Asset(
                    release=release.get("tag_name", ""),
                    name=name,
                    url=raw_asset.get("browser_download_url", ""),
                )
            )
    if not matches:
        return None
    return sorted(matches, key=_asset_score, reverse=True)[0]


def _install(asset: Asset, *, dry_run: bool) -> None:
    print(f"selected {asset.name} from {asset.release}", flush=True)
    print(asset.url, flush=True)
    if dry_run:
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", asset.url])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--torch-tag", default=None, help="Override detected torch tag, for example torch2.10")
    parser.add_argument("--cuda-tag", default=None, help="Override detected CUDA major tag, for example cu13")
    parser.add_argument("--cxx11-abi", choices=["TRUE", "FALSE"], default=None)
    parser.add_argument("--python-tag", default=None, help="Override Python ABI tag, for example cp312-cp312")
    parser.add_argument("--platform-tag", default=None, help="Override platform tag, for example linux_x86_64")
    parser.add_argument("--version", default=None, help="Require an exact causal-conv1d version, for example 1.6.1")
    parser.add_argument("--optional", action="store_true", help="Skip when no matching wheel exists instead of failing.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        detected_torch_tag, detected_cuda_tag, detected_cxx11_abi = _torch_tags()
    except SystemExit as exc:
        if args.optional:
            print(f"warning: {exc}", flush=True)
            return 0
        raise

    torch_tag = args.torch_tag or detected_torch_tag
    cuda_tag = args.cuda_tag or detected_cuda_tag
    cxx11_abi = args.cxx11_abi or detected_cxx11_abi
    python_tag = args.python_tag or _python_tag()
    platform_tag = args.platform_tag or _platform_tag()

    print(
        "causal_conv1d_wheel_query "
        f"repo={args.repo} torch={torch_tag} cuda={cuda_tag} cxx11abi={cxx11_abi} "
        f"python={python_tag} platform={platform_tag}",
        flush=True,
    )
    asset = _find_asset(
        repo=args.repo,
        torch_tag=torch_tag,
        cuda_tag=cuda_tag,
        cxx11_abi=cxx11_abi,
        python_tag=python_tag,
        platform_tag=platform_tag,
        version=args.version,
    )
    if asset is None:
        version_label = f", version={args.version}" if args.version else ""
        message = (
            f"No prebuilt causal-conv1d wheel found for torch={torch_tag}, cuda={cuda_tag}, "
            f"cxx11abi={cxx11_abi}, python={python_tag}, platform={platform_tag}{version_label}"
        )
        if args.optional:
            print(f"warning: {message}", flush=True)
            return 0
        raise SystemExit(message)
    _install(asset, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
