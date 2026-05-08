"""Top-level setup for FlexTrain.

Most metadata lives in ``pyproject.toml``. This file exists so that
``pip install -e .`` (or ``pip install .``) also builds the two
in-tree helper packages under ``helpers/``:

* ``matmul_dispatcher``  (cuBLASLt dispatcher, CMake + CUDA)
* ``transmission_scheduler``  (DP scheduler used by the working-set solver, C ext)

Both are first-party FlexTrain helpers, kept as separate Python
packages because they ship native code and are useful on their own.
"""
from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.editable_wheel import editable_wheel


REPO_ROOT = Path(__file__).resolve().parent
HELPERS_DIR = REPO_ROOT / "helpers"
HELPER_PACKAGES = ("matmul_dispatcher", "transmission_scheduler")
HELPER_PTH_NAME = "flextrain_helpers.pth"


def _write_helper_pth() -> None:
    purelib = Path(sysconfig.get_path("purelib"))
    purelib.mkdir(parents=True, exist_ok=True)
    pth_path = purelib / HELPER_PTH_NAME
    helper_paths = [
        str((HELPERS_DIR / name).resolve())
        for name in HELPER_PACKAGES
    ]
    payload = "".join(f"{path}\n" for path in helper_paths)
    if pth_path.exists() and pth_path.read_text() == payload:
        return
    pth_path.write_text(payload)
    print(f"[flextrain setup] wrote helper path file: {pth_path}", flush=True)


def _build_helper(name: str) -> None:
    pkg_dir = HELPERS_DIR / name
    if not pkg_dir.exists():
        raise RuntimeError(f"Missing helper package: {pkg_dir}")
    print(f"[flextrain setup] building helper: {name} ({pkg_dir})", flush=True)
    setup_py = pkg_dir / "setup.py"
    if not setup_py.exists():
        raise RuntimeError(f"Missing helper setup.py: {setup_py}")
    # Build the helper's native extension in place instead of nesting
    # another package install. This avoids recursive pip/setuptools
    # editable-install paths, which can fail in environments where
    # `python -m pip` is unavailable inside the build hook.
    subprocess.check_call(
        [sys.executable, str(setup_py), "build_ext", "--inplace"],
        cwd=str(pkg_dir),
    )


def _build_all_helpers() -> None:
    if os.environ.get("FLEXTRAIN_SKIP_HELPERS") == "1":
        print("[flextrain setup] FLEXTRAIN_SKIP_HELPERS=1, skipping helper builds")
        return
    for name in HELPER_PACKAGES:
        _build_helper(name)
    _write_helper_pth()


class _BuildPyWithHelpers(build_py):
    def run(self) -> None:
        _build_all_helpers()
        super().run()


def _print_post_install_hint() -> None:
    print(
        "[flextrain setup] To fetch prebuilt flash-attn and causal-conv1d "
        "wheels (matched on python/torch/cuda/compute-cap), run:\n"
        "    flextrain-post-install\n"
        "after pip install completes. The build-isolation env pip uses "
        "for editable installs cannot reach the user environment, so "
        "wheel installs must happen as a separate user-env step.",
        flush=True,
    )


class _DevelopWithHelpers(develop):
    def run(self) -> None:
        _build_all_helpers()
        _print_post_install_hint()
        super().run()


class _InstallWithHelpers(install):
    def run(self) -> None:
        _build_all_helpers()
        _print_post_install_hint()
        super().run()


class _EditableWheelWithHelpers(editable_wheel):
    def run(self) -> None:
        _build_all_helpers()
        _print_post_install_hint()
        super().run()


setup(
    cmdclass={
        "build_py": _BuildPyWithHelpers,
        "develop": _DevelopWithHelpers,
        "install": _InstallWithHelpers,
        "editable_wheel": _EditableWheelWithHelpers,
    },
)
