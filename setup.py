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
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.editable_wheel import editable_wheel


REPO_ROOT = Path(__file__).resolve().parent
HELPERS_DIR = REPO_ROOT / "helpers"
HELPER_PACKAGES = ("matmul_dispatcher", "transmission_scheduler")


def _build_helper(name: str) -> None:
    pkg_dir = HELPERS_DIR / name
    if not pkg_dir.exists():
        raise RuntimeError(f"Missing helper package: {pkg_dir}")
    print(f"[flextrain setup] building helper: {name} ({pkg_dir})", flush=True)
    setup_py = pkg_dir / "setup.py"
    if not setup_py.exists():
        raise RuntimeError(f"Missing helper setup.py: {setup_py}")
    # Install the helper directly via setuptools instead of nesting
    # another pip invocation. Some target environments can execute a
    # `pip` script without having an importable `pip` module, which
    # breaks recursive `pip install -e ...` calls during editable
    # builds.
    subprocess.check_call(
        [sys.executable, str(setup_py), "develop", "--no-deps"],
        cwd=str(pkg_dir),
    )


def _build_all_helpers() -> None:
    if os.environ.get("FLEXTRAIN_SKIP_HELPERS") == "1":
        print("[flextrain setup] FLEXTRAIN_SKIP_HELPERS=1, skipping helper builds")
        return
    for name in HELPER_PACKAGES:
        _build_helper(name)


class _BuildPyWithHelpers(build_py):
    def run(self) -> None:
        _build_all_helpers()
        super().run()


class _DevelopWithHelpers(develop):
    def run(self) -> None:
        _build_all_helpers()
        super().run()


class _InstallWithHelpers(install):
    def run(self) -> None:
        _build_all_helpers()
        super().run()


class _EditableWheelWithHelpers(editable_wheel):
    def run(self) -> None:
        _build_all_helpers()
        super().run()


setup(
    cmdclass={
        "build_py": _BuildPyWithHelpers,
        "develop": _DevelopWithHelpers,
        "install": _InstallWithHelpers,
        "editable_wheel": _EditableWheelWithHelpers,
    },
)
