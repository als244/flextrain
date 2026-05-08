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


def _install_flash_attn_wheels() -> None:
    """Best-effort fetch of prebuilt flash-attn wheels.

    flash-attention from-source builds take 30+ minutes; the wheels at
    https://github.com/mjun0812/flash-attention-prebuild-wheels are
    keyed by (python, torch, cuda) and skip the build entirely. This
    hook delegates to ``helpers/install_flash_attn_wheels.py``:

    * Always tries to install ``flash_attn`` (FA2). The wheel match is
      keyed on torch + CUDA + python detected at install time.
    * Tries to install ``flash_attn_3`` (FA3) only when the local GPU
      reports ``compute_capability[0] >= 9`` (Hopper sm_90+).

    Failures are non-fatal — the install proceeds without flash-attn.
    Set ``FLEXTRAIN_SKIP_FLASH_ATTN=1`` to skip this entirely.
    """
    if os.environ.get("FLEXTRAIN_SKIP_FLASH_ATTN") == "1":
        print(
            "[flextrain setup] FLEXTRAIN_SKIP_FLASH_ATTN=1, "
            "skipping flash-attn wheel install"
        )
        return

    try:
        import torch
    except ImportError:
        print(
            "[flextrain setup] torch not importable; skipping flash-attn "
            "wheel install (re-run pip install -e . after torch lands)",
            flush=True,
        )
        return

    if not torch.cuda.is_available():
        print(
            "[flextrain setup] no CUDA GPU detected; skipping flash-attn "
            "wheel install. Set FLEXTRAIN_SKIP_FLASH_ATTN=1 to silence.",
            flush=True,
        )
        return

    is_hopper = False
    try:
        cap = torch.cuda.get_device_capability(0)
        is_hopper = cap[0] >= 9
        print(
            f"[flextrain setup] detected sm_{cap[0]}{cap[1]}; "
            f"flash-attn 3 = {'enabled' if is_hopper else 'skipped (not Hopper sm90+)'}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[flextrain setup] could not query GPU capability ({e}); "
            "installing flash-attn 2 only",
            flush=True,
        )

    installer = HELPERS_DIR / "install_flash_attn_wheels.py"
    if not installer.exists():
        print(
            f"[flextrain setup] {installer} missing; skipping flash-attn install"
        )
        return

    packages = ["flash_attn"]
    if is_hopper:
        packages.append("flash_attn_3")
    for pkg in packages:
        cmd = [sys.executable, str(installer), "--package", pkg, "--optional"]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(
                f"[flextrain setup] flash-attn ({pkg}) wheel install failed "
                f"({e}); proceeding without it. Install manually via "
                f"`python {installer} --package {pkg}`.",
                flush=True,
            )


def _install_causal_conv1d_wheel() -> None:
    """Best-effort fetch of a prebuilt causal-conv1d wheel.

    causal-conv1d (Dao-AILab/causal-conv1d) ships prebuilt wheels on
    its GitHub releases keyed by (python, torch, cuda, cxx11-abi).
    Source builds compile CUDA kernels and take ~5-15 minutes; the
    wheel skips that. Used by FLA's Mamba-style state-space layers.

    Failures are non-fatal — the install proceeds without
    causal-conv1d. Set ``FLEXTRAIN_SKIP_CAUSAL_CONV1D=1`` to skip
    entirely.
    """
    if os.environ.get("FLEXTRAIN_SKIP_CAUSAL_CONV1D") == "1":
        print(
            "[flextrain setup] FLEXTRAIN_SKIP_CAUSAL_CONV1D=1, "
            "skipping causal-conv1d wheel install"
        )
        return

    try:
        import torch
    except ImportError:
        print(
            "[flextrain setup] torch not importable; skipping "
            "causal-conv1d wheel install",
            flush=True,
        )
        return

    if not torch.cuda.is_available():
        print(
            "[flextrain setup] no CUDA GPU detected; skipping "
            "causal-conv1d wheel install",
            flush=True,
        )
        return

    installer = HELPERS_DIR / "install_causal_conv1d_wheel.py"
    if not installer.exists():
        print(
            f"[flextrain setup] {installer} missing; skipping "
            "causal-conv1d install"
        )
        return

    cmd = [sys.executable, str(installer), "--optional"]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(
            f"[flextrain setup] causal-conv1d wheel install failed ({e}); "
            f"proceeding without it. Install manually via "
            f"`python {installer}`.",
            flush=True,
        )


class _BuildPyWithHelpers(build_py):
    def run(self) -> None:
        _build_all_helpers()
        super().run()


class _DevelopWithHelpers(develop):
    def run(self) -> None:
        _build_all_helpers()
        _install_flash_attn_wheels()
        _install_causal_conv1d_wheel()
        super().run()


class _InstallWithHelpers(install):
    def run(self) -> None:
        _build_all_helpers()
        _install_flash_attn_wheels()
        _install_causal_conv1d_wheel()
        super().run()


class _EditableWheelWithHelpers(editable_wheel):
    def run(self) -> None:
        _build_all_helpers()
        _install_flash_attn_wheels()
        _install_causal_conv1d_wheel()
        super().run()


setup(
    cmdclass={
        "build_py": _BuildPyWithHelpers,
        "develop": _DevelopWithHelpers,
        "install": _InstallWithHelpers,
        "editable_wheel": _EditableWheelWithHelpers,
    },
)
