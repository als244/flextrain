"""Build script for ``matmul_dispatcher``.

Two build targets:

1. ``libmatmul_dispatcher.so`` — the C++/CUDA shared library that owns
   the cuBLASLt context, algo cache, and ``dispatch_matmul`` C ABI.
   Built via CMake (CUDA toolchain integration).
2. ``matmul_dispatcher._dispatch_pyext`` — a CPython extension module
   that exposes ``matmul_fast`` (METH_FASTCALL) and links against
   ``libmatmul_dispatcher.so`` to call ``dispatch_matmul`` directly,
   bypassing ctypes for hot paths.

Both end up inside the ``matmul_dispatcher`` package directory so the
Python wrapper at ``matmul_dispatcher/__init__.py`` can dlopen the .so
and import the extension.
"""
import os
import subprocess
import sys
import shutil
import sysconfig
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


_HERE = Path(__file__).resolve().parent
_PKG_DIR = _HERE / "matmul_dispatcher"


# ---------------------------------------------------------------------------
# CMake-built C++/CUDA shared library.
# ---------------------------------------------------------------------------


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


# ---------------------------------------------------------------------------
# Custom build_ext: route CMake extensions through cmake, route plain
# Extensions (the CPython pyext) through the default builder.
# ---------------------------------------------------------------------------


class CMakeOrSetuptoolsBuild(build_ext):
    def run(self):
        # First build the CMake extension(s) so the .so is on disk
        # before the CPython extension links against it.
        cmake_exts = [e for e in self.extensions if isinstance(e, CMakeExtension)]
        other_exts = [e for e in self.extensions if not isinstance(e, CMakeExtension)]

        for ext in cmake_exts:
            self._build_cmake(ext)

        # Now let setuptools build the regular Extensions.
        self.extensions = other_exts
        super().run()
        # Restore for downstream tools.
        self.extensions = cmake_exts + other_exts

    def _build_cmake(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        extdir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--parallel"], cwd=self.build_temp
        )

        # Copy the built .so into the package source tree so editable
        # installs find it (CMake itself targets ``matmul_dispatcher/``
        # but explicit copy makes the contract obvious).
        for f in os.listdir(extdir):
            if f.startswith("libmatmul_dispatcher") and (
                f.endswith(".so") or f.endswith(".dll") or f.endswith(".dylib")
            ):
                src = extdir / f
                dst = _PKG_DIR / f
                if src.resolve() != dst.resolve():
                    print(f"Copying {src} -> {dst}")
                    shutil.copy(src, dst)


# ---------------------------------------------------------------------------
# CPython fast-path extension module.
# ---------------------------------------------------------------------------


pyext = Extension(
    name="matmul_dispatcher._dispatch_pyext",
    sources=[str(_HERE / "src" / "dispatch_pyext.cpp")],
    include_dirs=[str(_HERE / "src")],
    library_dirs=[str(_PKG_DIR)],
    libraries=["matmul_dispatcher"],
    runtime_library_dirs=["$ORIGIN"],
    extra_compile_args=["-O3", "-std=c++17"],
    language="c++",
)


setup(
    name="matmul_dispatcher",
    version="0.0.2",
    packages=find_packages(),
    package_data={"matmul_dispatcher": ["*.so", "*.dll", "*.dylib"]},
    ext_modules=[
        CMakeExtension("matmul_dispatcher.libmatmul_dispatcher"),
        pyext,
    ],
    cmdclass={"build_ext": CMakeOrSetuptoolsBuild},
    zip_safe=False,
    install_requires=["torch"],
)
