# FlexTrain helper packages

Two first-party Python packages with native code, kept in-tree and
built automatically by `pip install -e .` from the repo root.

| Package | Native code | Used by |
|---|---|---|
| `matmul_dispatcher` | C++17 + CUDA (CMake), wraps cuBLASLt | `flextrain.ops._kernels._matmul_dispatchers` |
| `transmission_scheduler` | Plain C extension (AVX2/scalar dispatch) | `flextrain.engine.active_model` (working-set DP solver) |

## Build requirements

* `matmul_dispatcher` — a working C++17 compiler, CMake ≥ 3.18, and
  the CUDA toolkit (so CMake's `find_package(CUDAToolkit)` resolves
  `cublasLt` / `cudart`).
* `transmission_scheduler` — any C compiler that supports `-O3 -fPIC`
  on Linux.

## Building manually

The top-level `setup.py` runs `pip install -e <helper>` for each one
during `pip install -e .`. To build a single helper directly:

```bash
cd helpers/matmul_dispatcher && pip install -e .
cd helpers/transmission_scheduler && pip install -e .
```

To skip building the helpers when iterating on Python-only changes:

```bash
FLEXTRAIN_SKIP_HELPERS=1 pip install -e .
```
