# TorchTitan Baseline Slot

Place a TorchTitan checkout here if you want the new harness to use it directly.
If this directory does not contain TorchTitan code, `baseline/scripts/install_backend.sh --backend torchtitan` fetches TorchTitan into the ignored `baseline/vendor/TorchTitan` directory. The launcher also falls back to `orig/baseline/torchtitan` when available.

The synthetic TorchTitan config registry lives at:

```text
baseline/backends/torchtitan/synthetic_registry.py
```

Install this backend in its own env:

```bash
baseline/scripts/install_backend.sh --backend torchtitan
```

The installer does not install `flextrain`. It installs Torch first, then TorchTitan dependencies, then editable-installs the TorchTitan checkout found in `baseline/TorchTitan` or `orig/baseline/torchtitan`.
