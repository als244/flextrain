# Megatron Baseline Slot

This directory is reserved for a Megatron Core checkout or local Megatron-specific assets.

The harness-generated command currently launches `baseline/backends/megatron/train.py` when present, otherwise it falls back to `orig/baseline/megatron/train.py`. Per-run HuggingFace-derived `model_dims.json` files are written into `baseline/runs/...`.

Install this backend in its own env:

```bash
baseline/scripts/install_backend.sh --backend megatron
```

The installer does not install `flextrain`. It installs Torch first, then Megatron Core and Transformer Engine dependencies, then matching prebuilt FlashAttention 2/3 wheels when available.
