# DeepSpeed Arctic / ALST Baseline Slot

The runnable synthetic-token entrypoint is:

```text
baseline/backends/deepspeed_arctic/train_synthetic.py
```

Use `baseline/run_baseline.py --backend deepspeed_arctic ...`; the harness writes the bf16 DeepSpeed config, sequence-parallel settings, and ALST-related flags.

Install this backend in its own env:

```bash
baseline/scripts/install_backend.sh --backend deepspeed_arctic
```

The installer does not install `flextrain`. It installs Torch first, then the DeepSpeed/HuggingFace dependencies, then matching prebuilt FlashAttention 2/3 wheels when available.

For sparse MoE HuggingFace models, use:

```bash
--moe-kernel-backend sonic
```

This loads `kernels-community/sonic-moe` with the Hugging Face `kernels` package and replaces compatible HF sparse MoE blocks after model load.
