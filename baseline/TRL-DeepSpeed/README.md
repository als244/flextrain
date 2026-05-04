# TRL + DeepSpeed Baseline Slot

The runnable synthetic-token entrypoint is:

```text
baseline/backends/trl_deepspeed/train_synthetic.py
```

Use `baseline/run_baseline.py --backend trl_deepspeed ...` instead of calling this script directly; the harness writes the bf16 DeepSpeed config and forwards memory/recompute flags.

Install this backend in its own env:

```bash
baseline/scripts/install_backend.sh --backend trl_deepspeed
```

The installer does not install `flextrain`. It installs Torch first, then TRL/DeepSpeed/Liger dependencies, then matching prebuilt FlashAttention 2/3 wheels when available.

For sparse MoE HuggingFace models, use:

```bash
--moe-kernel-backend sonic
```

This loads `kernels-community/sonic-moe` with the Hugging Face `kernels` package and replaces compatible HF sparse MoE blocks after model load.
