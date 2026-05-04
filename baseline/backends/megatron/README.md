# Megatron Backend Adapter

`baseline/run_baseline.py --backend megatron` generates a
`model_dims.json` from the HuggingFace model config and launches the
Megatron-Core training script at
`baseline/backends/megatron/train.py`. The script defaults to
precision-aware AdamW (bf16 main_params + bf16 exp_avg + bf16
exp_avg_sq) so the backend matches the bf16-master invariant the rest
of the harness enforces.

Script lookup order (the harness falls back to the second when the
first is missing):

1. `baseline/backends/megatron/train.py`
2. `orig/baseline/megatron/train.py`

Memory features are wired through the standard launcher flags
(`--recompute-granularity`, `--activation-offload`, `--optimizer-offload`,
`--offload-modules`, etc) — see the top-level
[baseline/README.md](../../README.md) for the full mapping.

Megatron-Core builds models from `model_dims.json`, not from HF model
classes, so `--moe-kernel-backend sonic` does not apply (Megatron-Core
has its own MoE-kernel selection).
