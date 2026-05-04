# Baseline Harness

This directory now has a unified launcher for synthetic-token training runs across:

- `megatrain`
- `torchtitan`
- `trl_deepspeed`
- `deepspeed_arctic`
- `megatron`

The top-level API is:

```bash
python baseline/run_baseline.py \
  --backend deepspeed_arctic \
  --model-path models/Llama-3.1-8B-Instruct \
  --seq-length 8192 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-steps 3 \
  --num-gpus 4 \
  --dry-run
```

Use `--backend all` to emit or run every backend command. Every run writes a `launch_plan.json`, generated backend configs, and `run.log` under `baseline/runs/...`.

For a complete Llama 3-family 128K maximum-memory-savings sweep with per-step throughput extraction, see `baseline/RUNBOOK_128K_MAX_MEMORY.md`.

## Installation Model

Running baselines does not require installing `flextrain` as a Python package. The launcher and synthetic datasets are repo-local source files; `baseline/run_baseline.py` adds the repo root to `PYTHONPATH`, and generated backend commands do the same. Each backend can therefore live in its own virtualenv with only that backend's dependencies installed.

Create an independent backend env with:

```bash
baseline/scripts/install_backend.sh --backend trl_deepspeed
baseline/scripts/install_backend.sh --backend deepspeed_arctic
baseline/scripts/install_backend.sh --backend torchtitan
baseline/scripts/install_backend.sh --backend megatron
baseline/scripts/install_backend.sh --backend megatrain
```

By default this creates `baseline/envs/<backend>`, installs Torch first from `https://download.pytorch.org/whl/cu126`, installs `baseline/requirements/<backend>.txt`, then installs matching prebuilt FlashAttention 2 and FlashAttention 3 wheels from `mjun0812/flash-attention-prebuild-wheels` when a wheel exists for the env's Python/CUDA/Torch/platform combination. It also installs `flash-linear-attention` in every backend env; HF-style/Qwen-hybrid backends additionally try a matching prebuilt `causal-conv1d` wheel from `Dao-AILab/causal-conv1d`.

MegaTrain and TorchTitan source checkouts are not vendored in git. The installer uses an existing checkout in `baseline/MegaTrain` or `baseline/TorchTitan` when present; otherwise it fetches them into ignored `baseline/vendor/...` directories. Override the source with `MEGATRAIN_REPO`, `MEGATRAIN_REF`, `TORCHTITAN_REPO`, or `TORCHTITAN_REF`.

Detect the local CUDA version with:

```bash
baseline/scripts/detect_cuda.py
```

On this machine, CUDA detection reports CUDA 13.1, which maps to PyTorch's `cu130` wheel tag. Use the detected CUDA wheel index when that is what you want:

```bash
baseline/scripts/install_backend.sh --backend trl_deepspeed --torch-index-url auto --recreate
```

Or set the Torch source explicitly for a cluster image:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  baseline/scripts/install_backend.sh --backend trl_deepspeed --recreate
```

If Torch is already provisioned in the env:

```bash
baseline/scripts/install_backend.sh --backend trl_deepspeed --skip-torch
```

Useful install controls:

- `--flash-version VERSION`: require an exact prebuilt FlashAttention package version, for example `--flash fa2 --flash-version 2.8.3`
- `--linear-attention {auto,strict,none}`: install `flash-linear-attention` for all backends and Qwen hybrid-attention deps where useful; `auto` warns when no exact `causal-conv1d` wheel exists, `strict` fails
- `--causal-conv1d-torch-tag TAG`: override the causal-conv1d wheel Torch tag; this is useful when an adjacent prebuilt wheel is known to work

Run one backend with its own env:

```bash
baseline/scripts/run_in_backend_env.sh trl_deepspeed \
  --model-path models/Llama-3.1-8B-Instruct \
  --seq-length 8192 \
  --num-gpus 4
```

`--backend all` is still useful for dry-run command generation, but independent installs usually mean activating/running one backend env at a time.

## Common Arguments

Required:

- `--backend {megatrain,torchtitan,trl_deepspeed,deepspeed_arctic,megatron,all}`
- `--model-path`: local HuggingFace model directory, for example `models/Qwen3-1.7B`
- `--seq-length`: synthetic token sequence length

Training shape:

- `--micro-batch-size`
- `--gradient-accumulation-steps`
- `--num-steps`
- `--num-gpus`
- `--learning-rate`
- `--weight-decay`
- `--seed`

Memory/recompute knobs exposed by the unified API:

- `--activation-checkpointing {none,selective,full,memory_budget}`
- `--activation-checkpoint-interval`
- `--activation-checkpoint-fraction`: fraction of decoder layers/blocks to checkpoint/recompute where the backend exposes a supported layer/block-selection API; true fractional values are rejected otherwise
- `--save-activation-layer-fraction`: compatibility with `orig/baseline/deepspeed --save_act_layer_frac`; converted to `1 - save_fraction`, then subject to the backend support checks below
- `--activation-offload {none,cpu}`
- `--optimizer-offload {none,cpu}`
- `--param-offload {none,cpu}`
- `--zero-stage {0,1,2,3}`
- `--tensor-parallel-size`
- `--pipeline-parallel-size`
- `--context-parallel-size`
- `--sequence-parallel-size`
- `--fsdp-shard-degree`
- `--fsdp-replicate-degree`
- `--recompute-granularity {selective,full}`
- `--recompute-method`
- `--recompute-num-layers`
- `--recompute-modules core_attn,mlp,layernorm`
- `--offload-modules qkv_linear,core_attn,attn_proj`
- `--num-grad-slabs`
- `--tiled-loss-shards`
- `--tiled-mlp`
- `--attn-implementation {flash_attention_2,auto,flash_attention_3,sdpa,eager}`: default is strict FlashAttention 2; pass `sdpa`/`eager` explicitly for fallback comparisons. MegaTrain only supports `flash_attention_2`, `sdpa`, and `eager`; `auto` skips FlashAttention 3 for that backend.
- `--moe-kernel-backend {hf,auto,sonic}`: for HF sparse MoE models, `sonic` replaces compatible HF MoE blocks with `kernels-community/sonic-moe`
- `--liger-kernel {auto,on,off}`: TRL Liger mode; `auto` enables it when `liger-kernel` and TRL support are installed
- `--use-liger-kernel`: legacy alias for `--liger-kernel on`

All generated DeepSpeed configs enable bf16 computation and, for ZeRO stages 1-3, bf16 master weights/gradients, bf16 optimizer states, and `optimizer.fp32_optimizer_states=false`. CPU optimizer/parameter offload entries use `pin_memory=true`. DeepSpeed launches also set `DS_SKIP_CUDA_CHECK=1` and `PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True,expandable_segments:True` unless those env vars are already set. TorchTitan is launched with `training.dtype=bfloat16` and `training.mixed_precision_param=bfloat16`. MegaTrain’s synthetic entrypoint constructs `CPUMasterConfig(dtype=torch.bfloat16)`.

Fractional activation checkpointing support is backend-dependent:

- `trl_deepspeed`: HF/TRL does not expose supported fractional layer checkpointing. The launcher rejects true fractional values (`0 < f < 1`); use `--activation-checkpointing none/full`.
- `deepspeed_arctic`: this path is still a HuggingFace model path, so true fractional values (`0 < f < 1`) are rejected. Use `--activation-checkpointing none/full` plus ALST knobs such as sequence parallelism, tiled loss, tiled MLP, and activation offload.
- `torchtitan`: fractional values are approximated by TorchTitan selective checkpointing every `round(1/f)` layers. `f=1` maps to full, `f=0` maps to none.
- `megatron`: fractional values map to Megatron full block recompute for `floor(num_layers * f)` layers. `f=1` maps to full uniform recompute, `f=0` maps to no recompute.
- `megatrain`: no true layer-fraction mode is exposed; true fractional values are rejected. Use `--activation-checkpoint-interval` to control recompute segment size.

## Synthetic Data

Synthetic examples are generated as random integer token IDs in `[0, vocab_size)`, where `vocab_size` is read from the model’s `config.json`. This avoids benchmarking tokenizer throughput or text-packing behavior. Labels are shaped to match each backend’s causal-LM API:

- HuggingFace/TRL/DeepSpeed/MegaTrain receive unshifted labels and let the model/backend shift.
- TorchTitan receives pre-shifted labels because its dataloader contract yields `(input, next_token_label)`.
- DeepSpeed ALST/Ulysses uses the DeepSpeed dataloader adapter to generate `shift_labels` when sequence parallelism is enabled.

## Backend Notes

### MegaTrain

Entrypoint:

```bash
python baseline/run_baseline.py --backend megatrain ...
```

Uses `baseline/MegaTrain` as the vendor tree and `baseline/backends/megatrain/train_synthetic.py` as the synthetic entrypoint. Extra knobs:

- `--activation-checkpoint-interval`: maps to MegaTrain `checkpoint_interval` (vendor default `4`, meaning checkpoint every N layers)
- `--activation-checkpoint-fraction`: true fractional values are rejected; prefer `--activation-checkpoint-interval`
- `--num-grad-slabs`: maps to MegaTrain gradient slab pool size
- `--backend-extra-arg --optimizer --backend-extra-arg deepspeed_cpu_adam`

### TorchTitan

Entrypoint:

```bash
python baseline/run_baseline.py --backend torchtitan ...
```

The harness looks for `baseline/TorchTitan`, then `baseline/vendor/TorchTitan`, then `baseline/torchtitan`, then the old `orig/baseline/torchtitan`. The synthetic registry is `baseline.backends.torchtitan.synthetic_registry` and currently covers TorchTitan’s built-in `llama3_8b`, `qwen3_1_7b`, `qwen3_32b`, and debug specs. For custom specs:

```bash
--torchtitan-module your.package.registry --torchtitan-config your_config
```

Important mappings:

- `--activation-checkpointing` -> `--activation_checkpoint.mode`
- `--activation-checkpoint-fraction` -> approximate selective interval via `--activation_checkpoint.selective_ac_option`
- `--activation-offload cpu` -> `--training.enable_activation_offload`
- `--optimizer-offload cpu` or `--param-offload cpu` -> `--training.enable_cpu_offload`
- tensor/pipeline/context/FSDP flags map to `--parallelism.*`

### TRL + DeepSpeed

Entrypoint:

```bash
python baseline/run_baseline.py --backend trl_deepspeed ...
```

Uses `SFTTrainer` with `dataset_kwargs={"skip_prepare_dataset": True}` so the token-native synthetic dataset is passed through directly. Important mappings:

- `--zero-stage`, `--optimizer-offload`, `--param-offload` -> generated DeepSpeed JSON
- `--activation-checkpointing` -> HF gradient checkpointing
- `--activation-checkpoint-fraction`: true fractional values are rejected because HF/TRL does not expose supported fractional layer checkpointing
- `--activation-offload cpu` -> TRL `activation_offloading` when available
- `--attn-implementation flash_attention_2` -> strict default for HF/TRL runs; `auto` may try FlashAttention 3 first and then fall back through FlashAttention 2, SDPA, and eager
- `--moe-kernel-backend sonic` -> loads `kernels-community/sonic-moe` through the Hugging Face `kernels` package and swaps compatible HF sparse MoE blocks
- `--liger-kernel auto` -> sets TRL `use_liger_kernel=True` only when `liger-kernel` is installed and the local TRL `SFTConfig` exposes the option
- Install with `baseline/scripts/install_backend.sh --backend trl_deepspeed`; this creates an isolated TRL/DeepSpeed env and installs Liger plus matching prebuilt FlashAttention wheels after Torch.

### DeepSpeed Arctic / ALST

Entrypoint:

```bash
python baseline/run_baseline.py --backend deepspeed_arctic --sequence-parallel-size 4 ...
```

Uses a custom HF + DeepSpeed loop in `baseline/backends/deepspeed_arctic/train_synthetic.py`. Important mappings:

- `--sequence-parallel-size > 1` enables DeepSpeed Ulysses SP registration and dataloader adaptation
- `--activation-checkpoint-fraction`: true fractional values are rejected because this path uses HuggingFace model checkpointing APIs
- `--zero-stage`, `--optimizer-offload`, `--param-offload` -> generated DeepSpeed JSON
- `--moe-kernel-backend sonic` -> loads `kernels-community/sonic-moe` through the Hugging Face `kernels` package and swaps compatible HF sparse MoE blocks
- `--activation-offload cpu` attempts the Arctic Training activation checkpoint offload monkey patch if installed
- `--tiled-mlp` attempts the DeepSpeed ALST Llama MLP tiling hook

### Megatron Core

Entrypoint:

```bash
python baseline/run_baseline.py --backend megatron ...
```

The harness generates a per-run `model_dims.json` from the HuggingFace config and launches the Megatron script. It prefers `baseline/backends/megatron/train.py` if present and otherwise falls back to `orig/baseline/megatron/train.py`. Important mappings:

- `--recompute-granularity`, `--recompute-method`, `--recompute-num-layers`, `--recompute-modules`
- `--activation-checkpoint-fraction` -> Megatron block recompute count
- `--activation-offload cpu` -> TE layer CPU offload flags
- `--offload-modules` -> Megatron fine-grained activation offload
- `--optimizer-offload cpu` -> Megatron optimizer CPU offload

## References Used For API Choices

- DeepSpeed bf16 config keys: https://www.deepspeed.ai/docs/config-json/
- DeepSpeed Arctic/ALST Ulysses integration: https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/
- Hugging Face Transformers DeepSpeed ALST guide: https://huggingface.co/docs/transformers/main/deepspeed_alst
- TRL SFTTrainer/SFTConfig: https://huggingface.co/docs/trl/en/sft_trainer
- FlashAttention prebuilt wheels: https://github.com/mjun0812/flash-attention-prebuild-wheels
- causal-conv1d prebuilt wheels: https://github.com/Dao-AILab/causal-conv1d/releases
- Flash Linear Attention: https://github.com/fla-org/flash-linear-attention
- Hugging Face kernels loader and SonicMoE kernel: https://github.com/huggingface/kernels and https://huggingface.co/kernels-community/sonic-moe
