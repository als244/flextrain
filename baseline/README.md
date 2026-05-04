# Baseline Harness

Unified launcher for synthetic-token training across these backends:

| Backend | Parallelism | Trainer / framework |
|---|---|---|
| `trl_deepspeed` | DeepSpeed ZeRO | TRL SFTTrainer (HF) |
| `trl_fsdp` | accelerate FSDP2 | TRL SFTTrainer (HF) |
| `deepspeed_arctic` | DeepSpeed + Ulysses SP | Custom HF + DeepSpeed loop |
| `megatrain` | CPU-master + grad slabs | MegaTrain (HF model wrapped in CPUMasterModel) |
| `torchtitan` | TorchTitan FSDP2 | TorchTitan native (registry-based) |
| `megatron` | Megatron-Core + TE | Megatron-Core native |

`trl_fsdp` is the apples-to-apples FSDP2 counterpart of `trl_deepspeed`:
identical SFTTrainer loop, only the parallelism plugin differs. Use it
for HF model families TorchTitan does not cover (e.g. Qwen3.5,
Qwen3.5-MoE).

The top-level launcher:

```bash
python baseline/run_baseline.py \
  --backend trl_deepspeed \
  --model-path models/Llama-3.1-8B-Instruct \
  --seq-length 8192 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-steps 3 \
  --num-gpus 4 \
  --dry-run
```

For a complete Llama-3-family 128K maximum-memory-savings sweep with
per-step throughput extraction, see
[`RUNBOOK_128K_MAX_MEMORY.md`](RUNBOOK_128K_MAX_MEMORY.md).

## Sweeps

Multi-backend evaluation sweeps are TOML-driven. Each sweep config has a
`[common]` section with run-wide settings (model path, sequence length,
num steps, num GPUs) and one `[<backend>]` section per backend with that
backend's memory-feature knobs (offloading, activation checkpointing,
sequence-parallel size, etc). The launcher walks the sections in order,
records failures without aborting the sweep, and emits a per-step
`throughput.csv` at the end.

```bash
# Run every backend section in the config:
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml

# Subset of backends:
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml \
    --backends trl_fsdp,trl_deepspeed

# Smoke-test commands without running them:
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml \
    --num-steps 1 --dry-run
```

Templating: any string value containing `${NUM_GPUS}` is substituted at
launch time (the launcher detects num_gpus from `nvidia-smi -L`, or you
can pass `--num-gpus N`). See
[`configs/llama3_128k_maxmem.toml`](configs/llama3_128k_maxmem.toml) for
the canonical example.

## Installation Model

Running baselines does not require installing `flextrain` as a Python
package. The launcher and synthetic datasets are repo-local source
files; `baseline/run_baseline.py` adds the repo root to `PYTHONPATH`,
and generated backend commands do the same.

The harness installs into **two conda envs** that you'll see in
`conda env list`:

| Conda env | Backends |
|---|---|
| `baseline_core` | `trl_deepspeed`, `trl_fsdp`, `deepspeed_arctic`, `megatrain`, `torchtitan` |
| `baseline_megatron` | `megatron` |

The five "core" backends share one env because their pip deps are
mutually compatible. Megatron lives alone because `transformer-engine`
pins torch tightly and historically conflicts with the deeper deps of
the HF backends. Override env names with `--env-name NAME` if you want
something else.

Set up the envs:

```bash
# baseline_core (one env covers five backends — re-running with a
# different --backend just adds that backend's vendor checkout if any):
baseline/scripts/install_backend.sh --backend trl_deepspeed
baseline/scripts/install_backend.sh --backend trl_fsdp
baseline/scripts/install_backend.sh --backend deepspeed_arctic
baseline/scripts/install_backend.sh --backend megatrain
baseline/scripts/install_backend.sh --backend torchtitan

# baseline_megatron (separate env):
baseline/scripts/install_backend.sh --backend megatron
```

What each invocation does (idempotent):

1. **Conda env**: creates the target env (`baseline_core` or
   `baseline_megatron`) with `python=3.12` if missing, else reuses it.
2. **Torch**: detects local CUDA via
   [`baseline/scripts/detect_cuda.py`](scripts/detect_cuda.py)
   (`nvidia-smi` → `nvcc` → `/usr/local/cuda/version.json`), maps to the
   right torch wheel index (`cu130` / `cu128` / `cu126` / `cu124` /
   `cu121`), and pip-installs `torch torchvision torchaudio` from that
   index.
3. **Requirements**: `pip install -r baseline/requirements/<env>.txt`
   (consolidated `baseline_core.txt` or `baseline_megatron.txt`).
4. **Vendor checkouts** (core env only): editable-installs MegaTrain
   and TorchTitan if either is present in `baseline/<Name>/` or the
   gitignored `baseline/vendor/<Name>/`; otherwise clones from the
   pinned upstream.
5. **FlashAttention**: resolves matching prebuilt FA2 + FA3 wheels from
   [`mjun0812/flash-attention-prebuild-wheels`](https://github.com/mjun0812/flash-attention-prebuild-wheels)
   based on the env's torch + CUDA + Python ABI + platform.
6. **`flash-linear-attention`** + matching prebuilt `causal-conv1d`
   wheel from
   [`Dao-AILab/causal-conv1d`](https://github.com/Dao-AILab/causal-conv1d/releases)
   (core env only). The `causal-conv1d` resolver probes the detected
   torch tag and walks back through earlier torch minors (default 2)
   so a fresh torch release without an exact prebuilt wheel still
   picks up the most recent ABI-compatible one.

Detect the local CUDA version manually:

```bash
baseline/scripts/detect_cuda.py
# cuda_version=13.1
# torch_cuda_tag=cu130
# torch_index_url=https://download.pytorch.org/whl/cu130
```

Override the torch wheel index for a cluster image:

```bash
baseline/scripts/install_backend.sh --backend trl_deepspeed \
  --torch-index-url https://download.pytorch.org/whl/cu130
```

Skip the torch reinstall (env already has the torch you want):

```bash
baseline/scripts/install_backend.sh --backend trl_deepspeed --skip-torch
```

Drop and recreate the conda env from scratch:

```bash
baseline/scripts/install_backend.sh --backend trl_deepspeed --recreate
```

Other install controls:

- `--env-name NAME`: target a non-default conda env (`baseline_core` /
  `baseline_megatron` are the defaults). The `BASELINE_CORE_ENV` and
  `BASELINE_MEGATRON_ENV` env vars override the defaults globally for
  both `install_backend.sh` and `run_in_backend_env.sh`.
- `--python-version VER`: python version for `conda create`. Default
  `3.12`. Only honoured when the env is being created from scratch.
- `--flash-version VERSION`: require an exact prebuilt FlashAttention
  package version, e.g. `--flash fa2 --flash-version 2.8.3`.
- `--linear-attention {auto,strict,none}`: install
  `flash-linear-attention` and Qwen `causal-conv1d` deps where useful.
  `auto` warns when no exact `causal-conv1d` wheel exists, `strict`
  fails.
- `--causal-conv1d-torch-tag TAG`: pin a specific `causal-conv1d` wheel
  torch tag (disables the automatic minor-version probing); only needed
  when you want a wheel other than the latest ABI-compatible one.

Run one backend through the dispatcher (auto-activates the right
conda env per backend):

```bash
baseline/scripts/run_in_backend_env.sh trl_deepspeed \
  --model-path models/Llama-3.1-8B-Instruct \
  --seq-length 8192 \
  --num-gpus 4
```

`run_in_backend_env.sh` activates `baseline_core` or `baseline_megatron`
based on the requested backend, then runs
[`baseline/scripts/check_cuda_compat.py`](scripts/check_cuda_compat.py)
before the backend launches — so a torch wheel built for a CUDA newer
than the installed driver fails fast with an actionable message instead
of an opaque CUDA error mid-run. Set `BASELINE_SKIP_CUDA_CHECK=1` to
bypass (useful for CI installs done off-GPU) or
`BASELINE_CUDA_CHECK_WARN_ONLY=1` to downgrade the failure to a warning.

If you'd rather activate the env yourself once and run multiple
backends:

```bash
conda activate baseline_core
python baseline/run_baseline.py --backend trl_fsdp ...
python baseline/run_baseline.py --backend trl_deepspeed ...
```

## Common Arguments

Required:

- `--backend {megatrain,torchtitan,trl_deepspeed,deepspeed_arctic,megatron,trl_fsdp,all}`
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
- `--attn-implementation {auto,flash_attention_3,flash_attention_2,sdpa,eager}`: default is `auto` — probes FA3 first, then FA2, then SDPA, then eager. Pass an explicit implementation to pin (e.g. `flash_attention_2` or `sdpa`) for fallback comparisons. MegaTrain only supports `flash_attention_2`, `sdpa`, and `eager`; under `auto` MegaTrain skips FA3 automatically.
- `--moe-kernel-backend {hf,auto,sonic}`: HF MoE expert kernel. `hf` keeps the model's default; `auto` enables HF's native [SonicMoE](https://huggingface.co/kernels-community/sonic-moe) via `model.set_experts_implementation("sonicmoe")` and falls back to `hf` if the model class doesn't expose it; `sonic` is strict and raises on failure. Applies to all four HF-loading backends (`trl_deepspeed`, `deepspeed_arctic`, `trl_fsdp`, `megatrain`); `torchtitan` and `megatron` use their own native MoE-kernel selection.
- `--liger-kernel {auto,on,off}`: TRL Liger mode; `auto` enables it when `liger-kernel` and TRL support are installed
- `--use-liger-kernel`: legacy alias for `--liger-kernel on`

### bf16 master weights / grads / optimizer states

Every backend in this harness defaults to **bf16 throughout** — bf16
parameters, bf16 gradients, bf16 optimizer states, no fp32 master copy.
Concretely:

- `trl_deepspeed`, `deepspeed_arctic`: generated DeepSpeed configs set `bf16.enabled=true`, `bf16.bf16_master_weights_and_grads=true`, `bf16.bf16_optimizer_states=true`, `optimizer.fp32_optimizer_states=false`. CPU optimizer/parameter offload entries use `pin_memory=true`.
- `trl_fsdp`: generated `accelerate_fsdp.yaml` sets `mixed_precision: bf16` and `fsdp_version: 2`; FSDP2's MixedPrecisionPolicy stores params/grads/opt in bf16 with no fp32 master.
- `torchtitan`: launched with `training.dtype=bfloat16` and `training.mixed_precision_param=bfloat16`.
- `megatron`: train script defaults `use_precision_aware_optimizer=True`, giving bf16 main_params + bf16 exp_avg + bf16 exp_avg_sq.
- `megatrain`: synthetic entrypoint constructs `CPUMasterConfig(dtype=torch.bfloat16)`.

DeepSpeed launches also set `DS_SKIP_CUDA_CHECK=1` and
`PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True,expandable_segments:True`
unless those env vars are already set.

### Fractional activation checkpointing support

- `trl_deepspeed`: HF/TRL does not expose supported fractional layer checkpointing. The launcher rejects true fractional values (`0 < f < 1`); use `--activation-checkpointing none/full`.
- `trl_fsdp`: same as `trl_deepspeed` — HF gradient_checkpointing is binary; rejected for `0 < f < 1`.
- `deepspeed_arctic`: this path is still a HuggingFace model path, so true fractional values (`0 < f < 1`) are rejected. Use `--activation-checkpointing none/full` plus ALST knobs such as sequence parallelism, tiled loss, tiled MLP, and activation offload.
- `torchtitan`: fractional values are approximated by TorchTitan selective checkpointing every `round(1/f)` layers. `f=1` maps to full, `f=0` maps to none.
- `megatron`: fractional values map to Megatron full block recompute for `floor(num_layers * f)` layers. `f=1` maps to full uniform recompute, `f=0` maps to no recompute.
- `megatrain`: no true layer-fraction mode is exposed; true fractional values are rejected. Use `--activation-checkpoint-interval` to control recompute segment size.

## Synthetic Data

Synthetic examples are generated as random integer token IDs in
`[0, vocab_size)`, where `vocab_size` is read from the model's
`config.json`. This avoids benchmarking tokenizer throughput or
text-packing behavior. Labels are shaped to match each backend's
causal-LM API:

- HuggingFace/TRL/DeepSpeed/MegaTrain receive unshifted labels and let the model/backend shift.
- TorchTitan receives pre-shifted labels because its dataloader contract yields `(input, next_token_label)`.
- DeepSpeed ALST/Ulysses uses the DeepSpeed dataloader adapter to generate `shift_labels` when sequence parallelism is enabled.

## Backend Notes

### TRL + DeepSpeed (`trl_deepspeed`)

Entrypoint:

```bash
python baseline/run_baseline.py --backend trl_deepspeed ...
```

Uses TRL `SFTTrainer` with `dataset_kwargs={"skip_prepare_dataset":
True}` so the token-native synthetic dataset is passed through directly.
Important mappings:

- `--zero-stage`, `--optimizer-offload`, `--param-offload` → generated DeepSpeed JSON
- `--activation-checkpointing` → HF gradient checkpointing
- `--activation-offload cpu` → TRL `activation_offloading` when available
- `--moe-kernel-backend sonic` → `model.set_experts_implementation("sonicmoe")` (HF native dispatch)
- `--liger-kernel auto` → sets TRL `use_liger_kernel=True` only when `liger-kernel` is installed and the local TRL `SFTConfig` exposes the option

### TRL + FSDP2 (`trl_fsdp`)

Entrypoint:

```bash
python baseline/run_baseline.py --backend trl_fsdp ...
```

Same TRL `SFTTrainer` loop as `trl_deepspeed`; the harness writes an
`accelerate_fsdp.yaml` (FSDP2 plugin: `fsdp_version=2`,
`fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP`,
`fsdp_sharding_strategy=FULL_SHARD` or `HYBRID_SHARD` depending on
`--fsdp-replicate-degree`) and launches via `accelerate launch
--config_file=...`. Important mappings:

- `--param-offload cpu` OR `--optimizer-offload cpu` → `fsdp_offload_params: true` (FSDP2 ties param + grad + opt offload together; both flags map to the same setting)
- `--activation-checkpointing != none` → `fsdp_activation_checkpointing: true` and HF `gradient_checkpointing`
- `--activation-offload cpu` → TRL `activation_offloading` when available
- `--fsdp-shard-degree`, `--fsdp-replicate-degree` → FSDP plugin sharding strategy
- `--moe-kernel-backend sonic` → `model.set_experts_implementation("sonicmoe")` (HF native dispatch)
- `--liger-kernel auto` → TRL `use_liger_kernel` when supported

### DeepSpeed Arctic / ALST (`deepspeed_arctic`)

Entrypoint:

```bash
python baseline/run_baseline.py --backend deepspeed_arctic --sequence-parallel-size 4 ...
```

Uses a custom HF + DeepSpeed loop in
`baseline/backends/deepspeed_arctic/train_synthetic.py`. Important
mappings:

- `--sequence-parallel-size > 1` enables DeepSpeed Ulysses SP registration and dataloader adaptation
- `--zero-stage`, `--optimizer-offload`, `--param-offload` → generated DeepSpeed JSON
- `--moe-kernel-backend sonic` → `model.set_experts_implementation("sonicmoe")` (HF native dispatch)
- `--activation-offload cpu` attempts the Arctic Training activation checkpoint offload monkey patch if installed
- `--tiled-mlp` attempts the DeepSpeed ALST Llama MLP tiling hook

### MegaTrain (`megatrain`)

Entrypoint:

```bash
python baseline/run_baseline.py --backend megatrain ...
```

Uses `baseline/MegaTrain` as the vendor tree (or fetches into
`baseline/vendor/MegaTrain` on first install) and
`baseline/backends/megatrain/train_synthetic.py` as the synthetic
entrypoint, which loads an HF model, optionally swaps in SonicMoE, then
wraps with `CPUMasterModel`. Extra knobs:

- `--activation-checkpoint-interval`: maps to MegaTrain `checkpoint_interval` (vendor default `4`, meaning checkpoint every N layers)
- `--num-grad-slabs`: maps to MegaTrain gradient slab pool size
- `--moe-kernel-backend sonic` → `model.set_experts_implementation("sonicmoe")` on the underlying HF model before wrapping
- `--backend-extra-arg --optimizer --backend-extra-arg deepspeed_cpu_adam`

### TorchTitan (`torchtitan`)

Entrypoint:

```bash
python baseline/run_baseline.py --backend torchtitan ...
```

The harness looks for `baseline/TorchTitan`, then
`baseline/vendor/TorchTitan`, then the old
`orig/baseline/torchtitan`. The synthetic registry is
`baseline.backends.torchtitan.synthetic_registry` and currently covers
TorchTitan's built-in `llama3_8b`, `qwen3_1_7b`, `qwen3_32b`, and debug
specs. For custom specs:

```bash
--torchtitan-module your.package.registry --torchtitan-config your_config
```

Important mappings:

- `--activation-checkpointing` → `--activation_checkpoint.mode`
- `--activation-checkpoint-fraction` → approximate selective interval via `--activation_checkpoint.selective_ac_option`
- `--activation-offload cpu` → `--training.enable_activation_offload`
- `--optimizer-offload cpu` or `--param-offload cpu` → `--training.enable_cpu_offload`
- tensor/pipeline/context/FSDP flags map to `--parallelism.*`

TorchTitan does not load HF model objects, so `--moe-kernel-backend
sonic` does not apply (TorchTitan has its own MoE-kernel selection).

### Megatron-Core (`megatron`)

Entrypoint:

```bash
python baseline/run_baseline.py --backend megatron ...
```

The harness generates a per-run `model_dims.json` from the HuggingFace
config and launches the Megatron script. It prefers
`baseline/backends/megatron/train.py` if present and otherwise falls
back to `orig/baseline/megatron/train.py`. Important mappings:

- `--recompute-granularity`, `--recompute-method`, `--recompute-num-layers`, `--recompute-modules`
- `--activation-checkpoint-fraction` → Megatron block recompute count
- `--activation-offload cpu` → TE layer CPU offload flags
- `--offload-modules` → Megatron fine-grained activation offload
- `--optimizer-offload cpu` → Megatron optimizer CPU offload

Megatron-Core builds models from `model_dims.json`, not from HF model
classes, so `--moe-kernel-backend sonic` does not apply (Megatron-Core
has its own MoE-kernel selection).

## MoE kernel routing (HF native)

For the four HF-loading backends, the SonicMoE kernel is reached through
HF's native dispatch — the harness no longer ships a custom
adapter/module-replacer. The path is:

1. Backend-specific train script loads the HF model via `AutoModelForCausalLM.from_pretrained`.
2. `baseline.backends.common.moe_kernel.apply_moe_kernel_backend(model, mode)` calls `model.set_experts_implementation("sonicmoe")` (introduced in [HF transformers PR #45433](https://github.com/huggingface/transformers/pull/45433)).
3. HF's modular `Experts` module rebinds its forward to `transformers.integrations.sonicmoe.sonicmoe_experts_forward`, which dispatches the [`kernels-community/sonic-moe`](https://huggingface.co/kernels-community/sonic-moe) kernel via `kernels.lazy_load_kernel("sonic-moe")`.

This route preserves the model's own router (no router replacement), is
DTensor-safe under FSDP2/EP, supports biased experts, and gracefully
no-ops for non-MoE configs. `mode="auto"` falls back to the model's
default kernel on any failure (transformers too old, model class without
`set_experts_implementation`, kernel can't load on this GPU/CUDA combo);
`mode="sonic"` is strict.

## References Used For API Choices

- DeepSpeed bf16 config keys: https://www.deepspeed.ai/docs/config-json/
- DeepSpeed Arctic/ALST Ulysses integration: https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/
- Hugging Face Transformers DeepSpeed ALST guide: https://huggingface.co/docs/transformers/main/deepspeed_alst
- TRL SFTTrainer/SFTConfig: https://huggingface.co/docs/trl/en/sft_trainer
- accelerate FSDP plugin (FSDP2): https://huggingface.co/docs/accelerate/usage_guides/fsdp
- HF transformers ExpertsInterface (sonicmoe dispatch): https://github.com/huggingface/transformers/pull/45433
- FlashAttention prebuilt wheels: https://github.com/mjun0812/flash-attention-prebuild-wheels
- causal-conv1d prebuilt wheels: https://github.com/Dao-AILab/causal-conv1d/releases
- Flash Linear Attention: https://github.com/fla-org/flash-linear-attention
- Hugging Face kernels loader and SonicMoE kernel: https://github.com/huggingface/kernels and https://huggingface.co/kernels-community/sonic-moe
