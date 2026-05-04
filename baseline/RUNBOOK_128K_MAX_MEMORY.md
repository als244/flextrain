# 128K Max-Memory Baseline Runbook

End-to-end recipe for running synthetic-token Llama-3-family training at
128K sequence length across all baseline backends, single-GPU, with the
most aggressive memory-savings knob each backend exposes.

The actual sweep is driven by [configs/llama3_128k_maxmem.toml](configs/llama3_128k_maxmem.toml)
and dispatched by [scripts/sweep.py](scripts/sweep.py). The runbook is
just the bootstrap: clone, install, point at a checkpoint, run.

## 0. Prereqs

- An H100 (or comparable) box with NVIDIA driver supporting CUDA 12.6+.
- A local Llama-3.1 / 3.3 checkpoint with native 128K context
  (`max_position_embeddings >= 131072` in `config.json`). Original Llama
  3 8B checkpoints are 8K context only and will not work without rope
  scaling.

## 1. Install conda envs (one-time)

The harness installs into two conda envs you'll see in `conda env list`:
`baseline_core` (covers `trl_deepspeed`, `trl_fsdp`, `deepspeed_arctic`,
`megatrain`, `torchtitan` — their pip deps are mutually compatible) and
`baseline_megatron` (Megatron-Core + transformer-engine, kept in its own
env because TE pins torch tightly).

```bash
cd "$(git rev-parse --show-toplevel)"

# Set up baseline_core (one invocation does all five core backends —
# the core conda env is reused after the first --backend creates it).
baseline/scripts/install_backend.sh --backend trl_deepspeed
baseline/scripts/install_backend.sh --backend trl_fsdp
baseline/scripts/install_backend.sh --backend deepspeed_arctic
baseline/scripts/install_backend.sh --backend megatrain
baseline/scripts/install_backend.sh --backend torchtitan

# Set up baseline_megatron (its own conda env).
baseline/scripts/install_backend.sh --backend megatron
```

What each invocation does (idempotent — re-running is safe):
- Creates the target conda env with `python=3.12` if not already present.
- Detects the local CUDA via `nvidia-smi` and installs torch from the
  matching wheel index (cu130 / cu128 / cu126 / cu124 / cu121).
- Installs the consolidated requirements file
  (`baseline/requirements/baseline_core.txt` or `baseline_megatron.txt`).
- Resolves and installs prebuilt FlashAttention 2 + 3 wheels matching
  the env's torch / CUDA / Python ABI / platform.
- For backends that need it: editable-installs the MegaTrain or
  TorchTitan vendor checkout (clones into `baseline/vendor/...` if no
  local checkout exists).
- Resolves and installs a matching prebuilt `causal-conv1d` wheel into
  `baseline_core` (probes the detected torch tag and walks back through
  earlier torch minors to find an ABI-compatible one).

You can verify the conda envs after install with:

```bash
conda env list
# # conda environments:
# #
# baseline_core         /home/.../miniconda3/envs/baseline_core
# baseline_megatron     /home/.../miniconda3/envs/baseline_megatron
```

If you want a specific FA wheel version pinned (rather than the latest
matching one), pass `--flash-version 2.8.3` etc. to the install command.

## 2. Point at your model

Edit `baseline/configs/llama3_128k_maxmem.toml` so `[common].model_path`
points at your local checkpoint (or copy the file and edit the copy).
The config has one section per backend with that backend's memory-feature
knobs already filled in.

## 3. Run the sweep

```bash
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml
```

What this does:
- Detects `num_gpus` from `nvidia-smi -L` (override with `--num-gpus N`).
- For each `[<backend>]` section in the config, dispatches one run via
  `run_in_backend_env.sh <backend>`. The CUDA pre-flight check runs
  first; FA3 is probed before FA2 (default `attn_implementation = auto`).
- A failure in one backend (OOM, missing dep, traceback) is logged and
  the sweep advances; it never aborts the whole comparison.
- At the end, prints a per-backend summary and writes
  `<sweep-root>/throughput.csv` from the per-backend `run.log` files.

The sweep root lands at:
```
baseline/runs/llama3_128k_maxmem_<timestamp>/
  trl_fsdp/                                # one dir per backend
    accelerate_fsdp.yaml                   # generated
    launch_plan.json
    run.log
  trl_deepspeed/
    deepspeed_bf16.json
    launch_plan.json
    run.log
  ...
  throughput.csv                           # combined per-step throughput
  llama3_128k_maxmem.toml                  # snapshot of the sweep config
```

## 4. Useful overrides

```bash
# Smoke test: 1 step, dry-run the commands
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml \
    --num-steps 1 --dry-run

# Run only one or two backends
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml \
    --backends trl_fsdp,trl_deepspeed

# Custom output dir (e.g. scratch FS)
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml \
    --output-root /scratch/$USER/sweep_$(date +%s)
```

## 5. Backend notes

All six backends default to **bf16 master weights / grads / optimizer
states** and FA3→FA2→SDPA→eager attention fallback (the `auto` default).
HF backends (`trl_deepspeed`, `deepspeed_arctic`, `trl_fsdp`,
`megatrain`) reach the SonicMoE kernel via HF's native
`model.set_experts_implementation("sonicmoe")` rather than a custom
adapter — set `--moe-kernel-backend sonic` (or `auto` for soft fallback)
to opt in.

| Backend | Parallelism | Memory features used in this sweep |
|---|---|---|
| `trl_fsdp` | TRL SFTTrainer + accelerate FSDP2 | FSDP CPU offload (params + grads + opt), FSDP activation checkpointing, TRL activation offload, Liger |
| `trl_deepspeed` | TRL SFTTrainer + DeepSpeed ZeRO-3 | DS param offload, DS opt offload, gradient checkpointing, TRL activation offload, Liger |
| `deepspeed_arctic` | Custom HF + DeepSpeed Ulysses SP | ZeRO-3, param/opt offload, sequence parallel, tiled MLP, tiled loss |
| `torchtitan` | TorchTitan FSDP2 | Activation checkpointing, activation/param/opt CPU offload, FSDP shard |
| `megatron` | Megatron + TE | Full uniform recompute, TE activation/weight CPU offload, optimizer CPU offload |
| `megatrain` | CPU-master + grad slabs | checkpoint_interval=4, deepspeed_cpu_adam optimizer |

`trl_fsdp` is the recommended HF-native baseline for model families
TorchTitan does not cover (Qwen3.5, Qwen3.5-MoE) — same SFTTrainer loop
as `trl_deepspeed`, FSDP2 instead of DeepSpeed ZeRO.

## 6. Interpreting results

`throughput.csv` columns: `backend, step, loss, step_time_s, tokens_per_s, log, line`.

- An OOM produces a `run.log` with no per-step lines — check the tail
  for `out of memory` / `CUDA out of memory`. The OOM is itself a
  meaningful "this backend cannot fit 128K @ 1 GPU at these settings"
  result; keep the log.
- If `trl_fsdp` and `trl_deepspeed` both run, their `tokens_per_s` is
  the apples-to-apples FSDP2 vs ZeRO-3 comparison (same TRL training
  loop, only the parallelism plugin differs).
- If a model's `config.json` lacks 128K context metadata (Llama 3 8B
  base, etc.), every backend will either fail or produce nonsense
  positions. Fix the model, not the runbook.
