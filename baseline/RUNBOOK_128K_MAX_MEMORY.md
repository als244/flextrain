# 128K Max-Memory Baseline Runbook

Synthetic-token Llama-3-family training at 128K, single-GPU, with the
most aggressive memory-savings knob each backend exposes.

> **Assumes** you've installed the harness with
> `baseline/scripts/install_backend.sh` (creates `baseline_core` and
> `baseline_megatron` conda envs). See
> [README.md](README.md#installation-model) if you haven't.

## TL;DR

```bash
# Run all six backends in the sweep config:
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml

# Or only specific backends:
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml \
    --backends trl_fsdp,trl_deepspeed
```

Outputs land in `baseline/runs/llama3_128k_maxmem_<timestamp>/<backend>/`,
with a combined `throughput.csv` at the sweep root.

## Pick your model

Edit
[`configs/llama3_128k_maxmem.toml`](configs/llama3_128k_maxmem.toml)'s
`[common].model_path` to point at your local checkpoint (or copy the
file first and edit the copy). The model needs native 128K context —
`max_position_embeddings >= 131072` in `config.json`. Llama 3.1 / 3.3
work; vanilla Llama 3 8B (8K context) does not.

## Pick your backends

`--backends` is a comma-separated subset of the backends declared in
the config:

| Use case | Command |
|---|---|
| Everything in the config | (no `--backends` flag) |
| TRL-based head-to-head: FSDP2 vs ZeRO-3 | `--backends trl_fsdp,trl_deepspeed` |
| Just one backend | `--backends megatrain` |
| HF-only (no TorchTitan / Megatron) | `--backends trl_fsdp,trl_deepspeed,deepspeed_arctic,megatrain` |

The launcher fails fast if you name a backend that has no `[<backend>]`
section in the config.

## Other overrides

```bash
# Faster smoke test (1 step instead of 5):
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml \
    --num-steps 1

# Print commands without running them:
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml \
    --backends trl_fsdp --dry-run

# Output to a scratch FS:
python baseline/scripts/sweep.py baseline/configs/llama3_128k_maxmem.toml \
    --output-root /scratch/$USER/sweep_$(date +%s)
```

`num_gpus` is auto-detected from `nvidia-smi -L`; override with
`--num-gpus N` if you want.

## What the launcher does

- Activates the right conda env per backend (`baseline_core` for
  `trl_*`/`deepspeed_arctic`/`megatrain`/`torchtitan`,
  `baseline_megatron` for `megatron`) via
  `run_in_backend_env.sh`.
- Runs the CUDA preflight check before each backend.
- Records per-backend pass/fail **without aborting the sweep** — one
  OOM doesn't kill the rest of the comparison.
- Writes per-step throughput to `<sweep-root>/throughput.csv` at the
  end.

Sweep root layout:

```
baseline/runs/llama3_128k_maxmem_<timestamp>/
  trl_fsdp/
    accelerate_fsdp.yaml      # generated launch config
    launch_plan.json
    run.log
  trl_deepspeed/
    deepspeed_bf16.json
    launch_plan.json
    run.log
  ...
  throughput.csv              # combined per-step throughput
  llama3_128k_maxmem.toml     # snapshot of the sweep config
```

## Interpreting results

`throughput.csv` columns: `backend, step, loss, step_time_s, tokens_per_s, log, line`.

- An OOM produces a `run.log` with no per-step lines — check the tail
  for `CUDA out of memory`. An OOM is itself a meaningful result
  ("this backend cannot fit 128K @ 1 GPU at these settings"); keep
  the log.
- `trl_fsdp` vs `trl_deepspeed` is the apples-to-apples FSDP2 vs ZeRO-3
  comparison (same TRL training loop, only the parallelism plugin
  differs).
- If the model's `config.json` lacks 128K context metadata, every
  backend will either fail or produce nonsense positions — fix the
  model, not the runbook.

## What each backend uses in this sweep

All backends default to **bf16 master weights / grads / optimizer
states** and FA3→FA2→SDPA→eager attention fallback (the `auto`
default). HF backends additionally route through
`model.set_experts_implementation("sonicmoe")` when the model exposes
it.

| Backend | Memory features in this config |
|---|---|
| `trl_fsdp` | FSDP2 CPU offload (params + grads + opt), FSDP activation checkpointing, TRL activation offload, Liger |
| `trl_deepspeed` | ZeRO-3, DS param + opt offload, gradient checkpointing, TRL activation offload, Liger |
| `deepspeed_arctic` | ZeRO-3, param/opt offload, sequence parallel, tiled MLP, tiled loss |
| `torchtitan` | Activation checkpointing, activation/param/opt CPU offload, FSDP shard |
| `megatron` | Full uniform recompute, TE activation/weight CPU offload, optimizer CPU offload |
| `megatrain` | `checkpoint_interval=4`, `deepspeed_cpu_adam` optimizer |

For the full per-backend flag mapping (recompute granularity, sequence
parallel, etc), see the top-level
[baseline/README.md](README.md#backend-notes).
