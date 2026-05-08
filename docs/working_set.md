# Working set tuning

The `WorkingSetConfig` decides what stays GPU-resident and what gets
offloaded to host between fwd and bwd / between bwd and the optimizer
step. The solver in `flextrain/core/working_set.py` picks one
automatically; this page covers what it does and how to override.

## What the solver decides

Given memory budgets (`max_gpu_mem_bytes`, `max_host_mem_bytes`,
leeways) and model shape:

| Knob | What it controls |
|------|------------------|
| `n_gpu_layers` | How many backbone-layer parameter copies live on GPU at once. The rest cycle in/out via H2D async copies. |
| `n_gpu_grads` | Same for gradient buffers. Usually = `n_gpu_layers`. |
| `n_gpu_opt_layers` | How many layers' opt state fits on GPU during the optimizer step. |
| `gpu_act_buffer_size` | Total bytes for the activation ring (also reused as the opt-state staging area). |
| `host_act_buffer_size` | Total bytes for activation persistence on host (high-tier saves). |
| `target_round_tokens`, `max_chunk_size` | How big each fwd/bwd chunk is. |
| Save level | One of {0, 1, 2, 3}. Lower = recompute more activations from tier-0 saves; higher = save more, recompute less. |

The solver picks save level + chunk size + how-many-resident-layers
jointly via a small dynamic-programming optimization that targets a
GPU memory budget.

## How to call the solver

```python
from flextrain.core.working_set import determine_working_set_config

ws = determine_working_set_config(
    model_dims={
        "d_model": 4096, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128,
        "expert_dim": 14336, "vocab_size": 128256, "n_layers": 32,
        "num_shared_experts": 0, "num_routed_experts": 0, "top_k": 0,
        "is_causal": True,
        "datatypes": {
            "embed": "bfloat16", "head_proj": "bfloat16",
            "attn_proj": "bfloat16", "expert_proj": "bfloat16",
            "router": "bfloat16", "norm": "bfloat16", "residual": "bfloat16",
        },
    },
    max_seq_len=4096, max_global_batch_tokens=8192,
    training_config={
        "master_weight_dtype": "bfloat16", "grad_dtype": "bfloat16",
        "opt_choice": "AdamW", "opt_dtype": "float32",
    },
    has_embed=True, has_head=True, num_local_layers=32,
    max_gpu_mem_bytes=int(24 * (1 << 30)),
    max_host_mem_bytes=int(110 * (1 << 30)),
    leeway_gpu_mem_bytes=int(2 * (1 << 30)),
    leeway_host_mem_bytes=int(4 * (1 << 30)),
    verbose=True,
)
```

`verbose=True` prints what the solver picked.

## Hand-overriding the plan

For tests / debugging, build a `WorkingSetConfig` directly:

```python
from flextrain.core.working_set import WorkingSetConfig

ws = WorkingSetConfig(
    target_round_tokens=2048,
    max_chunk_size=2048,
    max_training_chunks=4,
    max_total_round_tokens=2048,
    target_num_rounds=1,
    n_gpu_layers=8, n_gpu_grads=8, n_gpu_opt_layers=2,
    gpu_act_buffer_size=int(2 * (1 << 30)),
    host_act_buffer_size=int(4 * (1 << 30)),
    available_gpu_memory_bytes=int(24 * (1 << 30)),
    available_host_memory_bytes=int(110 * (1 << 30)),
    leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
    max_seq_len=2048, hardware_env={}, raw={},
)
```

## Save levels

* `save_level=0` — only tier-0 activations saved (RMSNorm rstd, MoE
  router state, etc). Forward is recomputed twice for almost
  everything during bwd. Lowest memory, highest compute.
* `save_level=3` — all tiers saved. Highest memory, lowest compute.
* `save_level=1`, `save_level=2` — intermediate tradeoffs. Tier-1
  saves the flash-attn intermediates; tier-2 adds large pre-projection
  tensors.

The engine guarantees that **loss curves are bit-identical across save
levels** for the same optimizer + initialization — the planner only
chooses where intermediates live, never what's computed.

## Common diagnostic outputs

The solver prints:
* `Comparing prior tokens per round: T with min chunk size...` — the
  initial chunk-size search.
* `Determined Initial Target Min Chunk Size Est ... of: M` — the
  arithmetic-intensity-bound minimum chunk size for this model. MoE
  models have higher floors (1.5K-2K tokens for 64-128 experts).
* `Selected Best Option: {...}` — final picks.
* `Could not find a valid configuration for seq len T; estimating
  max tokens per round to be N` — the budget is too small even for
  full offloading. Reduce model size or increase `target_round_tokens`.

## When the solver fails

Two common cases:

1. **MoE arithmetic-intensity floor exceeds budget.** Solution: bump
   `max_global_batch_tokens` up so the chunk size has room to satisfy
   the floor. For OLMoE with 64 experts, 4096 is comfortable.

2. **Not enough GPU memory for one full layer.** This indicates a
   too-large model for the GPU. Reduce model size, or reduce the
   `gpu_act_buffer_size` (less compute headroom for activations).

## When you should override

* You're benchmarking save-level invariance (force one save level
  explicitly via `force_saved_act_level`).
* You want to test offloading with a small model where the solver would
  pick all-resident — set `n_gpu_layers < n_layers` manually.
* You're prototyping a new arch and the solver doesn't know how to
  estimate its compute cost — work in pinned-config mode until you
  add the layer's `compute_cost` correctly.

## Memory budget heuristics

For an `N`-param model (excluding embed+head) with bf16 master, fp32
opt state:

| Component | Bytes |
|-----------|-------|
| Master weights (host) | `2 N` |
| Grads (host + GPU resident) | `2 N + 2 N * (n_gpu_grads / n_layers)` |
| Opt state (host) | `8 N` (AdamW fp32) or `2 N` (Muon bf16) |
| GPU compute slots | `2 N * (n_gpu_layers / n_layers)` |
| Activation ring | depends on save level & chunk size |

Hybrid Muon + AdamW saves ~40% on opt state vs pure AdamW for
transformer-shaped param distributions.
