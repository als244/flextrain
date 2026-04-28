# FlexTrain

Train transformer LLMs on hardware where the model doesn't fit in GPU
memory. FlexTrain rotates parameters, gradients, optimizer state, and
activations between GPU and host RAM via a working-set planner + DP
solver, so an 8B model trains end-to-end on a 24 GiB GPU without
DeepSpeed or FSDP.

Supported architectures: Llama-3, Qwen2/3, Qwen3-MoE, OLMoE, Qwen3-Next,
Gemma 2/3. See [`docs/architectures.md`](docs/architectures.md).

## Install

```bash
conda create -n flextrain python=3.12
conda activate flextrain
pip install torch triton
pip install -e .
```

The install builds two in-tree C/CUDA helpers (`matmul_dispatcher`,
`transmission_scheduler`); set `FLEXTRAIN_SKIP_HELPERS=1` to skip them
when iterating on Python-only code. Optional extras:
`-e ".[flash-attn]"` (Hopper/Blackwell), `-e ".[linear-attention]"`
(Qwen3-Next).

## Quickstart

```bash
python train.py \
  --model meta-llama/Llama-3.1-8B \
  --mode lora \
  --seq-len 1024 \
  --global-batch-tokens 1024
```

Auto-discovers your GPU + host memory budgets, probes hardware
(sustained TFLOPS / PCIe / mem bandwidth), downloads the model into
`models/` if needed, runs 20 steps on a tiny bundled SFT dataset, and
logs to `runs/<model>_<mode>_sl<seq_len>`.

Common flags:

| flag | what it does |
|---|---|
| `--mode {full,lora}` | full fine-tune or LoRA |
| `--use-muon` | use HybridMuonAdamW for `--mode full` (Muon on dense projections, AdamW elsewhere) |
| `--data-source {synthetic,json_sft}` | synthetic tokens or SFT JSONL |
| `--dataset path/or/repo` | local JSONL, HF repo, or http(s) URL |
| `--max-gpu-mem-gib N` / `--max-host-mem-gib N` | override auto-discovered budgets |
| `--force-save-level {0,1,2,3}` | force activation save-tier (debug) |
| `--save` | export `final.safetensors` after training |

Run `python train.py --help` for the full list.

### Learning rate schedule

Every run uses linear warmup → constant peak → cosine cooldown. Tune via:

| flag | default | what |
|---|---|---|
| `--lr` | 3e-5 (full), 1e-4 (lora), 1e-3 (`--use-muon`) | peak (max) LR |
| `--lr-warmup-pct` | 0.1 | fraction of steps spent ramping 0 → peak |
| `--lr-cooldown-start-pct` | 0.8 | fraction at which cosine cooldown begins |
| `--lr-final-pct` | 0.1 | final LR as a fraction of peak |

Example with a tighter warmup and a deeper cooldown:

```bash
python train.py --model models/Llama-3.1-8B --mode full \
  --seq-len 1024 --global-batch-tokens 524288 \
  --lr 5e-5 --lr-warmup-pct 0.03 --lr-final-pct 0.01
```

## Air-gapped compute nodes

If your training nodes have no internet, pre-stage the model + dataset
on a login node with `download.py`, then point `train.py` at the local
paths:

```bash
# Login node (has internet):
python download.py model meta-llama/Llama-3.1-8B --target models/Llama-3.1-8B
python download.py dataset HuggingFaceH4/no_robots --target datasets/no_robots.jsonl

# Compute node (no internet):
python train.py --model models/Llama-3.1-8B \
                --data-source json_sft \
                --dataset datasets/no_robots.jsonl \
                --seq-len 1024 --global-batch-tokens 1024
```

The dataset path normalizes common SFT schemas (`instruction/output`,
`prompt/completion`, chat-style `messages`, ...) into FlexTrain's JSONL
format. `download.py model --allow-patterns '*.safetensors' '*.json'`
skips redundant `pytorch_model.bin` shards.

## Python API

```python
from flextrain import from_pretrained
from flextrain.core.hw_probe import probe_hardware
from flextrain.optim.adamw import AdamW, AdamWHyperparams

probe = probe_hardware()  # ~14s; sustained TFLOPS / PCIe / mem-bw
am = from_pretrained(
    "models/Llama-3.1-8B",
    optimizer=AdamW(AdamWHyperparams(lr=3e-5)),
    max_seq_len=1024,
    max_global_batch_tokens=1024,
    max_gpu_mem_bytes=int(24 * (1 << 30)),
    max_host_mem_bytes=int(110 * (1 << 30)),
    hw_cost=probe.hw_cost,
    mem_bw_gbps=probe.mem_bw_gbps,
)

for batch in your_dataloader:
    am.fwd_bwd(batch)
    am.step()
```

For LoRA, pass `lora_targets="all"` (and optionally `lora_rank`,
`lora_alpha`).

## Layout

```
flextrain/
  core/      Layer/Block protocols, ActivationSchema, save-level DP solver,
             working-set sizer, hardware probe
  engine/    ActiveModel trainer, buffer manager, streams/events
  nn/        blocks/ (attention, FFN, MoE, RoPE, ...) + layers/ (full models)
  optim/     AdamW, Muon, HybridMuonAdamW
  ops/       FlexTrain-owned Triton kernels
  io/        HF weight load/save, per-arch adapters, download helpers
  bench/     parity tests + microbenchmarks
train.py     end-to-end CLI
download.py  pre-stage models/datasets for air-gapped nodes
```

## Documentation

| | |
|---|---|
| [architectures.md](docs/architectures.md) | supported HF configs |
| [working_set.md](docs/working_set.md) | how the planner picks chunk size, GPU layer counts, save tiers |
| [sft_vs_pretraining.md](docs/sft_vs_pretraining.md) | targets / loss-mask conventions |
| [dataset.md](docs/dataset.md) | data format + built-in JSON SFT source |
| [weights.md](docs/weights.md) | HF safetensors I/O, custom archs |
| [lora.md](docs/lora.md) | LoRA, MoE per-expert LoRA, HF PEFT parity |
| [implementing.md](docs/implementing.md) | adding a new block / layer / model |
| [dtypes.md](docs/dtypes.md) | compute / master / grad / opt-state dtypes |
| [optimizers.md](docs/optimizers.md) | AdamW / Muon / HybridMuonAdamW |

## Tests

Each test is a standalone script (no test runner yet):

```bash
python tests/test_save_level_parity.py        # bit-identical loss across save tiers
python tests/test_olmoe_1b7b_training.py      # OLMoE 1B-7B end-to-end on HF weights
python tests/test_random_init_pretraining.py  # cold-start on real Llama-3 tokens
python tests/test_muon_offloading_pretraining_moe.py  # Muon + offload parity
```
