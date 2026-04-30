# FlexTrain

Train transformer LLMs on hardware where the model doesn't fit in GPU
memory. FlexTrain rotates parameters, gradients, optimizer state, and
activations between GPU and host RAM via a working-set planner + DP
solver, so an 8B model trains end-to-end on a 24 GiB GPU without
DeepSpeed or FSDP.

Supported architectures: Llama-3, Qwen2/3, Qwen3-MoE, Qwen3.5,
Qwen3.5-MoE / Qwen3.6-MoE, OLMoE, Qwen3-Next, Gemma 2/3. See
[`docs/architectures.md`](docs/architectures.md).

### Verified end-to-end (RTX 3090 24 GiB GPU + 117 GiB host)

5-step training smoke on `datasets/mathinstruct.jsonl` with the
default `Instruction:/Response:` prompt template, `--max-seq-len 2048`,
mean-over-active-tokens loss (PyTorch `CrossEntropyLoss(ignore_index=-100)`
convention; matches HF / PEFT). LoRA-all at rank=16 unless noted.
Greedy generation also verified (coherent output, hits EOS naturally).

| Model | Params | Arch | Mode | Batch tokens | Loss curve (5 steps) |
|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | dense | LoRA | — | _not re-verified_ |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | — | _not re-verified_ |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | — | _not re-verified_ |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | — | _not re-verified_ |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 65k | 0.797 → 0.620 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | 65k | 0.744 → 0.455 |
| Qwen3.5-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | — | _not re-verified_ |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | — | _not re-verified_ |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | — | _not re-verified_ |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 65k | 0.743 → 0.541 |

Loss values reflect mean cross-entropy over response tokens
(positions where `targets != -100`); prior versions of this table
reported a different convention (mean over all tokens, including
prompt-position zeros) so older numbers are not directly comparable.

Additional models supported by the existing arch loaders (require a
larger machine to actually train): Qwen3.6-35B-A3B,
Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, Qwen3-Coder-30B-A3B-Instruct
(no new wiring needed; they reuse `Qwen3_5*` / `Qwen3Moe*` arch ids).

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
  --max-seq-len 1024 \
  --max-global-batch-tokens 1024
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
| `--truncate-long` | truncate response of records longer than `--max-seq-len` instead of dropping them (default: drop) |
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
  --max-seq-len 1024 --max-global-batch-tokens 524288 \
  --lr 5e-5 --lr-warmup-pct 0.03 --lr-final-pct 0.01
```

### Profiling with nsys

Wrap a window of steady-state steps for `nsys profile`:

```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  python train.py --model models/Llama-3.1-8B --mode full \
                  --max-seq-len 1024 --max-global-batch-tokens 524288 \
                  --steps 10 --profile-start-step 5 --profile-stop-step 7
```

`--profile-start-step` calls `cudaProfilerStart()` right before that
step begins; `--profile-stop-step` calls `cudaProfilerStop()` after it
ends (default = start + 2). nsys' capture range opens/closes on those
markers, so warmup and final-step teardown stay out of the report.
Each captured step is wrapped in an NVTX range so the timeline groups
by step.

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
                --max-seq-len 1024 --max-global-batch-tokens 1024
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
| [export.md](docs/export.md) | export to vLLM / sGLang / HF (full, LoRA adapter, merged) |

## Tests

Each test is a standalone script (no test runner yet):

```bash
python tests/test_save_level_parity.py        # bit-identical loss across save tiers
python tests/test_olmoe_1b7b_training.py      # OLMoE 1B-7B end-to-end on HF weights
python tests/test_random_init_pretraining.py  # cold-start on real Llama-3 tokens
python tests/test_muon_offloading_pretraining_moe.py  # Muon + offload parity
```
