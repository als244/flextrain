# FlexTrain

Train transformer LLMs on hardware where the model doesn't fit in GPU
memory. FlexTrain rotates parameters, gradients, optimizer state, and
activations between GPU and host RAM via a working-set planner + DP
solver, so an 8B model trains end-to-end on a 24 GiB GPU without
DeepSpeed or FSDP.

**Scope.** Single-GPU only. No distributed / multi-GPU support — the
working-set solver is the value proposition, not data or tensor
parallelism. The CUDA path is the only supported execution backend.
Tests are standalone scripts (no `pytest` runner yet).

**Supported architectures.** Llama 2 / 3 / 3.1+, Mistral, Qwen2,
Qwen3 (dense + MoE), Qwen3.5 (dense + MoE) / Qwen3.6, Qwen3-Next,
OLMoE — see [`docs/architectures.md`](docs/architectures.md). Each
arch has forward + backward + HF safetensors load. End-to-end smoke
runs on a 24 GiB GPU + 117 GiB host workstation are recorded in
[`docs/verified_runs.md`](docs/verified_runs.md).

## Install

```bash
conda create -n flextrain python=3.12
conda activate flextrain
pip install torch triton
pip install -e .
```

The install also:

* Builds two in-tree C/CUDA helpers (`matmul_dispatcher`,
  `transmission_scheduler`). Set `FLEXTRAIN_SKIP_HELPERS=1` to skip
  when iterating on Python-only code.
* Fetches prebuilt flash-attention wheels matching your
  (python, torch, CUDA) tuple from
  [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels).
  flash-attn 2 always; flash-attn 3 added on Hopper (sm 90+). Set
  `FLEXTRAIN_SKIP_FLASH_ATTN=1` to skip. Wheel-install failures are
  non-fatal — the rest of the install proceeds.
* Pulls in `flash-linear-attention` so Qwen3-Next / Qwen3.5 hybrid
  layers (Gated DeltaNet) work out of the box.

Optional extras: `-e ".[hopper]"` (`tilelang` for FLA on Hopper +
Triton ≥ 3.4 — works around a correctness bug in the default Triton
chunk_bwd kernel), `-e ".[causal-conv1d]"` (Mamba-style state-space
support — not used by current archs), `-e ".[peft]"` (HF PEFT for
LoRA parity tests), `-e ".[flash-attn-build]"` (build flash-attn
from source instead of the prebuilt wheel — slow; minutes-to-hours).

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
from flextrain.optim.adamw import AdamW, AdamWHyperparams

am = from_pretrained(
    "models/Llama-3.1-8B",
    optimizer=AdamW(AdamWHyperparams(lr=3e-5)),
    max_seq_len=1024,
    max_global_batch_tokens=1024,
    max_gpu_mem_bytes=24 << 30,
    max_host_mem_bytes=110 << 30,
)

for batch in your_dataloader:
    am.fwd_bwd(batch)
    am.step()
```

For LoRA, pass `lora_targets="all"` (and optionally `lora_rank`,
`lora_alpha`). For full fine-tuning with Muon on dense projections,
swap the optimizer for `flextrain.optim.HybridMuonAdamW`.

**Hardware probe.** `from_pretrained` runs `probe_hardware()` on the
first call (~10s) when `hw_cost` isn't provided, so the save-level DP
solver gets accurate TFLOPS / PCIe numbers. To avoid re-probing
across calls, cache the result and pass it explicitly:

```python
from flextrain.core.hw_probe import probe_hardware
probe = probe_hardware()  # ~10s; sustained TFLOPS / PCIe / mem-bw
am = from_pretrained(..., hw_cost=probe.hw_cost, mem_bw_gbps=probe.mem_bw_gbps)
```

`train.py` does this once at startup and reuses the result.

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
| [extending/](docs/extending/README.md) | adding a new model: tutorial + flow + per-level contracts (block / layer / chunk / model) |
| [dtypes.md](docs/dtypes.md) | compute / master / grad / opt-state dtypes |
| [optimizers.md](docs/optimizers.md) | AdamW / Muon / HybridMuonAdamW |
| [export.md](docs/export.md) | export to vLLM / sGLang / HF (full, LoRA adapter, merged) |
| [verified_runs.md](docs/verified_runs.md) | end-to-end smoke runs on the reference workstation |

## Tests

Each test is a standalone script (no test runner yet):

```bash
python tests/test_arch_parity.py              # FT-vs-HF loss + logit parity across archs
python tests/test_arch_lora_e2e.py --all      # LoRA-mode parity vs HF PEFT (all archs)
python tests/test_llama32_1b_parity.py        # Llama-3.2-1B end-to-end vs HF transformers
python tests/parity_qwen3_5_9b_35b_5step.py   # Qwen3.5-9B + 35B-A3B 5-step replay
```

Per-area suites live in subdirs: `tests/moe/` (MoE block / backend parity),
`tests/multi_chunk_dense_parity/` (chunked linear-attn correctness),
`tests/io/` (HF safetensors round-trip), `tests/cross_machine_parity/`
(replay across machines).
