# FlexTrain

A flexible-precision, working-set-aware training engine for transformer
language models. Built around AdaWS-style activation/parameter rotation
between GPU and host memory, with hand-written Triton kernels for the
hot paths and an explicit per-block "activation slot" abstraction that
makes recomputation, offloading, and mixed precision composable.

FlexTrain trains pretrained HF models (Llama-3, Qwen2/3, Qwen3-MoE,
OLMoE, Qwen3-Next, Gemma 2/3 — see [supported architectures](docs/architectures.md))
on hardware whose GPU memory is much smaller than the model — without
reaching for FSDP or DeepSpeed.

## Status

| Capability | Status |
|------------|--------|
| Llama-3.1-8B end-to-end on a 24 GiB GPU | ✓ |
| OLMoE-1B-7B end-to-end (HF weights, real data) | ✓ |
| Qwen3-MoE small-init parity vs naive PyTorch | ✓ |
| Qwen3-Next (alternating linear+full attention) | fwd+bwd blocks landed; full-arch end-to-end pending |
| Hybrid Muon + AdamW optimizer | ✓ (per-tensor classification) |
| MoE expert offloading | ✓ |
| LoRA fine-tuning (dense + per-expert MoE) | ✓ (Llama-3.1-8B E2E vs HF PEFT, OLMoE-1B-7B E2E) |
| Gemma 2 / 3 | pending |
| DDP / ZeRO-1 | pending |

See [docs/SESSION_NOTES.md](docs/SESSION_NOTES.md) for a running log of
decisions, findings, and known gaps.

## Installation

FlexTrain needs a CUDA-enabled PyTorch and a matching Triton. Use any
PyTorch build that targets your driver — there's no hard pin on a CUDA
version. Two C helpers ship in-tree under `helpers/` and are built
automatically by `pip install`:

* `matmul_dispatcher` — cuBLASLt dispatcher (CMake + CUDA, needs a C++17
  compiler and the CUDA toolkit's `cublasLt` / `cudart` available to
  CMake's `find_package(CUDAToolkit)`).
* `transmission_scheduler` — DP solver used by the working-set planner
  (a plain C extension).

```bash
# 1. Set up a conda env with PyTorch + Triton.
conda create -n flextrain python=3.12
conda activate flextrain
pip install torch triton

# 2. Install FlexTrain (also builds the two helpers under helpers/).
pip install -e .

# 3. (Optional) For Qwen3-Next / Qwen3.5 / Qwen3.6 hybrid linear-attention layers.
pip install -e ".[linear-attention]"
```

For Hopper / Blackwell GPUs you'll also want Flash Attention 3
(`pip install -e ".[flash-attn]"`) — the engine prefers it when
available and silently falls back otherwise.

If you need to skip the helper builds (e.g. iterating on Python-only
changes), set `FLEXTRAIN_SKIP_HELPERS=1` before running `pip install`.

## Quickstart

The shortest first run is the top-level `train.py` entrypoint, where the
user picks:

```bash
python train.py \
  --model models/Llama-3.1-8B \
  --mode lora \
  --seq-len 1024 \
  --global-batch-tokens 1024
```

That gives you:

* a real HF checkpoint
* one explicit choice between `full` and `lora`
* a real SFT run on the bundled sample dataset
* per-step logging for loss, tok/s, and memory
* no final checkpoint export unless you add `--save`
* explicit setup logs before the first training step

By default this command:

* uses `flextrain/configs/examples/data/tiny_math_sft.json`
* downloads the model into `models/` if needed
* runs 20 steps
* writes logs under `runs/<model>_<mode>_sl<seq_len>`

To train on synthetic tokens instead, switch the data source:

```bash
python train.py \
  --model models/Llama-3.1-8B \
  --mode lora \
  --seq-len 1024 \
  --global-batch-tokens 1024 \
  --data-source synthetic
```

`--synthetic-seq-len` is optional; if omitted it defaults to `--seq-len`.

To train on your own dataset, keep using `--dataset`. If the file exists
locally, FlexTrain uses it directly. If it does not exist, FlexTrain
first tries to download/materialize it and then trains from the local
JSONL it creates:

```bash
python train.py \
  --model models/Llama-3.1-8B \
  --mode lora \
  --seq-len 1024 \
  --global-batch-tokens 1024 \
  --dataset open-r1/OpenR1-Math-220k
```

That download path is meant for SFT-style datasets. It normalizes common
schemas like `instruction/output`, `prompt/completion`,
`question/answer`, and chat-style `messages` into a local JSONL before
training.

Examples:

```bash
# Full fine-tuning
python train.py \
  --model models/Llama-3.1-8B \
  --mode full \
  --seq-len 1024 \
  --global-batch-tokens 1024

# LoRA fine-tuning
python train.py \
  --model models/Llama-3.1-8B \
  --mode lora \
  --seq-len 2048 \
  --global-batch-tokens 2048
```

You should see setup messages like:

* `Preparing model from ...`
* `Model is ready. Building tokenizer-backed SFT data source...`
* `Starting training loop: ...`

After that, the first training step can still take a while on large
models before the first `[step ...]` line appears.

### YAML Path

If you want the fully explicit YAML-driven path, use:

```bash
python -m flextrain train flextrain/configs/examples/llama3_8b_math_sft.yaml
```

### Fine-tune a local HF checkpoint

If you already have a local HF model directory, the shortest path is:

```bash
python train.py \
  --model /path/to/your/model \
  --mode lora \
  --seq-len 1024 \
  --global-batch-tokens 1024
```

You can still switch to a custom dataset with `--dataset path/to/data.json`.
If that path does not exist, FlexTrain treats `--dataset` as a dataset
spec and materializes it first.

### Python API

For programmatic use, the recommended entry point is
`flextrain.from_pretrained`. It reads a local HF model directory,
builds the engine, picks a working-set config that fits your hardware,
loads weights, and applies any arch-specific fixups.

```python
import torch
from flextrain import from_pretrained
from flextrain.bench.parity import _Seq, _flextrain_step
from flextrain.optim.adamw import AdamW, AdamWHyperparams

opt = AdamW(AdamWHyperparams(lr=3e-5))
am = from_pretrained(
    "models/Llama-3.1-8B",
    optimizer=opt,
    max_seq_len=1024,
    max_global_batch_tokens=1024,
    max_gpu_mem_bytes=int(24 * (1 << 30)),
    max_host_mem_bytes=int(110 * (1 << 30)),
    device="cuda:0",
)

for batch in your_dataloader:
    seqs = [_Seq(s.tokens) for s in batch]
    for d, s in zip(seqs, batch):
        d.targets = s.targets
    loss = _flextrain_step(am, seqs)
```

For LoRA, add `lora_targets="all"` (and optionally `lora_rank`,
`lora_alpha`). `from_pretrained` initializes LoRA so the model starts
at base behavior before the first update.

## Documentation

* [Architectures](docs/architectures.md) — what's supported, with HF
  config keys.
* [SFT vs pretraining](docs/sft_vs_pretraining.md) — how the two flows
  differ in `seq.targets` masking, weight init, hyperparams.
* [Dataset format](docs/dataset.md) — token / target / loss-mask
  conventions, plus the built-in local JSON SFT source.
* [Weight I/O](docs/weights.md) — HF safetensors load / save, custom
  arch specs, expert-stacking hooks, post-load weight permutations.
* [LoRA fine-tuning](docs/lora.md) — `LoRAWrapperLayer` API, MoE
  per-expert LoRA, runnable Llama-3.2-1B example, HF PEFT parity.
* [Implementing a new block / layer / model](docs/implementing.md) —
  the protocol contracts (ParamSpec, ActivationSchema, Layer protocol),
  save levels, save-tier conventions, when to use `slot.aux`,
  `chunk.extra`, etc.
* [Datatypes](docs/dtypes.md) — compute / master / grad / opt-state
  dtypes, where each is honored, and recommended defaults per role.
* [Optimizers](docs/optimizers.md) — AdamW, Muon, HybridMuonAdamW.
* [Working set tuning](docs/working_set.md) — `n_gpu_layers`,
  `n_gpu_grads`, `n_gpu_opt_layers`, save levels, target tokens.

## Layout

```
flextrain/
  core/         # Layer/Block protocols, ActivationSchema, save-level solver
  engine/       # ActiveModel (the trainer), buffer manager, host backends
  nn/
    blocks/     # composable units (attention, FFN, MoE FFN, RMSNorm, RoPE,
                # gated-DeltaNet, etc.)
    layers/     # full transformer layers (Llama, Qwen3, Qwen3-MoE,
                # OLMoE, Qwen3-Next, ...)
  optim/        # AdamW, Muon, HybridMuonAdamW
  ops/          # FlexTrain-owned Triton kernels
    _kernels/   # private impl modules
  io/
    arch/       # per-architecture HF weight maps + config adapters
    hf_weights.py
  bench/        # parity testing + microbenchmarks
  cli.py        # CLI wrapper for HF training (alpha)
tests/
docs/
orig/           # reference port from the AdaWS prototype (not on PYTHONPATH
                # for new code; only some test references reach into orig
                # for cross-checks against the original Python layers)
```

## Tests

The most rigorous correctness gates:

* `tests/test_save_level_parity.py` — bit-identical loss curves across
  save levels. Catches silent activation-recompute bugs.
* `tests/test_olmoe_1b7b_training.py` — OLMoE-1B-7B end-to-end on
  real HF weights + real data; FT step-0 matches HF transformers.
* `tests/test_random_init_pretraining.py` — cold-start regime on real
  Llama-3 tokens.
* `tests/test_muon_offloading_pretraining*.py` — Muon + offloading
  parity for both dense and MoE.
* `tests/test_olmoe_engine_parity.py`,
  `tests/test_qwen3_moe_engine_parity.py` — small-init engine parity
  for MoE archs vs naive PyTorch.
* `tests/test_gated_deltanet_*.py` — Qwen3-Next linear-attention block
  parity (fwd + bwd).

Run individually with `python tests/<name>.py`. There is no test runner
config yet — each test is a standalone script.
