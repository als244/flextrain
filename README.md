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

## Quickstart: fine-tune from a HF checkpoint

The recommended entry point is `flextrain.from_pretrained` — point it
at any HF model directory and it builds the engine, picks a working-set
config that fits your hardware, loads weights, and applies any
arch-specific fixups. By default it sets up **full fine-tuning**;
add `lora_targets="all"` for LoRA.

### Full fine-tuning

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

for batch in your_dataloader:                  # see docs/dataset.md
    seqs = [_Seq(s.tokens) for s in batch]
    for d, s in zip(seqs, batch):
        d.targets = s.targets
    loss = _flextrain_step(am, seqs)
```

### LoRA fine-tuning

```python
opt = AdamW(
    AdamWHyperparams(lr=1e-4, beta1=0.9, beta2=0.95, weight_decay=0.0),
    state_dtype=torch.float32,
)
am = from_pretrained(
    "models/Llama-3.1-8B",
    optimizer=opt,
    max_seq_len=1024, max_global_batch_tokens=1024,
    max_gpu_mem_bytes=int(24 * (1 << 30)),
    max_host_mem_bytes=int(110 * (1 << 30)),
    device="cuda:0",
    lora_targets="all", lora_rank=16, lora_alpha=16.0,
)
# from_pretrained has already initialized LoRA (A ~ N(0, 0.02), B = 0)
# so the model starts at base behavior; just train.
for batch in your_dataloader:
    seqs = [_Seq(s.tokens) for s in batch]
    for d, s in zip(seqs, batch):
        d.targets = s.targets
    loss = _flextrain_step(am, seqs)
```

Under the hood `from_pretrained` reads `config.json`, looks up the
registered architecture, plumbs through everything that needs
arch-specific handling (RoPE scaling for Llama 3.1+ YARN, Q/K
halved→pair permutation, tied embeddings on small Llama-3.2
variants, …), and hands you a configured engine. To use a
pre-built `LlamaBlockConfig` directly instead, see
[docs/implementing.md](docs/implementing.md).

## Documentation

* [Architectures](docs/architectures.md) — what's supported, with HF
  config keys.
* [SFT vs pretraining](docs/sft_vs_pretraining.md) — how the two flows
  differ in `seq.targets` masking, weight init, hyperparams.
* [Dataset format](docs/dataset.md) — token / target / loss-mask
  conventions and how to plug in your own data.
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
