# Loss-curve parity harness

Correctness check for the FlexTrain training engine: compare per-step
loss against a pure-PyTorch reference on the same real-data stream,
under multiple memory / scheduling configurations.

If the engine is correct, every working-set config produces the same
loss trajectory as naive PyTorch modulo bf16 noise. If any config
diverges, something in the scheduling (activation offload, weight
ring, grad ring, opt-state ring, KV refresh, ...) is broken.

## Quick start

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flextrain
cd ~/Documents/FlexTrain
PYTHONPATH=. python tests/test_loss_curve_parity.py
```

Runs ~10 minutes end-to-end on a 3090. Output looks like:

```
====================================================
#  SETTING: S1: baseline (100 steps, lr=5e-4, d=512, L=6)
====================================================
  Data: 100 steps, ~384 tokens/step (actual total=49687), seq lens in [64, 256]
  Running naive PyTorch baseline...
  === config: A. fast path (all on-device, 1 chunk/round) ===
    -> ft loss: avg(first 3)=10.7325, avg(last 3)=8.2040
  ...
========================================================================
  LOSS CURVE PARITY SUMMARY
========================================================================
  config                                             final loss   Δ vs naive
  ------------------------------------------------------------------------
  naive PyTorch baseline                                 8.2031
  A. fast path (all on-device, 1 chunk/round)            8.2040      +0.0009
  B. multi-chunk (many chunks/round, on-device)          8.1770      -0.0261
  ...
```

## What it's testing

Eight working-set configs, each stressing a different engine path:

| Config | What it stresses |
|-|-|
| **A** Fast path | Every (layer, chunk) pair on-device. Baseline. |
| **B** Multi-chunk | Many chunks per round, still on-device. |
| **C** Multi-round | Multiple gradient-accumulation rounds per step. |
| **D** Host offload | Tight activation ring forces host offload during forward. |
| **E** Weight ring rotation | `N_P < n_layers` — weight prefetch in/out of ring. |
| **F** Grad ring rotation | `N_G < n_layers` — grad offload + re-fetch. |
| **G** Opt-state ring rotation | `N_opt < n_layers` — opt-state prefetch during `step()`. |
| **H** Sequence spans chunks | Single sequence split across multiple chunks (tests KV refresh during backward). |

At three settings (S1: baseline, S2: bigger model, S3: 2× steps + 2× LR).

**Every FT config must match the naive PyTorch reference within a
windowed-mean tolerance (10-step running avg).** If any config
drifts, it's a real bug.

## Reusable API

The harness lives in [`flextrain/bench/parity.py`](parity.py). The
public surface is four classes:

```python
from flextrain.bench import (
    ModelShape, WorkingSetSpec, LossCurveParityConfig,
    run_loss_curve_parity,
)

# 1. Model shape (defaults are a small 6-layer Llama with d_model=512).
shape = ModelShape(
    d_model=512, n_layers=6, n_heads=8, n_kv_heads=2,
    head_dim=64, expert_dim=1024, vocab_size=50432,
)

# 2. Working-set specs to compare.
working_sets = [
    WorkingSetSpec(
        label="A. fast path",
        n_gpu_layers=shape.n_layers, n_gpu_grads=shape.n_layers,
        n_gpu_opt_layers=shape.n_layers,
        gpu_act_buffer_size=256 * 1024 * 1024,
        host_act_buffer_size=0,
        max_chunk_size=512,
        target_round_tokens=512,
        max_total_round_tokens=1024,
        max_training_chunks=4,
    ),
    # ... more specs
]

# 3. Overall run config.
cfg = LossCurveParityConfig(
    shape=shape,
    working_sets=working_sets,
    n_steps=100,                # optimizer steps
    target_tokens_per_step=384, # roughly this many tokens per step
    min_seq_len=64, max_seq_len=256,
    lr=5e-4,
    init_seed=4242,
    shard_path="path/to/fineweb_train_000001.bin",
    device="cuda:0",
)

# 4. Run it.
result = run_loss_curve_parity(cfg)

# 5. Inspect + validate.
result.print_summary()
result.assert_all_match(window=10, windowed_atol=0.10)
```

`result` is a `LossCurveParityResult` with two useful attributes:

- `result.naive_curve` — list[float], per-step naive loss
- `result.ft_curves` — dict[label, list[float]] per FT config

Plus:

- `result.print_summary()` — prints a final-loss table.
- `result.assert_all_match(window=10, windowed_atol=0.10)` — raises
  AssertionError if any windowed trajectory diverges from naive OR
  any other FT config by more than `windowed_atol`.

## Customizing

### Use a different data source

`shard_path` must point to a FineWeb-format `.bin` file (GPT-2
tokenizer, uint16 tokens, 256-int32 header, EOT=50256). See
[orig/fineweb.py](../../orig/fineweb.py) to generate one. If you want
a totally different source, swap in your own stream by monkeypatching
`flextrain.bench.parity._generate_sequence_stream` — it must return
`list[list[_Seq]]` where `_Seq.tokens` is a CPU int64 tensor.

### Bigger / smaller model

Edit `ModelShape`. A 768-d 8-layer stress config:

```python
shape = ModelShape(
    d_model=768, n_heads=12, head_dim=64, n_layers=8,
    expert_dim=1536, vocab_size=50432,
)
```

Remember: the `gpu_act_buffer_size` in every `WorkingSetSpec` must be
≥ the max per-layer AdamW opt-state bytes (the activation buffer is
repurposed as the opt ring during `step()`). For AdamW
(2 × fp32 = 8 bytes/param), that's roughly
`8 × per_layer_params_bytes`. The engine raises at init time if this
is violated, so it's safe to just increase the budget until
construction succeeds.

### Different working-set sweeps

Build a list of `WorkingSetSpec` with the knobs you want to vary
(`n_gpu_layers`, `n_gpu_grads`, `n_gpu_opt_layers`,
`gpu_act_buffer_size`, `host_act_buffer_size`, `max_chunk_size`,
`target_round_tokens`, `max_total_round_tokens`,
`max_training_chunks`) and pass them to `LossCurveParityConfig`.

### Different step counts / LR

`cfg.n_steps` and `cfg.lr`. The windowed tolerance
(`assert_all_match(windowed_atol=...)`) should scale roughly with
`n_steps × lr` — bf16 noise compounds linearly per step.

## Interpreting results

**All configs within `atol` of naive:** engine is correct under the
configs tested. Move forward.

**One config differs but others match naive:** that config's code
path has a bug. Read which config, then trace:

- A (all on-device): forward/backward compute, head, embed.
- B: multi-chunk loop structure (chunk rotation within a single
  round).
- C: gradient accumulation across rounds (`_zero_grad` flag
  ordering, grad offload timing).
- D: host offload path (`send_home` / `fetch_home` and the activation
  ring rotation during backward).
- E: weight ring rotation during forward + backward.
- F: grad ring offload/prefetch during backward.
- G: opt-state ring prefetch during `step()`.
- H: KV context refresh during backward across sequence groups.

**All configs diverge from naive but agree with each other:** something
is wrong with the shared forward/backward compute (not scheduling).
Check kernels or loss functions.

**Two specific configs diverge with each other but each matches naive:**
extremely unlikely in practice; would indicate concurrency-specific
state depending on exact config.

## Why pure-PyTorch reference

The naive reference is an independent implementation written from the
math, not a port of FlexTrain or orig kernels. This means:

- Forward paths use stock `torch` ops (matmul, softmax, RMSNorm via
  `rsqrt(pow(2).mean(-1))`, SwiGLU via `F.silu() * up`).
- RoPE uses the **pair-interleave** convention (matches Triton kernel;
  NOT HuggingFace's halved-split convention — see
  [`docs/internal/NOTES.md`](../../docs/internal/NOTES.md) [FINDING 6]).
- Attention is causal-masked SDPA in fp32.
- AdamW is `torch.optim.AdamW` (canonical PyTorch implementation).

If naive and FlexTrain agree on a real dataset's loss trajectory,
both are computing the same thing. Orig's kernels or scheduling
could have a subtle bug that FlexTrain inherited; the naive reference
catches that — this is how we found the RoPE convention hazard and
the stream-context leakage (see `docs/internal/NOTES.md` findings list).

## Known bf16 drift characteristics

At `lr=5e-4`, 100 steps, d=512, L=6:
- Fast-path vs naive: windowed max |Δ| ≤ 0.03
- Cross-config spread: ≤ 0.06

At `lr=1e-3`, 200 steps, d=512, L=6 (stress):
- Fast-path vs naive: windowed max |Δ| ≤ 0.07
- Cross-config spread: ≤ 0.15, dominated by the sequence-spans-chunks
  config (flash-attn processes split sequences through a slightly
  different numerical path than whole sequences).

If your run shows substantially larger deltas, investigate.
