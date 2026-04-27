# SFT vs pretraining

FlexTrain has no separate "SFT mode" or "pretraining mode" — both flow
through the same `_flextrain_step(am, seqs)` API. The only thing that
changes is **how you build `seq.targets`** and (optionally) what you
load as the starting weights.

This page documents both flows side-by-side.

## What FlexTrain sees

A training step is a list of `_Seq` objects, each holding:

```python
seq.tokens   # (T,) int64 — input token IDs
seq.targets  # (T,) int64 — for each prediction position, the target
             # token ID, or -100 to ignore that position in the loss
seq.seq_id   # int — opaque identifier (informational)
```

The engine packs these into chunks, runs forward + backward, and
returns the per-token-averaged cross-entropy loss across active
positions (`targets[i] != -100`).

That's the whole API. Pretraining vs SFT differs only in:

1. What's in `tokens` (raw text vs. prompt+response).
2. What's in `targets` (every-position vs. response-only).
3. Initial weights (random init vs. pretrained checkpoint).
4. Hyperparameters (lr, schedule).

## Pretraining (next-token prediction on raw text)

**Goal**: train a model from scratch (or continue pretraining a
checkpoint) on a generic text corpus. Loss is computed at every
position.

```python
from flextrain.bench.parity import _Seq

def make_pretraining_seq(token_stream: list[int]) -> _Seq:
    """Standard next-token prediction: target[i] = tokens[i+1] for all i.
    The last position has no next token; mark it -100."""
    tokens = torch.tensor(token_stream, dtype=torch.int64)
    targets = torch.roll(tokens, -1)
    targets[-1] = -100   # roll wraparound — exclude from loss
    seq = _Seq(tokens)
    seq.targets = targets
    return seq
```

Real-world pretraining sources:
- **Pre-tokenized binary shards** (FineWeb, RefinedWeb): pull raw token
  streams of fixed/variable length, no further processing.
- **Streaming tokenization**: read raw text → tokenizer → token stream.

For **continued pretraining** of a HF checkpoint, also call
`am.load_hf(...)` after building the engine. The optimizer starts
fresh at step 0 unless you also load opt state.

## SFT (supervised fine-tuning, instruction following)

**Goal**: teach a pretrained model to follow instructions. Loss is
masked over the prompt and computed only on the response.

```python
def make_sft_seq(prompt: str, response: str, tokenizer) -> _Seq:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    total_ids = prompt_ids + response_ids

    tokens = torch.tensor(total_ids, dtype=torch.int64)
    targets = torch.roll(tokens, -1)
    # Mask prompt positions: the prediction AT prompt position i is
    # not penalized for failing to predict prompt position i+1.
    targets[: len(prompt_ids)] = -100
    targets[-1] = -100   # roll wraparound
    seq = _Seq(tokens)
    seq.targets = targets
    return seq
```

The `targets[: len(prompt)] = -100` is the critical SFT bit. Without
it, the model learns to reproduce the prompt — which is bad (it's
already in the input) and dilutes the training signal on the actual
response.

`tests/test_llama32_1b_parity.py:_pull_step_batches` is a working
SFT-style data loader for the MathInstruct dataset (`Problem: ...
Solution: ...` format). It's used by every E2E test in the repo as
a real-data signal.

## Side-by-side

| | Pretraining | SFT |
|---|---|---|
| Tokens | Raw corpus | Prompt + response |
| Target masking | Only the last position is `-100` (roll wraparound) | Prompt positions + last all `-100`; only response in loss |
| Starting weights | Random init OR pretrained checkpoint | Pretrained checkpoint |
| LR | Higher (3e-4 to 1e-3 typical) | Lower (1e-5 to 1e-4) |
| LR schedule | Long warmup + cosine decay | Short warmup + linear decay |
| Steps | Millions | Thousands–hundreds of thousands |
| Sequence length | Long (2048–8192+) | Short (512–4096) typically |
| LoRA? | Rare; usually full param training | Common; LoRA fine-tune is the default for size/$ |

## LoRA SFT

The most common deployment combination. Standard recipe:

```python
# 1. Build base layer + wrap with LoRA.
backbone = [
    LoRAWrapperLayer(LlamaBlock(i, cfg), lora_targets="all",
                     rank=16, alpha=16.0, dims=dims)
    for i in range(n_layers)
]
am = ActiveModel(...)

# 2. Load pretrained HF base weights.
am.load_hf("models/Llama-3.2-1B", strict=False)
# ... (Q/K halved->pair perm, B-zero init, etc — see docs/lora.md)

# 3. Build SFT batches with prompt masking.
step_batches = []
for prompt, response in your_dataset:
    seq = make_sft_seq(prompt, response, tokenizer)
    # Pack into batch (multiple seqs per step typical) ...

# 4. Train.
for batch in step_batches:
    loss = _flextrain_step(am, batch)
```

See [docs/lora.md](lora.md) for a full runnable example with HF
weight loading + Q/K permutation + LoRA init.

## DPO / preference training

Not directly supported as a one-call API yet — it requires custom
loss (chosen-vs-rejected log-prob ratios) outside the standard
cross-entropy that `_flextrain_step` does. Roadmap item.

## Where the loss comes from

Inside `_flextrain_step`:

```
total_active = sum of (targets[i] != -100) across all seqs in the batch
per_token_loss = sum_over_active(cross_entropy(logits[i], targets[i])) / total_active
```

This is identical to HF transformers' default loss when you pass
`labels=targets` with `-100` on ignored positions, modulo the engine's
chunk-packing semantics (which match — the engine sums across chunks
and divides at the end).

So if you compare against HF transformers on the same masked targets,
loss values match within bf16 noise.

## Common questions

**Q: My pretraining loss curves are unstable / spiking.** Most likely
the warmup is too short or LR too high. The
`test_random_init_pretraining.py` recipe (3e-4, no warmup, 20 steps)
is for fast correctness checks, NOT a real training recipe. For real
pretraining: cosine schedule with 1-5% warmup and peak LR around 3e-4.

**Q: My SFT loss starts at 8+ instead of 1-2.** Either (a) you're
predicting the prompt (forgot the `-100` mask), or (b) you didn't load
pretrained weights and the model is at random init. SFT only makes
sense from a trained base.

**Q: Should I use LoRA or full fine-tuning for SFT?** LoRA is the
default for instruction-tuning pretrained models — much smaller
optimizer state, smaller checkpoints, easier to merge multiple
adapters. Use full fine-tuning when (a) the base model is small
enough that LoRA's parameter savings don't matter, or (b) you're
making large distribution shifts where rank 16 isn't expressive
enough.
