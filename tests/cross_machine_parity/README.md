# Cross-machine logit parity (HF capture → FT replay)

Two-script harness for verifying FlexTrain's forward pass matches the
HuggingFace reference implementation, **across machines**.

Use case: a model that fits comfortably under `transformers` on one
box (e.g. an H100 with 80 GB) doesn't have to fit a side-by-side
`AutoModelForCausalLM` and a live FT engine on the same other box
(e.g. an RTX 3090 with 24 GB). Capture HF's logits on the big box;
replay against FT on whichever machine you actually run training.

The capture covers both:

* the **prompt** (chat-templated, so it matches the actual
  training/inference path), and
* the full **greedy continuation** (autoregressive decode to EOS or
  `--max-new-tokens`).

So 100% argmax-agreement across the captured sequence is a strong
end-to-end statement: every prompt-position prediction matches HF,
*and* every step of a long autoregressive decode produces the same
token HF would produce.

---

## What the bundle file contains

The capture script saves a single `.pt` file with these keys:

| key                       | dtype          | shape         | description                                                  |
|---------------------------|----------------|---------------|--------------------------------------------------------------|
| `input_ids`               | int64          | `(T,)`        | prompt + greedy-generated tokens, full sequence              |
| `prompt_T`                | int            | scalar        | prompt length; generated = `input_ids[prompt_T:]`            |
| `output_ids`              | int64          | `(n_gen,)`    | just the generated tokens (convenience)                      |
| `prompt_decoded`          | str            | —             | what HF saw after `apply_chat_template`                      |
| `output_decoded`          | str            | —             | the greedy continuation, decoded                             |
| `full_decoded`            | str            | —             | prompt + generated, decoded                                  |
| `logits`                  | bf16           | `(T, V)`      | per-position next-token logits from HF eager fwd             |
| `last_argmax_per_step`    | int64          | `(n_gen,)`    | argmax produced at each generation step (sanity check)       |
| `vocab_size`, `model`, `model_path_arg`, `prompt`, `dtype_used_for_fwd`, `n_generated`, `no_chat_template` | — | — | metadata |

`logits[i]` is the model's next-token distribution given context
`input_ids[:i+1]`. Positions `[0, prompt_T-1]` come from one big
forward over the full sequence; positions `[prompt_T, T-1]` are
recorded at each greedy-decode step (and verified to match the same
positions of the final full-sequence forward via an internal sanity
check inside the capture script).

File size scales as `T * vocab_size * 2 bytes`. For Qwen3.5-27B
(`vocab_size=248320`) at `T=138` (a small-prompt + ~120-token
decode), the file is ~65 MiB. Easy to ship.

---

## Step 1 — capture on the HF-friendly machine

```bash
PYTHONPATH=. python tests/cross_machine_parity/hf_capture_logits.py \
    --model models/Qwen3.5-27B \
    --prompt "What is the capital of France?" \
    --max-new-tokens 256 \
    --out hf_capture_27b_france.pt
```

Useful flags:

* `--max-new-tokens N` — caps greedy decode length. Stops early on EOS.
* `--no-chat-template` — bypass chat template; tokenize prompt as raw
  text. Use this if you want the comparison to mirror what training
  saw (continuous-pretraining-style targets) rather than chat-format
  inference.
* `--max-prompt-tokens N` — truncate the chat-templated prompt before
  generation. Useful for stress-testing long prompts.
* `--dtype {bfloat16,float16,float32}` — defaults to bf16 to match
  FT's training default.

The script prints the decoded prompt + every Nth decoded token + the
full decoded output before saving.

It also runs an internal sanity check: the next-token logits captured
during the autoregressive loop are compared against the corresponding
positions of a final full-sequence forward — they should match (HF
eager fwd is deterministic with `use_cache=False`). The script prints
the max|Δ| of that internal cross-check.

---

## Step 2 — replay + compare on the other machine

After moving the bundle file across machines:

```bash
PYTHONPATH=. python tests/cross_machine_parity/ft_replay_compare_logits.py \
    --capture hf_capture_27b_france.pt \
    --model models/Qwen3.5-27B \
    --gpu-gib 22.5 --host-gib 110
```

The replay script:

1. Builds the FT engine in **LoRA mode** by default (frozen base,
   small adapter rank 8, alpha 8). Frozen base + zero-init adapters
   means the forward computes the **un-adapted pretrained model** —
   which is exactly what the HF capture used. LoRA mode is preferred
   here because it has a much smaller GPU baseline footprint, so
   bigger models fit alongside the (T, V) logit comparison tensors.
   Pass `--no-lora` to use full-FT mode instead (heavier baseline).
2. Runs **one** FT forward over the full saved `input_ids`. No
   autoregression on the FT side — all the autoregressive work is
   already baked into HF's saved logits.
3. Compares HF and FT logits position-by-position. All comparison
   math runs on GPU (the FT logits tensor has stream-ownership quirks
   that make `.to('cpu')` raise `cudaErrorInvalidValue` on some
   setups; doing the math on GPU sidesteps that).

### What the replay prints

* **Diff (full / prompt / generated)** — `max|Δ|`, `mean|Δ|`, `rel-error`
  for the three slices. `rel-error = max|Δ| / ‖HF logits‖`.
  Acceptance: rel < 5e-3 (bf16 noise floor for an entire model
  forward).
* **Per-position max|Δ| across generated region** — bucketed into 4
  bins so you can see whether drift is *growing* across autoregressive
  depth (it shouldn't, if there's no compounding bug). Each bin
  reports max + median |Δ|.
* **Argmax agreement** — % of positions where HF's argmax equals FT's,
  for full / prompt / generated. **The strongest single signal**:
  this is the token a generator would emit. 100% means HF and FT
  produce identical greedy decodes.
* **First generated-region argmax mismatch** — if any. Tells you
  exactly where a divergence first appeared.
* **Next-token CE loss [full/prompt/generated]** — HF vs FT. This is
  what you'd see in training. If FT is correct, both should be ~equal.

### Reading the output

For Qwen3.5-27B at the time of writing (post-`18b7e09`):

```
=== Diff (full sequence) ===
  logits[full]      max|Δ|=1.4062  mean|Δ|=0.0275  rel=6.310e-05  [OK]

  argmax agreement [full]:      100.00% (138/138)
  argmax agreement [prompt]:    100.00% (17/17)
  argmax agreement [generated]: 100.00% (121/121)

  next-token CE [full]:      HF=0.1310  FT=0.1308
```

`max|Δ|=1.4` is the largest single-element absolute error across the
~34M-element `(T, V)` grid — bf16 noise on the long tail of the
distribution, not on the argmax. `mean|Δ|=0.027` is the typical
element-level error. `rel=6e-5` is the same number normalized by the
HF logit-tensor norm (~22000 for `T=138, V=248320`). Both stacks
agree on every token.

If you see `argmax agreement [generated] < 100%`, look at "first
generated-region argmax mismatch" — that's where the divergence
appears, and the four buckets of per-position drift will tell you
whether it's a one-off bf16 stumble or progressive compounding.

---

## Common gotchas

* **`apply_chat_template` returns a `BatchEncoding`** (not a plain
  dict) on some transformers versions, which iterates as
  `['input_ids', 'attention_mask']` rather than the token ids
  themselves. Capture script handles this; if you copy it as a
  template for other tests, mirror that normalization.
* **mRoPE on Qwen3.5**: the model is multimodal; HF expands 2D
  position_ids to 3-axis mRoPE internally. For text-only inputs the
  3 axes carry the same positions, so mRoPE collapses to standard
  partial-RoPE — which is what FT applies. No special handling
  needed at the capture/replay layer.
* **Tied embeddings**: Qwen3.5 has `tie_word_embeddings: false`, so
  `lm_head.weight` is in safetensors. For a future model with tied
  embeddings, the capture script would still work (HF resolves the
  tie internally); the FT side already mirrors `embed.t()` into the
  head buffer if needed.

---

## What this is *not*

* **Not a backward-pass test.** It only verifies forward parity. If
  you want to test gradients vs HF, see the per-arch e2e LoRA tests
  (`tests/test_phase2_lora_e2e_*` and `tests/test_arch_lora_e2e.py`).
* **Not a replacement for `tests/qwen3_5_per_layer_diff.py`.** That
  test localizes the *layer* where divergence appears — useful when
  you've changed FT layer internals and want to bisect. This harness
  is end-to-end: prompt → logits.
* **Not for stochastic decoding.** Greedy-only by design. Sampling
  would diverge HF and FT immediately on different RNG streams.
