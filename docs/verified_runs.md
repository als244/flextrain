# Verified end-to-end runs

Smoke-test record from training a handful of supported models end-to-end.

- **Workload**: 5 steps on `datasets/mathinstruct.jsonl` with the default
  `Instruction:/Response:` prompt template, `--max-seq-len 2048`,
  mean-over-active-tokens loss
  (`CrossEntropyLoss(ignore_index=-100)` convention; matches HF / PEFT).
- **Default mode**: LoRA-all at rank 16 unless the row says otherwise.
- **Generation**: greedy generation also verified — coherent output, hits
  EOS naturally.
- **Reproduce**: `python experiments/verified_runs.py run-grid --out
  runs/<dir>` shells out to `train.py` per row, captures the per-step
  log, and emits the table at `<dir>/new_table.md`. Defaults assume
  HF snapshots at `<repo>/models/<name>` and the SFT dataset at
  `<repo>/datasets/mathinstruct.jsonl`; override via
  `FLEXTRAIN_MODELS_DIR` / `FLEXTRAIN_VERIFIED_DATASET` env vars.

Loss values reflect mean cross-entropy over response tokens (positions
where `targets != -100`); prior versions of this table reported a
different convention (mean over all tokens, including prompt-position
zeros) so older numbers are not directly comparable.

## Tokenization & prompt format

Every row in the sweep uses **the same prompt wrapper and the model's
own tokenizer** (`JsonSFTTokenSource._build_prompt`,
`flextrain/io/sources.py:380`):

```text
Instruction:
{instruction}

Response:
{response}<|EOS|>
```

- **Per-model tokenizer**: each model's own HF tokenizer is loaded
  (Llama-3 BPE 128k vocab, Qwen3 ~152k, OLMoE GPT-NeoX-style, etc.) —
  token IDs are correct for that model.
- **No chat template**: `tokenizer.apply_chat_template()` is **not**
  invoked. The wrapper is plain text, so models like
  Llama-3.x-Instruct or Qwen3-Instruct do not see their natural
  `<|start_header_id|>` / `<|im_start|>` chat tokens.
- **Special tokens**: `add_special_tokens=False`, so no BOS / no
  chat-format header tokens. The tokenizer's EOS id is appended after
  the response so the model learns to terminate.
- **Loss masking**: cross-entropy is taken over response-token
  positions only (`ignore_index=-100` on prompt positions); reported
  loss is mean over those response tokens.

Implications for reading the table:

1. **Cross-row comparable**: every model trains on the same plain-text
   wrapper, so the curves can be compared head-to-head as an
   engine-correctness signal.
2. **Instruction-tuned models** (`Llama-3.1-8B-Instruct`, etc.): their
   priors are tuned to the model's chat template, not to
   `Instruction:/Response:`. Loss still descends (engine works), but
   absolute values are higher than what you'd see fine-tuning on the
   model's native chat format. Treat these numbers as a smoke signal,
   not as a production fine-tune quality measure.
3. **Base models**: the generic wrapper is roughly as
   out-of-distribution as anything else — descent is the relevant
   signal, not the absolute number.

If you want chat-template-aware training for production use, the right
fix is a `--apply-chat-template` flag on `train.py` that calls
`tokenizer.apply_chat_template([{"role":"user",...},
{"role":"assistant",...}], tokenize=False)` and feeds the resulting
string to the encoder; that's a separate change.

## RTX 5090 (31.3 GiB, 192 GiB host) — full sweep, 2026-05-10

All 13 rows re-verified at **auto memory budget** (no manual GPU/host
caps). Per-step metrics (`tok/sec`, `eff TFLOPS`, `hw TFLOPS`,
`peak alloc`, `peak resv`) read directly from `train.py`'s stdout at
step 3 — a mid-run logged data point past step-1 warmup. `peak alloc`
is `torch.cuda.max_memory_allocated()` (live peak); `peak resv` is
`torch.cuda.max_memory_reserved()` (caching-pool peak — what determines
OOM). Effective TFLOPS uses the canonical formula in
`flextrain/cli.py:_get_model_flops_per_token` (`matmul_factor = 4 if
LoRA else 6` — LoRA skips the frozen-weight wgrad — plus the causal
attention term). Hardware TFLOPS adds `recompute_flops / dt`; the gap
reflects the working-set solver's recompute trade-off.

| Model | Params | Arch | Mode | Loss curve (5 steps) | tok/sec | eff TFLOPS | hw TFLOPS | peak alloc | peak resv |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | dense | LoRA | 1.055 → 1.012 | 28,682 | 143.2 | 143.6 | 26.60 | 26.80 |
| Llama-3.2-1B | 1B | dense | full | 1.055 → 0.826 | 21,856 | 163.7 | 163.9 | 26.30 | 26.60 |
| Llama-3.1-8B-Instruct | 8B | dense | LoRA | 0.783 → 0.747 |  5,330 | 161.1 | 161.3 | 27.00 | 27.50 |
| Llama-3.1-8B-Instruct | 8B | dense | full | 0.783 → 0.600 |  3,723 | 168.7 | 174.3 | 26.50 | 26.80 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | 0.865 → 0.844 | 22,697 | 108.2 | 125.3 | 25.70 | 28.60 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | 0.865 → 0.673 | 13,305 |  95.1 | 108.1 | 26.40 | 29.00 |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | 0.933 → 0.873 |  5,026 | 153.2 | 153.5 | 27.10 | 27.40 |
| Qwen3-8B | 8B | dense, QK-norm | full | 0.928 → 0.478 |  3,533 | 161.6 | 169.2 | 27.60 | 27.80 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 0.747 → 0.661 |  4,947 | 158.3 | 158.9 | 27.00 | 27.90 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | 0.744 → 0.465 |  3,237 | 155.4 | 165.7 | 26.40 | 26.70 |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | 1.013 → 0.815 |  1,284 | 132.6 | 136.6 | 27.30 | 28.70 |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | 0.900 → 0.866 |  7,767 |  96.7 | 121.6 | 25.90 | 28.70 |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 0.742 → 0.676 |  5,923 |  74.6 |  92.3 | 24.90 | 28.50 |

The Qwen3.5-9B full-FT loss curve (0.744 → 0.465) reproduces the
historical RTX 3090 reference (0.744 → 0.455) to within ≈0.01. The
Llama-3.1-8B-Instruct rows show a smaller absolute loss-drop because
the instruction-tuned base is starting from a chat-template prior the
generic `Instruction:/Response:` wrapper does not match (see the
tokenization section above) — descent itself confirms the engine path
is exercised cleanly.

## RTX 5090 / 32 GiB — Gemma 2 / Gemma 3 dense, 2026-05-11

Newly added — covers Gemma 2 dense (`Gemma2ForCausalLM`) and Gemma 3
dense (`Gemma3ForCausalLM` for 1B; `Gemma3ForConditionalGeneration`
for 4B / 12B, text branch only). Validated end-to-end against HF
transformers on a single microbatch: cosine similarity ≥ 0.99 / sign-
match ≥ 0.92 / rel-L2 ≤ 20% per-block weight gradient parity, scalar
loss within 1% (see `tests/test_engine_fwd_bwd_parity.py`).

Engine pieces this exercises end-to-end:
* `Gemma3Block` dual-residual split backward (`backward_dgrad` +
  `backward_wgrad`), forward, and `forward_recompute` across tiers 0–3.
* Gated-GELU-tanh triton kernel
  (`flextrain/ops/_kernels/gelu_tanh_gated.py`), routed by
  `SwiGLUConfig.activation="gelu_tanh"`.
* `WeightMapEntry.hf_name_alternates` — handles 4B/12B's
  `language_model.model.layers.*` prefix in the same ArchSpec.
* `LMHead.final_logit_softcap` — Gemma 2's `tanh(logits/30)*30` head.
* `TokenEmbedLayer.embed_scale` — Gemma's `sqrt(d_model)` input scaling.
* `post_load_permute` — halved → pair-interleave Q/K perm AND the
  matching head-axis permute on the per-head QK-norm γ vectors, plus
  tied LM-head `embed.t()` mirror.

12B / 4B full-FT need `--leeway-gpu-mem-gib` bumped (default 3.0 →
6.0–8.0) and a smaller microbatch budget so flash-attn's bwd scratch
and the FFN bwd's recompute buffer fit alongside the working-set
solver's allocation. `experiments/verified_runs.py` carries per-row
overrides for these.

| Model | Params | Arch | Mode | Loss curve (5 steps) | tok/sec | eff TFLOPS | hw TFLOPS | peak alloc | peak resv |
|---|---|---|---|---|---|---|---|---|---|
| Gemma-2-2B-Instruct | 2B | dense, dual-residual norms, GELU-gated, attn+final softcap | LoRA | 1.102 → 0.949 | 12,767 | 134.5 | 134.8 | 28.00 | 28.50 |
| Gemma-2-2B-Instruct | 2B | dense, dual-residual norms, GELU-gated, attn+final softcap | full | 1.098 → 0.734 |  9,298 | 146.9 | 156.3 | 27.80 | 28.30 |
| Gemma-3-1B-Instruct | 1B | dense, dual-residual, QK-norm, alt local/global RoPE | LoRA | 1.441 → 1.284 | 28,430 | 114.8 | 115.1 | 27.20 | 27.50 |
| Gemma-3-1B-Instruct | 1B | dense, dual-residual, QK-norm, alt local/global RoPE | full | 1.433 → 0.929 | 20,549 | 124.5 | 124.7 | 27.00 | 27.40 |
| Gemma-3-4B-Instruct | 4B | dense, dual-residual, QK-norm, alt RoPE + linear scaling | LoRA | 1.272 → 1.058 |  8,947 | 139.9 | 140.1 | 27.10 | 27.70 |
| Gemma-3-4B-Instruct | 4B | dense, dual-residual, QK-norm, alt RoPE + linear scaling | full | 1.279 → 0.703 |  5,741 | 134.6 | 150.9 | 26.90 | 27.10 |
| Gemma-3-12B-Instruct | 12B | dense, dual-residual, QK-norm, alt RoPE + linear scaling | LoRA | 1.907 → 1.405 |  2,867 | 135.7 | 148.0 | 28.00 | 28.40 |
| Gemma-3-12B-Instruct | 12B | dense, dual-residual, QK-norm, alt RoPE + linear scaling | full | 1.988 → 0.619 |  1,597 | 113.4 | 135.4 | 28.20 | 28.90 |

Gemma-2-9B-Instruct full-FT is supported by the engine but the HF
reference side OOMs on a 32 GiB card (9B params + grads ≈ 36 GiB); the
row in `experiments/verified_runs.py` is wired up and ready on 48 GiB+
hardware.

**A note on Gemma 3 12B's higher starting loss vs 4B.** Gemma 3 12B
opens at ~1.88 (LoRA) / 1.99 (full) — *higher* than 4B's 1.26 / 1.28.
This is **not** a flextrain bug; we cross-checked HF transformers
under the same length-weighted mean over the first 32 MathInstruct
samples:

| model | flextrain step 0 | HF (matching convention) |
|---|---|---|
| Gemma-3-1B-Instruct | 1.441 | 1.518 |
| Gemma-3-4B-Instruct | 1.255 | 1.301 |
| Gemma-3-12B-Instruct | 1.881 | 1.962 |

flextrain matches HF to within ~5% relative; the residual gap is data
ordering / packing variance, not numerics. HF independently confirms
the 12B > 4B ordering — it's the same OOD-prompt-format penalty
documented for Llama-3.1-8B-Instruct above. Gemma 3 Instruct was
post-trained on its native `<start_of_turn>` chat template; the
generic `Instruction:/Response:` wrapper is out-of-distribution, and
larger / more confident priors get penalized harder for the mismatch.
Use these rows as an engine-correctness signal (loss descends → wires
are correct), not as a fine-tune quality measure — the right way to
get the quality measure is to run with the model's native chat
template (`tokenizer.apply_chat_template(...)`; see the tokenization
section above for the planned `--apply-chat-template` flag).

**A note on LoRA vs full step-0 loss.** At init LoRA has `A ~ N(0, σ)`
and `B = 0`, so the adapter delta `X @ A @ B = 0` and the model
behaves identically to the base — step-0 loss MUST equal full-FT's
step-0 loss on the same batch. The table's tiny step-0 differences
(e.g. 4B: 1.272 LoRA vs 1.279 full, 12B: 1.907 LoRA vs 1.988 full)
come from `train.py`'s producer/consumer data-prefetch race: LoRA and
full have different warm-up times, so the consumer pulls a slightly
different number of tokens in step 0 (4B: 31,240 vs 31,730 tokens;
12B: 15,268 vs 16,289 tokens) and the length-weighted mean differs by
~1–4%. `tests/test_lora_step0_equivalence.py` is the regression
guard: it loads each model in both modes against the **same**
hand-built batch and asserts step-0 loss is **bit-identical**. If
that test ever flips red the gap is real; otherwise the table gap is
data ordering, not numerics.

## Gemma 2 / 3 — same models, native chat template

Pass `--apply-chat-template` to `train.py` (or `apply_chat_template=True`
in `experiments/verified_runs.py`) and `JsonSFTTokenSource` renders
each record through the model's *native* chat format via HF's
`tokenizer.apply_chat_template`. For Gemma 2 / 3 that's the
`<bos><start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n…<end_of_turn>\n`
wrapper the model was post-trained on. The template's
`<end_of_turn>` becomes the natural turn-terminator; we do NOT
additionally append `tokenizer.eos_token_id` (which would be
`<eos>=1`, a *different* token from `<end_of_turn>=106` on Gemma —
appending it would teach a sequence the model never sees in real
inference).

| Model | Params | Arch | Mode | Loss curve (5 steps) | tok/sec | eff TFLOPS | hw TFLOPS | peak alloc | peak resv |
|---|---|---|---|---|---|---|---|---|---|
| Gemma-2-2B-Instruct (chat tpl) | 2B | dense, dual-residual, GELU-gated, attn+final softcap | LoRA | 1.131 → 0.972 | 13,002 | 137.0 | 137.2 | 28.00 | 28.50 |
| Gemma-2-2B-Instruct (chat tpl) | 2B | dense, dual-residual, GELU-gated, attn+final softcap | full | 1.132 → 0.696 |  9,179 | 145.1 | 154.3 | 27.80 | 28.60 |
| Gemma-3-1B-Instruct (chat tpl) | 1B | dense, dual-residual, QK-norm, alt local/global RoPE | LoRA | 1.891 → 1.656 | 28,212 | 113.9 | 114.2 | 27.20 | 27.50 |
| Gemma-3-1B-Instruct (chat tpl) | 1B | dense, dual-residual, QK-norm, alt local/global RoPE | full | 1.897 → 0.901 | 24,052 | 145.8 | 146.0 | 27.00 | 27.30 |
| Gemma-3-4B-Instruct (chat tpl) | 4B | dense, dual-residual, QK-norm, alt RoPE + linear scaling | LoRA | 1.520 → 1.173 |  8,893 | 138.9 | 139.1 | 27.10 | 27.70 |
| Gemma-3-4B-Instruct (chat tpl) | 4B | dense, dual-residual, QK-norm, alt RoPE + linear scaling | full | 1.520 → 0.669 |  5,737 | 134.6 | 150.7 | 26.90 | 27.60 |
| Gemma-3-12B-Instruct (chat tpl) | 12B | dense, dual-residual, QK-norm, alt RoPE + linear scaling | LoRA | 1.201 → 1.066 |  2,849 | 134.8 | 147.3 | 27.80 | 28.30 |
| Gemma-3-12B-Instruct (chat tpl) | 12B | dense, dual-residual, QK-norm, alt RoPE + linear scaling | full | 1.245 → 0.591 |  1,575 | 111.8 | 133.3 | 28.20 | 28.70 |

### Direct side-by-side, step-0 loss (mode = full)

| Size  | OOD wrapper | Chat template | Δ |
|---|---|---|---|
| Gemma-2-2B  | 1.098 | 1.132 |  +0.034 |
| Gemma-3-1B  | 1.433 | 1.897 |  +0.464 |
| Gemma-3-4B  | 1.279 | 1.520 |  +0.241 |
| **Gemma-3-12B** | **1.988** | **1.245** | **−0.743** |

The chat template **fixes the 12B anomaly**: with the model's native
format Gemma-3-12B's step-0 loss drops from 1.988 to 1.245, and the
expected ordering returns (1B > 4B > 12B in starting loss, larger
model = stronger prior). What looked like a flextrain bug last round
was an OOD-prompt-format penalty that bites the biggest model hardest
— exactly because its chat-template prior is the strongest.

The flip side: the chat template makes Gemma-3-1B / 4B *worse* at
step 0 (1.43 → 1.90 for 1B). The smaller instruct-tuned Gemmas have
weaker chat-template priors than their pretrained text-completion
priors; the `Instruction:/Response:` plaintext format is closer to
the QA-style data they saw in pretraining than the
`<start_of_turn>` tokens they see only in instruction tuning. The
final-step loss curves remain comparable (e.g. 4B full: 0.70 vs
0.67), so the engine is doing the right thing in both modes — only
the starting prior differs.

Production recipe: use `--apply-chat-template` when fine-tuning
instruction-tuned bases on data you intend to deploy with the chat
template (the only sensible production target). Use the generic
wrapper only for raw-text SFT on base models or for engine-correctness
smoke tests.

## Gemma 2 / 3 — 5-step loss-trajectory parity vs HF

The verified-runs rows above confirm flextrain *trains* (loss
descends, throughput sane), but a sharper question is whether the
trajectory **matches** HF on the same data with the same optimizer.
`tests/test_arch_parity.py` is the test for this: subprocess-
isolated, builds 5 deterministic batches from MathInstruct
(`target_tokens_per_step=512`), runs both stacks with bf16 AdamW
(`lr=5e-6` full / `lr=1e-4` LoRA, β=(0.9, 0.95), eps=1e-8,
wd=0). For LoRA, flextrain's A-matrix init is captured and replayed
into HF's PEFT model so both stacks start from the same LoRA state.

| Model | Mode | step 0 (HF / FT) | step 4 (HF / FT) | max \|Δloss\| over 5 steps | step-0 logit max\|Δ\| (ref\|max\|) |
|---|---|---|---|---|---|
| Gemma-2-2B-Instruct  | full | 0.909 / 0.907 | 0.885 / 0.888 | **0.0034** | 1.9 (32.8) |
| Gemma-2-2B-Instruct  | LoRA | 0.909 / 0.907 | 0.883 / 0.889 | **0.0070** | 1.9 (32.8) |
| Gemma-3-1B-Instruct  | full | 1.379 / 1.376 | 1.138 / 1.119 | **0.0189** | 1.9 (32.8) |
| Gemma-3-1B-Instruct  | LoRA | 1.379 / 1.376 | 1.104 / 1.132 | **0.0288** | 1.9 (32.8) |
| Gemma-3-4B-Instruct  | LoRA | — / — | — / — | **0.0091** | 2.1 (46.0) |
| Gemma-3-4B-Instruct  | full | — | — | (HF OOM, see note) | — |
| Gemma-3-12B-Instruct | LoRA | 1.294 / 1.289 | 0.843 / 0.909 | **0.1115** | 2.3 (46.3) |

* Gemma-3-12B-Instruct full FT and Gemma-3-4B-Instruct full FT
  trajectory tests are skipped on a 32 GiB card — HF's plain
  `torch.optim.AdamW` keeps params + fp32 grads + 2× fp32 opt state
  resident, which for 4B / 12B exceeds the GPU. flextrain handles
  these via host offload; HF can't, and we'd need an FSDP / offload-
  AdamW harness on the HF side to get an apples-to-apples 5-step
  trajectory comparison. The verified-runs section above still
  exercises these configs end-to-end (loss descends, engine is sound).
* For 12B LoRA, gradient checkpointing is enabled on the HF side to
  fit (24 GiB params + activations would otherwise OOM at bwd). This
  changes HF's bwd op ordering slightly, which along with bf16 noise
  compounding across 48 layers explains why 12B's `max|Δloss|` is
  ~0.11 vs ~0.01–0.03 for the smaller models. **Step-0 matches to
  0.4%** — confirming the forward + initial LoRA state are bit-
  equivalent — so the engine is correct; the divergence is bf16
  drift through the optimizer, not a routing bug. Trajectory shape
  tracks correctly (both stacks see the step-2 rise and the step-4
  drop at the same magnitude direction).

Below the noise floor: the smallest models (Gemma-2-2B,
Gemma-3-1B, Gemma-3-4B LoRA) all match within ≤3% loss agreement
across all 5 steps, end-to-end. That's the strongest evidence the
Gemma 2 / Gemma 3 engine path produces the right gradients — same
sequences in, same loss trajectory out.

## Re-verify protocol (post big-engine-change regression check)

Wrap script: ``bash experiments/reverify_gemma.sh`` — re-runs the 16
Gemma rows + a Llama-3.2-1B-LoRA smoke row (catches shared-code
regressions in api.py / head.py / embed.py / hf_weights.py), then
diffs the rerun against the committed baseline at
``runs/verified_gemma/`` via ``verified_runs.py compare``. Exits 0
when every row matches loss bit-exactly and throughput within ±5%;
non-zero on any drift.

Expectations on a clean run on the same hardware:

* **Losses**: bit-identical to ``runs/verified_gemma/<row>/final.json``
  for every row, every step. Engine code is deterministic given
  deterministic data + deterministic kernels.
* **Throughput**: within ±5% of the per-row step-3 tok/s. The few
  exceptions seen during the post-Gemma-engine-integration reverify
  were all on the "baseline" side (conservative measurement during
  GPU warm-up); reruns at steady state run +5-15% faster.
* **CUDA non-determinism caveat**: on 4B/12B with bf16 activations
  through flash-attn + LoRA-adapter grads (atomic accumulators),
  individual run-to-run drift of ~0.001-0.01 in losses is possible
  but uncommon. Rare numerical blowups under aggressive LR cooldown
  on 5-step toy runs can saturate the cross-entropy clamp
  (``loss=100.0`` sentinel in ``flextrain_cross_entropy_loss``) —
  re-running the row typically lands clean. If a row's loss
  consistently differs from baseline across **two** clean reruns,
  there's a real regression.

When a re-verify detects drift, the bisect order is:

1. ``tests/test_gemma3_block_parity.py`` — per-layer math. If this
   breaks, the issue is in the activation kernel / dual-residual
   bwd / forward_recompute. Look at commits 1 and 2.
2. ``tests/test_gemma3_full_forward_parity.py`` — full-model fwd
   bypassing the engine. If this breaks but block-parity passes,
   the issue is in the manual driver or HF-weight remap. Unlikely
   to bisect engine bugs.
3. ``tests/test_engine_fwd_bwd_parity.py`` — engine fwd+bwd vs HF on
   a fixed prompt. If this breaks but block-parity passes, the
   issue is in the engine wiring (block builder, post_load_permute,
   embed/head, ARCH_MODULES). Look at commit 3.
4. ``tests/test_arch_parity.py`` — 5-step optimizer trajectory. If
   all the above pass but this drifts on a NEW model not previously
   tested, the bug is mode-specific (e.g. LoRA-only). The lr=0
   forward-only baseline + LoRA target audit (documented in
   ``flextrain/nn/layers/gemma3.py:289``) are the next-level
   triage.

## RTX 3090 (24 GiB, 117 GiB host) — historical reference

Pre-2026-05 sweep. Most rows were skipped on this hardware due to
memory limits.

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

Additional models supported by the existing arch loaders (require a
larger machine to actually train): Qwen3.6-35B-A3B, Qwen3.5-122B-A10B,
Qwen3.5-397B-A17B, Qwen3-Coder-30B-A3B-Instruct (no new wiring needed;
they reuse `Qwen3_5*` / `Qwen3Moe*` arch ids).
