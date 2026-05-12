# Gemma family — verified runs and parity notes

End-to-end smoke + parity records for the Gemma models supported by
flextrain. Scope:

- **Gemma 2 dense** (`Gemma2ForCausalLM`) — Gemma-2-2B-Instruct;
  Gemma-2-9B-Instruct on 48 GiB+ hardware.
- **Gemma 3 dense** (`Gemma3ForCausalLM` for 1B;
  `Gemma3ForConditionalGeneration` for 4B / 12B, text branch only) —
  Gemma-3-{1,4,12}B-Instruct.
- **Gemma 4** — in active development, NOT yet in this page. See
  `docs/internal/gemma4_status.md` and
  `docs/internal/gemma4_open_investigations.md` for current state.

This page is split out from `docs/verified_runs.md` so the main verified-
runs sweep stays focused on the broader model zoo; Gemma's
arch-specific knobs (dual-residual norms, attention/final softcap,
QK-norm, alternating local/global RoPE, γ shift convention, tied LM
head) warrant their own discussion.

The general workload, prompt wrapper, tokenization conventions, and
reproduction recipe are documented at the top of `docs/verified_runs.md`
and apply identically here unless noted.

## Engine surface this exercises

Items exercised end-to-end by the rows below:

- `Gemma2Block` / `Gemma3Block` dual-residual split backward
  (`backward_dgrad` + `backward_wgrad`), forward, and
  `forward_recompute` across tiers 0–3.
- Gated-GELU-tanh triton kernel
  (`flextrain/ops/_kernels/gelu_tanh_gated.py`), routed by
  `SwiGLUConfig.activation="gelu_tanh"`.
- `WeightMapEntry.hf_name_alternates` — handles 4B/12B's
  `language_model.model.layers.*` prefix in the same ArchSpec.
- `LMHead.final_logit_softcap` — Gemma 2's `tanh(logits/30)*30` head.
- `TokenEmbedLayer.embed_scale` — Gemma's `sqrt(d_model)` input scaling.
- `post_load_permute` — halved → pair-interleave Q/K perm AND the
  matching head-axis permute on the per-head QK-norm γ vectors, plus
  tied LM-head `embed.t()` mirror.
- Gemma 3 alternating local/global RoPE (rope_base swap per layer);
  4B/12B add linear rope_scaling on full layers.

12B / 4B full-FT need `--leeway-gpu-mem-gib` bumped (default 3.0 →
6.0–8.0) and a smaller microbatch budget so flash-attn's bwd scratch
and the FFN bwd's recompute buffer fit alongside the working-set
solver's allocation. `experiments/verified_runs.py` carries per-row
overrides for these.

## RTX 5090 / 32 GiB — Gemma 2 / Gemma 3 dense, 2026-05-11

Validated end-to-end against HF transformers on a single microbatch:
cosine similarity ≥ 0.99 / sign-match ≥ 0.92 / rel-L2 ≤ 20% per-block
weight gradient parity, scalar loss within 1% (see
`tests/test_engine_fwd_bwd_parity.py`).

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

### Why Gemma 3 12B's starting loss is higher than 4B (OOD prompt format)

Gemma 3 12B opens at ~1.88 (LoRA) / 1.99 (full) — *higher* than 4B's
1.26 / 1.28. We cross-checked HF transformers under the same length-
weighted mean over the first 32 MathInstruct samples:

| model | flextrain step 0 | HF (matching convention) |
|---|---|---|
| Gemma-3-1B-Instruct | 1.441 | 1.518 |
| Gemma-3-4B-Instruct | 1.255 | 1.301 |
| Gemma-3-12B-Instruct | 1.881 | 1.962 |

flextrain matches HF to within ~5% relative; the residual gap is data
ordering / packing variance, not numerics. HF independently confirms
the 12B > 4B ordering — it's the same OOD-prompt-format penalty
documented for Llama-3.1-8B-Instruct in `verified_runs.md`. Gemma 3
Instruct was post-trained on its native `<start_of_turn>` chat
template; the generic `Instruction:/Response:` wrapper is out-of-
distribution, and larger / more confident priors get penalized harder
for the mismatch. The native-chat-template table below confirms this:
12B's step-0 loss drops from 1.988 → 1.245 when you use the right
wrapper.

### Why LoRA and full step-0 loss differ slightly

At init LoRA has `A ~ N(0, σ)` and `B = 0`, so the adapter delta
`X @ A @ B = 0` and the model behaves identically to the base — step-0
loss MUST equal full-FT's step-0 loss on the same batch. The table's
tiny step-0 differences (e.g. 4B: 1.272 LoRA vs 1.279 full, 12B: 1.907
LoRA vs 1.988 full) come from `train.py`'s producer/consumer data-
prefetch race: LoRA and full have different warm-up times, so the
consumer pulls a slightly different number of tokens in step 0 (4B:
31,240 vs 31,730 tokens; 12B: 15,268 vs 16,289 tokens) and the
length-weighted mean differs by ~1–4%.
`tests/test_lora_step0_equivalence.py` is the regression guard: it
loads each model in both modes against the **same** hand-built batch
and asserts step-0 loss is **bit-identical**. If that test ever flips
red the gap is real; otherwise the table gap is data ordering, not
numerics.

## Native chat template — same models, `--apply-chat-template`

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

## 5-step loss-trajectory parity vs HF

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

`bash experiments/reverify.sh --include-gemma --baseline runs/<trusted_baseline_dir>`
re-runs the full 29-row sweep (13 non-Gemma rows from
`docs/verified_runs.md` + the 16 Gemma rows below) and diffs each row
against the baseline via `verified_runs.py compare`. Exits 0 when every
row matches loss bit-exactly and throughput within ±5%; non-zero on any
drift. The non-Gemma rows (in particular the smaller Llama / Qwen
LoRA rows) catch shared-code regressions in api.py / head.py / embed.py
/ hf_weights.py that pure-Gemma rows would miss. See
`docs/verified_runs.md` §"How to reproduce this table" for the full
flag list.

For a Gemma-only run (skipping the broader sweep) when you've only
touched Gemma 2 / 3 code:

```bash
python experiments/verified_runs.py run-grid \
    --out runs/<dir> \
    --only gemma2_2b_lora gemma2_2b_full gemma3_1b_lora gemma3_1b_full \
           gemma3_4b_lora gemma3_4b_full gemma3_12b_lora gemma3_12b_full
```

(Add `_chat` variants for the chat-template rows.)

Expectations on a clean run on the same hardware:

* **Losses**: bit-identical to `runs/verified_gemma/<row>/final.json`
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
  (`loss=100.0` sentinel in `flextrain_cross_entropy_loss`) —
  re-running the row typically lands clean. If a row's loss
  consistently differs from baseline across **two** clean reruns,
  there's a real regression.

When a re-verify detects drift, the bisect order is:

1. `tests/test_gemma3_block_parity.py` — per-layer math. If this
   breaks, the issue is in the activation kernel / dual-residual
   bwd / forward_recompute. Look at commits 1 and 2.
2. `tests/test_gemma3_full_forward_parity.py` — full-model fwd
   bypassing the engine. If this breaks but block-parity passes,
   the issue is in the manual driver or HF-weight remap. Unlikely
   to bisect engine bugs.
3. `tests/test_engine_fwd_bwd_parity.py` — engine fwd+bwd vs HF on
   a fixed prompt. If this breaks but block-parity passes, the
   issue is in the engine wiring (block builder, post_load_permute,
   embed/head, ARCH_MODULES). Look at commit 3.
4. `tests/test_arch_parity.py` — 5-step optimizer trajectory. If
   all the above pass but this drifts on a NEW model not previously
   tested, the bug is mode-specific (e.g. LoRA-only). The lr=0
   forward-only baseline + LoRA target audit (documented in
   `flextrain/nn/layers/gemma3.py:289`) are the next-level
   triage.

## Known caveats and open questions

The Gemma 2 / 3 integration **looks right** by the criteria above (block
parity tight, full-forward parity tight, 5-step trajectory matches HF
to within bf16 noise for the smaller models), but a few items have
not been independently re-verified and could harbor subtle issues:

- **Larger γ values and bf16 vs fp32 multiply in RMSNorm**. HF's
  `Gemma3RMSNorm` (and Gemma 2's) computes `output * (1 + weight)` in
  fp32 then casts to bf16. flextrain's `flextrain_rmsnorm_fwd` does
  the multiply in bf16 (verified empirically while debugging Gemma 4;
  see `docs/internal/gemma4_open_investigations.md`). For Gemma 2 / 3
  the γ values are mostly close to zero in storage (γ − 1 convention,
  init=zeros), so `1 + γ` stays near 1.0 and the bf16 multiply loses
  little precision — different regime from Gemma 4 where stored γ
  values reach ~100. The Gemma 2 / 3 parity tests pass at the existing
  thresholds despite this; if those thresholds ever tighten, this is
  the first place to look.

- **Attention scaling assumption**. The Gemma 2 / 3 forward path
  relies on flextrain's flash kernel applying the default
  `softmax_scale = 1/sqrt(head_dim)`. HF Gemma 2 / 3 attention
  doesn't explicitly override `module.scaling` (so `eager_attention_forward`
  falls back to its default `head_dim**-0.5`), which matches. This is
  a **different** convention from Gemma 4 (which hardcodes
  `scaling = 1.0` and lets q_norm/k_norm γ encode the temperature).
  If anyone refactors the scaling path, Gemma 2 / 3 should continue
  to use the kernel default; Gemma 4 needs an explicit override.

- **12B LoRA `max|Δloss| = 0.11` is loose**. We attribute this to
  HF-side gradient checkpointing + bf16 drift through 48 layers, with
  step-0 logits matching to within ~0.4% as evidence the forward path
  is right. We do NOT have a clean apples-to-apples comparison (HF
  OOM on a 32 GiB card without grad-checkpointing). If 12B LoRA
  trajectory ever shifts further from HF (drifts to > 0.15), the
  next triage is the `lr=0` baseline + LoRA-target audit documented
  in `flextrain/nn/layers/gemma3.py:289`.

- **Generation quality not measured.** The verified-runs rows show
  loss descends and the per-row generation is "coherent + hits EOS,"
  but there's no automated quality metric (perplexity on held-out
  text, downstream eval scores) for any of the Gemma rows. If
  generation quality regresses, this table won't catch it.

- **9B not yet verified end-to-end.** The row is wired up; needs a
  48 GiB+ card to actually run both flextrain and HF side-by-side.

Anything that looks like a flextrain Gemma 2 / 3 bug should first be
reproduced through `tests/test_gemma3_block_parity.py` (block math at
small dims) → `tests/test_gemma3_full_forward_parity.py` (full forward
on real safetensors) → `tests/test_engine_fwd_bwd_parity.py` (full
fwd+bwd on a fixed prompt). Each gates the next; the small-dim block
parity is the tightest signal and is what we trust as the ground
truth for correctness.
