# Gemma 3 dense — status & continuation notes

**Last updated:** 2026-05-11
**Approved plan file:** `/home/shein/.claude/plans/have-a-good-look-zazzy-mountain.md`

## What this work is

Finish Gemma 3 dense (text-only) end-to-end in flextrain with forward AND backward parity vs HuggingFace transformers, validated on three local checkpoints:

- `models/Gemma-3-1B-Instruct` — `Gemma3ForCausalLM`, plain `model.layers.*` weight prefix, no vision tower, no rope_scaling.
- `models/Gemma-3-4B-Instruct` — `Gemma3ForConditionalGeneration`, text weights under `model.language_model.layers.*`, `rope_scaling: {factor: 8.0, rope_type: linear}`, SigLIP vision tower (ignored).
- `models/Gemma-3-12B-Instruct` — same shape as 4B, larger.

Multimodal, Gemma 4, and Gemma 3n (E2B/E4B PLE+MatFormer+AltUp) are **deferred to a separate session** per the user.

## What's already in the repo (do not duplicate)

- **Forward implementations** for `Gemma2Block` and `Gemma3Block` are complete and tested (`flextrain/nn/layers/gemma2.py`, `gemma3.py`). Gemma 3 = Gemma 2 dual-residual norm topology + per-head QK-norm inside `GQAAttentionBlock` (`cfg.qk_norm=True`).
- **HF weight map** + `γ + 1` post-load shift in `flextrain/io/arch/gemma3.py` covering the plain `Gemma3ForCausalLM` prefix.
- **Per-layer rope-base alternation** (local/global) via `build_gemma3_backbone` in `gemma3.py` and `sliding_window_pattern` integer extraction in the HF config translator.
- **Layer alternation reference** (`Qwen3_5LayerConfig` + dual-builder pattern) in `flextrain/nn/layers/qwen3_5.py` and `flextrain/io/arch/qwen3_5.py` if we want to mirror it.
- **Reference split-backward implementation** for a Llama-style block in `flextrain/nn/layers/llama.py:274-457`.

## Key constraints from the user

1. **Do not modify activation schemas of shared blocks** (`GQAAttentionBlock.fields()`, `SwiGLUFFN.fields()`, `RMSNormBlock.fields()`, etc.). Schemas are load-bearing structure and other archs depend on them.
2. **Layer-owned ActivationFields are fine** — adding new fields to `Gemma3Block`'s schema (declared inside `Gemma3Block.__init__` and concatenated with the block-owned fields) is acceptable. This is the pattern used for the existing `x_inp` and `x_mid` fields.
3. **Keyword-arg behavior changes to shared blocks are acceptable** (e.g., the rope.py linear-scaling fix below).
4. **Scope is 1B / 4B / 12B parity for text-only.** Vision tower and rare Gemma 3 27B-specific knobs (`query_pre_attn_scalar` mismatch with `head_dim`) are out of scope for this session.
5. **Gradient parity, not just logit parity**, is required.

## Progress so far (landed; no commit yet)

### `flextrain/nn/blocks/rope.py`
- `build_rope_inv_freq` now actually applies `inv / factor` when `rope_scaling["rope_type"] == "linear"`. Previously it was a documented no-op. This is keyword-arg behavior only — the function signature is unchanged.
- `build_partial_rope_inv_freq` extended similarly for partial-rotary archs (Qwen3.5/3.6/Qwen3-Next). Same behavior contract; no signature change.

### `flextrain/nn/layers/gemma3.py`
- `Gemma3BlockConfig.attn_logit_softcap` and `final_logit_softcap` defaults changed from `50.0` / `30.0` (Gemma 2 vintage) to `None`. Type widened to `float | None`. HF Gemma 3 sets both to `null`.
- Added `rope_scaling: object | None = None` field to `Gemma3BlockConfig`.
- `_build_attn` now threads `rope_scaling` into both `GQAAttentionConfig` and `GQASlidingWindowAttentionConfig`, and maps `None → 0.0` for the softcap at the boundary so the upstream config (typed `float`) is untouched.

**No edits to any shared block file.** No edits to any other arch's layer file.

## TODOs (ordered with dependencies)

State as of this checkpoint:

| # | Step | Status |
|---|---|---|
| 0 | Read llama.py split-bwd + norm.py / attention.py / ffn_dense.py bwd APIs | ✅ done |
| 3a | Add `attn_scale` (query_pre_attn_scalar) plumbing | ⏸ deferred (not needed for 1B/4B/12B) |
| 3b | Add linear rope_scaling | ✅ done (rope.py) |
| 3c | Fix `Gemma3BlockConfig` softcap defaults + thread `rope_scaling` | ✅ done (gemma3.py) |
| 5a | `tests/test_gemma3_block_parity.py` (autograd-reference oracle) | ⬜ **NEXT** |
| 1 setup | Add `a_only` / `ffn_only` tier-0 ActivationFields | ⬜ next |
| 1 main | Hand-roll Gemma2/Gemma3 dual-residual backward (split form) | ⬜ next |
| 2 | Implement `Gemma2Block.forward_recompute` / `Gemma3Block.forward_recompute` for tiers 0-3 | ⬜ |
| 4a | Add `WeightMapEntry.hf_name_alternates` for multi-prefix loading | ⬜ |
| 4b-4f | Gemma3 block builder, `post_load_permute`, register in `ARCH_MODULES` | ⬜ |
| 5b | `tests/test_gemma3_1b_parity.py` (full-model fwd + bwd vs HF) | ⬜ |
| 5c | `tests/test_gemma3_multimodal_parity.py` (4B/12B via language_model prefix) | ⬜ |
| 5d | Remove gemma skip in `tests/test_arch_parity.py:406-408` | ⬜ |
| 6 | End-to-end smoke: `from_pretrained` + 5 steps on Gemma-3-1B-Instruct | ⬜ |

## Design decisions (recorded so next session doesn't relitigate)

### 1. New tier-0 fields `a_only` and `ffn_only` go on `Gemma3Block` / `Gemma2Block` schemas, not on shared block schemas.

In Gemma's forward, attn and FFN are called with `zero_resid` (`ctx.scratch(...).zero_()`) so that the unfused outputs are recovered separately. Post-norm sublayers consume the unfused outputs:
```
a_only   = attn.fwd(zero, pre_attn_norm(x))       # = slot.xo when residual was 0
ffn_only = ffn.fwd(pre_ffn_norm(x_mid), zero)     # written to out_tensor=x, NOT persistent
```

Backward needs `a_only` for `post_attn_norm.bwd`'s pre-norm input, and `ffn_only` for `post_ffn_norm.bwd`'s pre-norm input. Three options were considered:

- **A. New tier-0 ActivationFields on the Gemma layer's schema** — `a_only` and `ffn_only`, copied during fwd. Cost: 2 × T × d_model bf16 per layer per chunk. **Chosen — simplest, matches the existing `x_mid` pattern.**
- B. Reconstruct in `forward_recompute` from `slot.attn_result + w_o` and `swiglu(x1,x3) + w_2`. Cheaper memory but tangled recompute chain.
- C. Stash in `slot.aux`. Rejected because aux lifetime guarantees aren't safe across host-offload of the slot.

Both fields will be declared inside `Gemma3Block.__init__` (and `Gemma2Block.__init__`), concatenated alongside `(x_inp,)` and `(x_mid,)` in the schema via `concat_fields`. **No changes to `GQAAttentionBlock.fields()` or `SwiGLUFFN.fields()`.**

### 2. Dual-residual backward — split form

Mirror Llama's split exactly. `backward` is a 4-line delegating shim over `backward_dgrad` + `backward_wgrad`.

**Gradient flow (verified against forward in `gemma3.py:190-222`):**

```
# Outer FFN residual: out = x_mid + post_ffn_norm(ffn_only)
dh3                            = dout
dx_mid_outer                   = dout
dffn_only, g_post_ffn_norm    += post_ffn_norm.bwd(dh3, slot.ffn_only, slot.post_ffn_norm_rstd)
dpre_ffn_norm_h                = ffn.bwd(dffn_only, ...)        # ffn.bwd inputs dy/d(W2·swiglu); zero_resid in fwd means no extra term
dx_mid, g_pre_ffn_norm        += pre_ffn_norm.bwd(dpre_ffn_norm_h, slot.x_mid, slot.pre_ffn_norm_rstd,
                                                  dx_accumulator=dx_mid_outer)

# Outer attn residual: x_mid = x + post_attn_norm(a_only)
dh2                            = dx_mid
dx_outer                       = dx_mid
da_only, g_post_attn_norm     += post_attn_norm.bwd(dh2, slot.a_only, slot.post_attn_norm_rstd)
dpre_attn_h                    = attn.bwd(da_only, ..., attn_norm_output=pre_attn_norm_fwd_output)
dx, g_pre_attn_norm           += pre_attn_norm.bwd(dpre_attn_h, slot.x_inp, slot.pre_attn_norm_rstd,
                                                   dx_accumulator=dx_outer)
return dx
```

- **Inline Wgrads in `backward_dgrad`** (no recomputed-RMSNorm operand needed): `g_o` (in `attn.bwd`), `g_2` (in `ffn.bwd`), all four `g_*_norm` (in the four `RMSNormBlock.bwd` calls), `g_q_norm`/`g_k_norm` (in `attn.bwd` when `cfg.qk_norm=True`).
- **Deferred to `backward_wgrad`**: `g_q`, `g_k`, `g_v`, `g_1`, `g_3` — they need recomputed `pre_*_norm` outputs as left operand. Ferried in `BackwardIntermediates.aux["pre_attn_norm_fwd_output"]` and `"pre_ffn_norm_fwd_output"`.

### 3. γ-shift convention

`_gemma3_post_load_hook` already adds `+1` to every γ at load (`flextrain/io/arch/gemma3.py:14-29`). The bwd math sees canonical γ. **No shift correction inside the layer.** Verified: derivative of `(γ - 1)` w.r.t. canonical γ is 1, so HF γ-grad == FT γ-grad.

### 4. Gemma 2 vs Gemma 3 backward

Identical except Gemma 3 sets `cfg.qk_norm=True` on the attention block. The QK-norm bwd is already implemented inside `GQAAttentionBlock.bwd` (attention.py:704-735) and triggers automatically when `cfg.qk_norm=True`. The Gemma layer just needs to pass `attn_norm_output` (the recomputed pre-norm output) to `attn.bwd`. **Implement Gemma 2 first** because no QK-norm = fewer moving parts to debug.

### 5. Multi-prefix HF loading for 4B/12B

`Gemma3ForConditionalGeneration` (4B/12B) wraps text weights under `model.language_model.layers.*`. Two options:

- **Chosen: Add `hf_name_alternates: tuple[str, ...] = ()` to `WeightMapEntry`** in `flextrain/io/hf_weights.py`. Loader tries `hf_name` first, falls back to alternates. Minimal diff (~30 lines), reusable (Qwen3.5 already does this manually in `_resolve_layer_prefix`).
- Rejected: per-arch `name_resolver` callback (more complex, less idiomatic).

### 6. Test pyramid

5a (block parity vs autograd) → 5b (1B full-model parity vs HF transformers) → 5c (4B/12B language-model-only parity) → 5d (re-enable existing arch parity loop for Gemma 3).

## Recommended next-session execution order

**Start with the autograd-reference test** — this is the oracle for the bwd. Writing it first means each gradient gets validated as it's added.

1. **`tests/test_gemma3_block_parity.py`** (oracle):
   - Build one `Gemma2Block` (no QK-norm) with random weights, `d_model=128, n_heads=4, n_kv=2, head_dim=32, expert_dim=256`, full attention, T=64 tokens.
   - Build a torch-autograd reference module that replicates the dual-residual fwd math in plain torch (or use HF's `Gemma2DecoderLayer` directly).
   - Forward: assert `max|out_FT − out_ref|` ≤ bf16 noise (~5e-3).
   - Set `dout = randn`; run FT `backward`; run autograd `loss.backward()`; assert per-tensor grad parity for all 12 grads: `g_q, g_k, g_v, g_o, g_1, g_2, g_3, g_pre_attn_norm, g_post_attn_norm, g_pre_ffn_norm, g_post_ffn_norm` (+ `g_q_norm, g_k_norm` for the Gemma 3 variant).
   - Loop over: {full, sliding} × {tier 0, 1, 2, 3}.

2. **Step 1 setup** — Edit `gemma2.py` schema and forward:
   - Declare `a_only_field = ActivationField("a_only", lambda n, d: (n, cfg.d_model), cfg.compute_dtype, tier=0)`.
   - Declare `ffn_only_field` similarly.
   - Insert into `concat_fields([..., (a_only_field,), self.post_attn_norm.fields(), (x_mid,), ..., (ffn_only_field,), self.post_ffn_norm.fields(), self.ffn.fields()])`.
   - In `forward`, after `a_only = self.attn.fwd(...)`, call `slot.a_only.copy_(a_only.view(-1, cfg.d_model))`. Same for `ffn_only` after `self.ffn.fwd(...)`.
   - Repeat in `gemma3.py`.

3. **Step 1 main** — Implement `Gemma2Block.backward_dgrad` / `backward_wgrad` / `backward` shim. Use the gradient-flow pseudocode above. Validate against test 1 after each grad added.

4. **Step 1 for Gemma 3** — Copy the Gemma 2 bwd verbatim into `Gemma3Block` (the only difference is QK-norm, which is handled inside `attn.bwd` when `cfg.qk_norm=True`).

5. **Step 2** — `forward_recompute` for tiers 0-3. Pattern: read `slot.has(name)`, recompute missing fields in dependency order. Note: with `a_only` / `ffn_only` at tier 0, they're always saved; no recompute needed for them.

6. **Step 4a** — Add `hf_name_alternates` to `WeightMapEntry`. Update loader in `flextrain/io/hf_weights.py` to try each alternate in order.

7. **Step 4b-4f** — Block builder mirroring `gemma2.py`. `post_load_permute` (Q/K halved→pair + tied head mirror + QK-norm head-internal perm). Register both `Gemma3ForCausalLM` and `Gemma3ForConditionalGeneration`. Add `"gemma3": gemma3` to `ARCH_MODULES`.

8. **Step 5b** — Full-model 1B parity. Load `models/Gemma-3-1B-Instruct` via `from_pretrained`. Run one fwd+bwd on a short prompt. Compare logits, per-token CE, embedding-grad, final-norm γ-grad, and per-layer grads (sample `g_q`, `g_v`, `g_1`, `g_2`) vs HF transformers.

9. **Step 5c** — 4B/12B with language-model-only prefix resolution.

10. **Step 5d** — Remove the `arch_id in ("Gemma2ForCausalLM", "Gemma3ForCausalLM")` skip at `tests/test_arch_parity.py:406-408`.

11. **End-to-end smoke** — `from_pretrained(...)` + 5 training steps on a tiny SFT config, confirm loss decreases.

## Testing plan (summary)

| Test | What it catches |
|---|---|
| `tests/test_gemma3_block_parity.py` (Step 5a) | Per-grad math errors. Validates dual-residual bwd against autograd. Loop over (full, sliding) × (tier 0..3). |
| `tests/test_gemma3_1b_parity.py` (Step 5b) | Weight loading, post_load_permute, RoPE convention, embedding+final-norm grads, full-model integration. 1B with `Gemma3ForCausalLM` plain prefix. |
| `tests/test_gemma3_multimodal_parity.py` (Step 5c) | Multi-prefix loading (`model.language_model.*`), `text_config` wrapping, linear rope_scaling. 4B + 12B. |
| `tests/test_arch_parity.py` (Step 5d, gate removed) | Existing LoRA + full-FT loops over MathInstruct prompts, loss-curve agreement over N steps. |
| `tests/test_arch_lora_e2e.py` (existing) | LoRA wrapper integration, save-tier-dependent numerical drift via `forward_recompute` vs `forward` agreement. |
| Manual smoke: 5-step training loop | End-to-end engine integration (working-set solver, save-tier DP, optimizer step). |

## Files this work touches

**Will be edited:**
- `flextrain/nn/layers/gemma2.py` — schema additions, `forward` save calls, `backward`/`backward_dgrad`/`backward_wgrad`, `forward_recompute`.
- `flextrain/nn/layers/gemma3.py` — same as gemma2.py. (Partially edited at checkpoint: `Gemma3BlockConfig` softcap+rope_scaling, `_build_attn` boundary mapping.)
- `flextrain/io/arch/gemma3.py` — block builder, `post_load_permute`, `register_block_builder` call, `BLOCK_BUILDER`, register HF arch IDs for both `Gemma3ForCausalLM` and `Gemma3ForConditionalGeneration`. Extend `hf_config_to_hyperparams` to surface `rope_scaling` and the `sliding_window_pattern` integer.
- `flextrain/io/arch/__init__.py` — add `"gemma3": gemma3` to `ARCH_MODULES`, drop the existing exclusion comment.
- `flextrain/io/hf_weights.py` — add `hf_name_alternates` to `WeightMapEntry`.
- `tests/test_arch_parity.py:406-408` — remove gemma skip after bwd lands.

**Will be created:**
- `tests/test_gemma3_block_parity.py`
- `tests/test_gemma3_1b_parity.py`
- `tests/test_gemma3_multimodal_parity.py`

**Already touched (committed in the partial-progress checkpoint):**
- `flextrain/nn/blocks/rope.py` — `build_rope_inv_freq` / `build_partial_rope_inv_freq` linear-scaling support.
- `flextrain/nn/layers/gemma3.py` — `Gemma3BlockConfig` defaults + `_build_attn` rope_scaling pass-through.

**Will NOT be edited (per user constraint):**
- `flextrain/nn/blocks/attention.py`, `attention_gated.py`, `ffn_dense.py`, `ffn_moe*.py`, `norm.py`, `linear_attn.py`, `lora.py` — shared blocks; schemas off-limits.
- Other archs' layer files (`llama.py`, `mistral.py`, `qwen*.py`, `olmoe.py`, `gemma2.py`'s schema — only its bwd will be added).
