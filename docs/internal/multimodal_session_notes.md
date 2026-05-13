# Multimodal — autonomous session notes

Working notes from the autonomous implementation session that follows the foundation-scaffolding session of 2026-05-12.

**Status doc (authoritative):** `docs/internal/multimodal_status.md`
**Approved plan:** `~/.claude/plans/please-take-your-time-delegated-parrot.md`

This file is a running log of:
* design decisions made on the fly,
* bugs or oddities encountered,
* deferred work (with concrete pointers to where to pick it up),
* anything the user should review after the autonomous session.

## Issues encountered

### 4. ChunkSeqRef unwrap missing in `_gather_round_sequences` (2026-05-13, fixed; SEVERITY: HIGH -- encoder never ran)

**Symptom:** Full-model loss parity vs HF was at 4.48% rel_err even after the vision-encoder bf16 inv_freq fix (which gave byte-exact encoder parity). Per-position embed diagnostic showed cos_min = -0.19 between HF post-scatter inputs_embeds and flextrain post-splice text_emb -- essentially random orientation, NOT bf16 noise.

**Root cause:** `MultimodalInputLayer._gather_round_sequences` was iterating ``chunk.seqs`` and treating each element as a Sequence:

```python
for s in seqs:
    mod_in = getattr(s, "modality_inputs", None) or {}
```

But ``chunk.seqs`` is a list of :class:`flextrain.engine.schedule.ChunkSeqRef`, NOT :class:`Sequence` directly. ``ChunkSeqRef.seq`` carries the actual Sequence. So ``getattr(ref, "modality_inputs", None)`` always returned None, ``per_encoder_inputs`` stayed empty, and the encoder was **never invoked**. The splice ran with no encoder cache (no-op), so image-position embeds stayed at the placeholder-token's embed-table lookup (a meaningless default vector).

The smoke test's "finite loss = pass" criterion didn't catch this -- the LM just produced high-loss predictions from the default vector and ran to completion.

**Fix:** unwrap to ``ref.seq`` (or fall back if no ``.seq`` attribute, for unit tests that pass flat Sequences directly).

**Impact:** loss parity went from 4.48% rel_err -> 1.70% rel_err.

**Lesson:** "smoke test passes (no crash)" is a very weak signal for multimodal code. Always verify with a value-level diagnostic (per-position embed comparison) before declaring victory.

### 3. Engine missing `_mm_weights` on ``embed_ctx`` (2026-05-13, fixed; latent)

**Symptom:** After fixing #4, the next run raised ``RuntimeError: MultimodalInputLayer.setup_round: ctx is missing the _mm_weights attribute``. Setup_round needs the encoder weights dict.

**Root cause:** My initial design routed encoder weights via ``ctx._mm_weights`` (side-channel rather than a positional kwarg). The engine never set it.

**Fix:** ``ActiveModel._setup_round`` attaches both ``embed_ctx._mm_weights = self.buffers.gpu_embed_params`` and ``embed_ctx._mm_grads = self.buffers.gpu_embed_grads`` before calling ``setup_round``. Harmless for text-only paths since TokenEmbedLayer doesn't read those attributes.

### 1. `mm_encoder_peak_bytes` not threaded through `_pick_chunk_size` (2026-05-13, fixed)

**Symptom:** post-MRoPE reverify on `qwen3_30b_a3b_lora` and `qwen3_5_moe_35b_a3b_lora` crashed with `NameError: name 'mm_encoder_peak_bytes' is not defined` at `flextrain/core/working_set.py:1522`.

**Root cause:** I threaded `mm_encoder_peak_bytes` through `determine_working_set_config` -> `_baseline_gpu_activation_memory` (call sites at lines 1255 and 1518). The line-1518 call is INSIDE `_search`, a nested function in `_pick_chunk_size`. `_pick_chunk_size`'s signature didn't have `mm_encoder_peak_bytes`, so the closure couldn't see it, and `_search` failed at the second `_baseline_gpu_activation_memory` call site.

The non-MoE rows succeeded because they hit the first call site (line 1255) earlier in `determine_working_set_config`'s scope, and never reached `_pick_chunk_size`'s search retry path.

**Fix:** added `mm_encoder_peak_bytes: int = 0` to `_pick_chunk_size`'s signature; threaded through from `determine_working_set_config`'s call site.

**Test:** retrigger reverify and confirm both MoE rows now run + match baseline drift profile.

**Lesson:** when threading a new kwarg through a function with nested closures, audit all the closure capture sites, not just the immediate signature.

## Deferred / follow-up work

### MRoPE pure-PyTorch vs Triton kernel: 1 bf16 ULP

`apply_rope_mrope_fwd` differs from `apply_rope_partial_fwd` (the Triton kernel) by up to 1 bf16 ULP per element (1/64 ≈ 1.56e-2 at magnitude ~1) due to reduction-order quantization. The algebraic formula is identical -- it's purely a rounding-direction difference. This is fine for forward parity vs HF (also at most 1 bf16 ULP from the kernel result) but means a "degenerate-3D = standard partial RoPE" identity isn't bit-exact. The dispatch on `seq_positions.shape[-1]` in `attention_gated._rope_fwd` keeps text-only chunks on the Triton path so this never affects text-only training.

`tests/test_mrope_block.py::test_degenerate_3d_matches_partial_rope` enforces the bound (< 2 ULP).

### Vision encoder bf16 parity -- two cascading "wrong reference" bugs (2026-05-13)

This is the most subtle bug we've hit. It took TWO rounds of user pushback on the "this is bf16 noise" rationalization to fully unwind.

**Round 1 -- standalone .to(bf16) reference.**

* Symptom: `tests/test_qwen_vl_vit_forward.py` showed cos_min 0.92, mean 0.995 between flextrain encoder and HF `Qwen3_5VisionModel`. The test built HF as `Qwen3_5VisionModel._from_config(vc).to("cuda").to(torch.bfloat16)`. I framed this as "bf16 24-layer noise floor".
* User pushback (correct): cos 0.92 is too low for a 24-layer ViT — likely a real bug.
* Investigation: HF registers `inv_freq` as a NON-PERSISTENT buffer (`register_buffer(..., persistent=False)`). `.to(bf16)` casts the buffer (`persistent=False` only controls state_dict inclusion, NOT the `.to()` cast behavior). So under standalone-cast, HF's effective `inv_freq` is bf16. flextrain was using fp32 inv_freq, which mismatches HF's standalone reference.
* "Fix" (round 1): added a `dtype=` kwarg to `_build_vision_rotary_inv_freq`; encoder forward passed `dtype=cfg.compute_dtype` (bf16).
* Result on standalone test: byte-exact (cos=1.0000, abs=0.0). Looks great. **But this matched the wrong precision policy.**

**Round 2 -- production `from_pretrained` reference.**

* Symptom: even with the "byte-exact" encoder fix and two other engine-side bugs fixed, full-model loss parity was 1.70% rel_err. Per-layer LM forward diagnostic showed `pre_LM_input` cos_min=0.5847 — only image positions diverged (text positions were byte-exact). Encoder was producing wrong outputs in the full-model context despite being "byte-exact" in the standalone test.
* User pushback (correct): "1.70% gap seems too high; look at inter-layer residuals." This pushback was the trigger.
* Investigation: instrumented `Qwen3_5VisionRotaryEmbedding.inv_freq.dtype` in both contexts.
  - `Qwen3_5VisionModel._from_config(vc).to(bf16)` → `inv_freq` is **bf16** (standalone test path).
  - `AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype=bf16)` → `inv_freq` is **fp32** (production path).
* Why: `from_pretrained` calls `_init_weights` on the loaded model. For Qwen3_5VisionModel, `_init_weights` re-initializes the rotary buffer in fp32 AFTER the bf16 cast. So in production, the buffer ends up fp32 regardless of `torch_dtype`. The standalone `_from_config` path doesn't go through `_init_weights` post-cast, so the cast survives.
* Final fix: `_build_vision_rotary_inv_freq` defaults to fp32 (matching production HF), and the encoder forward passes `torch.float32` explicitly (not `cfg.compute_dtype`).
* Test updated to use `AutoModelForImageTextToText.from_pretrained(..., torch_dtype=bf16, attn_implementation="eager")` as the reference, NOT the standalone-cast variant. Asserts the loaded HF `inv_freq` is fp32 first, then runs the byte-exact comparison.

**Result:**
* Vision encoder standalone parity (vs production HF reference): **byte-exact** — cos=1.0000, abs=0.0.
* Full-model `pre_LM_input` parity: **byte-exact** — cos=1.0000, abs=0.0 (encoder+splice produce identical embeddings).
* Per-layer LM cos: 0.9999 (layer 0) → 0.9980 (layer 23) — pure bf16 accumulation through 24 LM layers.
* End-to-end single-image loss parity: **0.11% rel_err** (was 1.70% before this final fix; was 4.48% before all three bug fixes). Now BELOW the text-only LM floor (0.50%).

**Lessons:**
1. "This is bf16 noise" is a tempting conclusion that hides real precision-policy bugs. Any reference test should explicitly show what precision policy it's testing against. The 0.92 cos floor would have been a smoking gun for "different reference paths use different precision policies" if I'd checked `inv_freq.dtype` once in each path.
2. `_from_config(...).to(bf16)` is NOT a valid stand-in for `from_pretrained(torch_dtype=bf16)` for any model whose `_init_weights` does post-load buffer re-initialization. ALWAYS prefer `from_pretrained` for parity references; if you must use `_from_config`, document why and audit every non-persistent buffer.
3. Per-position diagnostics (`pre_LM_input` cos by token type) catch bugs that aggregate metrics smear over. Text positions at cos=1.0 with image positions at cos=0.58 is unmistakably "the encoder output is wrong in this context"; an overall cos=0.99 mean would have hidden it.

### Phase 1.5 -- 3-D MRoPE position generation in chunk preparation

Today chunk prep emits `(T, 1)` positions for every chunk. For multimodal forward parity vs HF we need per-token `(t, h, w)` derived from each image's `grid_thw`. HF reference: `Qwen3VLTextModel.get_rope_index`. Add this as a follow-up extension to `_populate_mm_chunk_extras` (or alongside it).

## Numerical / non-determinism notes from the 2026-05-12 sweep

For reference when reading future reverify diffs:

* 5/13 rows bit-exact vs baseline `runs/reverify_20260512_193355`.
* 7/13 rows tiny drift Δ 0.0002–0.0033 (all Qwen-family) — kernel non-determinism in partial-RoPE + QK-norm reductions. Unchanged from pre-existing flextrain behavior.
* 1/13 (`llama_3_1_8b_full`) hit a step-3 NaN sentinel. Steps 1-2 byte-identical to baseline; isolated re-run converged normally (Δ ≤ 0.0004 vs baseline). Transient bf16 instability, not a code regression.

The Qwen drift bound (≈ Δ 0.003 on the worst row) is the relevant threshold when reading future reverifies — anything bigger should be investigated.
