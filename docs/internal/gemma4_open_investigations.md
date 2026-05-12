# Gemma 4 — open investigations, suspicions, and architectural nuances

**Last updated:** 2026-05-12
**Companion to:** `gemma4_status.md` (high-level status),
`gemma4_design_notes.md` (Stage 2/3 design)

This is the working notes file for **what we don't yet trust**. The
high-level status doc tracks progress; this file tracks open bugs,
suspicions, untested assumptions, and Gemma-4-specific nuances I've
discovered that could be (or are) sources of bug.

Treat this as the authoritative "what's still wrong" log. Update it
after every investigation, even — especially — if the result is
"we ruled this out."

## Current symptom

`tests/test_gemma4_31b_forward_parity.py` runs end-to-end against the
real 31B safetensor (60 layers, 100s runtime). With **tight thresholds
(cos ≥ 0.99)**, the test fails. Per-layer cos profile after the
already-landed fixes (RMSNorm convention, x-mutation, SDPA scale=1.0):

| Layer slice | cos range | sign_match | rel_l2 |
|---|---|---|---|
| L0 sliding | 0.944 | 0.85 | 0.34 |
| L1–L4 sliding | 0.99–0.9999 | 0.84–0.97 | 0.014–0.14 |
| L5 global | 0.975 | 0.86 | 0.25 |
| L6–L10 sliding | 0.99+ | 0.93–0.97 | 0.02–0.07 |
| L11–L29 globals (every 6th) | 0.95–0.99 | 0.70–0.91 | 0.13–0.37 |
| Mid sliding | 0.99+ | 0.90–0.97 | 0.04–0.10 |
| L41 global | 0.92 | 0.73 | 0.40 |
| L47 global | 0.94 | 0.79 | 0.33 |
| L53 global | **0.70** | 0.71 | 0.81 |
| L57, L58 sliding | 0.96–0.97 | 0.94 | 0.22–0.27 |
| L59 global (last) | **0.55** | 0.62 | 1.22 |

**For comparison:** Gemma 3 4B/12B forward parity gets `cos > 0.999`
across the full 26–48 layer stack with the same `_diffstats` /
`_compare` helpers. The bf16-noise floor at this depth is **not**
cos=0.94. Something is genuinely off in Gemma 4.

## High-suspicion open items (most likely bugs)

### 1. Attention scaling on the flash path (sliding layers — HIGH)

**Suspicion**: this is the dominant cause of per-layer drift on sliding
layers and probably a contributor on globals too.

HF Gemma 4 sets `Gemma4TextAttention.scaling = 1.0`
(`modular_gemma4.py:920`). The eager attention path multiplies scores
by exactly this value (`eager_attention_forward` in
`modeling_gemma4.py:779`), OVERRIDING the default `1/sqrt(head_dim)`
factor. The model's q_norm / k_norm γ vectors encode the effective
scaling instead:

* L0 (sliding, head_dim=256): q_norm γ ≈ 1.02 (uniform scalar across
  256 channels). k_norm γ ≈ 0.122 (uniform). Effective Q·K scaling ≈
  0.122 (in the k_norm γ).
* L5 (global, head_dim=512): q_norm ≈ 1.02, k_norm ≈ 0.062. Effective
  ≈ 0.062.
* All q_norm / k_norm tensors for Gemma 4 31B are **uniform scalars**
  (a single value broadcast across head_dim) — not per-channel γ. This
  was a surprise; check
  ``model.language_model.layers.{L}.self_attn.{q,k}_norm.weight``
  in the safetensor — they're length-`head_dim` but every entry
  equals the same scalar.

flextrain's **flash kernel** (sliding layers, `flextrain_attention_fwd`
in `flextrain/ops/_kernels/attention.py:629`) hardcodes
`softmax_scale = q.shape[-1] ** -0.5`. There's no kwarg to override it.
This applies an EXTRA `1/sqrt(head_dim)` factor on top of what's in
k_norm γ:

```
HF effective scale = γ_k ≈ 0.122
FT effective scale = γ_k × (1/√d) ≈ 0.122 × 0.0625 ≈ 0.0076 for sliding
```

Ratio: ~16× softer softmax on FT side for sliding. This is a
substantial divergence in attention pattern. Softmax-saturation argues
against it being catastrophic (the argmax-attended position is often
the same regardless of temperature), but the OFF-PEAK attention mass
does differ, which propagates noise into the residual stream.

flextrain's **SDPA fallback** (globals, head_dim=512) was updated to
pass `scale=1.0` matching HF's behavior. Confirmed via
`F.scaled_dot_product_attention(..., scale=1.0, enable_gqa=True)`.

**What to try next:**

* (a) Modify the flextrain attention fwd/bwd to accept an explicit
  `softmax_scale` kwarg (touches `flextrain/ops/_kernels/attention.py`
  — SHARED with other archs). Default keeps current `1/√d` so existing
  callers unaffected. Then `Gemma4AttentionBlock` passes `softmax_scale=1.0`.
* (b) Stay local to Gemma4: pre-scale `slot.xq` by `sqrt(head_dim)`
  before the kernel call, scale it back after. Two bf16 multiplies per
  layer; trivial cost. But slot.xq is saved for bwd — need to ensure
  bwd sees the un-scaled value. Cleanest: scale into a scratch tensor
  and pass scratch to the kernel.
* (c) Don't fix the scaling — the q_norm/k_norm γ values pre-scale Q/K
  enough that softmax saturation handles most of the divergence. This
  is what's happening now and it's producing cos≈0.94. **Not good
  enough** per the user (correctly: Gemma 3 hits cos > 0.999).

**Open question for option (a)**: does the flash kernel chain
(`_flash4_fwd`, `_flash3_fwd`, `_flash2_fwd`, `_eager_attention_fwd`,
each at lines 195+, 236+, 275+, 313+ of `attention.py`) plumb
`softmax_scale` from `flash_attn_varlen_func`? Need to check. The
underlying flash-attn library DOES expose a `softmax_scale` parameter;
flextrain just doesn't surface it.

### 2. L59 catastrophic drop (cos=0.55) — HIGH

The drop from L58 (cos=0.93) to L59 (cos=0.55) is a single-step
collapse, not gradual drift. L59 is the LAST layer, the LAST global,
and has the smallest `layer_scalar=0.0364`. Things to check:

* **Is L59 supposed to write `shared_kv_states[59]`?**
  Yes: `store_full_length_kv = True` because L59 is the last
  full_attention index. HF writes; flextrain doesn't. **This is fine**
  because Gemma 4 31B has `num_kv_shared_layers=0` (no layer ever
  READS shared_kv_states), so the write is dead code. Verified.

* **Is the layer_scalar amplifying small relative errors?** With
  `scalar=0.036`, the output magnitude is ~0.036× the unscaled
  output. If FT and HF agree to within ε at the pre-scalar stage,
  the post-scalar agreement is still ε in absolute terms but ε/0.036
  in relative terms = 27× worse rel_l2. **But cos is scale-invariant.**
  So cos shouldn't drop just from the small scalar.

* **Hypothesis**: the small layer_scalar means L59's contribution to
  the residual is small (~3.6%); the layer's job is essentially to
  pass through. Cumulative drift from earlier layers dominates. But
  L58 has cos=0.93 (already drifted) and applies its own scalar
  (0.75). The drop to 0.55 isn't fully explained by drift alone.

* **Untested**: is there something special about reading/writing
  `shared_kv_states` that affects the LAYER's output? Looking at
  HF's flow: the dict is updated but not read by L59's own attention.
  Should be irrelevant.

* **Untested**: is the SDPA path correct for L59 specifically? L59 is
  the 10th global layer. Same code path as L5, L11, …, L53. cos
  progressively degrades through globals (L5=0.97 → L59=0.55). This
  suggests cumulative drift via the SDPA path, not an L59-specific bug.

**To investigate**: compare FT pre-scalar L59 output to HF pre-scalar
L59 output. If pre-scalar disagrees significantly, the math is wrong;
if pre-scalar agrees and only post-scalar diverges, it's a scaling
artefact. This is the next concrete debugging step.

### 3. RMSNorm γ × x precision (medium suspicion)

HF Gemma 4 RMSNorm does the γ multiply in fp32 then casts to bf16
(`modeling_gemma4.py:182-186`). flextrain's
`flextrain_rmsnorm_fwd` does the multiply in bf16 (verified
empirically: kernel output bit-matches a manual `(x_fp * rstd).to(bf16) * w_bf16`,
NOT a manual `((x_fp * rstd) * w_fp32).to(bf16)`).

For Gemma 4 31B some γ channels in `input_layernorm` are HUGE
(observed max=105 at one position; uniform per-tensor γ for q_norm /
k_norm are smaller, around 1.0 / 0.06–0.12).

Practical impact: bf16 multiply of γ=105 × x_normed introduces ~1 ULP
of error per channel relative to fp32 multiply. Compounds over 60
layers and 4 norms per layer. Probably a contributor to drift but not
the dominant cause of cos=0.94.

**Open question**: should we use fp32 multiply in our RMSNorm kernel?
Would touch shared code; affects all archs. Or: add a Gemma 4-specific
RMSNorm path. Punt for now.

### 4. q_norm / k_norm uniform-scalar γ assumption (low-but-real)

When I dumped Gemma 4's q_norm / k_norm γ tensors, every entry
equals the same scalar (`q_norm.weight[i] == q_norm.weight[0]` for
all i). This was a surprise — typically RMSNorm has per-channel γ.

If the model was actually trained with a scalar (not vector) γ, then
the safetensor storing it as a length-`head_dim` vector is wasteful
but harmless. flextrain's RMSNorm treats it as a vector and applies
per-channel multiply — same numerical effect.

**Open question**: is this a Gemma 4 design choice (q/k_norm are
explicitly scalar in the source code somewhere)? Or is it
trained-into-the-weights coincidence? Doesn't matter for forward, but
matters for backward: g_q_norm grad reduces to a single scalar in
either case; per-channel γ would diverge under further training while
scalar γ wouldn't.

**Action**: ignore for now. We don't lose anything by treating it as
per-channel γ. Note for Stage 3 grad parity: the per-channel grad
should still match HF (HF treats it as per-channel too).

## Medium-suspicion items (potential bugs, harder to localize)

### 5. Proportional partial-RoPE math (medium)

For Gemma 4 global layers (full_attention, rope_type="proportional",
partial_rotary_factor=0.25, head_dim=512), we rotate channels
`[0, 128)` and pass through channels `[128, 512)`. The inv_freq is
built via `flextrain/nn/blocks/rope.py:build_partial_rope_inv_freq`
with `head_dim=512` (proportional denominator) — numeric-exact against
HF's `_compute_proportional_rope_parameters` per
`tests/test_rope_proportional.py` (8 tests pass).

flextrain's pair-interleave layout vs HF's halved layout: my
`post_load_permute` (and the test's `_load_layer_weights_gemma4`)
permutes the rotated channels of W_q / W_k. Specifically:

```
FT layout per head: [pair0_re, pair0_im, pair1_re, pair1_im, ...,
                     pair_{rope_angles-1}_im, hf_passthrough_chunk_a, hf_passthrough_chunk_b]
```

Where:
* The rotated prefix `[0, 2*rope_angles)` has FT[2i]=HF[i],
  FT[2i+1]=HF[half + i] for i in `[0, rope_angles)`.
* The non-rotated suffix has HF positions `[rope_angles, half) ∪ [half + rope_angles, head_dim)`
  in natural HF order.

`tests/test_gemma4_loader.py::test_partial_halved_to_pair_perm_partial_31b_global`
verifies the permute is a valid bijection and that the rotated prefix
maps to the right HF positions. **What's NOT yet verified**: that the
NUMERICAL output of FT's `apply_rope_partial_fwd(slot.xq, ..., rot_dim=128)`
on permuted weights matches HF's `apply_rotary_pos_emb(query_states, cos, sin)`
on un-permuted weights, for a real Q tensor.

**Action**: write a focused numerical check. Take random Q, K. Apply
FT's permute + flextrain partial-rope. Apply HF's halved-rope on
un-permuted. Compare element-wise after un-permuting FT back to
halved layout. **This is the most likely undetected math bug.**

### 6. Sliding-window mask construction (low)

HF Gemma 4 sliding attention uses `sliding_window=1024`. With T=23 in
the test, the window doesn't actually clip anything (every causal
position is within 1024 of i). So sliding window correctness isn't
exercised by this test. The mask construction in
`_build_attn_mask_for_layer` would matter for T > 1024.

**Action**: not blocking Stage 2. Note for Stage 4 (longer prompts)
or anyone running the test at higher T.

### 7. HF attention implementation choice (low)

We force `text_cfg._attn_implementation = "eager"` on the outer
text_cfg before constructing each HF layer. This should give us the
pure-PyTorch attention path. We rely on this; if HF silently picks
SDPA or flash internally, mask shape and scaling behavior might
differ.

**Action**: verify HF actually uses eager. Could add an assertion in
the test that the layer's `_attn_implementation` is `"eager"` after
construction.

### 8. shared_kv_states dict accumulating across layers (low)

The test passes a single `shared_kv_states = {}` dict and reuses it
across all 60 layer calls. With `num_kv_shared_layers=0`, no layer
READS from this dict. Layers L=58 (sliding) and L=59 (global) WRITE
to it (their `store_full_length_kv=True`). Both writes happen but are
never read.

**Action**: not a correctness issue for 31B Instruct. Note for any
future Gemma 4 model with `num_kv_shared_layers > 0`.

## Lower-suspicion (likely correct, but document for completeness)

### 9. Token embedding scaling

flextrain scales the embedding output by `sqrt(d_model) = sqrt(5376) ≈ 73.3`
to match HF's `Gemma3TextScaledWordEmbedding` (which Gemma 4 inherits via
`Gemma4TextScaledWordEmbedding`). Implemented in the test driver
(`x_input = (embed_w[input_ids].float() * sqrt(d_model)).to(bf16)`).
**Same as Gemma 3 — works there with cos > 0.999**, so this isn't the
Gemma-4-specific bug.

### 10. Causal mask

4D additive `(1, 1, T, T)` with `torch.finfo(fp32).min` at disallowed
positions. Used by HF eager attention. Standard pattern from
Gemma 3 forward parity test. Verified by the high cos values on most
sliding layers.

### 11. layer_scalar application order

HF: `hidden_states *= self.layer_scalar` AFTER `residual + post_ffn_layernorm(ffn_out)`.
FT: same in `Gemma4Block.forward` final lines.
Order matches.

### 12. tie_word_embeddings

Gemma 4 has `tie_word_embeddings=True`. The lm_head shares weights
with the token embedding. Handled in arch loader's `post_load_permute`
(mirror embed.t() into w_head_proj when head is zeros). Verified at
small scale by the loader tests.

## Changes that landed during Stage 2 bring-up (cumulative)

1. `flextrain/nn/blocks/attention_gemma4.py`:
   * **SDPA fallback** for `head_dim > 256` (flash kernel limit).
     Module-local; no changes to shared `GQAAttentionBlock`.
   * **SDPA passes `scale=1.0`** matching HF's
     `Gemma4TextAttention.scaling = 1.0`. (Flash path still uses
     hardcoded `1/√d` — open item #1.)
   * **`bwd` raises NotImplementedError when SDPA path was used**
     (eager-bwd via autograd graph stitch deferred to Stage 3).

2. `flextrain/io/arch/gemma4.py:_gemma4_post_load_hook`:
   * **No `+1` γ shift** — Gemma 4's `Gemma4RMSNorm` uses
     `output * weight` directly (vs Gemma 3's `output * (1 + weight)`),
     so the safetensor stores canonical γ. The original hook copy-pasted
     Gemma 3's `+1` shift and was wrong. The hook is now a no-op kept
     for symmetry.

3. `tests/test_gemma4_31b_forward_parity.py`:
   * `_load_layer_weights_gemma4` no longer applies `+1` to RMSNorm γ.
   * `_load_global_weights_gemma4` no longer applies `+1` to final_norm γ.
   * `_drive_flextrain_one_layer` **clones `x_in`** before calling
     `block.forward` because the block mutates its input
     (FFN passes `out_tensor=x` to reuse the residual buffer). In
     production this is fine — the engine doesn't re-read the layer
     input post-call — but the parity test feeds the same `x` to both
     FT and HF.
   * `_build_hf_layer_isolated` **stopped copying text_cfg** via
     `__class__(**to_dict())`. The round-trip drops the
     leading-underscore `_attn_implementation` attribute, putting the
     rope module and the layer out of sync.
   * **Tolerances are tight (`cos ≥ 0.99`)** — we will not loosen
     these. The test failing is the signal.

4. `flextrain/api.py`:
   * Added `Gemma4ForCausalLM` and `Gemma4ForConditionalGeneration` to
     `_ARCH_MODULE_OVERRIDES` so `from_pretrained` routes to
     `flextrain/io/arch/gemma4.py`.

## Gemma 4 architectural nuances (recorded so we don't re-discover)

### RMSNorm

* `Gemma4RMSNorm.forward(x)`: `(_norm(x.float()) * weight.float()).type_as(x)` — fp32 normalize, fp32 multiply by γ, cast to input dtype. With_scale init=ones.
* No `1+weight` trick (unlike Gemma 3). Safetensor stores canonical γ.
* `v_norm` is `Gemma4RMSNorm(with_scale=False)` — no γ, RMS division only.

### Attention

* `Gemma4TextAttention.scaling = 1.0` hardcoded. Overrides eager's default `1/sqrt(head_dim)`.
* q_norm / k_norm γ are stored as length-`head_dim` vectors but in the
  Instruct checkpoint they're uniform scalars (all entries equal).
  q_norm value ≈ 1.02 across the model; k_norm value 0.06–0.12 depending
  on layer (smaller on globals).
* V-RMSNorm is `with_scale=False` — γ-free; just RMS-divides V's last axis per-head.

### Attention head dim split

* Sliding layers: `head_dim = config.head_dim = 256`,
  `num_key_value_heads = 16`.
* Global layers (full_attention): `head_dim = config.global_head_dim = 512`,
  `num_key_value_heads = config.num_global_key_value_heads = 4`.
* The same `self.head_dim` (set at __init__) is used for Q, K, AND V
  reshape — Q, K, V all have the same per-head dim per layer. V-dim is
  NOT independently configurable (no MLA-style asymmetry).

### k_eq_v (global layers, num_global_key_value_heads != num_key_value_heads)

When `config.attention_k_eq_v=True` AND layer is `full_attention`:

* `v_proj` is `None` — no W_v parameter exists for this layer.
* `value_states = key_states_pre_norm` (the W_k @ x output, BEFORE k_norm and rope).
* The same `key_states_pre_norm` tensor then has k_norm and rope
  applied to become `key_states_post_norm_post_rope`.
* `value_states` goes through `v_norm` (no γ) → final V.
* So K-path and V-path BOTH read from the SAME pre-norm tensor but
  apply DIFFERENT normalizations downstream.

### Per-layer-type RoPE

* Sliding: `rope_type="default"`, `rope_theta=10_000`, full rotation
  over head_dim=256.
* Global: `rope_type="proportional"`, `rope_theta=1_000_000`,
  `partial_rotary_factor=0.25` → rotate `int(0.25 × 512 // 2) = 64`
  channel pairs (128 channels), pass through remaining 384.
* Proportional rope inv_freq formula: `base ** (-2i / head_dim)` for
  i in `[0, rope_angles)`, padded with zeros to length head_dim/2.
  Note `head_dim` in the denominator, NOT `rot_dim`. flextrain's
  `build_partial_rope_inv_freq` supports both (default uses rot_dim;
  proportional uses head_dim via the new `head_dim` kwarg).

### layer_scalar

* `Gemma4TextDecoderLayer.layer_scalar` is `register_buffer("layer_scalar", torch.ones(1))` —
  a buffer, not a Parameter. NOT in HF's `named_parameters()` and
  NOT trained by the optimizer.
* In the 31B Instruct checkpoint, layer_scalar values are NON-trivial
  (e.g. L0=0.0894, L1=0.0654, L29=0.555, L59=0.0364). Some get small,
  some stay near 1. CANNOT be defaulted to 1.0.
* Multiplied at the very END of each decoder layer's forward, after
  the FFN residual add.
* flextrain stores it as a Python float on `Gemma4Block.layer_scalar`;
  loaded via `Gemma4Block.set_layer_scalar(value)` in arch.gemma4's
  `post_load_permute` (which has access to `am.backbone[L]` and the
  safetensor path via `am._hf_source_path`).
* No grad path: HF doesn't differentiate through `layer_scalar`
  because it's a buffer. flextrain doesn't either.

### Multimodal wrapper prefix

* `Gemma4ForConditionalGeneration` (the public 31B / 26B-A4B class)
  saves text weights under `model.language_model.layers.{i}.*`. NOT
  `language_model.model.layers.{i}.*` (the Gemma 3 4B/12B convention).
  Different wrapping order.
* `model.embed_tokens.weight` is at `model.language_model.embed_tokens.weight`.
  `model.norm.weight` at `model.language_model.norm.weight`.
* Vision tower weights at `model.vision_tower.*` and
  `model.embed_vision.*` are ignored at load time (the loader's
  "leftover" diagnostic lists them but they're not in our weight map).

### Final logit softcap

* `final_logit_softcapping = 30.0`. Applied as
  `logits = tanh(logits / 30) * 30` after the LM head.
* `attn_logit_softcapping` is `None` (disabled). Gemma 2 had 50.0;
  Gemma 4 turned it off.

### Knobs in the config that are inactive for 31B Instruct

These exist in `Gemma4TextConfig` and the modular source treats them
as live branches, but the Instruct checkpoint disables them. flextrain
should ASSERT they're inactive when loading, so we don't silently
half-support them:

* `num_kv_shared_layers = 0` — no layer reads `shared_kv_states`.
* `hidden_size_per_layer_input = 0` — no Per-Layer Embeddings (PLE).
* `use_double_wide_mlp = False` — KV-shared layers don't get a 2× MLP.
* `enable_moe_block = False` — no per-layer MoE branch (the parallel
  experts-with-dense-FFN topology Maarten's article describes).
  This is True for the 26B-A4B variant which is OUT OF SCOPE.
* `use_bidirectional_attention = "vision"` — text path stays causal.

## What we've verified (so we don't re-verify)

* RMSNorm kernel produces bit-exact output for the bf16-multiply variant
  on real Gemma 4 weights (verified element-by-element vs manual fp32
  baseline; the discrepancy with HF is the bf16 vs fp32 multiply
  precision, not a kernel bug).
* `build_partial_rope_inv_freq` with `rope_type="proportional"` matches
  HF's `_compute_proportional_rope_parameters` element-exact (8 unit tests).
* `_partial_halved_to_pair_perm` is a valid bijection and reduces to
  Gemma 3's full-rope permute when `rope_angles = head_dim/2`.
* Block-level fwd + bwd parity at small dims is tight (cos > 0.998 on
  every gradient; covers V-norm + k_eq_v + proportional partial rope
  + dual-residual + 4 save tiers).
* flextrain's `from_pretrained` builds the 31B backbone correctly
  (smoke test in `tests/test_gemma4_loader.py`).
* The `_drive_flextrain_one_layer` driver mutates its input via the
  block's FFN call — now cloned in the driver.
* HF `Gemma4RMSNorm` uses `weight` directly (no `+1`); confirmed by
  reading the generated `modeling_gemma4.py` source AND by element-
  matching HF's `input_layernorm(x)` to a manual fp32 RMSNorm with γ
  applied directly.

## Tests that EXIST and are passing

* `tests/test_rope_proportional.py` (8) — proportional rope math vs HF.
* `tests/test_gemma4_block_parity.py` (12) — block fwd + bwd parity
  at small dims, cos > 0.998 on every gradient.
* `tests/test_gemma4_loader.py` (7) — config translation, per-layer
  builder, permute math, ArchSpec registration.

## Tests that EXIST and are currently FAILING

* `tests/test_gemma4_31b_forward_parity.py` (1) — passes the
  end-to-end pipeline but per-layer cos drops below the
  `cos ≥ 0.99` threshold on many layers. **This is the next
  investigation.**

## Suggested investigation order (when next session picks up)

1. **Verify rope math on a real layer.** Take random Q. Apply FT's
   permute + flextrain partial-rope. Apply HF's halved-rope on
   un-permuted. Un-permute FT back. Element-equal? If no, that's a
   sub-cos-1.0 source confirmed.
2. **Add `softmax_scale` kwarg to `flextrain_attention_fwd`** (and
   the corresponding bwd path). Default keeps current `1/√d` for
   non-Gemma-4 callers (shared-block-safe). Wire from
   `Gemma4AttentionBlock` with `softmax_scale=1.0`. Re-run; expect
   sliding cos to jump significantly (probably > 0.999).
3. **Compare FT pre-scalar vs HF pre-scalar at L59.** Isolates whether
   the cos=0.55 is a math bug at L59 or a magnification artefact of
   the small layer_scalar.
4. **If still residual drift at globals**: investigate SDPA backend
   selection (math vs memory-efficient kernels can differ in bf16
   precision); confirm HF eager is what's running (assertion-check
   `text_cfg._attn_implementation`).
5. **If everything is right but cos < 0.999**: bf16 vs fp32 γ multiply
   in RMSNorm. Decide whether to switch the kernel or accept the
   noise floor.

## What this session's work is NOT yet

* Production-ready forward at 31B (still has correctness gaps).
* Backward parity at 31B scale (Stage 3, blocked on Stage 2 being clean).
* End-to-end smoke / 5-step SFT (Stage 4, blocked on Stages 2+3).

## What this session's work IS

* A complete, correctly-wired forward path at small dims (block parity
  tight, validated against autograd reference).
* A complete arch loader registered in `ARCH_MODULES`.
* A 31B integration test scaffold that runs end-to-end. Failing at
  cos > 0.99 — the failure mode is informative, not silent.
* This notes file: the open-issue inventory so the next session
  doesn't waste cycles re-deriving what's been ruled out.
