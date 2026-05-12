# Gemma 4 — remaining stages: design + todos

**Last updated:** 2026-05-12 (in-session working doc; live)
**See also:** `docs/internal/gemma4_status.md` (high-level status / handoff)

This doc is the active design + implementation notes for Stages 2-5.
The status doc tracks where we are; this doc spells out HOW the
remaining tests / harnesses should be built.

## Current state (recap)

Stages 0, 1a, 1b, 1c, 1d are done — block-level fwd+bwd parity vs
torch autograd reference, plus the HF arch loader + permute math.
**66 tests passing, 0 regressions.** Files on disk:

- `flextrain/nn/blocks/rope.py` (edited: proportional branch)
- `flextrain/nn/blocks/attention_gemma4.py` (created)
- `flextrain/nn/layers/gemma4.py` (created)
- `flextrain/io/arch/gemma4.py` (created)
- `flextrain/io/arch/__init__.py` (edited: registered)
- `tests/test_rope_proportional.py` (8 tests)
- `tests/test_gemma4_block_parity.py` (12 tests)
- `tests/test_gemma4_loader.py` (7 tests)
- `docs/internal/gemma4_status.md`, `gemma4_in_flight.md` (memory)

What this validates so far:
- ✅ Block math (fwd + bwd) for both sliding (V-norm + standard W_v
  path + full rope) and global (V-norm + k_eq_v fold + proportional
  partial rope) variants. cos > 0.998, sign > 0.95, rel_l2 < 8% on all
  weight grads.
- ✅ Recompute paths for save tiers 0/1/2/3 — `forward_recompute`
  correctly rebuilds dropped fields.
- ✅ HF config translation, per-layer block-config selection, optional
  `w_v` handling, permute math for full and partial rope.
- ✅ Proportional rope formula exactly matches HF.

What's NOT yet validated:
- ❌ End-to-end weight loading against the real 31B safetensors.
- ❌ Flextrain's full forward on 31B matches HF's reference forward.
- ❌ Gradient parity against HF on a real prompt + loss.
- ❌ A 5-step SFT training trajectory matches HF.

## Remaining stages

### Stage 2 — `tests/test_gemma4_31b_forward_parity.py`

**Goal**: Confirm flextrain's full forward on `Gemma-4-31B-Instruct`
matches HF's reference per-layer (and on the final logits) within
the Gemma-3 tolerances.

**Why this is non-trivial**: 31B in bf16 ≈ 62 GB. HF can't load the
whole model on one GPU (max consumer is 80 GB H100 / A100; many
have 24-48 GB). The test must reference HF *one layer at a time*.

**Design — "layerwise streaming HF" reference**:

```
1. Build flextrain via api.from_pretrained("models/Gemma-4-31B-Instruct").
   Host RAM ~62 GB; GPU ~ N_P × layer_size_bf16 (working set, ~few GB).
2. Pick a short prompt (T=48 tokens, same as Gemma 3 12B test).
3. Drive flextrain forward through ALL 60 layers; capture each
   block's output on CPU into a list ft_outputs[60].
4. Build HF reference layer-by-layer:
   a. text_cfg = AutoConfig.from_pretrained(model_dir).get_text_config()
   b. rope_module = Gemma4TextRotaryEmbedding(text_cfg).to("cuda")
      (constructed once — handles both sliding + full rope curves)
   c. embed_w = read_safetensor_tensor(
        f, "model.language_model.embed_tokens.weight"
      ).to("cuda")
   d. x_hf = embed_w[input_ids] * sqrt(hidden_size)
      # ^ Gemma's embed scale; matches HF's Gemma3TextScaledWordEmbedding
      # (modular_gemma4.py:1152 inherits from Gemma3).
   e. shared_kv_states = {}
   f. for L in range(text_cfg.num_hidden_layers):
        layer_type = text_cfg.layer_types[L]
        hf_layer = Gemma4TextDecoderLayer(text_cfg, layer_idx=L).to("cuda")
        _load_hf_layer_weights(hf_layer, L, model_dir)   # see helpers below
        position_ids = torch.arange(T).unsqueeze(0).to("cuda")
        cos, sin = rope_module(x_hf, position_ids, layer_type=layer_type)
        attn_mask = _build_attn_mask(T, layer_type, text_cfg)
        x_hf_in = x_hf.unsqueeze(0)  # add batch dim
        x_hf_out = hf_layer(
            hidden_states=x_hf_in,
            position_embeddings=(cos, sin),
            attention_mask=attn_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=None,
        )
        # HF returns either Tensor or tuple; unwrap.
        if isinstance(x_hf_out, tuple): x_hf_out = x_hf_out[0]
        x_hf_out = x_hf_out.squeeze(0)
        _compare(f"fwd[L{L}]", ft_outputs[L], x_hf_out, ...)
        x_hf = x_hf_out
        del hf_layer; gc.collect(); torch.cuda.empty_cache()
5. Final norm + lm_head + softcap:
   - final_norm_w = read_safetensor_tensor(f, "model.language_model.norm.weight") + 1.0
   - h_normed = rmsnorm_fp32(x_hf, final_norm_w, eps=text_cfg.rms_norm_eps)
   - logits_hf = h_normed @ embed_w.t()   # tied
   - if text_cfg.final_logit_softcapping:
         logits_hf = tanh(logits / cap) * cap
   - Compare flextrain's logits to logits_hf.
```

**Helpers to write**:

- `_load_hf_layer_weights(hf_layer, L, model_dir)`:
  Open safetensor index, find which shard each weight lives in,
  read by name, copy into the corresponding `hf_layer.*.weight` /
  `hf_layer.*.layer_scalar` slot. Handles:
  - `self_attn.{q,k,v}_proj.weight` (v_proj absent on global layers)
  - `self_attn.{q,k}_norm.weight`, `v_norm` has no weight (with_scale=False)
  - `mlp.{gate,up,down}_proj.weight`
  - `input_layernorm.weight`, `post_attention_layernorm.weight`,
    `pre_feedforward_layernorm.weight`, `post_feedforward_layernorm.weight`
  - `layer_scalar` (per-layer scalar buffer; **critical** — non-1.0 on
    the 31B Instruct checkpoint).

- `_build_attn_mask(T, layer_type, text_cfg)`:
  4D causal mask `(1, 1, T, T)` for SDPA path. For sliding layers,
  additionally mask positions where `i - j > sliding_window`.
  HF expects float mask with 0 at attend positions and -inf elsewhere
  (or equivalent boolean mask depending on implementation choice).
  Safer: use `torch.finfo(dtype).min` instead of literal -inf to dodge
  NaN propagation in softmax.

- `_attn_implementation`: HF defaults to SDPA on modern PyTorch.
  Force `text_cfg._attn_implementation = "eager"` for the layer-in-
  isolation case to avoid SDPA's stricter mask shape requirements and
  to match the simplest reference math.

**Risks + mitigations**:

| Risk | Mitigation |
|---|---|
| 62 GB host RAM needed for flextrain `from_pretrained` | Check via `psutil.virtual_memory()`; skip test if too low. Document the threshold. |
| HF `Gemma4TextDecoderLayer` constructor needs full text_cfg | Use `AutoConfig.from_pretrained(model_dir).get_text_config()` — already validated this path in Stage 1d. |
| `shared_kv_states` dict semantics surprising | num_kv_shared_layers=0 in 31B Instruct → `is_kv_shared_layer` is False for all layers → `shared_kv_states[layer_idx]` write happens only if `store_full_length_kv` is True (which depends on layer_type matching the LAST of its type before sharing starts). Since no sharing, this is mostly cosmetic. Initialize empty dict and let HF populate it. |
| flextrain forward might OOM on GPU at 31B working set | Match the user's existing `from_pretrained` defaults (force_saved_act_level / chunk size) — same path Gemma 3 12B test uses. T=48 is small enough that working set fits. |
| HF `Gemma4TextRotaryEmbedding(text_cfg)` registers buffers we don't control | Set requires_grad=False on all params/buffers after construction. For init: HF's `__init__` calls the rope init function eagerly so we should get correct inv_freq buffers without needing `_init_weights`. |
| `layer_scalar` buffer alignment | flextrain's Gemma4Block.layer_scalar is a Python float (not a tensor); HF's is `register_buffer("layer_scalar", torch.ones(1))`. Both apply the same multiply at end-of-layer. Loader writes the safetensor value into both sides identically. Already covered by Stage 1d. |
| HF returns either Tensor or `BaseModelOutputWithPast` | The layer's forward returns just a Tensor (no caching path). Defensive `if isinstance(..., tuple): x = x[0]`. |
| Per-layer mask construction matches HF | Worth replicating from a tiny `Gemma4TextModel` forward run (T=8, eager) to confirm the mask shape and value convention. |

**Tolerances**: same as Gemma 3 forward parity test
(`tests/test_gemma3_full_forward_parity.py`):
- per-layer hidden state: `cos > 0.999, rel_l2 < some-loose-bound`
- final logits: `cos > 0.999`

**Test runtime**: dominated by 60 × (HF layer construct + safetensor
read for that layer + 1 fwd pass at T=48). Maybe 1-3 minutes total.
Mark `@pytest.mark.slow` so it's opt-in.

**Test parametrization**: just one test at first
(`test_gemma4_31b_forward_parity`). Can add T=128 / longer prompt
variants later. The block parity already covers the math; this test
is the integration gate.

### Stage 3 — `tests/test_gemma4_31b_grad_parity.py`

**Goal**: Confirm flextrain's gradient on 31B matches HF's reference
per-parameter, on the same short prompt as Stage 2.

**Why this is non-trivial**: HF `model.backward()` materializes
grads for all 31B params → ~62 GB of grad tensors → certain OOM. Even
on CPU, full HF fwd+bwd at this scale would take a long time and need
~120 GB RAM (params + grads).

**Design — "layerwise reverse-mode HF rebuild"**:

```
1. Run flextrain fwd → CE loss on (input_ids, labels=input_ids).
2. Run flextrain bwd, stashing per-layer:
     - x_in_L (the hidden state input to layer L; available from slot.x_inp pre-bwd)
     - dx_in_L (the grad coming INTO layer L during bwd; captured at the
       start of block.backward)
     - per-param grads g_W (already computed; copy to CPU)
3. For each layer L in reverse:
     a. Build ONE HF Gemma4TextDecoderLayer (same as Stage 2).
     b. Load its weights from safetensor + set requires_grad=True on all params.
     c. Build position_embeddings / mask exactly as in Stage 2.
     d. y = hf_layer(x_in_L.unsqueeze(0), position_embeddings=..., ...)[0].squeeze(0)
     e. y.backward(dx_in_L)   # populates .grad on every param
     f. Collect q_proj.weight.grad, etc., map to flextrain layout via
        _map_hf_grad_to_ft_layout (existing helper from
        tests/test_gemma3_full_fwd_bwd_parity.py:370-405, extended for
        partial-rotated-channels permute and absent w_v on globals).
     g. Compare cos / sign / rel_l2 against flextrain's per-layer grads.
     h. del hf_layer; gc; cuda.empty_cache.
4. Embedding + final-norm grad: small enough to do as a CPU HF
   fwd+bwd on a synthetic tiny stack (or a single-prompt T=8 dummy
   that uses the real embed + norm but no decoder layers). Compare
   to flextrain's accumulated embed.grad / final_norm.grad.
```

**Helpers to reuse from Gemma 3**:
- `_diffstats`, `_compare` (tolerances `cos > 0.98, rel_l2 < 0.2`)
- `_map_hf_grad_to_ft_layout` — needs extension:
  - Skip `w_v` mapping on global (k_eq_v=True) layers.
  - For Q/K wgrads on global layers: only the rotated channels need
    the inverse pair-interleave→halved permute. Non-rotated tail
    channels passed through in natural HF order during load, so the
    grad's natural HF order is correct as-is at those positions.

**Tier choice**: This test uses save_tier=max so all activations are
saved (no recompute path exercised). The recompute paths are already
covered by `test_gemma4_block_recompute_then_backward_parity` × tier 0-3.

**Risks**:

| Risk | Mitigation |
|---|---|
| flextrain bwd captures + CPU offloads per-layer × 60 ≈ many GB | Save selectively — only x_in_L and dx_in_L are needed, both T × d_model bf16 = ~0.5 MB each per layer. Per-param grads can stay on the host master copy that flextrain produces. |
| HF backward at one layer might still allocate intermediate activations | T=48, one layer → kilobytes of activations. Fine. |
| Subtle disagreement on what "dx into layer L" means | flextrain captures via `block.backward(dx, ...)` entry. HF gives grad via the layer's `forward_hook` -> `backward_hook` -> `grad_output[0]`. Both are dL/d(layer-L output). Same Gemma 3 invariant — see `tests/test_gemma3_full_fwd_bwd_parity.py:139-146`. |
| layer_scalar grad: HF doesn't have one (it's a buffer); flextrain doesn't either | Confirm both treat as constant. Already handled in Stage 1d. |

**Tolerances**: same as Gemma 3 full fwd+bwd parity
(`tests/test_gemma3_full_fwd_bwd_parity.py:65-78`):
- `PARITY_COS_TOL = 0.98`
- `PARITY_SIGN_TOL = 0.92`
- `PARITY_REL_L2_TOL = 0.2`
- `LOSS_REL_TOL = 5e-3`
- Tiny-grad escape: when `ref_scale < 1e-4`, only require `cos > 0.5`
  (γ vectors with near-zero grad).

### Stage 4 — 5-step SFT smoke

**Goal**: End-to-end engine integration check. Confirm flextrain can
load 31B, take 5 SFT steps, and the loss curve looks sane vs HF.

**Design**: Mirror `runs/verified_gemma_rerun2/`.

- Pick 5 short prompts from MathInstruct (same set Gemma 3 used).
- `from_pretrained("models/Gemma-4-31B-Instruct", lora_targets=(...), ...)` —
  LoRA-only so we don't need to allocate full-FT optimizer state for 31B.
- Run 5 steps with `am.fwd_bwd` + `am.step` (AdamW).
- Compare per-step loss to HF reference (CPU run with same LoRA config
  via PEFT, same prompts and seed). Allow same drift as Gemma 3 12B
  LoRA (~10% rel by step 5, decreasing with smaller LR).

**Risks**:
- Memory: 31B base + LoRA adapters + AdamW state for LoRA only ≈ 70 GB
  host. Fits if user's machine has ≥96 GB system RAM.
- LoRA targets: same 7-projection set as Gemma 3 (`w_q, w_k, w_v, w_o,
  w_1, w_2, w_3`). For global layers, `w_v` is absent — LoRA wrapper
  must skip that target on those layers. Test by inspecting
  `am.backbone[L].lora_targets` for L ∈ {sliding, global}.

### Stage 5 — drop gemma skip in `tests/test_arch_parity.py:406-408`

One-line change. Re-runs the cross-arch parity sweep including Gemma 4.
Only meaningful after Stages 2-4 are all green.

## Open design questions

1. **Should `Gemma4Block.layer_scalar` be a tensor instead of a Python
   float?** Currently a float. A tensor would let it move with the
   layer between host/device and survive serialization. Float is
   simpler; works because we read its value once at load time and use
   it as a constant. Decision: keep as float unless a future use case
   needs a tensor.

2. **LoRA on `w_v_norm`?** It doesn't exist as a parameter (V-norm has
   no γ). LoRA wrapper iteration naturally skips it. No action needed.

3. **Stage 4 lr=0 baseline?** Gemma 3 included an `lr=0` LoRA run to
   isolate forward parity from optimizer-side bf16 noise (see
   `docs/internal/gemma3_status.md` §"SKEPTICISM..."). Do the same on
   31B if Stage 4 shows surprising drift.

4. **Test ordering**: Run Stage 2 first. Only proceed to Stage 3 once
   Stage 2 is fully green. Stage 3 depends on flextrain's forward
   matching HF; if forward is off, gradient parity will too.

## Working order for the rest of this session

1. ☐ Write a smoke test that actually runs `from_pretrained` on
   `Gemma-4-31B-Instruct` with `load_weights=False` first (validates
   the arch loader path end-to-end without needing the safetensors
   in RAM). If `load_weights=False` mode isn't supported, skip and
   move to step 2.
2. ☐ Implement `_load_hf_layer_weights` and `_build_attn_mask`
   helpers in a new test file.
3. ☐ Implement the layerwise streaming HF harness in
   `tests/test_gemma4_31b_forward_parity.py`. Start with a CPU-only
   variant on a tiny synthetic Gemma 4 config (T=16, n_layers=4,
   d_model=64) to validate the harness math — this gives us a fast
   feedback loop before unleashing on 31B.
4. ☐ Run on real 31B safetensors with T=48.
5. ☐ If memory becomes a problem, document the threshold and skip
   the heavy test on under-resourced machines.
6. ☐ Update `gemma4_status.md` to mark Stage 2 done.

## Notes on tooling

- `safe_open(path, framework="pt")` is the safetensors entry. Get
  `f.get_tensor(name)` for one tensor at a time — never loads the
  full shard into RAM.
- HF's `Gemma4TextDecoderLayer` lives at `transformers.models.gemma4.modeling_gemma4`.
  `Gemma4TextRotaryEmbedding` is in the same module.
- `am.backbone[L].base` unwraps the LoRA wrapper when present;
  plain `am.backbone[L]` when not.
- The HF safetensor index is at
  `models/Gemma-4-31B-Instruct/model.safetensors.index.json`. Already
  confirmed in Stage 1d that this maps every tensor to its shard.
- `am._hf_source_path` is set by `am.load_hf` and reused by
  `post_load_permute`. We can also read it ourselves in tests.
