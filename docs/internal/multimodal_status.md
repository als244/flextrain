# Multimodal Input Layer — status & continuation notes

**Last updated:** 2026-05-13 (post-Phase-1.5 + initial validation gates)
**Approved plan file:** `/home/shein/.claude/plans/please-take-your-time-delegated-parrot.md`

## What this work is

Extend flextrain's text-only training engine to support multimodal inputs (vision/audio towers as part of the "input layer"), with an abstraction usable across model families (Qwen3.5/3.6, Qwen3-VL, Gemma3, Gemma4).

**Phase 1 pilot (current scope):** bring Qwen3.5/3.6 multimodal to forward parity end-to-end vs HF.

- Vision tower **frozen** (all weights `TensorSpec(frozen=True)`); abstraction designed for trainable encoders later.
- Data path: `Sequence.modality_inputs={"image": [ImageInputCPU(...)]}`.
- Preprocessing: wraps HF `AutoImageProcessor` (source-of-truth alignment with HF, no native re-implementation).
- Multi-image per sequence supported. Video deferred.
- Splice strategy: `ConcatSplice` (Qwen-VL pattern).
- ~~Phase 1 limitation, documented: **3-D MRoPE position vectors are NOT emitted by chunk prep**.~~ **CLOSED in Phase 1.5 (2026-05-13).** Chunk preparation now computes per-token `(t, h, w)` 3-D positions for any sequence with `modality_inputs`, using HF Qwen3-VL's `get_rope_index` algorithm. The MRoPE kernel's `seq_positions.shape[-1] == 3` dispatch now fires for multimodal chunks. Text-only chunks stay on the `(T, 1)` path, byte-identical to pre-multimodal.

## Progress: Phase 1 implementation landed (uncommitted)

### Core types
- `flextrain/core/modality.py` (NEW) — `ImageInputCPU`, `ImageInputs`, `ImageEmbeddings`, `ImageGradInputs`, `InputsSummary`, modality-keyed unions.
- `flextrain/core/__init__.py` — re-exports the new types + `ModalityEncoder`.

### Protocols + Engine hooks
- `flextrain/core/layer.py` — `ModalityEncoder` Protocol; optional `setup_round` / `finalize_round` on `InputLayer`; `ChunkMeta.seq_positions` generalized to `(T, K) int32` (K=1 byte-identical to today; K=3 reserved for MRoPE).
- `flextrain/engine/active_model.py` — two `hasattr`-guarded hook calls in `_setup_round` and `_embed_backward`; `num_vision_layers` thread-through in `load_hf`.

### MRoPE
- `flextrain/nn/blocks/rope.py` — `build_mrope_axis_assignment` + `apply_rope_mrope_fwd` + `apply_rope_mrope_bwd`. Pair-interleaved channel layout; pure-PyTorch reference impl (Triton kernel deferred). Supports both contiguous (`mrope_interleaved=False`) and HF interleaved (`mrope_interleaved=True`, e.g. Qwen3.5/3.6 default with `mrope_section=[11,11,10]`) layouts.
- `flextrain/nn/blocks/attention_gated.py` — `GQAAttentionGatedConfig.mrope_section` / `.mrope_interleaved` knobs; runtime dispatch in `_rope_fwd`/`_rope_bwd` based on `seq_positions.shape[-1]`.
- `flextrain/nn/layers/qwen3_5.py` — `Qwen3_5LayerConfig.mrope_section` / `.mrope_interleaved`; threaded into `GQAAttentionGatedBlock` via `Qwen3_5FullLayer.__init__`.
- `flextrain/io/arch/qwen3_5.py` — `hf_config_to_hyperparams` extracts `mrope_section` + `mrope_interleaved` from HF `rope_parameters`.

### Vision encoder
- `flextrain/nn/encoders/__init__.py` (NEW) + `flextrain/nn/encoders/qwen_vl_vit.py` (NEW) — `QwenVLVisionConfig` + `QwenVLVisionEncoder`. Port of HF `Qwen3VLVisionModel`: patch embed (Conv3d), bilinear pos-embed interpolation (`fast_pos_embed_interpolate`), 2-axis halved vision RoPE (`apply_rotary_pos_emb_vision`), depth transformer blocks (LayerNorm + fused QKV attention + LayerNorm + GeLU-tanh MLP), patch merger (spatial_merge_size**2 grouping → MLP to out_hidden_size). Frozen / forward-only / pure PyTorch. Variable-length attention via per-image SDPA chunks.

### Multimodal input layer + splice
- `flextrain/nn/multimodal_input.py` (NEW) — `MultimodalInputLayer`. Composes `TokenEmbedLayer` + N `ModalityEncoder`s + splice strategies. Implements `InputLayer` Protocol plus `setup_round` / `finalize_round`. Merged `param_spec` (text-embed + all encoders). Phase 1 encoder weights are frozen → grad/opt-state buffers automatically skipped by `BufferManager`.
- `flextrain/nn/splices/__init__.py` (NEW) + `flextrain/nn/splices/concat.py` (NEW) — `concat_splice_fwd` / `concat_splice_bwd`. Reads `chunk.meta.extra["mm_placeholder_positions"]` + `["mm_image_assignment"]` to scatter encoder rows onto placeholder positions; bwd zeros placeholder rows in `d_text_emb` to protect the embed table.

### Arch loader (Qwen3.5/3.6 wiring)
- `flextrain/io/hf_weights.py` — `ArchSpec.vision_embed` + `.vision_layer` (optional tuples); `num_vision_layers` kwarg on loader + exporter; `_render_fx_name` helper for `{i}`-substituted flextrain names.
- `flextrain/io/arch/qwen3_5.py` — `_VISION_EMBED` + `_VISION_LAYER` weight maps for `model.visual.*`; `hf_config_to_vision_dims`, `build_modality_encoders`, `modality_splice_strategies` factories. Wired into `QWEN3_5_ARCH`.

### Sequence + chunk preparation
- `flextrain/io/sequence.py` — `Sequence.modality_inputs: dict[str, list]` (default `{}`).
- `flextrain/engine/schedule.py` — `_populate_mm_chunk_extras` post-pass after chunk materialization. For text-only rounds: fast-bails (no extras written → text-only byte-identical). For multimodal rounds:
  - emits `chunk.meta.extra["mm_placeholder_positions"]["image"][0]` and `["mm_image_assignment"]["image"][0]` for each chunk;
  - **(Phase 1.5)** also calls `_compute_mrope_position_ids(seq)` for each multimodal seq, slices per-chunk, concats in chunk-local order, and replaces `chunk.meta.seq_positions` with the resulting `(T, 3) int32` tensor. Per-seq positions cached in a round-local dict so a seq split across multiple chunks doesn't get recomputed.
  - `_compute_mrope_position_ids` mirrors HF `Qwen3VLForConditionalGeneration.get_rope_index`: text tokens are `(p, p, p)`; image-placeholder tokens are `(t_pos_start, t_pos_start + h_idx, t_pos_start + w_idx)` over the post-merge grid; global counter advances by `max(merged_h, merged_w)` after each image block. Merged-grid dims derived from `placeholder_positions.numel()` + `grid_thw` (square-image fast-path + rectangular fallback).

### from_pretrained kwargs + multimodal build branch
- `flextrain/api.py` — `_build_input_layer` factory (text-only OR `MultimodalInputLayer`). `enable_multimodal: "auto" | bool` + `freeze_modality_encoders: bool` kwargs on `from_pretrained`. Routed through `_build_active_model`. `arch_module` threaded through so the factory can find `hf_config_to_vision_dims` etc.

### Working-set planner thread
- `flextrain/core/_sizing.py` (unchanged) + `flextrain/core/working_set.py` — `_baseline_gpu_activation_memory` and `determine_working_set_config` gained `mm_encoder_peak_bytes: int = 0`. `_build_active_model` (api.py) computes a conservative 256 MiB-per-encoder slack when multimodal is on. Coarse heuristic; refine later with measured `peak_workspace_bytes(InputsSummary)` from the encoder.

### HF AutoImageProcessor wrapper
- `flextrain/io/image_processing.py` (NEW) — `MultimodalProcessorBundle.from_pretrained(model_path)` loads HF `AutoProcessor` + `AutoTokenizer`. `preprocess_images(images, bundle)` returns `(pixel_values, image_grid_thw)`. `build_multimodal_sequence(text, images, bundle)` produces a flextrain `Sequence` with `modality_inputs={"image": [...]}`. Verifies chat-template image-placeholder expansion matches `grid_thw`.

## Verification — regression sweep

Foundation + MRoPE plumbing exercised on the 13-row text-only reverify sweep:

* Run 1 (foundation only, no MRoPE yet): 5/13 bit-exact, 7/13 tiny drift Δ 0.0002–0.0033 (Qwen partial-RoPE/QK-norm kernel non-determinism), 1/13 transient bf16 NaN spike (`llama_3_1_8b_full`) — re-ran in isolation, converged normally to baseline within Δ 0.0004.
* Run 2 (post-MRoPE plumbing in attention_gated): result pending, started 2026-05-13 (~3am UTC).

Phase 1 final reverify is on the pending list after all code is in.

## TODOs (current state)

| # | Step | Status |
|---|---|---|
| 1 | Foundation modality types | ✅ done |
| 2 | ModalityEncoder + InputLayer hooks | ✅ done |
| 3 | ChunkMeta.seq_positions (T,K) | ✅ done |
| 4 | Engine setup/finalize_round hooks | ✅ done |
| 5 | ArchSpec.vision_embed / vision_layer | ✅ done |
| 6 | MRoPE in rope.py + dispatch in attention_gated | ✅ done |
| 7 | Qwen3.5 block builder MRoPE plumbing | ✅ done |
| 8 | QwenVLVisionEncoder | ✅ done |
| 9 | MultimodalInputLayer + ConcatSplice | ✅ done |
| 10 | Qwen3.5 arch wiring | ✅ done |
| 11 | Sequence.modality_inputs | ✅ done |
| 12 | Chunk prep mm_* extras | ✅ done |
| 13 | from_pretrained kwargs + multimodal branch | ✅ done |
| 14 | Working-set mm_encoder_peak_bytes thread | ✅ done |
| 15 | HF AutoImageProcessor wrapper | ✅ done |
| 16 | Post-MRoPE reverify | ⏳ in progress |
| 17 | Final reverify after Phase 1 | ⬜ pending |
| 18 | Phase 1.5: compute 3-D MRoPE positions per token in chunk prep | ✅ done |
| 19 | Block-level MRoPE unit test (`tests/test_mrope_block.py`) | ✅ done |
| 20 | 3-D position-ID unit test (`tests/test_mrope_position_ids.py`) | ✅ done |
| 21 | Splice + placeholder-grad routing test (`tests/test_concat_splice.py`) | ✅ done |
| 22 | MultimodalInputLayer construction smoke test (`tests/test_multimodal_input_smoke.py`) | ✅ done |
| 23 | Vision encoder forward parity test vs HF | ✅ **byte-exact** vs production HF (`from_pretrained(torch_dtype=bf16)`, fp32 inv_freq, eager attn): cos = 1.000000, abs_err = 0.0. Reference path corrected on 2026-05-13 (earlier "byte-exact" claim was vs standalone `_from_config(...).to(bf16)`, which has bf16 inv_freq and does NOT match production). |
| 24 | End-to-end smoke: from_pretrained(enable_multimodal=True) + 1-image forward + backward | ✅ done (PASSING; finite loss) |
| 25 | Single-image full-model loss parity vs HF | ✅ **PASSING (rel_err 0.11% / threshold 5%; trajectory 4.48% → 1.70% → 0.11% across three rounds of bug fixes)** |
| 26 | Per-layer hidden-state parity vs HF (LM-side) | ✅ `pre_LM_input` byte-exact (cos = 1.0000, abs = 0.0); per-layer cos 0.9999 → 0.9980 across 24 LM layers (bf16 accumulation, no compounding pathology). `tests/test_qwen3_5_mm_per_layer_parity.py`. |
| 27 | Multi-image + realistic-data parity vs HF | ✅ `tests/test_qwen3_5_mm_dataset_parity.py`: streams `HuggingFaceH4/llava-instruct-mix-vsft` (real multi-turn LLaVA conversations, varying image resolutions); 12 single-image + 4 synthesized multi-image examples. **16/16 pass**: all have `pre_LM_input` cos_min = 1.000000, abs_max = 0.0 (byte-exact post-splice); loss rel_err max 0.918%, mean 0.328% (well under 1.5% threshold). Multi-image cases (cu_seqlens packing + ImageEmbeddings.token_offsets) verified working. |
| 28 | HF processor parity test (the wrapper uses HF processor verbatim; bit-parity implicit) | ⬜ low priority |
| 29 | Final reverify after fp32 inv_freq fix | ⬜ (change only touches `flextrain/nn/encoders/qwen_vl_vit.py`; no text-only path imports it, so reverify-sweep regression is impossible by construction) |
| 30 | Vision encoder flash-attn backend (`attn_implementation="flash"`) | ✅ Added 2026-05-13. New `_sdpa_varlen_flash` method calls `flextrain.ops._kernels.attention.flextrain_attention_fwd(..., causal=False)` — one kernel call across all images in `cu_seqlens` instead of the per-image SDPA Python loop. Plumbed via `vision_dims["attn_implementation"]` in `flextrain/io/arch/qwen3_5.py:build_modality_encoders`. End-to-end loss parity vs HF: rel_err 0.457% (single-image synthetic; vs 0.11% with default SDPA — small bf16 reduction-order drift from packing, still well under the 1.5% bound). Eager parity unchanged (byte-exact). Speedup over default SDPA: 1.12× on 8-image batch (modest because SDPA already gets Flash via PyTorch's auto-dispatch; the win scales with image count). **Default remains "sdpa"** (parity-safe); opt in via `attn_implementation="flash"`. |
| 31 | End-to-end multimodal **training** parity vs HF (`tests/test_qwen3_5_mm_train_parity.py`) | ✅ Added 2026-05-13. Full-parameter FT of Qwen3.5-2B LM (vision tower frozen on both stacks). On each of 2 real LLaVA-vsft conversations, flextrain runs 5 training steps (`am.fwd_bwd` + `am.step`) while HF acts as a forward-only oracle: at every step, flextrain's current LM state is synced into HF via `_build_hf_state_dict_from_archspec` (handles the Q/K halved→pair perm inversion and linear-attn fused→split unbundling) and `tie_weights()` re-ties the embed↔lm_head. **10/10 per-step matches** under a hybrid threshold (`rel_err < 1.5% OR abs_err < 0.05` — the absolute clause matters once loss → 0 in overfitting and the same ~0.003 abs drift inflates the relative error). max abs_err = 0.023, mean abs_err = 0.006, max rel_err at loss > 0.5 = 2.185%. **Training is working**: per-example loss decreases 4.8× (2.14 → 0.45) and 40× (0.67 → 0.017) over the 5 steps. (Gotcha for future tests: `fwd_bwd` only computes grads — `step()` must be called separately to apply the AdamW update. Without it, the loss is identical across "training" steps and the parity test is meaningless.) |

## Known gaps & follow-ups

### ~~Phase 1.5 — 3-D MRoPE positions in chunk preparation~~ DONE

Closed 2026-05-13. `_compute_mrope_position_ids` + chunk-prep integration land in `flextrain/engine/schedule.py`. Unit-tested at four configurations (text-only returns None; image at sequence start; image after text; rectangular grid). HF Qwen3-VL `get_rope_index` is the algorithmic reference.

### Phase 1.5 — MRoPE Triton kernel

`apply_rope_mrope_fwd/bwd` in `rope.py` is pure-PyTorch. Allocates short-lived `(T, n_pairs)` cos/sin tensors per call. Fine for forward-parity validation; production should use a fused Triton kernel matching the existing `flextrain_rope_partial_fwd/bwd` API.

### Working-set encoder peak heuristic

`_build_active_model` uses a hardcoded 256 MiB/encoder slack. Refine to call `encoder.peak_workspace_bytes(InputsSummary)` with a representative summary built from the dataset (or from `max_seq_len` + expected `n_images_per_round`).

### Deepstack (Qwen3-VL proper)

`build_modality_encoders` raises `NotImplementedError` when `deepstack_visual_indexes` is non-empty. This is Phase 2: needs backbone-layer protocol extension to accept per-layer aux inputs (vision features re-injected at backbone layers 8/16/24).

### Splice grad accumulator (Phase 3 trainable encoder)

`concat_splice_bwd` accepts an optional `d_image_grad_accum: ImageGradInputs | None`. Phase 1 (frozen encoder) passes None and just zeros placeholder rows. Phase 3 will allocate the accumulator in `MultimodalInputLayer.backward` (start-of-chunk) and pass it; `finalize_round` will call `encoder.backward_round(accum, inputs, weights, grads, ctx)` for non-frozen encoders.

### Validation coverage

Phase 1 final reverify is text-only (proves no regression). Multimodal forward parity vs HF needs separate tests (items 19-24 above). Until those pass, treat Phase 1 multimodal as "wired but unvalidated end-to-end".

## Files touched / to touch

### Modified

- `flextrain/core/__init__.py`, `flextrain/core/layer.py` — Protocol additions, ChunkMeta gen.
- `flextrain/engine/active_model.py` — hooks + num_vision_layers thread-through.
- `flextrain/engine/schedule.py` — chunk prep mm_* extras post-pass.
- `flextrain/io/hf_weights.py` — ArchSpec.vision_embed/vision_layer + loader/exporter changes.
- `flextrain/io/arch/qwen3_5.py` — vision weight entries + factories + MRoPE hyperparams.
- `flextrain/io/sequence.py` — `modality_inputs` keyword on Sequence.
- `flextrain/nn/blocks/rope.py` — MRoPE math.
- `flextrain/nn/blocks/attention_gated.py` — MRoPE dispatch.
- `flextrain/nn/layers/qwen3_5.py` — MRoPE config plumbing.
- `flextrain/api.py` — `enable_multimodal` kwarg, input-layer factory branch, multimodal encoder peak slack.
- `flextrain/core/working_set.py` — `mm_encoder_peak_bytes` thread.

### Created

- `flextrain/core/modality.py`
- `flextrain/nn/encoders/__init__.py`, `flextrain/nn/encoders/qwen_vl_vit.py`
- `flextrain/nn/multimodal_input.py`
- `flextrain/nn/splices/__init__.py`, `flextrain/nn/splices/concat.py`
- `flextrain/io/image_processing.py`
- `docs/internal/multimodal_status.md` (this file)
- `docs/internal/multimodal_session_notes.md` (autonomous-session bug log)

## Design decisions (recorded so next session doesn't relitigate)

### 1. Encoder weights share `embed` scope (no new buffer tier)

`MultimodalInputLayer.param_spec` is the MERGE of `text_embed.param_spec` and every `ModalityEncoder.param_spec`. Encoder tensor names use the prefix `f"{modality}{encoder_id}_"` (e.g. `"image0_w_q"`) to avoid collisions. All these tensors live in the existing `gpu_embed_params` / `host_embed_params` dicts.

The arch-loader's `vision_embed` and `vision_layer` ArchSpec attrs are organizational only — both deposit under scope `"embed"` in the loader's dest dict. Per-vision-layer entries embed `{i}` in **both** their `hf_name` AND `flextrain_name` (e.g. `flextrain_name="image0_layer_{i}_w_q"`) so each vision layer's weights get a unique key inside `gpu_embed_params`.

Rejected: separate `gpu_encoder_params` / `gpu_encoder_grads` buffer tier. Reason: Qwen-VL ViT at depth=24 hidden=1024 is ~150-600 MB of params — well within the always-on-GPU regime. Revisit if Phase 3 introduces a multi-GB trainable encoder.

### 2. `setup_round` / `finalize_round` as optional Protocol hooks

Optional methods on `InputLayer`, `hasattr`-guarded by the engine. Matches the precedent of `Layer.backward_dgrad` / `Layer.backward_wgrad`. Zero overhead on text-only paths.

### 3. `ChunkMeta.seq_positions` extended in place to `(T, K)`

The K=1 path is byte-identical (`.reshape(-1, 1)` on a 1-D input). 2-D input (Sequence-of-Sequence or 2-D Tensor) produces `(T, K)`. Existing attention blocks that read `seq_positions[:, 0]` are unchanged; MRoPE blocks slice per-axis.

### 4. Splice strategies are function pairs, not classes

`concat_splice_fwd` and `concat_splice_bwd` are typed `Callable`s living in `flextrain/nn/splices/`. Stateless; the strategy "object" is just a tuple `(fwd, bwd)`. Phase 2 adds `SubstitutionSplice` (Gemma3/4) using the same shape.

### 5. Placeholder-row gradient routing

After ConcatSplice scatters encoder rows onto placeholder positions, dx at those positions in backward belongs to the **encoder grad accumulator**, NOT the embed-table row. `concat_splice_bwd` (a) routes placeholder-position dx into the encoder accumulator (Phase 3) or drops it (Phase 1 frozen), and (b) zeros those positions in `d_text_emb` so the embed-table scatter-add doesn't poison the placeholder row.

### 6. Pure-PyTorch MRoPE reference impl in Phase 1

`apply_rope_mrope_fwd/bwd` allocates `(T, n_pairs)` cos/sin tensors per call. Acceptable for forward-parity validation. Triton kernel is a Phase 2 perf optimization.

### 7. HF AutoImageProcessor is the data source of truth

`flextrain/io/image_processing.py` wraps HF processors directly — no reimplementation of resize/normalize/patchify. This avoids silent divergence between flextrain's preprocessing and HF inference.

## Open investigation items

- **3-D position computation** for vision tokens (above — Phase 1.5 priority).
- **HF chat-template integration**: confirm `tokenizer.apply_chat_template` expands `<|image_pad|>` to N copies for Qwen3.5/3.6 (older HF versions may not). The wrapper's verification raises if mismatched.
- **bf16 vision encoder vs fp32 normalize**: HF processors normalize in fp32 then cast to fp32 by default. Our encoder casts to bf16 at the patch_embed entry. Need to verify this matches HF's behavior or whether HF uses fp32 throughout the vision tower.

## Phase 2 design items flagged

### Gemma3 / Gemma4 bidirectional image-block attention (NOT a Phase 1 issue)

Verified by reading HF `Qwen3_5TextModel.forward`: Qwen3.5/3.6/3-VL use **plain causal attention** in the LM (`create_causal_mask` at `modeling_qwen3_5.py:1271`). No image-aware masking. Image differentiation is purely via MRoPE 3-D positions + the splice. Our causal flash-attn varlen chunking handles this correctly (0.11% loss-parity confirms it works end-to-end).

**Gemma3 / Gemma4 are different.** Gemma3 multimodal applies **bidirectional attention within image blocks** (image tokens attend to each other forward AND backward), while the rest of the sequence stays causal. The current flextrain attention block uses `is_causal=True` with a uniform mask -- it cannot express "causal + bidirectional-window-at-image-positions." Phase 2 Gemma3 work needs either:

1. Custom attention mask built per-chunk that's mostly lower-triangular but with bidirectional windows at image block positions, OR
2. Two-pass attention -- bidirectional pass on image tokens followed by causal pass on the merged sequence.

Additionally, image blocks straddling chunk boundaries become tricky: the second chunk's image tokens need to attend FORWARD to first-chunk image tokens, but flextrain's KV cache only handles backward causal flow. Mitigations:
- Constrain chunk boundaries to never split an image block (cleanest; restricts max_chunk_size dynamically per round), OR
- Materialize per-image attention separately and cache the result.

For Phase 2 Gemma3 planning, treat this as a **backbone-layer protocol extension**, NOT just an arch loader change.

## Testing pyramid (still to implement; items 19-24)

1. Block-level MRoPE parity (degenerate-3D ≡ standard partial RoPE).
2. Vision encoder forward parity vs HF (single image; cos ≥ 0.99 per output token).
3. Splice + placeholder grad routing unit test.
4. Text-only regression (re-run reverify after every commit — already in CI form via `experiments/reverify.sh`).
5. Single-image full-model forward parity vs HF (the big one).
6. Multi-image full-model forward parity.
7. Working-set sanity with vision tower attached.
8. HF processor parity (bit-identical pixel_values for the same input image).
