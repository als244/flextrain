# Diffusion transformer (DiT) — design notes

Goal: support training of bidirectional diffusion transformers (DiT,
PixArt-α/Σ, SD3 / Flux MM-DiT subset) on FlexTrain's working-set
engine.  These models share the engine's per-layer/per-chunk
abstraction but differ from causal LMs in attention, loss, and the
input/output pipeline.  This doc plans the integration step by step
and flags what's reusable, what needs to change, and what new tests
are required.

## Key differences vs. causal LMs

| Aspect | Causal LM (Llama, Qwen, …) | Diffusion transformer |
|---|---|---|
| Attention | Causal (lower-triangular) | **Bidirectional** (no mask, all-to-all) |
| Sequence order | Sequence is meaningful (positions 0..T) | Patches in arbitrary 2-D / 3-D order; positional encoding is spatial |
| Input | Token IDs → embedding lookup | Image latent → patch-conv → `(num_patches, d_model)` |
| Conditioning | None per-block (just residual stream) | Time-step + (class / text) conditioning, applied via AdaLN per block |
| Output | Logits over vocab → CE | `(num_patches, patch_dim²·channels)` → patch-unpatchify → noise prediction → MSE |
| Loss | Cross-entropy on next token | MSE between predicted noise and target noise (or v-prediction, x₀-prediction) |
| Loss masking | `targets == -100` ignore-index | Usually no mask (every patch contributes), but classifier-free guidance training drops conditioning sometimes |
| Per-token signal | `seq.targets[i]` | `seq.target_noise[t, i]` (patch-level) |

## What's reusable from the existing engine

Almost everything below the block / loss boundary:

* **`ActiveModel` engine** — chunking, save-level solver, working-set
  rotation, GPU/host buffer rings, optimizer plumbing. No changes.
* **`LMHead` → replace** with a `DiTHead` that does the final
  ``RMSNorm + linear → patch-unpatchify``. Same protocol shape (one
  ``forward_backward`` taking residual stream and emitting per-token
  loss + dx upstream).
* **`TokenEmbedLayer` → replace** with `PatchEmbedLayer` — a
  patch-conv that turns ``(B, C, H, W)`` into
  ``(num_patches, d_model)``.  Owns time/conditioning embeddings
  alongside the patch projection.
* **`SwiGLUFFN`, `RMSNormBlock`** — reused as-is.
* **`GQAAttentionBlock`** — reused with `is_causal=False`. Flash-attn 2
  supports ``causal=False``, so the kernel call site already covers
  bidirectional attention; only the block config needs flipping.
* **LoRA wrapper** — works unchanged on the new block (it doesn't
  care about attention masking or loss type).

## What needs new code

### 1. Bidirectional attention block

Just `GQAAttentionBlock` with `is_causal=False`.  Sanity check:

* `flextrain_attention_fwd` already takes `causal: bool` — verify the
  bwd kernel path doesn't shortcut on causal=True. One call site to
  audit (`flextrain/ops/_kernels/attention.py`). **Likely zero-effort
  change.**
* RoPE for spatial positions: 2-D RoPE with separate frequencies for
  H and W axes.  The current 1-D RoPE kernel takes `seq_positions:
  (T, K_rope)` and a 1-D `inv_freq` array; **K_rope=2 path is
  half-implemented** (the kernel hardcodes `K_rope=1`). To handle 2-D
  RoPE cleanly, either:
  * Extend the kernel to loop over `K_rope` (small change), or
  * Pre-rotate per-axis in two separate kernel calls (zero kernel
    change; one extra reshape).  Cleaner for the first cut.

### 2. AdaLN block (modulation)

DiT applies ``y = scale(t) * norm(x) + shift(t)`` instead of plain
RMSNorm. Per-block it's a small linear ``t_emb → (scale, shift)``
projection of the time embedding, plus per-block ``γ, β`` adds.

Plan:

* Add `AdaLNBlock(eps, t_emb_dim, hidden_dim)`. Owns one ``w_ada
  : (t_emb_dim, 2 * hidden_dim)`` for ``[scale; shift]``.
* Activation schema: stash ``rstd`` of the inner RMSNorm (tier 0) and
  ``t_emb`` for that block (tier 0).  Also stash ``norm_x`` (the
  RMSNorm output) at tier 1 since we'll need it in bwd.
* Forward: ``norm_x = rmsnorm(x); t_proj = t_emb @ w_ada; scale,shift
  = chunk(t_proj, 2, dim=-1); return scale * norm_x + shift``.
* Backward: split incoming `dy` into ``d_scale = dy * norm_x``,
  ``d_shift = dy``, ``d_norm_x = dy * scale``; back through RMSNorm to
  get ``dx``; back through ``w_ada`` matmul to get ``dt_emb`` and
  ``g_w_ada``.

Two AdaLN per block (pre-attn, pre-FFN), or one shared. Standard DiT
uses **separate pre-attn / pre-FFN AdaLN** plus a third ``gate_attn``
/ ``gate_mlp`` from the same projection (so ``w_ada`` produces 6
chunks, not 2).

The standard DiT block is:

```
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    = chunk(t_emb @ w_ada, 6, dim=-1)
y_attn = attn(modulate(rmsnorm(x), shift_msa, scale_msa))
x = x + gate_msa * y_attn
y_mlp = mlp(modulate(rmsnorm(x), shift_mlp, scale_mlp))
x = x + gate_mlp * y_mlp
```

### 3. Time / conditioning embedding layer

The "input layer" needs to produce **two** outputs per chunk:

* The patched residual stream `(num_patches, d_model)`.
* The shared time/class conditioning vector `t_emb : (d_model_t,)` (or
  per-sample `(B, d_model_t)` if multiple samples per chunk).

Options:

* **(a) Stash t_emb in `LayerContext.extra`** so every block can read
  it. Engine doesn't currently expose a per-step scalar conditioning;
  add a `LayerContext.cond` field. Cheap.
* **(b) Broadcast t_emb to every patch** (waste memory but no
  protocol change).

(a) is cleaner. One field in `LayerContext` keyed
``conditioning: dict[str, torch.Tensor]`` lets future architectures
add their own conditioning (FiLM, cross-attention K/V cache, etc.).

### 4. DiTHead

Final norm + linear projection back to patch space. Like LMHead but
without softmax / cross-entropy. Two parameters: ``w_final_norm``
(RMSNorm γ) and ``w_unpatchify`` (linear ``(d_model, patch_dim²·C)``).

The loss is computed elsewhere (the engine's `LossFn` interface
already takes pluggable losses; an ``MSELoss`` exists in
`flextrain/nn/loss.py`). The head emits per-token loss = per-patch
MSE; the loss fn just needs to consume `(predicted_noise,
target_noise)` instead of `(logits, target_token_id)`.

`MSELoss` already exists in `flextrain.nn.loss` — it wants
`labels: (T', V)` (per-token target vector). We pass
`token_slice.labels = target_noise.reshape(T', patch_dim²·C)`. Fits
the existing API cleanly.

### 5. Bidirectional autograd reference for testing

For causal LMs we have naive PyTorch reference implementations. For
DiT we need:

* A small naive ``naive_dit_block`` that uses
  ``nn.functional.scaled_dot_product_attention(..., is_causal=False)``
  and AdaLN modulation.
* Block-level math parity: random `(T, d_model)` input + random
  `t_emb`, run FT block fwd+bwd, run naive autograd, compare:
  ``dW_q``, ``dW_k``, ``dW_v``, ``dW_o``, ``dW_1/2/3``, ``dW_ada``,
  ``d_t_emb``, ``dx``.
* End-to-end pixel-space MSE on ImageNet-256 (small DiT-S/2 first,
  then DiT-XL/2 8B-equivalent if it fits).

### 6. Dataset / batching

* Inputs: pre-extracted VAE latents (e.g. SD-VAE 8× downsample of
  256×256 RGB → `(C=4, 32, 32)` latents → `(num_patches, d_model)`).
* Per-step: sample noise ε, sample timestep t, build x_t = α_t·x_0 +
  σ_t·ε, run model to predict ε_θ(x_t, t), compare with ε.
* `_Seq` extension: replace `.tokens` (int64 token IDs) with `.x_t`
  (fp16/bf16 patch latents) and `.target_noise`. The engine doesn't
  care about the underlying type — only about `.tokens` shape (T,).
  Need to confirm the engine doesn't `.long()` cast anywhere.
  *Audit item.*

## Step-by-step implementation plan

1. **AdaLNBlock + tests.** Math parity vs autograd.  Reusable in
   isolation.
2. **2-D RoPE.** Either kernel update (loop over `K_rope`) or two
   sequential 1-D applications wrapped in a helper. Math parity vs
   reference.
3. **DiTBlock layer.** Composes AdaLN + GQAAttentionBlock(is_causal=False) + AdaLN + SwiGLUFFN. Math
   parity at block level. Run 2-3 random inputs, compare to naive
   autograd.
4. **PatchEmbedLayer + DiTHead.** Patch projection + unpatchify. Math
   parity (round-trip should be lossless after fixed weights).
5. **ArchSpec + ``DiT_block_builder``.** Register
   ``"DiTForImageGeneration"`` (or whatever the HF arch ID is for
   DiT-XL/2 if it ships).  Plumb time-embedding init and class-label
   conditioning.
6. **Dataset adapter.** Audit `flextrain.bench.parity._Seq` and the
   engine packing pipeline to confirm patch latents (fp16) flow
   through unchanged. Build a `_DiTSeq` if needed.
7. **End-to-end smoke.** DiT-S/2 (33M params) on a tiny ImageNet
   subset for ~100 steps. Compare loss curve to a reference
   implementation (e.g. facebookresearch/DiT or PixArt-α).
8. **LoRA on DiT** — should "just work" via `LoRAWrapperLayer` wrapping
   `DiTBlock` (LoRA targets all 2-D linears: q, k, v, o, w_1/2/3,
   w_ada). Test: image-prompt fine-tuning.

## Open questions

* **2-D RoPE in flash-attn**: flash-attn 2 doesn't apply RoPE
  internally — that's our existing ``apply_rope_fwd`` + matmul to
  Q/K. But for 2-D, the inv_freq is an `(D/2,)` vector that combines
  axis-x and axis-y components depending on dim index. Need to decide
  the convention (axial — half the dims encode x, half encode y —
  vs. RoPE-2D — alternating). Check what the target arch (DiT-XL/2,
  PixArt) actually uses.
* **Classifier-free-guidance training**: 10% of training steps drop
  the class label (or text embedding). Implementation: caller
  randomizes per-batch and zeros out the conditioning before passing
  to the engine. No engine change needed.
* **VAE latent precision**: latents are typically fp32; do we keep
  them fp32 in the residual stream input or cast to bf16 before
  patch-embed? Standard DiT keeps the VAE encode in fp32 and casts to
  bf16 for transformer compute. Mirror that.
* **Heterogeneous backbones**: can a single `ActiveModel` mix DiT
  blocks (bidirectional, MSE loss) with a causal-LM head? Probably
  not worth supporting — the loss functions are different and the
  conditioning channel is DiT-specific. Keep the engine flexible
  enough that the `LossFn` and `LayerContext` extensions don't break
  causal flows.

## Test strategy (matches existing repo conventions)

Mirror the test patterns the repo already uses:

* `tests/test_dit_block_math.py` — naive autograd parity for one
  block (analogous to `test_lora_wrapper_math.py`).
* `tests/test_dit_engine_smoke.py` — engine fwd+bwd on a 4-layer
  small-init DiT, compare to naive multi-layer DiT.
* `tests/test_dit_pixel_e2e.py` — DiT-S/2 on ImageNet-256 for ~100
  steps, save loss curve + a few sample images. Reference a known
  open implementation (DiT or PixArt-α).
* `tests/test_dit_lora_smoke.py` — apply LoRA via
  `flextrain.from_pretrained(..., lora_targets="all")`, verify loss
  drops on 1 step (matches the pattern from `test_lora_moe_smoke`).

Cross-stack parity reference: a small forward-only HF DiT load (if
HF ships one) or the official `facebookresearch/DiT` reference. Use
the same logit-capture / loss-capture diagnostic approach as
`tests/test_lora_8b_diagnostics.py`. With `is_causal=False`, the
top-K disagreeing-position analysis is even more important since
divergence can come from anywhere in the sequence (not just causal
boundary).
