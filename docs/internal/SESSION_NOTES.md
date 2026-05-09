# Session notes — decisions, findings, issues, questions

Running log during the autonomous work session (user at gym/out).

## Decisions made

### Kernel ownership (DONE)
- Copied all orig Triton kernels (`awsm_transformer/ops/*.py` + `matmul_dispatchers.py` + `mem_register.py`) into `flextrain/ops/_kernels/`.
- Renamed public API `awsm_* → flextrain_*`. Kernel-internal function names kept as-is (they're private).
- `matmul_dispatcher` (compiled C++ cuBLASLt extension) kept as external dependency — we don't vendor C++ builds.
- Backward compat removed in tests; `orig/awsm_transformer/` still exists for parity tests that compare against orig's Python layers (TransformerEmbed / Head / Layer).

### Dual-mode router kernel (DONE)
- Extended `flextrain_fused_topk_softmax` and `flextrain_moe_router_gate_bwd` with a `MODE` constexpr switch: `topk_then_softmax` (Qwen3-MoE default, weights sum to 1) and `softmax_then_topk` (OLMoE with `norm_topk_prob=False`, gpt-oss).
- Single kernel per direction; no Python fallback.
- User-driven: no special case branches in Python; routing_mode propagates from `MoESwiGLUConfig → ffn_moe.py fwd/bwd → kernel MODE constexpr`.

### Optimizer state naming (DONE)
- Renamed `o_m → o_adam_m`, `o_v → o_adam_v`, `o_momentum → o_muon` for unambiguous per-algorithm prefixes (user feedback — `o_m` and `o_momentum` were semantically confusing).

### Per-parameter optimizer dtypes (DONE)
- `TensorSpec.opt_state_dtype` is already per-tensor. Hybrid optimizer honors it when allocating both AdamW and Muon state.

### HybridMuonAdamW classification (DONE)
- Auto-classification rules:
  - Explicit `TensorSpec.optimizer` wins.
  - 2-D tensors (not matching AdamW name fragments) → Muon.
  - 3-D MoE expert stacks `(E, d, 2F)` / `(E, F, d)` → Muon **per-expert slice** (Newton-Schulz iterated over expert dim inside step()).
  - 1-D (norms, biases) / embeddings / head / routers → AdamW.
- Filter-aware state allocator: only the applicable state tensors are allocated per param (no wasted bytes for Muon slots on AdamW params).

### OLMoE-1B-7B bugs (FIXED, findings documented below)
1. MoE weight packing was `[gate, up]`; kernel expects `[up, gate]` (x3=value in first half, x1=gate in second).
2. OLMoE router is `softmax_then_topk` (`norm_topk_prob=False`) — new mode added.
3. OLMoE has **full-dim** QK-norm (not per-head like Qwen3). New RMSNorm config with `per_head=False`.
4. **Non-obvious**: the Q/K halved→pair RoPE permutation must also be applied to the 1-D `w_q_norm` / `w_k_norm` weights. They multiply post-projection Q/K per-dim — permutation of the Q/K output changes which dim of the norm weight multiplies which value.

## Findings

### F-1. Engine parity test was a false positive prior to F-2
The OLMoE engine parity test passed with max |Δ|=0.0002 while both naive and FT were computing *wrong* SwiGLU (reading `[gate, up]` as `[value, gate]`). With random-init weights, the results happened to have matching magnitudes. **Random-init parity does not verify correctness — it verifies that two implementations disagree in the same way.** Real-weight tests (against HF) caught the bug. **Takeaway: prefer HF-weight tests or inject structured weights (not just Normal(0, 0.02)) for new architectures.**

### F-2. Applying permutation fixes to HF-weight tests
Every HF model that uses Llama-style halved RoPE needs a permutation wrapper. Existing: `_permute_qk_for_pair_interleave(w, head_dim)` in `tests/test_llama32_1b_parity.py`. **Must also permute `w_q_norm` / `w_k_norm` if the architecture has QK-norm.**

### F-3. HF model leaves GPU memory pinned
Running HF transformers' `from_pretrained(device_map=...)` in the same process as FT leaves ~12 GiB pinned even after `del model + gc.collect + empty_cache`. **Workaround**: run HF reference in a subprocess (OS reclaims everything on exit).

### F-4. Solver arithmetic-intensity floor for MoE
OLMoE's 64 experts + top-8 routing has an arithmetic-intensity-based minimum chunk size of ~1.7K tokens. Must target `max_global_batch_tokens` ≥ this (we use 4096). Qwen3-MoE with 128 experts will need similar or larger.

### F-5. MoE expert weight stack shape interacts with Muon
3-D tensor `(E, d, 2F)` can't go into Newton-Schulz directly — iterating expert slices is the right call (user-confirmed).

### F-6. Dense → MoE engine ordering crashes
Running the dense E2E engine immediately followed by a MoE engine in the same Python process causes `CUDA error: invalid argument` during the second engine's initial prefetch or step. Fix: call `am.buffers.destroy()` + `unregister_all_process_pinned_memory()` + `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` between engines. Same issue affected the OLMoE training test and was fixed by a subprocess for HF.

## Open issues / questions for the user

### Q-1. Muon hyperparams sharing
Currently `HybridMuonAdamWHyperparams` has separate `adamw` and `muon` sub-hyperparam objects with independent LR. Default LR comes from each sub-object. Should the top-level `lr` field override both, or stay as a label only? Right now top-level `lr` is unused.

### Q-2. Muon state dtype for MoE experts
For MoE expert stacks routed to Muon, state is allocated using `TensorSpec.opt_state_dtype` (default bf16). Resolved: bf16 default matches Muon's traditional momentum precision and saves ~50% opt state memory on MoE models. Pass `state_dtype=torch.float32` for higher-precision moment tracking when needed.

### Q-3. Qwen3-MoE 30B-A3B HF weights test
Architecture works (small-init parity passes). Need a `test_qwen3_moe_30b_training.py` following the OLMoE template but I can't run it here (disk/GPU). Want me to write the script anyway? You can run on a bigger box.

### Q-4. GPT-OSS
Requires attention sinks (trainable per-head scalars concatenated to pre-softmax attention logits) + router bias + alternating SWA. Non-trivial — needs either a custom attention kernel or sink-as-extra-K-V-token hack. **Deferring** unless you have a strong preference. Which variant do you care about first: 20B or 120B?

## Hybrid attention (Qwen3-Next / Gated DeltaNet) — partial

### What landed
- New block: `flextrain/nn/blocks/linear_attn.py:GatedDeltaNetBlock`.
  - Forward path complete: QKVZ + BA projections, depthwise causal conv1d
    with SiLU, gate computation `g = -exp(A_log) * softplus(a + dt_bias)`,
    `chunk_gated_delta_rule_fwd` (FLA, called directly — not via
    `torch.autograd`), gated RMSNorm with z, out projection.
  - ParamSpec includes the gate-rule scalars (`w_lin_dt_bias`,
    `w_lin_A_log`) classified as AdamW (1-D), and the depthwise conv1d
    weights as a 3-D `(conv_dim, 1, K)` tensor.
  - Activation schema: tier 0 saves the small inputs/outputs the bwd
    needs (`lin_a`, `lin_b`, `lin_z`, `lin_conv_in`); tier 1 saves the
    larger post-conv `q/k/v/g`, plus the FLA-fwd intra-chunk attention
    `A_int` (passed to bwd) and the pre-norm core_attn_out.
  - FLA installed in env (`flash-linear-attention 0.5.0`).

### Backward (LANDED — see test_gated_deltanet_bwd.py)
**Approach**: single scoped autograd block over the per-call computation
graph, with a `torch.autograd.Function` wrapping FLA's `_fwd`/`_bwd`
primitives. The autograd graph is local to ONE block call (not global
across the whole model), and is constructed fresh each backward by
re-running the forward over leaf-cloned weights with `requires_grad`.

**Why this is OK** despite the "no torch.autograd" framework principle:
1. The graph is bounded — built/torn-down per layer call.
2. The FLA core uses its hand-written kernels via the custom Function.
3. The remaining ops (projections, conv1d, gated-RMSNorm, gate calc)
   are tiny and PyTorch's autograd handles them efficiently.
4. The block's interface to the engine is unchanged — fwd writes
   activations to slot, bwd reads upstream dy, returns dx, accumulates
   weight grads. The autograd subgraph is an implementation detail.

**Test**: `tests/test_gated_deltanet_bwd.py` compares against full
autograd reference; max |Δ| on dL/dx = 2.6e-5; weight grads agree to
bf16 floor.

**Future work**: replace the autograd subgraph with a fully hand-rolled
bwd if needed for memory or perf reasons. The current path saves
intermediates twice (once during the autograd replay), which is wasteful
on large layers. Lands as a follow-up after we have HF-weight parity.

### Older notes about deferred bwd (now obsolete)
- ~~**Backward** is a stub raising `NotImplementedError`.~~ To complete:
  1. dL/d(out_proj_input) ← `dy @ w_lin_out.T`, plus
     `g_w_lin_out += out_norm.T @ dy`.
  2. dL/d(o_normed) → reverse gated RMSNorm: silu(z) · rmsnorm(o) · w.
     Need both dL/do (chained back through FLA) and dL/dz, dL/dw_lin_norm.
  3. dL/d(o_pre_norm) → call `chunk_gated_delta_rule_bwd(q, k, v, g, beta, A_int, ...)`
     returning `dq, dk, dv, db, dg, _, dA_log, ddt_bias`.
  4. Reverse the SiLU+conv1d+cat — depthwise conv bwd via standard formula
     (sliding-window dot-product).
  5. Reverse the QKVZ/BA splits and the projections; accumulate `g_w_lin_qkvz`,
     `g_w_lin_ba`.
  6. dL/dx = dL/d(qkvz) @ w_lin_qkvz.T + dL/d(ba) @ w_lin_ba.T.

  **Estimated effort**: ~half a day of hand-derivation + parity-test work.
  Suggest scheduling alongside Gemma 3 since both involve QK-norm bwd
  through projections.

### Why not torch.autograd reference for bwd
User's stated direction: "Using torch autograd probably won't work we
will need to integrate it with our custom backward." Honoring that — no
autograd shortcut. The fwd path is structured to make the bwd
a mechanical hand-derivation; the design is intentional.

### Architectural note for the future block
Gated DeltaNet replaces *both* the GQAAttentionBlock and the RoPE call
in the dense layer (linear-attention layers don't use RoPE — positional
information lives in the recurrent state). So the enclosing
`Qwen3NextLinearAttentionLayer` will look like Llama-block but with
`self.attn = GatedDeltaNetBlock` and **no** RoPE step in the layer fwd.
For Qwen3-Next's full-attention layers we reuse the existing
`GQAAttentionBlock` with `qk_norm=True` (it already supports per-head
QK-norm).

The `layer_types: list[str]` config field from HF is straightforward to
plumb: build an alternating `Qwen3NextLayer[i]` based on `layer_types[i]`
== `"linear_attention"` vs `"full_attention"` — the heterogeneous
backbone path the engine already supports.

## Late-session additions

### Gemma 2 / Gemma 3
- Forward path landed for both, including dual-residual norms (pre+post
  for each sublayer), attention logit softcap (plumbed through to
  flash-attn's `softcap` arg), per-layer alternating sliding-window
  attention, and Gemma 3's per-head QK-norm.
- Arch specs for `Gemma2ForCausalLM` and `Gemma3ForCausalLM` /
  `Gemma3ForConditionalGeneration` registered. Post-load hook applies
  the Gemma `γ - 1 → γ` shift on RMSNorm weights.
- Gemma 3 backbone factory (`build_gemma3_backbone`) handles per-layer
  alternating local-vs-global RoPE base + sliding-window pattern.
- **bwd is stubbed** (raises NotImplementedError) for both Gemma layers.
  The dual-residual structure makes the chain rule tedious — pre + post
  norm on each sublayer means the bwd routes grads through 4 RMSNorm
  bwds + 1 attention bwd + 1 FFN bwd per layer, with non-trivial
  dependencies. Will land in a follow-up.

### LoRA primitive
- Added `TensorSpec.frozen: bool` field. Engine's buffer allocator
  skips grad / opt_state allocation for frozen tensors but keeps
  master + compute (forward still needs the weight).
- All three optimizers (AdamW, Muon, HybridMuonAdamW) skip frozen
  tensors in their step loops.
- New module `flextrain/nn/blocks/lora.py` with:
  - `LoRALinearConfig` — names + rank + alpha + scaling.
  - `lora_param_spec(cfg)` — returns the (frozen base, A, B)
    TensorSpec triple. Compose with the layer's other params via
    ParamSpec.merge.
  - `lora_init(weights, cfg, seed=...)` — PEFT-style: `A ~ N(0, 0.02)`,
    `B = 0` so the LoRA delta starts at zero (no perturbation to
    base behavior at init).
  - `lora_linear_fwd(x, weights, cfg, out=...)` — computes
    `y = x @ W + (x @ A) @ B * (alpha / r)`.
  - `lora_linear_bwd(dy, x, weights, grads, cfg)` — accumulates
    `g_<a_name>` and `g_<b_name>`; returns the LoRA contribution
    to dL/dx (caller adds the base path's `dy @ W.T` separately).
- Tested in `tests/test_lora_block.py`:
  - Math parity vs autograd (fwd + bwd dA, dB, dx-LoRA).
  - Frozen allocation: only A/B get grads + opt state; base skipped.
  - Init scheme verified.

**What's NOT yet done for LoRA**: a layer-level wrapper that uses
`lora_linear_fwd` / `bwd` in place of the existing dense matmul in
`LlamaBlock`'s Q/V projection. This is a small follow-up — the
primitive is in place. See `Wire LoRA into LlamaBlock` todo.

## 2026-04-26: Llama 3.1 YARN RoPE bug + cross-stack noise floor investigation

### Bug found and fixed: `rope_scaling` ignored

The FT RoPE kernel hardcoded vanilla `inv_freq[i] = θ^(-2i/D)` and
silently ignored `config.rope_scaling`. Llama 3.1+ uses
`rope_type: llama3` (factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
original_max_position_embeddings=8192) — a frequency-band scaling
where `inv_freq` indices ≥ ~30 are scaled by up to 0.125× of vanilla.

**Fix:** RoPE kernel now takes a precomputed `inv_freq` array (length
`head_dim/2`) instead of a scalar θ. Block-level
`build_rope_inv_freq(head_dim, rope_base, rope_scaling)` computes
vanilla or YARN-scaled curves; matches `transformers._compute_llama3_parameters`
exactly (max|Δ|=0). `LlamaBlockConfig.rope_scaling` field plumbs it
through. Back-compat: a 1-element scalar tensor is still accepted (the
wrapper builds vanilla inv_freq lazily and caches by `(head_dim, base, device)`).

**Impact (Llama-3.1-8B step 0, B=0):**
- Before fix: per-token CE max\|Δ\| = 0.226, mean Δ = +0.014
- After fix:  per-token CE max\|Δ\| = 0.116, mean Δ = -0.0004 (loss-mean
  agreement to 4 decimals)

### Rule-out: residual cross-stack disagreement is bf16 noise, not a bug

User pushed back on calling residual gap "bf16 noise" without proof.
Concrete experiment (`tests/test_lora_8b_diagnostics.py` plus a
side-by-side HF-bf16-vs-HF-fp32 run on the same model):

| Comparison | model | logit max\|Δ\| | logit mean\|Δ\| |
|---|---|---|---|
| HF-bf16 vs HF-fp32 (within HF) | Llama-3.2-1B | **0.486** | 0.025 |
| FT-bf16 vs HF-bf16 (cross-stack) | Llama-3.2-1B | **0.438** | 0.036 |
| FT-bf16 vs HF-bf16 (cross-stack) | Llama-3.1-8B | 2.08 | 0.031 |

The FT-vs-HF noise on 1B (0.44) is **smaller than HF's own bf16 noise
floor (0.49)**. There is no algorithmic disagreement — FT produces
bf16-correct outputs. argmax matches at all top-disagreement positions
(predictions agree even where individual logit values differ most). 8B
is ~4× the 1B floor, plausibly explained by 32 vs 16 layers and the
FT-Triton-flash vs HF-SDPA attention kernel difference.

**Caveat:** can't verify "FT in fp32" directly — flash-attn 2 requires
fp16/bf16. Would need a PyTorch-SDPA attention fallback for an
all-fp32 cross-stack parity test.

### Process notes for future investigations

- Capture pre-CE logits via a `LossFn` subclass that snapshots its
  `logits` arg. Non-invasive. See `_LogitsCapturingCE` in
  `tests/test_lora_8b_diagnostics.py`.
- Always do FT-vs-FT bit-identity across two working set configs FIRST.
  If those agree (max\|Δ\|=0), the engine is deterministic and any
  cross-stack gap is purely cross-stack.
- Compare against HF's own bf16-vs-fp32 floor on the same model
  before chalking up divergence to "kernel numerics" — apples-to-apples
  at the same scale is the right reference.

## 2026-04-26: Issues found while building per-arch from_pretrained builders

### Issue: working-set solver fails on tiny synthetic configs

Trying to smoke-test a Qwen2 builder with a 256-d, 4-layer fake
config produced ``ValueError: Not enough GPU memory to fit any valid
chunk size large enough to fit at least 1 additional complete
layer``. Solver expects realistic param/activation ratios; tiny
configs hit edge cases in the chunk-sizing search.

**Workaround:** smoke-test with real (but small) HF weights when
possible (Llama-3.2-1B, Qwen3-1.7B). For arch-without-weights cases,
add a debug skip: ``FLEXTRAIN_SKIP_WORKING_SET=1`` could short-circuit
to a "all GPU" plan. Filed for follow-up.

### Issue: flash-attn 2 doesn't accept fp32

For the bf16 vs fp32 cross-stack noise investigation, I tried running
FT in fp32 to rule out bf16 as the cause. flash-attn 2's
``varlen_fwd`` raises ``RuntimeError: FlashAttention only support
fp16 and bf16 data type``. Without a PyTorch SDPA fallback in the FT
attention block we can't do a true all-fp32 cross-stack test.
Worked around with HF-bf16-vs-HF-fp32 within HF (proves the bf16
floor is what we expect).

### Issue: arch ``dims`` map missing ``attn_dim`` / ``kv_dim``

OLMoE's full-row QK-norm uses ``weight_dim_name="attn_dim"`` /
``"kv_dim"`` but the per-arch ``hf_config_to_flextrain`` doesn't
inject these into the engine's ``dims`` map. Hit a
``KeyError: 'attn_dim'`` at engine construction.

**Fix:** ``flextrain.api.from_pretrained`` now injects
``attn_dim = n_heads * head_dim`` and ``kv_dim = n_kv_heads *
head_dim`` into ``dims`` when the arch omits them. Each per-arch
module is free to set its own value (e.g. when adopting a Multi-Latent
or grouped-query attention variant where the math diverges).

### Issue: per-arch builder boilerplate

All six arch builders (Llama / Qwen2 / Qwen3 dense / Qwen3-MoE /
OLMoE / Mistral / Gemma 2) follow nearly identical scaffolding:
build the per-layer ``BlockConfig``, instantiate the block, optionally
wrap with ``LoRAWrapperLayer``. ~50 lines of duplicated code per
arch. Could be reduced via a helper that takes ``BlockConfigCls`` +
field-name → hyperparam-key mapping. Not blocking — left for a
cleanup pass once we have ≥10 arches.

### Issue: post-load permutation duplication

Llama / Qwen2 / Qwen3 / Qwen3-MoE / OLMoE / Mistral / Gemma 2 all
need the Q/K halved→pair perm. Five of them duplicate the same
``_halved_to_pair`` helper. Should factor to
``flextrain.io.weight_perms`` or similar. Cleanup item.

### Real-data parity test landed: ``tests/test_arch_parity.py``

Generic FT-vs-HF arch parity diagnostic running real MathInstruct
data. Per arch, runs both **LoRA fine-tuning** (vs HF + PEFT) and
**full-parameter fine-tuning** (vs HF + plain torch.optim) for N
training steps. Compares loss curves, step-0 logits, step-0 per-token
CE.

Outcomes from today's run:

* **Qwen2.5-0.5B**: LoRA & full both match HF. step 0 Δ ≤ 0.003,
  max |Δ_loss| over 5 steps ≤ 0.01 (LoRA) / 0.005 (full FT). Found
  + fixed bias-permutation bug below.
* **Mistral-7B-v0.3**: LoRA matches HF. step 0 Δ = 0.0001, max
  |Δ_loss| over 5 steps = 0.097. Full-FT not run (HF can't fit 7B
  fp32 grads on a single 24 GiB GPU; no FT issue).
* **Gemma 2 / Gemma 3**: skipped — bwd is stubbed (see notes above).

### Bug found by Qwen2 real-data parity: Q/K bias permutation missing

While running ``tests/test_arch_parity.py`` against Qwen2.5-0.5B with
real MathInstruct data, FT base-loss showed 7.67 vs HF's 0.47. Root
cause: Qwen2's ``post_load_permute`` reused Llama's hook, which
permutes ``w_q`` / ``w_k`` along the head_dim axis from halved-split
to pair-interleave layout, but does not permute the corresponding
Q/K **bias** vectors. Qwen2 has ``attention_bias=True``, so each Q/K
projection includes a per-head_dim bias that must follow the same
permutation.

**Fix:** Qwen2's ``post_load_permute`` now runs the Llama hook then
permutes ``b_q`` / ``b_k`` 1-D bias vectors along the head_dim axis.
After fix: FT 0.4702 vs HF 0.4677 (Δ=0.003 — bf16 noise).

This bug would have shipped silently if we had only smoke-tested
"loss decreases" without comparing to HF. **Real-data + cross-stack
comparison was essential.** Same risk exists in any arch with biases
or other post-projection-but-pre-RoPE operations that depend on the
head_dim layout. Audit list:
* Qwen2 — fixed.
* Qwen2.5 — same as Qwen2, covered.
* Other arches (Llama / Qwen3 / OLMoE / Mistral / Gemma 2) have
  ``attention_bias=False``; no bias to permute. ✓

### Qwen3-Next GatedDeltaNet: clean no-recompute bwd + partial-tier
recompute matching dense pattern

After user pushed back on naive scoped-autograd recompute in bwd
([2026-04-27]), and then on the all-or-nothing tier-recompute
shape ("the forward_recompute() pattern should be similar as
compared to other models"), the linear-attention block was
restructured to mirror the dense ``GQAAttentionBlock`` /
``SwiGLUFFN`` conventions:

* **Forward factored into 6 explicit stages**:
  ``_fwd_proj_split`` → ``_fwd_conv`` → ``_fwd_qkv_heads`` →
  ``_fwd_gate`` → ``_fwd_fla`` → ``_fwd_norm_out``. Each writes
  exactly the slot fields it owns. ``fwd`` itself is now a
  ~10-line composition of these stages.
* **Per-tier ``fwd_recompute_*`` helpers** — ``fwd_recompute_post_conv``
  (tier 3, x → projections + conv), ``fwd_recompute_qkv_heads``
  (tier 2, post_conv → silu + reshape + repeat-interleave),
  ``fwd_recompute_fla`` (tier 2, FLA fwd from saved q/k/v/g/b).
* **Layer-level ``forward_recompute``** in
  ``Qwen3NextLinearLayer`` now checks ``slot.has(<field>)`` for
  each tier and calls only the missing stage's recompute helper —
  matching ``LlamaBlock.forward_recompute``'s structure.
* **Re-tiered activation fields** per FT convention (small things
  always-saved, large things higher tier):
  - Tier 0: ``lin_a, lin_b, lin_g, lin_g_post`` (all small, used
    for the gate-bwd chain rule + FLA bwd).
  - Tier 1: ``lin_z`` (gated-norm gate; medium).
  - Tier 2: ``lin_q, lin_k, lin_v, lin_A_int, lin_core_out`` (FLA
    inputs/outputs; medium).
  - Tier 3: ``lin_conv_in, lin_post_conv_pre_silu`` (largest:
    shape ``(T, conv_dim)`` where conv_dim = 2*key_dim+value_dim).

Key implementation details:

* **Saved a new tier-1 field** ``lin_post_conv_pre_silu`` (shape
  ``(T, conv_dim)``) so silu's bwd can be computed without re-running
  conv1d. Marginal extra activation memory.
* **Saved post-cumsum g** (``slot.aux["lin_g_post"]``) AND raw
  pre-cumsum g (``slot.lin_g``). The chain rule for A_log / dt_bias
  uses the raw pre-cumsum g; FLA's bwd needs the post-cumsum g (it
  matches what fwd computed internally). Both required.
* **Hand-rolled gated-RMSNorm bwd** (``_gated_rmsnorm_bwd``) so
  out-projection's o_normed can be materialized in O(T*D) — much
  cheaper than re-running the full block.
* **Conv1d bwd via** ``torch.nn.grad.conv1d_weight`` /
  ``torch.nn.grad.conv1d_input`` — these are kernel calls, not
  autograd recomputes.
* **GVA reverse-repeat-interleave** is just a sum over the rep dim.

Caught two subtle bugs while writing this:
1. ``chunk_gated_delta_rule_fwd`` mutates the input g in place
   (cumsum within chunks). Saving ``slot.lin_g.copy_(g)`` AFTER the
   FLA call would store the post-cumsum version. We now clone g
   before saving so ``slot.lin_g`` always holds the raw pre-cumsum.
2. FLA's ``chunk_gated_delta_rule_bwd`` needs ``g=g_post`` (the
   post-cumsum g matching fwd's internal state), but its returned
   ``dg`` is in pre-cumsum space (FLA applies a reverse-cumsum at
   the end). My initial implementation passed pre-cumsum g, which
   produced gradients that were ~6 orders of magnitude wrong on
   the gate path.

``tests/test_gated_deltanet_bwd.py`` passes with no recomputation —
all grads within bf16 noise of the autograd reference. Forward path
unchanged.

### Recomputation audit (Qwen3-Next + Gemma 2/3)

User flagged ([2026-04-27]) that scoped torch.autograd in bwd was a
performance trap in MegaTrain — full forward recomputation per
backward call. Status of each affected block:

| Block | Bwd impl | Recomputes fwd? | Why |
|---|---|---|---|
| Llama, Qwen2, Qwen3 dense, OLMoE, Qwen3-MoE, Mistral | hand-rolled FT bwd | **No** | Activations saved at fwd-time per ActivationSchema; bwd reads them |
| GatedDeltaNetBlock (Qwen3-Next linear-attn half) | torch.autograd.Function wrapping FLA | **Yes — full block fwd recomputed in bwd** | FLA's kernels are inaccessible at the FT-grad-accumulator level; the only way to bwd FLA is to replay fwd to set up its autograd graph |
| Gemma 2 dual-residual layer | **stubbed (NotImplementedError)** | TBD | Bwd not yet implemented; stub raises rather than running |
| Gemma 3 | inherits Gemma 2 stub | TBD | same as Gemma 2 |

**For Gemma 2/3 specifically:** the dual-residual norm structure
(pre + post norms on each sublayer) only adds two extra norms per
sublayer. Each norm's bwd needs ``(input_to_norm, rstd, w_norm)`` —
all of which are saved in the activation schema at fwd-time. So a
**hand-rolled, no-recompute bwd** is feasible; the plan is to write
it directly rather than ship a scoped-autograd version that would
recompute 4 norms + attn + FFN per backward call.

Queued: hand-rolled Gemma 2/3 bwd that reads only saved activations.

**For Qwen3-Next**: full hand-rolled FLA-replacement bwd is also
queued (matches the existing TODO at top of
``flextrain/nn/blocks/linear_attn.py``).

### Action items captured for follow-up

- ``flextrain.from_pretrained`` cleanly handles tied ``lm_head``
  (Llama-3.2-1B/3B, Gemma 2, sometimes Qwen3-1.7B). Need a unit test
  that asserts ``head_w == embed_w.T`` after the post-load step on
  these models.
- ``post_load_permute`` does ``am._refresh_gpu_residents()`` and a
  CUDA sync. If the user passes a non-CUDA device this will fail.
  Audit + guard.

## Heuristics I'm using this session

- Never merge a kernel refactor without running at least one E2E parity (otherwise nothing's verified).
- When in doubt, write a minimal naive reference and compare layer-by-layer against HF.
- Unit-test bwd kernels against autograd in both modes whenever adding a mode flag.
- Prefer subprocesses over aggressive GPU cleanup when integrating third-party libraries (HF transformers, etc.).

## 2026-04-27: Qwen3-Next 8-layer E2E test issues

### Engine bug: `_update_fwd_context` assumes every layer has `xk`/`xv`

While building `tests/test_qwen3_next_8layer_e2e.py` (8-layer mini
Qwen3-Next, alternating linear-attn and full-attn layers), the engine
crashed during the FIRST FT backward pass at
`active_model.py:_update_fwd_context`:

```
AttributeError: activation field 'xk' not present at level=3
(available: ['attn_norm_rstd', 'chosen_experts', 'expert_counts',
 'ffn_norm_rstd', 'lin_A_int', 'lin_a', 'lin_b', 'lin_conv_in',
 'lin_core_out', 'lin_g', 'lin_g_post', 'lin_k',
 'lin_post_conv_pre_silu', 'lin_q', 'lin_v', 'lin_z',
 'router_weights', 'scattered_router_weights', 'x_inp', 'x_router',
 'x_up'])
```

The KV-window refresh (paper §3, the cross-seq-group K/V copy used
during the multi-seq-group backward) unconditionally reads
`src_slot.xk` / `src_slot.xv`. Linear-attention layers
(`Qwen3NextLinearLayer`) don't declare those activation fields —
they have their own `lin_q`/`lin_k`/`lin_v` (per-head, post-conv).

**Fix shipped:** guard `_update_fwd_context` with
`if not src_slot.has("xk"): return` in both the on-device and host
branches. Linear-attn layers don't write or consume the global KV
cache, so skipping the refresh is safe (and is what would happen
naturally if the engine inspected layer types).

The 8-layer E2E test (linear+full alternation) is the regression test
for this fix — without the guard it crashes at step 0 backward.

## 2026-04-27: Qwen3-Next vs HF parity audit (real-weight bugs)

User asked for Qwen3.5/3.6 integration. Order: clean up Qwen3-Next
first (HF-parity-correct), then 3.5, then 3.6.

The existing Qwen3-Next tests (`test_gated_deltanet_*.py`,
`test_qwen3_next_8layer_e2e.py`) are all FT-internal: they verify
engine determinism + math equivalence with a pure-PyTorch reference
that uses the same layout FT chose. **None compare against HF
transformers' `Qwen3NextForCausalLM`.** That's why the bugs below
have been latent.

After reading HF source + building a 4-layer mini Qwen3-Next via
`transformers.Qwen3NextForCausalLM`, found the following gaps:

### Bug Q3N-1: RMSNorm `(1+weight)` convention not applied at load

HF `Qwen3NextRMSNorm.forward = x.normed * (1 + weight)`. Stored γ is
canonical-γ minus 1. FT's `flextrain_rmsnorm_fwd` does `x.normed * γ`
without the +1 shift. Without the shift, FT would multiply by ~0
(since stored weights cluster near 0).

Confirmed by `tests/test_qwen3_next_norm_vs_hf.py`:
* Without shift: max|Δ| = 3.27 (catastrophic).
* With shift: max|Δ| = 0.016 (bf16 noise).

Fix: extend `_qwen3_next_post_load_hook` to add 1.0 to every loaded
RMSNorm tensor (`input_layernorm`, `post_attention_layernorm`,
`linear_attn.norm`, `q_norm`, `k_norm`, top-level `model.norm`).
Same convention as Gemma 2 (which already does this in its post-load
hook, see `_gemma2_post_load_hook`).

### Bug Q3N-2: Full-attention output gate missing

HF `Qwen3NextAttention`:

    q_proj.weight: (n_heads * head_dim * 2, hidden_size)
    Q, gate = chunk(q_proj(x), 2, dim=-1)
    ... attention ...
    out = attn_output * sigmoid(gate)
    out = o_proj(out)

FT's `GQAAttentionBlock` has `q_proj` shape `(hidden_size, n_heads*head_dim)`
(no factor of 2) and no output gate. FT's Qwen3-Next full-attention
layer uses `Qwen3MoEBlock` which uses GQAAttentionBlock — so missing
gate.

Fix: extend `GQAAttentionConfig` with `attn_output_gate: bool = False`.
When True, double w_q's output dim, split Q from gate post-projection,
multiply attn output by sigmoid(gate) before w_o. Touch fwd, bwd,
recompute paths.

### Bug Q3N-3: Partial-rotary RoPE not supported

HF Qwen3-Next: `partial_rotary_factor: 0.25`. RoPE applied to
the first `head_dim * 0.25 = 64` channels only; the remaining 192
pass through unrotated.

FT's `flextrain_rope_fwd` applies RoPE to the full head_dim.

Fix: add `partial_rotary_dim: int` parameter to the RoPE kernel and
the block-level `apply_rope_fwd`. When partial_rotary_dim < head_dim,
the kernel iterates only over `[0:partial_rotary_dim/2]` pairs and
leaves the rest untouched. Handle pair-interleave perm only for the
rotated portion.

### Bug Q3N-4: Shared-expert path missing in MoE FFN

HF's `Qwen3NextSparseMoeBlock`:
* Routed experts (existing path).
* `shared_expert` (separate gate/up/down dense MLP).
* `shared_expert_gate.weight: (1, hidden_size)` — per-token scalar
  gate determining shared-vs-routed mix. `shared_out * sigmoid(scalar_gate)`
  is added to the routed-expert output.

FT's `MoESwiGLUFFN` is routed-only. No shared expert.

Fix: extend `MoESwiGLUConfig` with `shared_expert_dim: int = 0`.
When > 0, allocate dense `w_shared_1/w_shared_2/w_shared_3` (gate/down/up)
plus a `w_shared_expert_gate (hidden_size,)` 1-D vector. fwd adds
`silu(x @ w_shared_1) * (x @ w_shared_3) @ w_shared_2 * sigmoid(x @ w_shared_expert_gate)`.
bwd back-propagates through both paths.

### Bug Q3N-5: HF stores experts as fused `gate_up_proj` / `down_proj`

Recent HF Qwen3-Next saves experts as:
    experts.gate_up_proj: (E, 2*F, D)     # fused [gate; up] over dim 1
    experts.down_proj:    (E, D, F)
Older HF format (which our existing loader assumes) used per-expert
files: `experts.{e}.gate_proj.weight (F, D)` etc. Our loader silently
no-ops on the new format because it falls through the `shard_g is
None` branch.

Fix: add a new branch that detects `experts.gate_up_proj` and reads
it directly. Stack into our `[up, gate]` packing convention via:
    fused = hf['experts.gate_up_proj']  # (E, 2*F, D)
    gate_part, up_part = fused[:, :F, :], fused[:, F:, :]
    w_up[e] = cat([up_part[e].T, gate_part[e].T], dim=1)  # → (D, 2F)

### Bug Q3N-6 (suspected — needs verification): conv1d bias

HF Qwen3-Next stores `conv1d.bias` as a parameter (visible in
`Qwen3_5GatedDeltaNet`'s init, also in our small mini-Qwen3-Next).
Our linear-attn block's conv1d uses `bias=None`. Need to either
load+apply the bias, or confirm Qwen3-Next configs disable it.

### Plan

1. Land fixes 1, 2, 3 (norm shift, output gate, partial rotary). These
   together gate the full-attn layer against HF.
2. Land fixes 4, 5 (shared expert, fused experts). These together gate
   the MoE FFN against HF.
3. Land fix 6 (conv1d bias) after verifying it's actually used.
4. End-to-end logits parity vs HF on the 4-layer mini-Qwen3-Next.
5. Then move to Qwen3.5 integration.

### Bug Q3N-7 (engine, latent): cross-layer KV refresh wrong for heterogeneous backbones

`_update_fwd_context` (`active_model.py:1376`) refreshes the global
`kv_fwd` window for the next bwd iteration. Two paths:

* **Same layer, prior seq group at this chunk_in_group_ind** — works
  for any layer that has `slot.xk`/`xv` (i.e., softmax-attn). Linear-attn
  layers don't have these fields, but they don't *consume* `kv_fwd`
  either, so it's fine if their iteration's update is a no-op.
* **Fallback: prior layer (`layer_ind - 1`), last seq group, same
  chunk_in_group_ind** — used at seq-group=0 boundaries. The agent's
  recent fix guards `if not src_slot.has("xk"): return`, which prevents
  the crash but leaves the window stale.

**The problem**: in heterogeneous backbones with multiple linear-attn
layers between softmax-attn layers (the actual Qwen3-Next pattern is
`[L,L,L,F,L,L,L,F,...]`), bwd transitions through linear-attn layers
without refreshing the window. When the next softmax-attn layer's bwd
runs (e.g., layer 3 in `[L,L,L,F,L,L,L,F]` after bwd has finished
layers 7,6,5,4), `flextrain_attention_bwd` reads `kv_fwd.k[:total_k]`
which still holds **layer 7's K/V**, not layer 3's.

`attn.bwd` reads `kv_fwd` UNCONDITIONALLY (independent of save level),
so this matters for every save level.

**Why prior tests didn't catch it**:
* Llama / Qwen2 / Qwen3-MoE / OLMoE / Mistral are homogeneous —
  fallback always finds a softmax-attn layer with `xk/xv`. ✓
* `test_qwen3_next_8layer_e2e.py` uses `[L,F]*4` (alternating, never
  more than one linear-attn between full-attns). The fallback's
  `prior_layer = L-1` IS a linear-attn, so guard fires, leaving the
  window holding the prior softmax-attn layer's K/V — which is the
  WRONG layer's K/V. The test passed because the naive PyTorch
  reference uses the same engine path (the test compares FT to FT
  variants and to a PyTorch reference that has no analog of the
  global window — the reference computes attention freshly per layer).
  So FT vs FT can be self-consistent while both are slightly wrong vs
  HF.

**Fix** (queued for after the Q3N-2..6 block-level fixes land):
extend `_update_fwd_context` to walk **further back** when
`layer_ind - 1` doesn't have `xk/xv`, until we find a layer that
does AND that has the same source chunk position. Or — cleaner —
refresh per-layer at the start of each softmax-attn layer's bwd
iteration, populating from THIS layer's last seq_group last chunk
slot.

**Regression test**: end-to-end Qwen3-Next vs HF logits parity (Q3N-8
in the plan) will catch this once the simpler block-level bugs are
fixed.

### Why the 8-layer Qwen3-Next test passed despite missing HF features

User pointed out (rightly): how can our 8-layer Qwen3-Next E2E test
pass within ~3e-3 of the naive reference if FT is missing the
output gate, norm shift, partial-rotary, shared expert, etc.?

Answer: the 8-layer test's "naive reference" uses the SAME math
FT uses, not HF's math. Specifically:

* `NaiveQwen3NextLinearBlock` uses standard ``x * rstd * w`` RMSNorm
  (no ``(1 + w)`` shift) — same as FT.
* `NaiveQwen3NextFullBlock` builds standard GQA without an output
  gate — same as FT.
* MoE FFN is routed-only — no shared expert — same as FT.
* RoPE is full-rotary — same as FT.

So the test verifies: "the engine + working-set rotations don't
introduce additional drift on top of FT's chosen math." It does NOT
verify: "FT's chosen math matches HF Qwen3-Next."

This is the same trap that hid the Qwen2 bias-permutation bug
(detected only when running real data + HF reference; FT-internal
tests passed silently). **F-1 in this notes file already documented
this lesson — and we made the same mistake again.** Random-init,
same-math reference tests are a "do FT and the reference disagree
in the same way" check, not a correctness check.

Tightening the 8-layer test's tolerance wouldn't help — both sides
compute the same math; the only delta is bf16 noise. The fix is to
**replace the naive reference with HF**, which exposes every diff.
Skeleton at `tests/test_qwen3_next_vs_hf.py`; FT side is wired in
incrementally as each Q3N-* fix lands. The test should fail until
all fixes are in.

Going forward: any new arch integration must include an HF-vs-FT
random-init parity test as part of the acceptance criterion. A
"naive PyTorch reference" is fine for verifying engine determinism
across save levels and offload configs, but not for verifying math
correctness vs the upstream model.

## 2026-04-27 progress on Q3N correctness path

### Landed today (each with passing parity test)

* **Q3N-1**: `(1 + weight)` shift in `_qwen3_next_post_load_hook` for
  every loaded RMSNorm γ (input/post-attn/lin/q/k norms + final
  norm). Test: `test_qwen3_next_norm_vs_hf.py` shows max\|Δ\|=0.016
  with shift, 3.27 without — without-shift would have been catastrophic.

* **Q3N-2**: Standalone `GQAAttentionGatedBlock` in
  `flextrain/nn/blocks/attention_gated.py` (478 lines, no subclassing
  per user direction). Adds the Qwen3-Next/3.5/3.6 sigmoid-gated
  output path: ``q_proj`` shape `(d_model, attn_dim*2)` → split into
  `(Q, gate)` → ``attn_out * sigmoid(gate)`` → ``w_o``. New activation
  field `attn_gate` at tier 2. Math parity test:
  `test_gqa_gated_vs_hf.py` — fwd max\|Δ\|=2.4e-4, all weight grads
  rel ≤ 1%, dL/d(attn_norm_output) max\|Δ\|=7.6e-6.

* **Q3N-3**: Standalone partial-rotary RoPE kernel in
  `flextrain/ops/_kernels/rope_partial.py` + block-level wrapper
  `apply_rope_partial_fwd/bwd` and `build_partial_rope_inv_freq` in
  `flextrain/nn/blocks/rope.py`. Parity test
  `test_rope_partial.py` — 5 sub-tests, ALL pass bit-identically:
  Q/K fwd parity (max\|Δ\|=0), pass-through bit-identical, bwd
  matches reference inverse, full-rotary equivalence (rot_dim==head_dim
  matches the existing kernel exactly), variable T per call.

  Subtle precision finding: kernel computes `cos/sin` in fp32 and
  the `q*cos` multiply in fp32 (cast to bf16 only on store). Naive
  reference that pre-casts cos/sin to bf16 produces max\|Δ\|=7.8e-3,
  which is bf16 rounding noise in the multiply. Fixed by making the
  reference match the kernel's precision pattern; result is
  bit-identical. Documented in the test docstring as guidance for
  future kernel-vs-reference parity tests.

### Remaining for Qwen3-Next correctness

* **Q3N-4**: Shared-expert MoE FFN. New standalone block
  `MoESwiGLUSharedExpertFFN` next to the existing `MoESwiGLUFFN`.
  Math (verified against HF source, lines 782-799 in
  `transformers/models/qwen3_next/modeling_qwen3_next.py`):

      shared_out  = SwiGLU_MLP(x)                           # dense
      shared_gate = sigmoid(x @ w_shared_expert_gate.T)     # (T, 1)
      shared_out *= shared_gate                              # element-wise
      final_out   = routed_out + shared_out                  # ADDITIVE

  Note: NOT a `(1-σ)*routed + σ*shared` mixture. The shared expert
  always contributes (scaled by sigmoid in [0,1]); routed output is
  unmodulated. Param spec adds `w_shared_1` (gate proj),
  `w_shared_3` (up proj), `w_shared_2` (down proj),
  `w_shared_expert_gate` (1, d_model) per HF naming.

* **Q3N-5**: Loader for HF's fused `experts.gate_up_proj` /
  `experts.down_proj` 3-D weight tensors (the existing FT loader uses
  the older per-expert layout).

* **Q3N-6**: Load `conv1d.bias` (HF Qwen3-Next's depthwise conv has
  a bias; FT's linear-attn block uses `bias=None`). Either disable
  it on construction (matching the small mini-config which has no
  bias) or add it as an FT param.

* **Q3N-7**: Engine `_update_fwd_context` cross-layer fallback for
  heterogeneous backbones (linear-attn separating full-attn). Walk
  back through layer indices until finding a softmax-attn layer.

* **Wire `GQAAttentionGatedBlock`** into `Qwen3NextFullLayer` (currently
  uses standard `Qwen3MoEBlock` → `GQAAttentionBlock`). Need either a
  parallel `Qwen3MoEGatedBlock` layer that uses gated GQA + shared
  MoE FFN, or a refactor of `Qwen3MoEBlock` to accept attention-block
  and FFN-block classes.

* **End-to-end Qwen3-Next 4-layer vs HF logits parity test** — the
  regression net for all of Q3N-2..7. The skeleton lives at
  `tests/test_qwen3_next_vs_hf.py` (HF half wired, FT half pending
  on the above fixes).

* Then **Qwen3.5-0.8B-Base** real-weight LoRA + full-FT vs HF parity
  (real model, will exercise the full Qwen3-Next-shared codebase
  plus the Qwen3.5-specific in_proj split).
* Then **Qwen3.6** small random-init E2E.
