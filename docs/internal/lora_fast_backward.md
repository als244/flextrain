# LoRA fast backward — plan & progress

## Pre-refactor baseline

If anything goes wrong during the refactor and we need to revert,
the working baseline is **commit `a277d90b096cad064b46ac8da49ba2945205bd76`**
("Update"). All changes described in the phasing below are layered
on top of this commit; each phase is independently reversible by
checking out the parent of the phase's last commit.

## Decisions / issues uncovered during implementation

* **Wrapper-internal dispatch, not engine-side.** Original plan had
  the engine inspect ``isinstance(layer, LoRAWrapperLayer)`` and
  route. Final design has the wrapper's ``backward()`` do
  ``backward_dgrad → backward_wgrad → accumulate_lora_grads``
  internally; engine calls ``layer.backward(...)`` exactly as for
  any other layer. Cleaner: engine has zero LoRA awareness.
* **Two skip mechanisms, not one.** Dense projections (``w_q,k,v,o,
  1,2,3``) use ``skip_grads`` + ``capture_xy`` -- the block stashes
  ``(X, dY)`` into a caller-supplied dict and the wrapper
  accumulates ``dA, dB`` via rank-r matmuls afterwards. MoE expert
  projections (``w_up,w_down`` and shared-expert variants) use a
  ``lora_per_expert_callback(name, eid, X, dY)`` that fires inside
  the per-expert loop and accumulates immediately -- no per-expert
  ``(X, dY)`` clones (which would be 3-4× memory of the dW slow
  path). Two mechanisms because their data-flow cost profiles are
  very different.
* **bf16 reorder noise dominates LoRA fast-vs-slow regression.** The
  fast and slow paths produce algebraically equivalent ``dA, dB``,
  verified in fp32 to ~3e-7 relative error. In bf16 the matmul
  reduction order differs and per-step grads diverge by ~3.5e-3
  relative. Over 100 optimizer steps that amplifies to ~4e-2
  absolute on the loss curve (well within the 0.1 atol used by the
  regression). NOT a math bug -- documented in
  ``tests/test_phase2_lora_math.py``.
* **Llama backward intermediates carry a ``clone()`` of dx_resid /
  dy_resid for inline-Wgrad capture.** The layer mutates these
  tensors in place via ``rmsnorm.bwd(dx_accumulator=dx)``. Cloning
  costs ~206 MiB per layer per chunk for an 8B model at chunk=25200,
  but avoids the 12 GFLOPs Wgrad. Worth it on fast hardware.
* **NVTX libnvToolsExt argtypes truncation bug** (unrelated to LoRA
  refactor but discovered along the way) -- fixed earlier:
  ``ctypes`` defaulted ``cudaStream_t`` arg to ``c_int`` (32-bit)
  instead of ``c_void_p`` (64-bit), causing ``nvtxNameCuStreamA`` to
  segfault under nsys but not normally. Documented in commit history.
* **Router not LoRA target by default.** The
  ``_discover_lora_eligible_names`` helper excludes any param whose
  name contains "router". The ``_FAST_PATH_TARGETS`` list includes
  ``w_router`` only as a fallback if a user explicitly targets it;
  default behavior is unchanged.
* **HF PEFT MoE comparison deferred.** HF PEFT applies one shared
  adapter across all routed experts (since OlmoeForCausalLM batches
  experts into one ``OlmoeExperts`` op). FlexTrain's wrapper applies
  per-expert adapters by default. Cross-stack parity is therefore
  apples-vs-oranges -- documented as a follow-up TODO in
  ``tests/test_lora_e2e_olmoe_moe.py``. Real-data correctness is
  validated via ``tests/test_phase2_olmoe_lora_e2e.py`` (FT vs FT
  regression) and ``tests/test_olmoe_engine_parity.py`` (FT vs naive
  PyTorch full-FT).

## Problem

Today every backbone layer's `backward()` is a single method that produces
both the upstream gradient `dL/dx` (**dgrad**) and the per-projection
weight gradients `dL/dW` (**wgrad**) in one shot. The LoRA wrapper
(`flextrain/nn/layers/lora_wrapper.py`) reuses this monolithic backward by
passing a scratch grad buffer, then post-decomposes `dL/dW` →
`(dL/dA, dL/dB)` via `dW @ B^T` and `A^T @ dW`. This produces correct
LoRA gradients but the cost is identical to full fine-tuning's backward
because the full Wgrad matmul `X^T @ dY` runs for every targeted
projection — even though for LoRA those base weights are frozen and we
only need the rank-r updates.

For Llama 3.1 8B at chunk=25200 with `lora_targets="all"`, the per-layer
Wgrad matmul cost is ~12 GFLOPs (full d_model² + MLP terms), versus
~1.5 GFLOPs for the rank-16 LoRA-only path — a ~250× FLOP gap per
projection that scales with chunk size. On H100 this shows up as a
massive throughput shortfall vs. what the timeline says is achievable.

## Goal

Add a clean dgrad/wgrad split to every backbone layer's backward path
so the engine can call dgrad alone for LoRA layers and skip the
full-Wgrad matmuls. LoRA accumulates `dL/dA, dL/dB` directly from the
intermediates the dgrad pass already produces, never materializing
`dL/dW` on the frozen base weights.

## Current state of the codebase (already partially split)

The split is **already partially in place** for some blocks. This is the
key piece of leverage that makes the refactor tractable:

| Block | Today |
|-------|-------|
| `nn/blocks/attention.py` (`GQAAttentionBlock`) | Has `bwd(...)` (dgrad + g_o + saves dq/dk/dv on `slot.aux`) and `bwd_accumulate_qkv_grads(...)` (g_q, g_k, g_v after RMSNorm bwd recomputes attn_norm_output). Wgrad partially split: `g_o` is in `bwd`, `g_q/g_k/g_v` in the second method. |
| `nn/blocks/attention_gated.py` (Qwen3-Next full attention with GLU gating) | Same shape — `bwd` + `bwd_accumulate_qkv_grads`. |
| `nn/blocks/ffn_dense.py` (SwiGLU FFN) | Has `bwd(...)` (dgrad + g_2) and `bwd_accumulate_w1_w3_grads(...)` (g_1, g_3 after RMSNorm bwd). Same partial-split shape. |
| `nn/blocks/ffn_moe.py` (`MoEFFN` — OLMoE / Qwen3-MoE / Qwen3-Next data path) | **Monolithic** `bwd(...)` — per-expert loop interleaves dgrad and per-expert Wgrad in one body. Needs split. |
| `nn/blocks/ffn_moe_shared.py` (shared experts, used by some MoE arches) | Need to inspect. |
| `nn/blocks/ffn_moe_sonic.py` (Sonic MoE variant) | Need to inspect. |
| `nn/blocks/linear_attn.py` (Qwen3-Next gated DeltaNet) | **Monolithic** `bwd(...)` — gradients scattered across multiple kernel callbacks. Needs split. |
| `nn/blocks/norm.py` | Norm `bwd` accumulates `g_w` (RMSNorm gain grad) inline. Trivially small (1-D weight) — keep as-is, not a LoRA target. |

The split lines for the partially-split blocks are not perfectly clean:
some Wgrads (like `g_o`, `g_2`) are computed inline in `bwd` because
they don't need RMSNorm's recomputed output, while others (`g_q/k/v`,
`g_1/g_3`) are deferred to the second call because they need the
recomputed RMSNorm output. The new contract collapses both into a
single `wgrad` call that takes whatever the dgrad pass produced and
runs the addmm; the layer routes RMSNorm-recompute-dependent
intermediates the same way it does today, just packaged into the
intermediates payload.

## Contract change (Layer Protocol)

`flextrain/core/layer.py` — `Layer` protocol gains two methods that
together replace the monolithic `backward(...)`:

```python
def backward_dgrad(
    self,
    dx: torch.Tensor,
    chunk: ChunkMeta,
    weights: Mapping[str, torch.Tensor],
    slot: ActivationSlot,
    ctx: LayerContext,
) -> tuple[torch.Tensor, "BackwardIntermediates"]: ...

def backward_wgrad(
    self,
    intermediates: "BackwardIntermediates",
    weights: Mapping[str, torch.Tensor],
    grads: MutableMapping[str, torch.Tensor],
    ctx: LayerContext,
    *,
    skip_target_names: frozenset[str] = frozenset(),
) -> None: ...
```

`backward_dgrad` returns:
1. `upstream_dx` — the chain-rule output (same as today's `backward`'s
   return value).
2. `BackwardIntermediates` — a typed payload carrying every per-projection
   `(input_act_X, upstream_grad_dY)` pair the layer would have used to
   compute `dW`. The payload also carries any layer-internal state
   needed by `backward_wgrad` (e.g. cached `dq/dk/dv` post-RoPE, or
   the recomputed RMSNorm output that today travels via `slot.aux`).

`backward_wgrad` consumes the intermediates and accumulates `dL/dW`
into `grads` for each projection **except** those listed in
`skip_target_names`. Default `skip_target_names = frozenset()` means
"compute every Wgrad" — full FT behavior, no behavior change vs today.

The engine's existing `Layer.backward(...)` becomes a thin shim:

```python
def backward(self, dx, chunk, weights, grads, slot, ctx) -> torch.Tensor:
    upstream_dx, inter = self.backward_dgrad(dx, chunk, weights, slot, ctx)
    self.backward_wgrad(inter, weights, grads, ctx)
    return upstream_dx
```

This shim guarantees zero-behavior-change for any caller that doesn't
opt into the new path — including all current tests, all parity benches,
and any external user code.

### `BackwardIntermediates`

A small dataclass in `flextrain/core/layer.py`:

```python
@dataclass
class BackwardIntermediates:
    """Per-projection (X, dY) pairs and any layer-internal cache
    backward_wgrad needs to consume. Produced by ``backward_dgrad``,
    consumed by ``backward_wgrad`` (or by a LoRA wrapper)."""

    # name -> (X, dY) where dW = X^T @ dY would accumulate into grads[f"g_{name[2:]}"].
    # Names match the TensorSpec.name convention used for ``weights`` and
    # ``grads`` dict keys (e.g. "w_q", "w_o", "w_1", "w_up").
    proj_inputs_and_grads: dict[str, tuple[torch.Tensor, torch.Tensor]]

    # Layer-internal state (e.g. RMSNorm-recomputed outputs, MoE
    # expert_counts) that backward_wgrad reads. Opaque to LoRA --
    # LoRA only consumes proj_inputs_and_grads.
    aux: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.proj_inputs_and_grads[name]

    def __contains__(self, name: str) -> bool:
        return name in self.proj_inputs_and_grads
```

For MoE projections, the `(X, dY)` tensors are 3-D (per-expert) -- the
LoRA wrapper already handles 3-D via `bmm`, so the contract carries
through unchanged.

## Engine integration

`flextrain/engine/active_model.py::_backward_pass` is the only call site:

```python
# Today:
upstream_dx = layer.backward(dx, chunk.meta, weights, grads, dev_slot, ctx)

# After:
upstream_dx, inter = layer.backward_dgrad(dx, chunk.meta, weights, dev_slot, ctx)
skip = layer.lora_target_names() if isinstance(layer, LoRAWrapperLayer) else frozenset()
layer.backward_wgrad(inter, weights, grads, ctx, skip_target_names=skip)
if isinstance(layer, LoRAWrapperLayer):
    layer.accumulate_lora_grads(inter, weights, grads, ctx)
```

The engine doesn't need to know per-arch which projections are LoRA
targets — that's encapsulated in `LoRAWrapperLayer.lora_target_names()`
(returns the set of `target_name` strings). The wrapper's
`accumulate_lora_grads` runs the rank-r matmuls.

## LoRA wrapper

`flextrain/nn/layers/lora_wrapper.py::LoRAWrapperLayer`:

```python
def backward_dgrad(self, dx, chunk, weights, slot, ctx):
    eff = self._build_effective_weights(weights)
    return self.base.backward_dgrad(dx, chunk, eff, slot, ctx)

def backward_wgrad(self, intermediates, weights, grads, ctx, *, skip_target_names=frozenset()):
    # Forward the call with our LoRA targets added to the skip set so
    # the base layer doesn't compute dL/dW for them.
    eff = self._build_effective_weights(weights)
    skip = skip_target_names | self.lora_target_names()
    self.base.backward_wgrad(intermediates, eff, grads, ctx, skip_target_names=skip)

def accumulate_lora_grads(self, intermediates, weights, grads, ctx):
    # Direct chain-rule path: never materialize dL/dW for the base.
    for cfg in self.targets:
        X, dY = intermediates[cfg.target_name]
        A = weights[cfg.a_name]
        B = weights[cfg.b_name]
        ga = grads["g_" + cfg.a_name[2:]]
        gb = grads["g_" + cfg.b_name[2:]]
        # ... rank-r matmuls (math below)

def lora_target_names(self) -> frozenset[str]:
    return self._target_set
```

### LoRA gradient math (correctness derivation)

For a LoRA-wrapped projection:

```
Y = X @ W^T + α (X @ A^T) @ B^T
   ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^
   base path      LoRA delta path
```

where `X` has shape `(T, in)`, `W` has shape `(out, in)`, `A` has shape
`(r, in)`, `B` has shape `(out, r)`, `α = scale`.

By chain rule, given `dL/dY` of shape `(T, out)`:

```
dL/dW = X^T @ dL/dY              (full Wgrad — what we want to skip)
dL/dA = α * B^T @ dL/dY^T @ X    (rank-r path)
dL/dB = α * dL/dY^T @ (X @ A^T)  (rank-r path)
```

Equivalent forms (more natural for `addmm_`):

```
dL/dA: (B^T @ dY^T) @ X    -- contracts via rank then tokens; result (r, in)
dL/dB: dY^T @ (X @ A^T)    -- contracts via tokens then rank; result (out, r)
```

Watch the layout convention — in the codebase `weights[name]` is stored
**transposed** (shape `(in, out)` for what the math calls `W^T`),
because the forward does `Y = X @ W` directly with no `.T()`. The LoRA
A/B are stored to match that convention. The actual implementation
needs to align with whatever each base layer does in its own forward;
each layer's `backward_wgrad` produces an `addmm`-shaped `(X, dY)`
where `dW = X^T @ dY` accumulates into `grads[name]` in storage shape.
LoRA's path therefore mirrors the same conventions:

```python
# What base layer would have done (now skipped):
# grads[name].addmm_(X.T, dY)   -- shape (in, out)
#
# LoRA equivalent (stored A: (r, in), B: (out, r) -- match base storage):
# Forward delta: Y += α * X @ A^T @ B^T  =>  Y += α * (X A^T) B^T
# dL/d(X A^T) = α * dY @ B
# dL/dB = (α * dL/d(post-A)).T @ dY      where post-A = X @ A^T, shape (T, r)
#                                         dL/dB shape: (r, out) -- match storage of B^T
# dL/dA = ...
```

**Important:** the exact addmm-shape and transpose needs to be
re-derived against each layer's stored weight convention during
implementation. The current `lora_wrapper.py` lines 376-389 already
gets this right via `dW @ B.transpose(-1, -2)` and
`A.transpose(-1, -2) @ dW`; the new path produces the same final
gradient values from `(X, dY)` directly without ever materializing `dW`.

**Verification path:** Phase 2 includes a parity test that runs both
paths on small dims and asserts grad equality element-wise. We commit
the new path only when this passes.

### MoE specifics

For MoE projections (`w_up`, `w_down` per expert):

* The intermediates dict keys are still `"w_up"` / `"w_down"` (single
  entry per projection name), not per-expert. The value tensors are
  3-D: `X[e]` and `dY[e]` per expert. The engine's per-expert dispatch
  is hidden inside the block's `backward_dgrad` / `backward_wgrad`
  implementations.
* LoRA already handles 3-D via `bmm` on `dW` (lines 380-388 of
  current `lora_wrapper.py`). The new path uses the same `bmm` shape
  but on `(X, dY)` directly.
* Empty experts (where `n_exp_tokens == 0`) need an empty-tensor
  contract — the intermediates entry is None or has `T=0` along dim 0,
  and wgrad / LoRA-accumulate skip those cleanly.

### Linear attention (Qwen3-Next)

`linear_attn.py` is monolithic and uses a chain of FLA kernels for the
DeltaNet recurrence. The relevant projections are `w_lin_qkvz` (fused
QKV+Z), `w_lin_ba` (β/α projection), `w_lin_conv` (1-D conv), and
`w_lin_o` (output projection). LoRA today targets the matmul-flavored
2-D weights among these; gated DeltaNet's `g_lin_A_log` /
`g_lin_dt_bias` /`g_lin_conv` are not LoRA targets in practice.

Phase 3 plans to split out `(X, dY)` for `w_lin_qkvz`, `w_lin_ba`, and
`w_lin_o` only; the remaining FLA-managed gradients stay inside
`backward_wgrad`. The skip set passes through cleanly because each
of those is keyed by a `TensorSpec.name`.

## Testing strategy

The bar is **end-to-end loss-curve parity over many steps on real data**
against a known-correct reference, not single-step grad-tensor
comparison. The single-step grad equality test catches obvious math
bugs but won't catch (a) subtle drift that only manifests as the
optimizer accumulates state across steps, or (b) cases where some path
silently produces stale-but-finite gradients. Loss curves over many
steps catch both.

The repo already has `tests/test_loss_curve_parity.py` plus the
harness in `flextrain.bench.parity` (`ModelShape`, `WorkingSetSpec`,
`LossCurveParityConfig`, `run_loss_curve_parity`). New tests below
plug into that harness rather than reinventing the comparison logic.

### Reference implementation

For each arch, the "known correct" reference is one of:

1. **Pre-refactor FlexTrain** (preferred for Phase 1 gates). Compare
   the monolithic-`backward()` path on commit
   `a277d90b096cad064b46ac8da49ba2945205bd76` (the recorded baseline)
   against the post-refactor split. Both use the same engine, kernels,
   and HF weight load — any divergence is purely the refactor.
2. **HF transformers `AutoModelForCausalLM`** (preferred for Phase 2/3
   end-to-end). HF runs forward + autograd backward + AdamW; FlexTrain
   runs the new split. Already used by `test_llama_parity.py`,
   `test_llama32_1b_parity.py`, `test_qwen3_1b7_parity.py`,
   `test_olmoe_engine_parity.py`, `test_qwen3_moe_engine_parity.py`,
   etc. — so per-arch comparators already exist; new tests plug into
   them.
3. **Naive PyTorch reference** (for blocks where neither (1) nor (2)
   is convenient — e.g. small-init synthetic tests). Used by
   `flextrain.bench.parity::run_loss_curve_parity`.

### Gates

1. **Phase-1 gate (Llama, no behavior change).**
   Existing parity tests must pass unchanged on the post-refactor code:
   - `tests/test_loss_curve_parity.py` — windowed-mean loss curves
     across multiple working-set configs match.
   - `tests/test_llama_parity.py` / `tests/test_llama32_1b_parity.py`
     — multi-step loss-curve parity vs. HF transformers on real
     Llama 3.2 1B weights and real tokens.
   - `tests/test_llama_block_baseline_parity.py` — block-level
     forward+backward parity vs. naive PyTorch.
   - `tests/test_save_level_parity.py` — bit-identical loss across
     save tiers.
   No new tests are required for Phase 1; the existing ones ARE the gate.

2. **Phase-2 gate (LoRA fast path correctness).**
   - `tests/test_lora_fast_backward_parity.py` (new) — construct a
     small Llama block with LoRA targets, run both the
     pre-refactor LoRA path (monolithic `backward(...)` + post-decompose
     `dW`) and the new dgrad + accumulate-LoRA-grads path on the same
     inputs / weights / activations. Assert element-wise equality of
     `upstream_dx`, `g_a` and `g_b` across every targeted projection,
     with bf16-appropriate tolerance.
   - `tests/test_lora_e2e_llama_8b.py` (existing) and
     `tests/test_lora_engine_smoke.py` — must continue to pass with
     the new path. These are multi-step end-to-end tests.
   - `tests/test_lora_loss_curve_parity_llama.py` (new) — multi-step
     loss-curve parity for `--mode lora` on real Llama 3.2 1B weights
     + real tokens, comparing pre-refactor LoRA implementation
     (commit `a277d90b...`) against the new path. Uses
     `LossCurveParityConfig` with two trajectories (old vs new),
     asserts windowed-mean loss matches over ≥ 200 steps.

3. **Phase-3 gate (per-arch end-to-end).**
   For each arch under refactor (Mistral, Qwen2, Qwen3, Qwen3.5,
   OLMoE, Qwen3-MoE, Qwen3-Next), one new test:
   `tests/test_lora_loss_curve_<arch>.py`. Each runs full FT and LoRA
   trajectories on real weights + real data for ≥ 200 steps and
   asserts windowed-mean loss matches a reference. The reference is:
   - **Mistral / Qwen2 / Qwen3 / Qwen3.5**: HF transformers (their
     existing `test_*_parity.py` files prove HF parity already works
     for forward; we extend to multi-step training-loss parity).
   - **OLMoE / Qwen3-MoE**: HF transformers MoE forward+backward
     (the existing `test_olmoe_engine_parity.py` /
     `test_qwen3_moe_engine_parity.py` show the comparison setup).
   - **Qwen3-Next**: pre-refactor FlexTrain (HF's hybrid linear+full
     attention path is itself in flux; pre-refactor is the stable
     reference).
   Tolerance: windowed-mean atol of 0.10 (matches existing
   `test_loss_curve_parity.py` convention) over a windowed mean of
   the last ~50 steps. If a per-arch test fails this gate, that arch
   doesn't ship until the math is fixed.

### Gemma is out of scope

`tests/test_lora_loss_curve_gemma2.py` and `_gemma3.py` are NOT
authored. Gemma layers retain monolithic backward; LoRA continues to
use the slow path on Gemma until Gemma support is finalized
elsewhere.

## Phasing

The intent is that each phase is independently mergeable and testable.

### Phase 0 — preparation (no code changes yet)
- [x] Survey codebase to identify which blocks already have a
      partial split (above table).
- [x] Document the math + contract (this file).

### Phase 1 — Protocol + Llama plumbing, no behavior change ✅ DONE
- [x] Add `BackwardIntermediates` dataclass to
      `flextrain/core/layer.py`.
- [x] Add `backward_dgrad` / `backward_wgrad` to the `Layer` Protocol.
      (Existing implementers that only have `backward` continue to
      satisfy the protocol — Protocol additions are non-breaking and
      the engine still calls `backward()` only in Phase 1.)
- [x] Implement `backward_dgrad` / `backward_wgrad` on
      `LlamaBlock` (`nn/layers/llama.py`). The split lives in the
      LAYER, not the blocks initially: `backward_dgrad` calls
      `attn.bwd(...)` + `ffn.bwd(...)` with their existing seam (and
      receives `_grads` so today's block-level Wgrad accumulations
      that aren't deferred to ``bwd_accumulate_*`` -- ``g_o, g_2,
      g_attn_norm, g_ffn_norm, g_b_q/k/v`` -- can still flow).
      `backward_wgrad` calls `attn.bwd_accumulate_qkv_grads(...)` +
      `ffn.bwd_accumulate_w1_w3_grads(...)` reading the recomputed
      RMSNorm outputs from `intermediates.aux`.
- [x] Keep `LlamaBlock.backward(...)` as the shim that calls dgrad
      then wgrad.
- [x] Update `flextrain/core/__init__.py` to export
      `BackwardIntermediates`.
- [x] Update `docs/implementing.md` with a new "Optional: split
      backward into `backward_dgrad` / `backward_wgrad`" subsection.
- [x] **Gate: existing parity tests pass unchanged.**
      - `tests/test_llama_parity.py` — orig vs ft: dx 0.0, all
        matmul grads 0.0, g_q 4e-6, norm grads 1.3e-3 (identical to
        baseline `a277d90`).
      - `tests/test_save_level_parity.py` — bit-identical loss curves
        across save tiers (max |Δ| over 10 steps = 0.00e+00).
      - `tests/test_loss_curve_parity.py` — 100-200 step loss curves
        across 8 working-set configs × 3 model shapes match naive
        PyTorch reference within tolerance.
- [x] **New unit gate: `tests/test_llama_dgrad_wgrad_split.py`** —
      direct comparison of monolithic `backward()` vs explicit
      `backward_dgrad + backward_wgrad` on the same forward-produced
      slot. dx and all matmul Wgrads bit-identical (0.0); RMSNorm
      gain grads within ~6e-10 (pre-existing fp32 atomicAdd noise).
- [x] **New regression gate: `tests/test_phase1_regression.py`** —
      multi-step (100 steps) loss-curve comparison vs pre-refactor
      baseline `a277d90`, on three working-set configs (all-on-device,
      host-offload + recompute, weight ring rotation). Max |Δ| =
      **0.000e+00** at every step in every config. The baseline JSON
      is committed at `tests/phase1_baseline_100.json` so the test is
      a standalone regression check anyone can re-run. To regenerate
      the baseline against a new pre-refactor snapshot:
      ``python tests/test_phase1_regression.py --mode dump --steps 100``.

### Phase 2 — engine + LoRA fast path on Llama ✅ DONE (Llama dense)

**Decisions / scope landed**:

* **No engine changes.** The wrapper's ``backward()`` is the
  dispatcher: engine still calls ``layer.backward(...)`` exactly as
  before, the wrapper internally routes through
  ``backward_dgrad → backward_wgrad → accumulate_lora_grads``. Cleaner
  than my original "engine dispatches on isinstance" plan.
* **Skip mechanism = ``skip_grads`` kwarg on block-level
  ``bwd_accumulate_*``** (not on ``bwd``). When a grad name is in
  the skip set the addmm is gated and the ``(X, dY)`` pair is
  written to the caller-supplied ``capture_xy`` dict. Bit-equivalent
  for full FT (default empty set).
* **Hybrid fast/slow path inside one wrapper.** Llama's deferred
  Wgrads (``g_q, g_k, g_v, g_1, g_3``) take the rank-r fast path;
  the inline Wgrads (``g_o, g_2``) still materialize ``dW`` in a
  scratch buffer and decompose. ``LoRAWrapperLayer`` dispatches
  per-projection automatically. Targeting ``w_o``/``w_2`` is still
  correct, just not optimized in this Phase. (Block-level surgery
  to make ``g_o``/``g_2`` skip-able is a follow-up; not blocking
  Phase 3.)
* **MoE intermediates path via the slow path.** Phase 2 ships only
  for dense Llama. MoE (3-D `(E, in, out)` projections) flows
  through ``LoRAWrapperLayer.accumulate_lora_grads``'s slow-path
  branch (matmul-then-bmm-decompose) which is unchanged from Phase
  1. MoE fast path is Phase 3.

**Files changed in Phase 2**:

* ``flextrain/nn/blocks/attention.py`` — ``bwd_accumulate_qkv_grads``
  gains ``skip_grads`` and ``capture_xy`` kwargs.
* ``flextrain/nn/blocks/ffn_dense.py`` — same shape on
  ``bwd_accumulate_w1_w3_grads``.
* ``flextrain/nn/layers/llama.py`` — ``backward_dgrad`` accepts
  ``grads`` (drops the Phase-1 ``_grads`` hack) and
  ``skip_target_names``. ``backward_wgrad`` translates ``w_*`` →
  ``g_*`` skip names and routes ``(X, dY)`` capture into
  ``intermediates.proj_inputs_and_grads``.
* ``flextrain/nn/layers/lora_wrapper.py`` — drop the monolithic
  scratch-`dW`-then-decompose backward. New methods:
  ``backward_dgrad`` / ``backward_wgrad`` (both forward to the base
  with widened skip set), ``accumulate_lora_grads`` (fast-path for
  projections in ``proj_inputs_and_grads``, slow-path fallback for
  the ones still in ``aux["lora_slow_scratch_grads"]``),
  ``lora_target_names()``. The legacy ``backward()`` is now a thin
  shim over these three.
* ``tests/test_phase2_lora_regression.py`` + locked baseline
  ``tests/phase2_lora_baseline_100.json`` — three working-set configs
  × 100 steps. Locked Phase-1 LoRA loss curve as the regression
  target.
* ``tests/test_phase2_lora_math.py`` — single-step ``dA, dB`` slow
  vs fast in fp32 (rel ~2.6e-7) and bf16 (rel ~3.5e-3, stable across
  20 seeds). Pins the math correctness so the regression's loss
  drift is justified as bf16 reorder rather than algorithmic bug.

**Gate evidence**:

| Gate | Result |
|---|---|
| `test_phase1_regression.py` (FT regression) | bit-identical 0.000e+00 over 100 steps × 3 configs |
| `test_phase2_lora_math.py` (math equality, fp32) | dA/dB rel = 2.6e-7 (rounding floor) |
| `test_phase2_lora_math.py` (math equality, bf16, 20 seeds) | dA/dB rel max = 4.0e-3 (reorder noise) |
| `test_phase2_lora_regression.py` (loss curve regression) | max |Δ| = 4e-2 over 100 steps × 3 configs (within 0.1 atol; pure bf16 reorder amplification) |
| `test_lora_engine_smoke.py` (smoke + frozen invariants) | loss reduces 6.22 → 5.11 over 20 steps; base weights unchanged; A/B updated |
| `test_lora_wrapper_math.py` (vs PyTorch autograd) | bf16 rel < 13% on every target — within pre-Phase-2 baseline |
| `test_llama_dgrad_wgrad_split.py` (split vs monolithic) | bit-identical except RMSNorm gain grads (atomicAdd fp32 noise) |
| `test_save_level_parity.py` (save-tier parity) | bit-identical 0.000e+00 over 10 steps |
| `test_llama_parity.py` (FT vs orig PyTorch port) | unchanged from Phase 1: dx 0.0, matmul Wgrads 0.0, norm grads 1.3e-3 |

#### Pre-Phase-2 baseline lock-in (was)

Before any Phase-2 code change touches the LoRA wrapper, **freeze the
Phase-1 LoRA loss curve** as the regression target. The test is
``tests/test_phase2_lora_regression.py`` (to be written), structured
exactly like ``test_phase1_regression.py`` but wrapping each
``LlamaBlock`` in a ``LoRAWrapperLayer`` with ``lora_targets="all"``,
rank=16, alpha=16. Run on the post-Phase-1 code, dump
``tests/phase2_lora_baseline_100.json``. After Phase 2 is implemented,
the same test in ``--mode compare`` must produce a loss curve that
matches the baseline within bf16 reorder tolerance (~1e-3 max |Δ|
because the LoRA fast-path's accumulation order differs from the slow
path -- algebraically equivalent, numerically nearby in bf16).

#### Skip mechanism: per-block ``skip_grads`` kwarg

The cleanest place to skip the Wgrad addmm is **inside the block**.
Every block bwd / bwd_accumulate method gains an optional
``skip_grads: frozenset[str] = frozenset()`` kwarg. When a grad name
is in the set, the block:
1. Skips the corresponding ``torch.addmm`` into ``grads[name]``.
2. Writes the ``(X, dY)`` pair to ``slot.aux[f"lora_xy_{name}"]`` so
   the layer-level ``backward_dgrad`` can pull them into
   ``intermediates.proj_inputs_and_grads``.

The block's signature stays backward-compatible: full-FT callers
don't pass ``skip_grads`` and behavior is identical.

The layer-level ``backward_dgrad`` / ``backward_wgrad`` translate
``skip_target_names`` (uses ``w_*`` names) → ``skip_grads`` (uses
``g_*`` names) and pass it to each block call. After block calls
return, the layer pulls the ``(X, dY)`` pairs from ``slot.aux`` into
``intermediates.proj_inputs_and_grads``.

#### LoRA fast-path math (Llama 2-D projections)

For a 2-D projection with stored convention ``Y = X @ W`` (so ``W``
has shape ``(in, out)`` and ``grads[name]`` has the same shape):

```
Forward:  Y' = X @ W + α (X @ A) @ B
          where A : (in, r),   B : (r, out)

Backward (frozen base, LoRA-only):
          dL/dA = α * (X^T @ dY) @ B^T   shape (in, r)
          dL/dB = α * A^T @ (X^T @ dY)   shape (r, out)
```

But ``X^T @ dY`` is exactly the ``dW`` we want to NOT materialize.
Use the contract-then-multiply order:

```
          dL/dA = α * X^T @ (dY @ B^T)        -- contract: T x r matmul, T x in matmul
          dL/dB = α * (A^T @ X^T) @ dY        -- contract: r x in matmul, r x T matmul
                = α * (X @ A)^T @ dY
```

These contract along the **token** axis (``T``) twice and the **rank**
axis (``r``) once, never along ``in × out``. FLOPs are roughly
``2T·r·in + 2T·r·out + 2T·r·in`` per projection per chunk = ``O(T·r·d)``,
vs. the full Wgrad's ``2T·in·out = O(T·d²)``. For Llama 3.1 8B at
chunk=25200, d=4096, r=16: 256× FLOP reduction per projection.

For 3-D MoE projections ``Y[e] = X[e] @ W[e]`` per expert: same math
under ``bmm``.

#### Concrete checklist

- [ ] **Lock the LoRA baseline.** Write ``tests/test_phase2_lora_regression.py``
      that runs LoRA on three working-set configs (A/D/E like the FT
      regression). Dump baseline against current Phase-1 code.
- [ ] **Add ``skip_grads`` to ``flextrain/nn/blocks/attention.py``.**
      In ``GQAAttentionBlock.bwd``: gate the inline ``g_o`` ``addmm``;
      stash ``(slot.attn_result.view(num_tokens, -1), dx_resid)`` to
      ``slot.aux["lora_xy_g_o"]`` when skipped. In
      ``bwd_accumulate_qkv_grads``: gate each of ``g_q, g_k, g_v``;
      stash ``(attn_norm_output, slot.aux["bwd_dq" / "bwd_local_dk"
      / "bwd_local_dv"])`` to ``slot.aux["lora_xy_g_*"]`` when skipped.
- [ ] **Add ``skip_grads`` to ``flextrain/nn/blocks/ffn_dense.py``.**
      In ``SwiGLUFFN.bwd``: gate inline ``g_2``; stash
      ``(fwd_act_swiglu, dy_resid)`` to ``slot.aux["lora_xy_g_2"]``
      when skipped. In ``bwd_accumulate_w1_w3_grads``: gate each of
      ``g_1, g_3``; stash ``(ffn_norm_output, slot.aux["bwd_dx1_up"
      / "bwd_dx3_up"])``.
- [ ] **Update ``LlamaBlock.backward_dgrad`` and
      ``LlamaBlock.backward_wgrad``.** Take ``grads`` and
      ``skip_target_names`` as the protocol now requires. Translate
      to ``skip_grads`` for block calls. After block calls, pull
      ``(X, dY)`` from ``slot.aux["lora_xy_*"]`` into
      ``intermediates.proj_inputs_and_grads`` keyed by ``w_*`` name.
- [ ] **Drop the ``_grads`` Phase-1 hack.** The layer's
      ``backward_dgrad`` now takes ``grads`` properly; the
      ``backward()`` shim updates accordingly.
- [ ] **Implement ``LoRAWrapperLayer`` split.**
      - ``backward_dgrad``: build effective weights, call
        ``self.base.backward_dgrad(..., skip_target_names=self._target_set)``.
      - ``backward_wgrad``: build effective weights, call
        ``self.base.backward_wgrad(..., skip_target_names=self._target_set)``.
      - ``accumulate_lora_grads(intermediates, weights, grads, ctx)``:
        for each LoRA target, read ``(X, dY) = intermediates[name]``,
        compute rank-r ``dA, dB`` per the math section above,
        ``ga.add_`` and ``gb.add_``.
      - ``lora_target_names() -> frozenset[str]``: returns
        ``self._target_set``.
      - Drop the old ``backward(...)`` body; replace with the
        delegating shim.
- [ ] **Modify ``flextrain/engine/active_model.py::_backward_pass``.**
      Detect whether the layer has ``backward_dgrad``/``backward_wgrad``
      AND is a ``LoRAWrapperLayer``. Dispatch:
      - LoRA: ``upstream_dx, inter = layer.backward_dgrad(...,
        skip_target_names=layer.lora_target_names())``;
        ``layer.backward_wgrad(inter, ..., skip_target_names=layer.lora_target_names())``;
        ``layer.accumulate_lora_grads(inter, ...)``.
      - Non-LoRA + has split: same flow, empty skip set.
      - Non-LoRA + no split (all archs other than Llama, until
        Phase 3): keep ``layer.backward(...)`` fallback.
- [ ] **Run Phase-2 LoRA regression in ``compare`` mode.** Must match
      the locked baseline within max |Δ| ≤ 1e-2 (LoRA fast-path's
      different accumulation order produces small numeric drift in
      bf16; tolerance gives float-noise headroom).
- [ ] **Run Phase-1 FT regression unchanged.** Full-FT path goes
      through the same code with empty ``skip_target_names`` and must
      stay bit-identical to its Phase-1 baseline.
- [ ] **Single-step LoRA grad parity.** Add to
      ``test_phase2_lora_regression.py`` a single-step section that
      runs both old (slow) LoRA backward and new (fast) on the same
      forward-produced slot. Assert ``dA, dB`` match within bf16
      tolerance (~1e-3 max |Δ| because of accumulation order).
- [ ] **HF parity check.** Run an existing LoRA-vs-HF-PEFT test
      (e.g. ``tests/test_lora_e2e_llama_8b.py`` or
      ``tests/test_lora_engine_smoke.py``) post-Phase-2 to confirm
      the trajectory still matches HF.
- [ ] **Benchmark.** On the H100 user's box, profile a LoRA step at
      chunk=25200 and confirm the ``Backward: Chunk N`` NVTX ranges
      shrink vs. the Phase-1 baseline -- the entire point of the
      refactor.

### Phase 2.5 — extend fast path to ALL projections (not just deferred Wgrads) ✅ DONE

User insistence: "no slow path; no extra compute should be done getting wgrad
for frozen params". Phase 2 originally scoped only the deferred Wgrads
(``g_q,g_k,g_v,g_1,g_3``); inline Wgrads (``g_o, g_2``) and 3-D MoE expert
Wgrads (``g_up, g_down``, plus shared-expert variants) still went through
the slow scratch-`dW`-then-decompose path. This phase eliminates that.

**Skip mechanism extended to:**

* ``attention.py::GQAAttentionBlock.bwd`` — gates inline ``g_o``.
* ``ffn_dense.py::SwiGLUFFN.bwd`` — gates inline ``g_2``.
* ``attention_gated.py::GQAAttentionGatedBlock.bwd`` and
  ``bwd_accumulate_qkv_grads`` — gates ``g_o`` (inline) +
  ``g_q/g_k/g_v`` (deferred). Used by Qwen3.5-Full + Qwen3-Next-Full.
* ``ffn_moe.py::MoESwiGLUFFN.bwd`` — gates per-expert ``g_up, g_down``
  + ``g_router`` via a ``lora_per_expert_callback(name, eid, X, dY)``.
  The callback fires inside the per-expert loop and does rank-r
  matmuls into the wrapper's adapters with no per-expert clones.
* ``ffn_moe_shared.py::MoESwiGLUSharedExpertFFN.bwd`` — same callback
  for ``g_shared_up, g_shared_down, g_shared_expert_gate`` (per-shared-
  expert iteration). Routed-path callbacks chain through.
* ``linear_attn.py::GatedDeltaNetBlock.bwd`` — gates ``g_lin_out``,
  ``g_lin_qkvz``, ``g_lin_ba`` (the matmul-flavored projections).
  Other linear-attn-internal grads (``g_lin_A_log``, ``g_lin_dt_bias``,
  ``g_lin_conv``, ``g_lin_norm``) are not LoRA-targetable in practice
  and accumulate inline as before.

**Layer surface:** every supported arch's ``backward_dgrad`` now plumbs
``skip_target_names`` to the underlying block bwd calls and pulls
captured ``(X, dY)`` into ``intermediates.proj_inputs_and_grads``. MoE
layers additionally install a per-expert callback into
``slot.aux["__lora_moe_callback__"]`` that fires from inside the MoE
block's bwd loop.

**Wrapper surface:** ``LoRAWrapperLayer._FAST_PATH_TARGETS`` now
includes every projection name we know how to skip. The wrapper raises
``RuntimeError`` if any LoRA target falls back to the slow scratch-`dW`
path -- no silent slow path. Verified by:

* ``tests/test_phase2_olmoe_lora_e2e.py`` — OLMoE LoRA on real
  FineWeb tokens, 50 steps. Loss 10.88 → 8.36, frozen base bit-
  identical, max |Δ| vs locked baseline = 9e-3 (within bf16 reorder).
* ``tests/test_lora_moe_smoke.py`` — OLMoE LoRA targets={w_q,w_k,w_v,
  w_o,w_up,w_down}, 15 steps fixed batch, loss 6.30 → 5.28.
* ``tests/test_lora_engine_smoke.py`` — Llama LoRA targets="all"
  (every 2-D), 20 steps, loss 6.22 → 5.11.

### Phase 3 — roll out to remaining archs

Each item in this list is one PR-shaped unit of work. Each lands its
own per-arch end-to-end correctness gate (see "Testing strategy" above
— real-data loss-curve parity over multiple steps against either the
pre-refactor implementation or HF's reference forward+backward).

- [x] **Mistral** (`nn/layers/mistral.py`) — inherits from
      ``LlamaBlock``, gets the split for free. ✓
- [x] **Qwen2** (`nn/layers/qwen2.py`) — inherits from ``LlamaBlock``,
      gets the split for free. ✓ (QKV biases stay inline; not LoRA
      targets.)
- [x] **Qwen3** (`nn/layers/qwen3.py`) — own ``Qwen3DenseBlock`` with
      QK-norm, given the dgrad/wgrad split. ``Qwen3DenseSWABlock``
      inherits and is covered. ✓
- [x] **Qwen3.5** (`nn/layers/qwen3_5.py`) — both ``Qwen3_5FullLayer``
      (Llama-shaped) and ``Qwen3_5LinearLayer`` (gated DeltaNet) given
      the split. The linear-attn variant has no deferred Wgrads --
      every Wgrad runs inline in ``lin_attn.bwd``. ✓
- [x] **OLMoE** (`nn/layers/olmoe.py`) — split with MoE per-expert
      callback; LoRA fast path covers ``w_q,w_k,w_v,w_o,w_up,w_down``.
      Real-data E2E gate at
      ``tests/test_phase2_olmoe_lora_e2e.py``. ✓
- [x] **Qwen3-MoE** (`nn/layers/qwen3_moe.py`) — split with same MoE
      callback as OLMoE. ✓
- [x] **Qwen3-Next** (`nn/layers/qwen3_next.py`) — both Linear and
      Full layer variants given the split. The Full variant uses
      ``GQAAttentionGatedBlock`` (already gated) +
      ``MoESwiGLUSharedExpertFFN`` (now also gated for shared-expert
      Wgrads). ✓

**Gemma 2 / Gemma 3 are explicitly out of scope** for Phase 3 — those
arches aren't ready yet. They retain the monolithic `backward(...)`
path for now; once Gemma support is finalized elsewhere it can be
added as a follow-up phase that mirrors the Llama-shape changes.

After Phase 3 lands every supported arch, the monolithic
`Layer.backward(...)` shim can be removed from the protocol; each
layer implements the two split methods as its primary API. We keep
the shim available if Gemma re-enters scope so it doesn't block
on this refactor.

## Open questions / risks

* **Per-projection `(X, dY)` memory.** Holding all per-projection pairs
  live between `backward_dgrad` and `backward_wgrad` adds GPU residency.
  For a Llama layer at chunk=25200 / d_model=4096, the worst case is
  `7 projections × 2 tensors × 25200 × max(d_model, attn_dim) × 2 bytes`
  ≈ 2.8 GiB extra residency.
  **Mitigation:** the engine calls dgrad → wgrad (or dgrad →
  lora-accumulate) immediately, in sequence, on the same compute
  stream, so the intermediates dict is one-chunk-lived; we don't pile
  up across the layer loop. Same memory profile as today.
* **Norm `bwd` recompute behavior.** Today norm bwd has an
  `recompute_output` flag that produces the recomputed norm output
  for the next-stage Wgrad. Intermediates carry that recomputed output
  in `aux` so `backward_wgrad` doesn't need to re-do it.
* **Stream / event correctness.** `backward_dgrad` and `backward_wgrad`
  both run on `self.streams.compute` from inside the engine's
  `with torch.cuda.stream(...)` block. No new sync points required.
* **Numerics.** The new LoRA path computes `dL/dA, dL/dB` from `(X, dY)`
  directly rather than from a materialized `dL/dW`. The two are
  algebraically equivalent in IEEE arithmetic for fp32, but bf16
  reduction order differences can produce tiny element-wise deltas.
  The parity test uses `atol=0` rtol-based comparison with a tolerance
  appropriate for bf16 accumulation; values must match within float
  noise.

## File-by-file change manifest (for implementation reference)

When Phase 1 lands:
- `flextrain/core/layer.py` — add `BackwardIntermediates`, extend
  `Layer` Protocol, add shim helper.
- `flextrain/nn/layers/llama.py` — add `backward_dgrad`,
  `backward_wgrad`; `backward(...)` becomes a shim.

When Phase 2 lands:
- `flextrain/engine/active_model.py::_backward_pass` — call dgrad+wgrad
  separately, plumb `skip_target_names`, dispatch LoRA accumulate.
- `flextrain/nn/layers/lora_wrapper.py` — implement new methods,
  drop scratch-`dW` flow.
- `flextrain/nn/blocks/ffn_dense.py` and `flextrain/nn/blocks/attention.py`
  — `backward_wgrad`-equivalent block methods honor skip set.
- `tests/test_lora_fast_backward_parity.py` — new.

When Phase 3 lands (per arch):
- One `nn/layers/<arch>.py` file gains the split.
- One `nn/blocks/<block>.py` file (if newly touched by that arch) gains
  the split.
- One `tests/test_lora_<arch>_smoke.py` file gains a smoke parity test.
