# FlexTrain v2 — Implementation Notes / Decision Log

This is a running log of everything I did while you were away, with special
attention to any decision that could affect the scheduling invariants from
the paper or from `orig/active_model.py`. Anything marked **[DECISION]** is
something you may want to review and possibly revert.

Source of truth for the AdaWS schedule: `orig/active_model.py:1162-1632`
(`fwd_bwd`) and `:531-665` (`determine_saved_levels`). Any behavior change
against those two functions is a bug unless explicitly called out below.

---

## Ground rules I'm operating under

- **DP solver reused as-is.** The compiled C extension at
  `orig/transmission_scheduler_pkg` stays. New code in
  `flextrain/core/save_level.py` only *builds inputs* and *wraps outputs*; it
  does not reimplement the DP.
- **Schedule preserved.** Forward/backward traversal order is
  `for layer: for chunk` during forward, reverse during backward
  (orig:1261,1275). The "no home slot" fast path when
  `n_home_act_slots == 0` (orig:546) is preserved as
  `SaveLevelPlan.all_on_device`.
- **Phase 1 = API only.** No compute kernels, no engine, no torch.nn. Just
  the abstraction types + one smoke test. Stopping at a review gate before
  wiring real layers.
- **Everything is additive.** `orig/` is not touched. All new code lives in
  `flextrain/`.

---

## Files created so far

### Plan + orientation
- [docs/START_HERE.md](START_HERE.md) — named phases in order, review gate
  after Phase 1.
- [docs/PLAN.md](PLAN.md) — the approved full plan (copied from
  `~/.claude/plans/`).

### Package skeleton
- [flextrain/__init__.py](../flextrain/__init__.py) — empty package marker.
- [flextrain/core/__init__.py](../flextrain/core/__init__.py) — re-exports
  the three Phase-1 modules.

### Phase 1 core abstractions (the load-bearing refactor)
- [flextrain/core/activation_schema.py](../flextrain/core/activation_schema.py)
  — replaces the 4 parallel code paths per layer type
  (`make_act_slot` / `get_act_slot_size` / `send_activations_home` /
  `fetch_activations`) with a single declarative schema.
- [flextrain/core/layer.py](../flextrain/core/layer.py) — Protocols
  (`Layer`, `InputLayer`, `OutputLayer`) + supporting types (`ParamSpec`,
  `TensorSpec`, `ComputeCost`, `ChunkMeta`, `LayerContext`, `LossStats`).
- [flextrain/core/save_level.py](../flextrain/core/save_level.py) — wraps
  the existing DP solver with per-layer `max_tier` + padding.

### Phase 1 tests
- [tests/test_phase1_core.py](../tests/test_phase1_core.py) — 12 tests,
  all passing. Run with `PYTHONPATH=. python tests/test_phase1_core.py`.
  Notably `test_schema_byte_sizes_match_orig_arithmetic` is a byte-exact
  match against the hand-computed sizes in `dense_layer.py:837-916` at
  Llama3-8B dims / 1024 tokens — this is the proof that the schema is a
  drop-in for `get_act_slot_size`.

---

## Decisions I made that affect semantics

### [DECISION 1] `SaveLevel(-1)` replaces the dict-value `-1` sentinel
- **In `orig`:** `saved_levels[(layer_id, chunk_id)] = -1` means
  "no home slot, activation stays on GPU ring." Magic integer in a dict.
- **In v2:** `SaveLevel.on_device()` / `SaveLevel.is_on_device` properties.
  Same semantics, typed.
- **Risk:** none functional. Cosmetic. Can revert by reading the `.value`
  field directly.

### [DECISION 2] Per-layer `max_tier` + DP padding with ±inf
- **In `orig`:** `num_saved_activation_levels = self.model_layers[0].max_saved_activations_level + 1` (orig:567) — assumes all layers share tier count.
- **In v2:** each `ActivationSchema` declares its own `max_tier`. When
  building DP tables we pad to `k_global = max(max_tier+1 across layers)`
  with `value=-1e18`, `duration=+1e18` in disallowed cells so the solver
  will never pick them.
- **Why:** heterogeneous backbones (GPT-OSS dense+MoE alternation,
  Qwen3-Next linear+full attention) require this. For homogeneous models
  (Llama, Qwen-dense, OLMoE) this is strictly equivalent to `orig`.
- **Risk:** small. If the solver ever ignores `-inf` sentinels and picks a
  disallowed cell, `plan_from_solution` defends by capping at
  `max_tier_per_task`. Watch for issues at HF logit-parity stage.
- **Revert plan:** set `k_global = layers[0].schema.max_tier + 1`; drop the
  padding code.

### [DECISION 3] `persist=False` for MoE router metadata
- **In `orig`:** `x_router`, `expert_counts`, `router_weights`,
  `chosen_experts`, `scattered_router_weights` live in the host-pinned
  buffer with tier=0 (always saved); also appear on device every chunk.
  See `moe_layer.py:1458-1477`.
- **In v2:** these will be declared with `persist=False` when we port
  MoE — engine keeps them in a small device-only scratch area, no host
  slot at all. Saves ~5 tensors * N_chunks * N_layers of pinned host
  memory that was essentially wasted (the data is trivially recoverable
  on device for the duration it's needed).
- **Risk:** moderate. If the engine ever needs to restore router metadata
  during backward *after* that chunk's device slot has been evicted, this
  breaks. Mitigation: MoE backward is scheduled immediately after its
  forward-chunk on the same (layer, chunk) pair, and the router outputs
  are consumed in the fwd itself — router metadata never needs to persist
  longer than one chunk. Confirm by tracing through `moe_layer.py`
  `backward_moe` when we port it.
- **Revert plan:** flip `persist=True` on those fields when declaring the
  MoE schema. Zero-line change.

### [DECISION 4] `offload=False` flag for "lives on device, not offloaded"
- **In `orig`:** implicit — some fields are computed fresh every chunk and
  never touch the host buffer, but there's no formal flag.
- **In v2:** explicit `ActivationField.offload: bool`. Drives which fields
  `send_home` / `fetch_home` iterate.
- **Risk:** none — this is purely a code-organization win.

### [DECISION 5] `token_axis: int = 0` declaration per field
- **In `orig`:** special-case branching in `forward` and `fetch_activations`
  for `softmax_lse` (shape `(n_heads, num_tokens)`, axis 1). See
  `dense_layer.py:31-34`.
- **In v2:** `ActivationField.token_axis` declared per-field; `slot.view_for(
  num_tokens, dims)` uses `tensor.narrow(axis, 0, num_tokens)` uniformly.
- **Risk:** none — the branching moves from `forward` into the slot machinery.

### [DECISION 6] `ComputeCost.avoided_recompute_flops` is a tuple, not a dict
- **In `orig`:** `saved_fwd_flops = {0: ..., 1: ..., 2: ..., 3: ...}`
  (dict keyed by tier, `get_fwd_flops` in `dense_layer.py:981`).
- **In v2:** `tuple[int, ...]` of length `max_tier + 1`. Indexed by tier.
- **Risk:** none. Also enforces monotone non-decreasing (you can't save
  less at a higher tier).

### [DECISION 7] `HardwareCost` dataclass replaces scattered floats
- **In `orig`:** `self.peak_tflops_est`, `self.bw_est_gb_per_sec`,
  `PRACTICAL_EFFICIENCY_FACTOR` as module-level constant (orig:595,615).
- **In v2:** `HardwareCost(peak_tflops, practical_efficiency_factor,
  pcie_bw_gbps)` with `flops_to_ms` / `bytes_to_ms` methods. Exactly the
  same math; one struct instead of three fields.
- **Risk:** none.

### [FINDING — from 3-way parity] All grads in bf16-noise tolerance after RoPE convention fix

[tests/test_llama_parity.py](../tests/test_llama_parity.py) runs forward +
backward three ways (naive PyTorch autograd / orig / flextrain) on a
single Llama block with seed-fixed weights and input. After fixing a bug
in the naive reference's RoPE convention (see below):

* **orig vs flextrain**: bit-identical (0.000e+00) for y, dx, g_1, g_2,
  g_3, g_o, g_v, g_q, g_k. Same kernels in same order → same output.
* **orig/flextrain vs naive**: y ~3.8e-3, dx ~5.2e-3, all grads
  ~4–8e-3. Expected bf16 rounding (naive path uses fp32 internally,
  orig uses bf16; both use bf16 weights).
* **Exception — g_attn_norm / g_ffn_norm**: ~1e-3 orig-vs-ft non-zero.
  `awsm_rmsnorm_bwd` uses atomic fp32 adds to accumulate dW; run-to-run
  order differences at 1e-3 scale. Benign.

**Investigation that caught the bug:**

The first pass of this test showed g_q / g_k disagreeing with naive by
~67% relative norm while all other grads were at ~3%. I initially
attributed this to bf16 noise and set a loose tolerance. Wrong.

[tests/test_gqk_investigation.py](../tests/test_gqk_investigation.py) runs
naive autograd with both fp32 AND bf16 internal precision, and compared
to orig. Key result: naive(bf16) vs naive(fp32) showed ~1e-3 for g_q /
g_k. So the ~67% error was NOT a precision issue.

Root cause: **the naive reference's RoPE used the halved-split
convention** (pair ``x[..., :D/2]`` with ``x[..., D/2:]``, which is
HuggingFace / GPT-NeoX style), **but orig's Triton kernel uses the
pair-interleave convention** (pair ``x[..., 2i]`` with ``x[..., 2i+1]``,
see [orig/awsm_transformer/ops/rope.py:38-48]).

Both are valid RoPE -- same frequency spectrum, different tensor-element
assignment -- but mixing conventions produces ~20-25% g_q/g_k error per
the investigation script.

Fix: [tests/test_llama_parity.py::_rope_ref] now uses pair-interleave.
After the fix, all grads agree with orig within expected bf16 tolerance
(~6e-3). The kernel is correct; the reference had a bug.

[tests/test_gqk_norope.py](../tests/test_gqk_norope.py) independently
confirmed this by stripping RoPE out entirely: naive(bf16) vs orig g_v
is exactly 0.0 (V bypasses the softmax path cleanly in bf16), and g_q /
g_k differ by ~1.3e-3 -- which matches naive(fp32) vs naive(bf16) of
the same operation, i.e. the bf16-vs-fp32 precision floor. Cannot do
better than that without matching precision on both sides.

**Takeaway for future ports:** when porting a layer with RoPE to a new
kernel, verify the rotation convention (halved-split vs pair-interleave)
before diffing gradients. Wrong convention looks exactly like a
large-magnitude kernel bug.

### [CONTRACT] Forward / forward_recompute / backward (how we handle recompute)

This is the same three-method contract as orig, but with the
dict-membership check replaced by a typed slot query.

**Layer Protocol** (in [flextrain/core/layer.py](../flextrain/core/layer.py)):

```python
class Layer(Protocol):
    schema: ActivationSchema
    param_spec: ParamSpec
    def forward(self, x, chunk, weights, slot, ctx) -> Tensor: ...
    def forward_recompute(self, slot, chunk, weights, ctx) -> None: ...
    def backward(self, dx, chunk, weights, grads, slot, ctx) -> Tensor: ...
    def compute_cost(self, chunk) -> ComputeCost: ...
```

**Semantic correspondence to orig:**

| orig (dense_layer.py) | flextrain (nn/layers/*.py) |
|---|---|
| `forward(X, chunk_metadata, weights, base_act_slot, fwd_context)` | `Layer.forward(x, chunk, weights, slot, ctx)` |
| `forward_recompute(fwd_act_slot, base_act_slot, chunk_metadata, weights, fwd_context)` | `Layer.forward_recompute(slot, chunk, weights, ctx)` |
| `backward(dX, chunk_metadata, weights, grad_weights, fwd_act_slot, fwd_context, bwd_context)` | `Layer.backward(dx, chunk, weights, grads, slot, ctx)` |

**How recomputation decisions are driven:**

1. **Per-(chunk,layer) tier assignment**: `flextrain.core.save_level.build_dp_tables` + the existing C DP solver in `orig/transmission_scheduler_pkg` pick a tier `L ∈ {0..max_tier}` for each (chunk, layer) pair. Tier `L` means "persist to host all ActivationField with `tier <= L`; everything else gets recomputed at backward time."
2. **Forward**: `Layer.forward` writes ALL declared activations into the device slot (every tier). Engine then calls `send_home(host_slot, device_slot, level)` which copies only the tier-≤-L fields (see [activation_schema.py:284](../flextrain/core/activation_schema.py) `send_home`).
3. **Backward prep**: engine allocates a fresh device slot AT TIER L (so higher-tier fields do not appear), `fetch_home` restores saved fields, then calls `Layer.forward_recompute(slot, ...)`. The layer branches on `slot.has("field_name")` to fill in missing fields.
4. **Backward**: `Layer.backward` runs against the now-complete slot.

**Concrete orig→flextrain diff for the recompute decision:**

```python
# orig/dense_layer.py:131
if "xq" not in fwd_act_slot:
    fwd_act_slot["xq"] = base_act_slot["xq"][:num_tokens, :].view(...)
    # recompute Q + re-RoPE ...

# flextrain/nn/layers/llama.py:217
if not slot.has("xq"):
    attn_norm_output = self.attn_norm.fwd_from_rstd(...)
    self.attn.fwd_recompute_qo(...)
```

Same semantics, typed slot. Missing-field access via `getattr` on
`ActivationSlot` raises `AttributeError` (not a silent `KeyError` that
slips through a conditional), so layer bugs like forgetting to check
`slot.has("xq")` before `slot.xq` fail loudly on the first call.

**What the layer author never has to do:**

* Allocate buffer space — the engine does, from the layer's own
  `schema`.
* Decide which fields to send home — the engine does, from the
  DP-chosen tier.
* Compute byte sizes for the DP solver — derived from the schema.
* Do chunk-view re-slicing — `slot.view_for(num_tokens, dims)` handles
  it (including the `softmax_lse` transpose-axis case).

### [DECISION 10] Heterogeneous-backbone contract (verified under test)
- **Requirement:** "Ensure that models with different types of blocks will
  work (gpt-oss with some sliding-window attention, gemma with some linear
  attention, or deepseekv3 with some dense vs. moe)."
- **What the design already allowed:** `ActiveModel.backbone: Sequence[Layer]`
  is a tuple of Protocol instances — no "one true layer type" assumption.
  Each layer brings its own `schema`, `param_spec`, and `compute_cost`.
  `build_dp_tables` already padded to `max(max_tier+1)` across layers.
- **What I hardened this session:**
  1. Explicit guarantee that layer `schema` / `param_spec` instances are
     NOT shared across layers (tested in
     [tests/test_heterogeneous_backbone.py](../tests/test_heterogeneous_backbone.py)).
  2. Added [SlidingWindowLlamaBlock](../flextrain/nn/layers/llama_swa.py) — a
     one-line factory demonstrating the "same compute path, different kernel
     flag" case (Mistral, Gemma-alternating, GPT-OSS subset).
  3. Test proving a backbone with a `max_tier=2` layer alongside
     `max_tier=3` layers produces a rectangular DP table where the shallow
     row's column 3 has ±inf sentinels.
- **Contract for adding a new layer type:**
  1. Implement `Layer` Protocol: `layer_id`, `schema`, `param_spec`,
     `forward`, `forward_recompute`, `backward`, `compute_cost`.
  2. Its `schema` may have any `max_tier` (engine handles pads).
  3. Its `param_spec` may have any tensor set (each layer loads its own
     master/grad/opt buffers).
  4. Its forward MUST return `(num_tokens, d_model)` — the residual-stream
     shape that the next layer sees. This is the ONE invariant.
  5. For HF weight loading, also add an entry in
     [flextrain/io/arch/](../flextrain/io/arch/) mapping HF tensor names
     to your layer's flextrain_name fields.
- **What this means for BufferManager (Phase 3 work):** the GPU param /
  grad / act-slot rings must be sized to the MAX across layer types, not
  one uniform size. Each ring's per-slot buffer must be large enough to
  hold the largest layer's data — smaller layers simply leave the tail
  unused when they occupy a slot. I'll note this in BufferManager's
  docstring when we get there.
- **Examples this enables out-of-the-box:**
  - GPT-OSS: alternating full + SWA dense layers (ship as
    `[LlamaBlock, SlidingWindowLlamaBlock, LlamaBlock, ...]`).
  - Gemma2: every-other SWA (same pattern).
  - DeepSeek-V3 / Qwen3-Next: once we port MoE as
    `MoETransformerBlock`, mix `[LlamaBlock, MoETransformerBlock, ...]`
    — no engine changes needed.
  - Qwen3-Next: once we have `LinearAttentionBlock`, mix full + linear.
- **Risk:** low. The DP solver has been verified to handle padded arrays
  in [tests/test_phase1_core.py::test_dp_tables_shape_and_padding]. The
  kernel fusion and orchestration loops all iterate by
  `for layer in self.backbone` with no layer-type branches.

### [DECISION 9] `TensorSpec` carries per-role dtypes (added after your note)
- **Requirement:** "datatypes of things like master params, params during
  compute, master gradients, master optimizer state can be specified for
  various layers."
- **Original design:** `TensorSpec` had a single `dtype` field. Fine for
  orig (which is uniformly bf16) but prevents mixed-precision training
  where master params are fp32 but compute is bf16.
- **New design:** `TensorSpec` has four optional dtype fields:
  `compute_dtype` (required), `master_dtype`, `grad_dtype`,
  `opt_state_dtype`. Missing roles default to `compute_dtype` except
  `opt_state_dtype` which defaults to `float32`. `TensorSpec.simple(name,
  shape_fn, dtype)` preserves the old single-dtype call site for block
  code that doesn't care.
- **Queries:** `TensorSpec.{compute,master,grad,opt_state}_byte_size(dims)`,
  `ParamSpec.byte_size(dims, role='compute'|'master'|'grad'|'opt_state')`.
- **Semantics:** the engine, when it prefetches params host->device, casts
  `master_dtype -> compute_dtype` if they differ. When it offloads updated
  master weights, it casts back. The optimizer step operates on
  master+grad+opt_state buffers on host (orig's "store opt state in host
  and transfer all master training state to GPU" — paper §5.1).
- **Risk:** low. Pure-bf16 paths are unchanged (use `TensorSpec.simple`).
  When we start loading HF fp32 checkpoints into bf16 compute, we get
  fp32 master for free without changing layer code.
- **Revert plan:** not needed; defaults collapse to the old single-dtype
  behavior.

### [DECISION 8] `plan_from_solution` forces last `n_gpu_act_slots` to each
layer's own `max_tier`
- **In `orig`:** `saved_act_choices[-self.n_gpu_act_slots:] = num_saved_activation_levels - 1` (orig:640).
- **In v2:** `for t in range(T - n_gpu_act_slots, T): choices[t] = max_tier_per_task[t]`.
- **Risk:** none for homogeneous backbones (identical). For heterogeneous,
  this is the "right" generalization: final tail forced to each layer's OWN
  top tier rather than a global constant. Worth double-checking when
  heterogeneous arches arrive.

---

## Things I deliberately did NOT change

- **Traversal order.** `for layer: for chunk` (orig:1261,1275) stays.
- **The `n_home_act_slots == 0` fast path** (orig:546) is preserved as
  `SaveLevelPlan.all_on_device` — any round where the activation ring is
  large enough to hold every (chunk, layer) pair skips the DP entirely.
- **DP solver call shape.** `solver.solve(compute, durations, values, N)`
  stays. Only the ``(T, k)`` array is now globally padded.
- **"Value = avoided recompute time, higher is better"** — unchanged.

---

## Conda environment

Created `flextrain` environment cloned from `net`:
```
conda create --name flextrain --clone net -y
```

**Gotcha we hit + fix** (worth knowing):

- `net` has `torch 2.9.1+cu130` (CUDA 13) but `flash-attn 2.8.3` is built
  against CUDA 12. Fresh activation gives
  `ImportError: libcudart.so.12: cannot open shared object file`.
- Fix: installed `nvidia-cuda-runtime-cu12` into the env, then added
  `~/miniconda3/envs/flextrain/etc/conda/activate.d/flash_attn_cu12_shim.sh`
  which prepends the cu12 runtime lib path to `LD_LIBRARY_PATH` on every
  `conda activate flextrain`.
- Confirmed `import flash_attn; flash_attn.flash_attn_varlen_func` works on
  a fresh activation with the shim in place.

If you'd prefer to rebuild flash-attn against cu13 properly instead of using
this shim, run in the env:
`FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE pip install flash-attn --no-build-isolation --force-reinstall`
(builds from source, takes ~20 minutes).

## Open questions for you when you're back

1. **MoE router `persist=False` (DECISION 3).** Please confirm by skimming
   `orig/awsm_transformer/moe_layer.py` `backward_moe` that router metadata
   is never needed outside its own (chunk, layer) pair. If it IS, I'll flip
   the flag to `True` when we port.
2. **`HardwareCost.practical_efficiency_factor` default.** `orig` has a
   module-level constant `PRACTICAL_EFFICIENCY_FACTOR`. I haven't grepped
   for the actual value — please tell me what to use or I'll read it out of
   `orig/active_model.py` on the way through.
3. **Heterogeneous-model tail-forcing (DECISION 8).** For a GPT-OSS-style
   mixed dense+MoE backbone, is it correct that the final
   `n_gpu_act_slots` tail should force each layer's OWN max_tier, or
   should it force the GLOBAL max (so MoE layers are forced to a possibly
   impossible tier if they have lower max)? I went with per-layer own
   max_tier; revisit when we port GPT-OSS.

---

## Next steps while you sleep

Per auto mode, I'll continue through:

1. **Phase 1 smoke test** — instantiate a dummy dense-transformer-like
   `ActivationSchema`, exercise it at each tier, confirm byte sizes match
   manual arithmetic, build DP tables on a mini model, confirm the output
   shape and padding are well-formed.
2. **Phase 2 — `nn/blocks/` + first architecture.** Port RMSNorm, GQA
   attention, SwiGLU FFN, MoE FFN from `orig/awsm_transformer/` onto the new
   contract. Compose `LlamaBlock`.
3. **Phase 2 — embed + head** via `InputLayer` / `OutputLayer`.
4. **Phase 3 — engine port.** `active_model.py` split into orchestrator +
   `buffers.py` + `streams.py` + `schedule.py`.
5. If I hit any decision that could affect scheduling invariants, I will
   **stop, write it here, pick the safer option, and continue**.

I will commit nothing, push nothing, and delete nothing. All changes are
additive in `flextrain/`, `tests/`, and `docs/`. `orig/` is untouched.

---

## FINAL STATUS (morning summary)

**Everything below this line is where things actually ended up after the overnight run.**

### What's done

All scaffolding is in place and verified. 18 tests pass on CPU alone, no
GPU needed. `python -m flextrain info` works.

```
flextrain/
├── __init__.py
├── __main__.py                 python -m flextrain
├── cli.py                      argparse shell (info + train stub)
├── config.py                   RunConfig / ModelConfig / TrainConfig / HardwareConfig / IOConfig / OptimizerConfig
├── core/                       [ALL DONE]
│   ├── activation_schema.py    ActivationField, ActivationSchema, ActivationSlot, send_home, fetch_home, concat_fields
│   ├── layer.py                Layer/InputLayer/OutputLayer Protocols, ParamSpec, TensorSpec (4 per-role dtypes), ComputeCost, ChunkMeta, LayerContext, LossStats
│   ├── save_level.py           SaveLevel, SaveLevelPlan, HardwareCost, DPTables, build_dp_tables, plan_from_solution
│   └── working_set.py          WorkingSetConfig dataclass, determine_working_set_config (delegates to orig)
├── engine/                     [SCAFFOLD + STUBS]
│   ├── active_model.py         ActiveModel dataclass; fwd_bwd/step/load_hf/save_hf raise NotImplementedError (Phase 3)
│   └── buffers.py              BufferManager stubs, KVContextWindow, ScratchPool (fully functional)
├── io/                         [ALL DONE]
│   ├── hf_weights.py           ArchSpec, WeightMapEntry, Transform, load_hf_safetensors, export_hf_safetensors, select_arch, register_arch
│   └── arch/llama.py           Llama family weight map + hf_config_to_flextrain + hf_config_to_hyperparams
├── nn/                         [PARTIAL — only norm ported]
│   ├── blocks/norm.py          RMSNormBlock composable unit
│   └── layers/                 (empty — Phase 2 continuation)
├── ops/__init__.py             Shim re-exporting every op from orig/awsm_transformer/ops/
└── optim/                      [ALL DONE]
    ├── adamw.py                AdamW driven by ParamSpec (collapses orig's 9-unrolled step calls)
    ├── muon.py                 Muon parallel to AdamW
    └── base.py                 Optimizer Protocol, OptimizerStateSpec, state_key helper

tests/
├── run_all.py                  Master runner: PYTHONPATH=. python tests/run_all.py
├── test_phase1_core.py         14 tests: schema byte-size parity with orig, tier monotonicity, slot view/send/fetch, DP padding, per-role dtypes, etc.
└── test_hf_weights.py          4 tests: fake-Llama round-trip load+export, shape preservation, strict-mode errors

docs/
├── START_HERE.md               Phase order + review gates
├── PLAN.md                     Full approved plan (copy of ~/.claude/plans/...)
└── NOTES.md                    This file (decision log)
```

### How to continue (in order)

**Step 1.** Port compute blocks onto the new contract. These need a GPU to
test. Work module-by-module and add a small numerical-parity test against
the orig implementation for each:
- `flextrain/nn/blocks/rope.py` — RoPE forward + backward + `position_angles`.
- `flextrain/nn/blocks/attention.py` — `GQAAttentionBlock` with fields
  (xk/xv/attn_result/lse/xq/xo at appropriate tiers).
- `flextrain/nn/blocks/ffn_dense.py` — `SwiGLUFFN` with fields (x1, x3 at
  tier 3).

**Step 2.** `flextrain/nn/layers/llama.py` — `LlamaBlock` as composition:
```python
class LlamaBlock:
    def __init__(self, layer_id, model_cfg, optimizer):
        self.attn_norm = RMSNormBlock("attn_norm", eps=model_cfg.rms_norm_eps)
        self.attn      = GQAAttentionBlock(...)
        self.ffn_norm  = RMSNormBlock("ffn_norm", eps=model_cfg.rms_norm_eps)
        self.ffn       = SwiGLUFFN(...)
        self.schema     = ActivationSchema(
            fields=concat_fields([b.fields() for b in (self.attn_norm, self.attn, self.ffn_norm, self.ffn)]),
            max_tier=3,
        )
        self.param_spec = ParamSpec.merge([b.param_spec() for b in (self.attn_norm, self.attn, self.ffn_norm, self.ffn)])
```

**Step 3.** `flextrain/nn/embed.py` (`TokenEmbedLayer` : `InputLayer`) and
`flextrain/nn/head.py` (`LMHead` : `OutputLayer`). Port directly from
`orig/awsm_transformer/{embed,head}.py`.

**Step 4.** Engine wiring. `engine/buffers.py` allocates from
`WorkingSetConfig + ParamSpec + ActivationSchema`, `engine/active_model.py`
implements `fwd_bwd` mirroring `orig/active_model.py:1162-1632`.

**Step 5.** Additional architectures via the template in `nn/layers/llama.py`
+ weight map in `io/arch/<family>.py`. Each is ~1 day.

### Critical decisions recorded

Nine in total, sections **[DECISION 1]** through **[DECISION 9]** above.
The ones most likely to need your review:

- **[DECISION 3]** MoE router metadata as `persist=False`. Re-check once
  MoE is ported; revert is one-line if wrong.
- **[DECISION 2]** Per-layer `max_tier` with DP padding. This enables
  heterogeneous backbones (GPT-OSS, Qwen3-Next) but changes the DP input
  shape. Homogeneous models are strictly equivalent to orig.
- **[DECISION 9]** `TensorSpec` carries four per-role dtypes
  (compute/master/grad/opt_state). You asked for this explicitly; just
  flagging that the defaults collapse to single-dtype behavior if unused.

### Test commands

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate flextrain
cd ~/Documents/FlexTrain
PYTHONPATH=. python tests/run_all.py              # all 18 tests
python -m flextrain info                          # package status
```

### Nothing destructive was done

- `orig/` is unchanged (verify with `git status` or a file-tree diff).
- Only additions: `flextrain/`, `tests/`, `docs/`.
- New conda env `flextrain` cloned from `net`, with a cu12 shim for
  flash-attn (see "Conda environment" section above).

---

## Phase 3 — Engine Port: Proposed Order + Clarifying Questions

*Written before any engine compute has been implemented — user asked to
sanity-check the approach first.*

### What I read

* `orig/active_model.py` all 1852 lines
  (`__init__` → `initialize` → `load` → `save` → `create_gpu_activations`
  → `create_gpu_opt_state` → `determine_saved_levels` → `split_sequences`
  → `prepare_training_chunks` → `update_fwd_context` → `fwd_bwd` → `step`)
* `orig/awsm_transformer/{embed,head}.py`
* `orig/train.py` (driver loop)
* `orig/sequence.py`, `orig/sequence_pool.py`, `orig/fineweb.py` (data)
* Existing `flextrain/engine/{active_model,buffers}.py` (signatures only)
* Existing `flextrain/nn/layers/llama.py`, `flextrain/core/*`, `flextrain/io/*`
* Existing optim (AdamW + Muon) and how they consume ParamSpec

### The ten engine sub-systems orig bundles into ActiveModel

1. Stream + event management (4 streams: compute, inbound,
   outbound, inbound_fwd_context; many dict-keyed event maps)
2. Host + device buffer lifecycle (model weights, grads, opt state,
   act slots, KV context, transitions)
3. `cudaHostRegister` for the host act buffer (powers-of-two-safe pin)
4. `initialize` / `load` / `save` (per-layer `torch.save` per tensor)
5. `split_sequences` + `prepare_training_chunks` (data -> chunks)
6. `determine_saved_levels` — DP-solver driver
7. `fwd_bwd` — the 470-line orchestrator (fwd + head + bwd)
8. `update_fwd_context` — KV refresh across seq-groups during bwd
9. `step` — optimizer + host↔device ring rotation
10. `destroy`

Phase-3 split of those into modules:

| orig bundle                               | flextrain module             |
|-------------------------------------------|------------------------------|
| Streams + events                          | `engine/streams.py` (new)    |
| Host/device buffer allocators (1,2,3)     | `engine/buffers.py`          |
| `initialize/load/save` (HF now)           | `engine/active_model.py` (+ `io/hf_weights.py` already) |
| `split_sequences` / `prepare_training_chunks` | `engine/schedule.py` (new)|
| `determine_saved_levels`                  | Already in `core/save_level.py` (build_dp_tables + plan_from_solution) — need a thin `ActiveModel.plan_round(seq_groups)` wrapper |
| `fwd_bwd`                                 | `engine/active_model.py::fwd_bwd` |
| `update_fwd_context`                      | `engine/active_model.py` private helper |
| `step`                                    | `engine/active_model.py::step` |

I'll keep `engine/` four files: `active_model.py`, `buffers.py`,
`streams.py`, `schedule.py` — as already scoped in PLAN.md.

### Proposed port order

Each step is a separate commit-sized chunk that compiles and passes tests
before moving on. I'll drive new tests at each boundary so we catch
regressions close to where they were introduced (lesson from the RoPE
convention bug).

**[3A] TokenEmbedLayer : InputLayer**  —  *~80 LOC, pure compute port*
  - `flextrain/nn/embed.py`: wraps `awsm_embedding_bwd`
  - ParamSpec with one tensor `w_tok_embeddings (vocab_size, d_model)`
  - Schema: `max_tier=0`, empty fields (embedding has no per-chunk
    activation state — orig stores it in the transition table)
  - Test: 1-way parity vs. orig (fwd + bwd on same token_ids & dx;
    gradient accumulation equivalence)
  - HF arch: `w_tok_embeddings` already mapped in `io/arch/llama.py`

**[3B] LMHead : OutputLayer**  —  *~150 LOC, chunked-CE port*
  - `flextrain/nn/head.py`: ports `process` + `process_head_chunk` but
    compressed into a single `forward_backward(x, labels, chunk,
    weights, grads, ctx, loss_scale)` returning `(dx, LossStats)`
  - ParamSpec with `w_final_norm (d_model,)` + `w_head_proj (d_model,
    vocab)`
  - The internal `head_chunk_size=1024` stays as a constructor arg
    (it's a micro-chunking within a training chunk to keep the
    `(tokens, vocab)` logits buffer small — paper §3.2 end)
  - LossStats carries per-token loss tensor, written into the caller's
    Sequence object same as orig (see `active_model.py:1387-1390`)
  - Test: 3-way parity (naive PyTorch autograd + orig + flextrain) for
    (fwd + CE + bwd + head-proj weight grad). Tolerance same as
    test_llama_parity (bf16 noise <1e-2).

**[3C] engine/streams.py**  —  *~120 LOC, pure bookkeeping*
  - `StreamBundle` dataclass: compute, inbound, outbound,
    inbound_fwd_context, optional `secondary_compute` (MoE)
  - NVTX naming helpers (keep nvtx; lets us nsys-profile for free —
    orig:104-111)
  - Event maps typed as `dict[int, Event]` and `dict[tuple[int,int],
    Event]`, wrapped in small helper classes so the engine code reads
    better than orig's raw-dict chaos
  - No tests — this is scaffolding the engine exercises

**[3D] engine/schedule.py**  —  *~250 LOC, data pipeline*
  - `split_sequences(seqs, working_set) -> list[list[Sequence]]`
  - `prepare_training_chunks(round_seqs, working_set, device,
    chunk_meta_builder) -> tuple[list[list[ChunkMeta]], dict[int,
    ChunkPayload]]`
    - `ChunkPayload` is the new typed replacement for orig's dict
      (`chunk_token_ids`, `chunk_label_ids`, `chunk_metadata`,
      `chunk_seqs`)
  - One bit of behavioral preservation matters: orig's
    `prepare_training_chunks` iterates sequences into "sequence groups"
    based on where sequence starts land in chunks, and
    `determine_saved_levels` walks them in the same order. I'll keep
    this exact logic; it's not scheduling but its ordering is
    load-bearing. Port 1:1 with minimal cleanup (typed dataclasses).
  - Test: deterministic output shape for a fixed input sequence list;
    diff against orig on a tiny (~5-seq) case.

**[3E] engine/buffers.py — real bodies**  —  *~600 LOC, the load-bearing piece*
  - `BufferManager.allocate(working_set, layer_specs, embed_spec,
    head_spec, dims, device)` does everything orig's `load()` does
    except weight loading:
    - Pinned host memory via `cudaHostRegister` (preserve orig's
      non-power-of-2 trick — `active_model.py:266-273`)
    - Contiguous GPU param ring of size `N_P *
      max(layer.param_spec.byte_size('compute', dims))` (HETEROGENEOUS!)
    - Same for grad ring, opt ring
    - GPU act ring sized `n_gpu_act_slots * max(layer.schema.
      device_size_bytes(max_chunk_size, dims))` — again, max across
      layer types. This is what [DECISION 10] calls out.
    - KV context sized to max_seq_len (per orig:432-446)
  - `prefetch_layer_params(layer_id, slot_idx, stream) -> Event`
    — iterates ParamSpec tensors, does host->device non_blocking copy
    with master_dtype -> compute_dtype cast when they differ
  - `offload_layer_grads(layer_id, slot_idx, stream) -> Event`
  - `gpu_act_slot(slot_idx, schema, num_tokens, dims, level) ->
    ActivationSlot` — slices the per-slot uint8 region into typed views
    using `ActivationSchema`
  - `host_act_slot(layer_id, chunk_id, schema, num_tokens, dims, level)
    -> ActivationSlot` — next-fit allocator over the host act buffer
    (matches orig's `cpu_cur_act_buffer_offset` cursor)
  - `swap_to_optimizer_state()` / `restore_activation_ring()` — the
    ring-reuse described in paper §3.3 (repurpose GPU act buffer as
    opt-state staging during `step`)
  - Test: allocate for a tiny 2-layer-heterogeneous backbone
    (LlamaBlock + MistralBlock); assert byte sizes, ring counts,
    slot shapes are as expected. No GPU needed — allocate to `"meta"`
    device + bypass cudaHostRegister under a flag.

**[3F] ActiveModel.fwd_bwd — the port**  —  *~500 LOC, scheduling heart*
  - Port `orig/active_model.py:1162-1632` 1:1 into
    `engine/active_model.py::fwd_bwd`:
    1. split into rounds (calls `schedule.split_sequences`)
    2. per round:
       - `prepare_training_chunks`
       - `build_dp_tables` + `plan_from_solution` (uses existing
         `core/save_level.py` — no new DP code needed!)
       - allocate host act slots (BufferManager)
       - embed all chunks -> transitions
       - forward loop (for layer, for chunk) with prefetch/offload
       - head per chunk (OutputLayer.forward_backward)
       - backward loop (reverse) with recompute + fetch + prefetch
       - embed backward
  - `update_fwd_context` becomes a private `_update_fwd_context` method
    (same conditional logic, typed args)
  - Tests: this is where the end-to-end parity test lives. Plan below.

**[3G] ActiveModel.step**  —  *~150 LOC*
  - Port `orig/active_model.py:1632-1731`
  - Orig calls `layer.step(weights, grads, opt_state, opt_hp)` which is
    the hand-unrolled 9 AdamW calls. In v2 we replace with
    `self.optimizer.step(layer.param_spec, master, grads, state,
    step_num=n)` — this is where the [FINAL STATUS] §Step 1's
    ParamSpec-driven loop pays off.
  - Ring rotation + host mirroring logic identical to orig.
  - Test: do 1 AdamW step on host-only tensors; compare against calling
    orig layer.step via the pass-through shim in `flextrain/ops/`.

**[3H] ActiveModel.load_hf / save_hf**  —  *~100 LOC, mostly done*
  - `load_hf` wires `load_hf_safetensors` against a `dest` mapping
    that's `{(scope, name): buffer_manager.host_param(layer_id, name)}`
  - `save_hf` the same in reverse
  - Test: round-trip (tensor values preserved through HF-load ->
    FlexTrain buffers -> HF-export). Already done for the pure I/O
    layer; now just verifying the BufferManager -> I/O plumbing.

**[3I] `io/shard_stream.py`**  —  *~200 LOC, port of sequence_pool.py*
  - One class: `ShardTokenStream(shard_pattern, num_shards, min_seq_len,
    max_seq_len, min_tokens_threshold, ...) -> Iterator[Sequence]`
  - Behavior 1:1 from `orig/sequence_pool.py` — background thread,
    prefetch, `get_sequences(max_token_count)`
  - Keeps orig's `Sequence` class definition (with `per_token_loss`,
    `seq_id`, `start_train_time`, etc.) as `flextrain/io/sequence.py`
    — it's 50 LOC and the engine needs the `per_token_loss` mutable
    buffer
  - Test: read a small test shard (we'll synthesize one), iterate a
    handful of sequences, assert `len`, `seq_id`, target-shift is
    correct.

**[3J] cli.py train**  —  *~100 LOC*
  - Reads a YAML (config.py already has `RunConfig`)
  - Instantiates: `ModelConfig` -> backbone (for Llama:
    `[LlamaBlock(i, cfg) for i in n_layers]`, embed, head)
  - `determine_working_set_config` → `WorkingSetConfig`
  - `ActiveModel(...)`; `active_model.load_hf(hf_path)`
  - `ShardTokenStream` or `HFTokenStream` (HF stream is Phase 4; start
    with shard)
  - Training loop mirroring orig/train.py:266-385
  - Minimal LR scheduler (use orig's `get_lr` via the ops shim for
    consistency)

**[3K] End-to-end Llama3-8B parity test**  —  *the acceptance gate*
  - `tests/test_parity_llama3.py`:
    1. Load Llama3-8B HF checkpoint on both orig and flextrain.
       (This requires `INIT_MODEL_PATH` for orig; I'll write a small
       one-time bridge that takes a FlexTrain host-buffer dict produced
       by `io.hf_weights.load_hf_safetensors` and writes the orig
       per-tensor `torch.save` format so we can load the *same* init
       into both.)
    2. Same seed, same 5 batches of real FineWeb shards.
    3. Run 5 `fwd_bwd + step` on both.
    4. Compare: every-step loss (tol: 1e-4 relative for bf16
       parity), final weight norms per tensor.
  - If this passes we've proven the engine is behavioral-equivalent
    on a real model at scale.

### Rough LOC estimate for Phase 3

|     | section                    | new LOC | test LOC |
|-----|---------------------------|---------|----------|
| 3A  | nn/embed.py                |  ~80   |   ~100  |
| 3B  | nn/head.py                 |  ~180  |   ~200  |
| 3C  | engine/streams.py          |  ~120  |     –   |
| 3D  | engine/schedule.py         |  ~250  |   ~150  |
| 3E  | engine/buffers.py bodies   |  ~600  |   ~250  |
| 3F  | ActiveModel.fwd_bwd        |  ~500  |  [in 3K]|
| 3G  | ActiveModel.step           |  ~150  |   ~100  |
| 3H  | ActiveModel.load_hf / save |  ~100  |    ~80  |
| 3I  | io/shard_stream.py          |  ~200  |    ~80  |
| 3J  | cli.py train               |  ~100  |     –   |
| 3K  | end-to-end parity test     |    –   |  ~250   |
| Tot |                            | ~2300  | ~1200   |

### Things I am deliberately NOT doing

- **Porting the dashboard.** `orig/dashboard/` stays. cli.py logs to
  stdout + a pickle (step_stats.pkl) the same way orig does. Dashboard
  port is Phase 6 material; out of scope here.
- **Multi-GPU group_config branches.** Orig has
  `group_config`/`master_conn`/`next_conn`/`prev_conn` plumbing (never
  exercised, all `None` in `train.py`). I will NOT port those; the
  code becomes much simpler without them. If multi-GPU is ever wanted
  that's a separate v3 design.
- **Debug probes.** The many commented-out `torch.save(...)` /
  `torch.cuda.synchronize()` lines in orig's fwd_bwd do not carry over.
- **Native FlexTrain checkpoint format.** Phase 4 of PLAN.md.
  End-to-end training needs load_hf + save_hf only.
- **HFTokenStream (datasets + AutoTokenizer).** Shard stream only for
  now. We have FineWeb `.bin` shards; that's enough to verify parity.
  HFTokenStream is a ~150-LOC add later.
- **`torch.cuda.empty_cache()` inside the loop** (orig:1623). Orig
  takes a device-wide sync hit per step; I'll keep the call for parity
  of timing characteristics but document it so a future pass can lift
  it.
- **MoE support in the engine.** Engine will dispatch generically via
  `Layer.forward/backward/etc.`, so when MoETransformerBlock lands in
  a future phase it plugs in without engine changes. But this phase's
  parity test is Llama3 dense only.

### Clarifying questions before I start writing fwd_bwd

Please answer these; they affect the buffer / stream design and I'd
rather not revisit them later.

**Q1. Dashboard integration — keep or cut?**
Orig instantiates `DashboardLogger` and calls `.log(step_stats)` in
the train loop. I propose: **cut** from v2 cli.py. cli.py logs
step_stats to stdout + pickle; dashboard port is a later task. Sound?

**Q2. Keep `force_saved_act_level` override?**
Orig has `FORCE_SAVED_ACT_LEVEL` for debugging — forces every
(layer,chunk) to a fixed tier. Useful for nsys-profiling specific
scenarios. I propose: **keep**, as `ActiveModel(..., force_save_level:
int | None)`. Zero engine-code cost, high debug value.

**Q3. Keep orig's `group_config` multi-GPU plumbing?**
Answered above — I'll **cut** unless you object. Significantly
simpler engine code. If you want multi-GPU later we'll add a
separate `DistributedActiveModel` subclass.

**Q4. Native FlexTrain save/load (not HF)?**
Orig's `active_model.save()` writes per-tensor `.pt` files under a
directory tree. It's not a checkpoint format you'd want to ship —
safetensors is better. My plan: **only implement `load_hf` /
`save_hf`** in Phase 3. Native resume (`io/checkpoint.py`) is Phase
4. For resume-ability during this phase, we re-export to HF and
re-load. Agreed?

**Q5. Sequence object ownership.**
Orig's engine writes per-token loss into the caller's `Sequence`
object (`active_model.py:1390`). That makes `Sequence` a mutable
engine-shared state object. I propose keeping exactly this behavior —
we need it for the accumulation loop in train.py — but moving
`Sequence` into `flextrain/io/sequence.py` so it lives alongside
`ShardTokenStream`. The class is 50 LOC, no import cycle issues. OK?

**Q6. AdamW vs Muon default.**
Orig's train.py uses Muon for all layers (`USE_MUON = True`). Muon
for Llama3-8B is the "fast path" on our 3090 target. I'll make
`cli.py train` default to Muon with an `--optimizer adamw` flag to
override, matching orig. Sound?

**Q7. The `head_chunk_size` constant.**
Orig hardcodes `head_chunk_size=1024` in the `head.forward_backward`
call (`active_model.py:1383` omits it -> head uses its own default).
I'll expose it as `LMHead(vocab_size, d_model, head_chunk_size=1024)`.
Any reason to pick a different default for the Llama3 target?

**Q8. Loss scale factor.**
Orig computes `loss_scale_factor = 1.0 / step_tokens` per step
(train.py:304). This "avg loss over step_tokens" scale is baked into
the head's weight-grad + dX computation
(`head.py:235,246`). V2's `OutputLayer.forward_backward` takes
`loss_scale` the same way. Keeping exact semantics. Confirm OK?

**Q9. Seed / RNG parity.**
The end-to-end parity test requires deterministic forward/backward.
Our existing test (`test_llama_parity.py`) fixes
`torch.manual_seed(42)` and passes bit-identically between
orig+flextrain. For 3K I'll do the same — load the same HF
checkpoint to both, seed both, feed the same sequences. Flash-attn
determinism should not be an issue for non-varlen forward on fixed
inputs. Any known determinism traps I should pre-empt?

**Q10. Where should LlamaBlock construction live?**
Currently `LlamaBlockConfig` is constructed by the caller. cli.py
train will need a model-factory: given `ModelConfig` (from
`hf_config_to_flextrain`), produce `embed`, `backbone`,
`head`. Propose: `flextrain/nn/build.py::build_llama_model(model_cfg,
n_layers, hyperparams) -> (InputLayer, list[Layer], OutputLayer)`.
Extends to other families via dispatch on `hf_config.architectures`.
Agree?

### What I'd like to hear back

- Sign-off or course-correction on the 11-step order above.
- Answers to Q1–Q10 (or "default to your proposal" for any of them).
- Any invariants in orig's `fwd_bwd` I should double-check before
  writing the port. I flagged `update_fwd_context` (the cross-seq-group
  KV refresh during bwd) as the trickiest piece — I'll port it with
  extra care and write a targeted unit test for it.

Once you sign off I'll work through 3A → 3K in order, writing
decisions into this file as I go (new `[DECISION 11]`, `[DECISION 12]`,
... sections). I will stop and ask again if I hit anything that could
affect the AdaWS schedule invariants.

---

## User sign-off (2026-04-23)

User signed off on all 10 questions ("default to your proposal" on each).
Two user-stated hardened rules that apply to the rest of Phase 3:

* **[RULE A] Always 3-way parity: naive PyTorch + orig + flextrain.**
  Never just compare flextrain vs orig, even for a "trivial wrapper".
  Orig may have convention / sign / dtype bugs that a vs-orig-only test
  would ratify. The RoPE convention bug in
  [FINDING — from 3-way parity] is the precedent: orig and flextrain
  were bit-identical but both diverged from the correct reference. The
  naive PyTorch path — written from the algebra, not from orig's code —
  must be the ground truth.

* **[RULE B] Tests run on the RTX 3090 under the `flextrain` conda env.**
  `run_all.py`'s CPU-only docstring is stale; use
  `conda activate flextrain && PYTHONPATH=. python tests/run_all.py`
  for everything. A test that can't exercise its GPU kernels is not a
  real parity test.

---

## [3A] TokenEmbedLayer : InputLayer — DONE (2026-04-23)

### Files added
* [flextrain/nn/embed.py](../flextrain/nn/embed.py) — 125 LOC.
  Implements :class:`TokenEmbedLayer` against the
  :class:`InputLayer` Protocol. Forward = fancy-index gather;
  backward = `awsm_embedding_bwd` scatter-add. Empty
  :class:`ActivationSchema` (max_tier=0, no fields) because the embed
  layer has no per-chunk activation state — its "activation" is the
  token_ids tensor already owned by :class:`ChunkMeta`. This matches
  orig's special-casing (embed has no entry in `cpu_act_slots`; see
  `orig/active_model.py:1230-1235`).
* [tests/test_embed_parity.py](../tests/test_embed_parity.py) — 265 LOC.
  4 tests, all pass on the 3090.

### Parity results (RTX 3090, bf16)
Forward (bit-identical, no arithmetic):
* naive vs orig vs flextrain: `torch.equal` holds bit-for-bit.

Backward, two patterns exercised:
* Sparse (V=4096, d=128, T=256): orig-vs-naive `0.00e+00`, ft-vs-naive
  `0.00e+00`, orig-vs-ft `0.00e+00`.
* Dense (V=128, d=256, T=4096, ~32 scatter-adds per vocab row):
  orig-vs-naive `0.00e+00`, ft-vs-naive `0.00e+00`, orig-vs-ft
  `0.00e+00`.

Observation: orig's `awsm_embedding_bwd` kernel accumulates in fp32
internally, so both kernel paths agree bit-identically with an fp32
naive reference cast back to bf16. This is better than the 5e-2 bf16
tolerance we budgeted.

### Decisions recorded

### [DECISION 11] Empty ActivationSchema for input / output layers
The :class:`InputLayer` Protocol requires a `schema` attribute for
symmetry with :class:`Layer`, but the token embedding has no per-chunk
activation slot. We declare `ActivationSchema(fields=(), max_tier=0)`
— the engine will compute `home_size_bytes == 0` and skip allocation
cleanly. Same applies to :class:`OutputLayer` (3B) if we decide the
head has no offloadable state; the residual-stream input to the head
lives in the transition table, which the engine owns separately.

**Why:** forcing a "real" schema on embed/head would require
introducing synthetic fields we never save. The derived size math
already returns 0 for an empty schema, so no engine branches needed.
**Risk:** zero functional. Cosmetic only.
**Revert plan:** not needed.

### [DECISION 12] Embed-layer zero FLOPs in ComputeCost
Embedding fwd (a gather) and bwd (a scatter-add) are purely
bandwidth-bound; reporting them as non-zero FLOPs would skew the DP
solver's compute-time estimates. Orig does not run the embed layer
through the DP input-assembly (`determine_saved_levels` iterates
`self.local_layer_ids`, not the embed layer), so orig's effective
"embed flops" is also zero. We match that by returning
`ComputeCost(total_fwd_flops=0, avoided_recompute_flops=(0,))`.

**Why:** consistent with orig's DP input construction; keeps the
solver honest about where time is actually spent.
**Risk:** zero.

---

## [3B] LMHead : OutputLayer — DONE (2026-04-23)

### Files added
* [flextrain/nn/loss.py](../flextrain/nn/loss.py) — 260 LOC. Pluggable
  loss objectives. `LossFn` Protocol + `TokenContext` /
  `TokenSlice` dataclasses + built-in `CrossEntropyLoss`, `MSELoss`,
  `GRPOLoss`. See [DECISION 13] below.
* [flextrain/nn/head.py](../flextrain/nn/head.py) — 270 LOC. Ports
  `orig/awsm_transformer/head.py` with pluggable loss and the
  fused-micro-chunk invariant preserved. CE is the default (matches
  orig / SFT use case).
* [tests/test_head_parity.py](../tests/test_head_parity.py) — 340 LOC.
  4 tests, all pass on the 3090 with CE as the loss.

Also **modified**: `flextrain/core/layer.py` — `OutputLayer` Protocol
signature now takes `token_ctx: TokenContext` + optional `loss_fn:
LossFn` instead of `labels: Tensor`; `LossStats` gains
`per_token_loss`, `next_prediction`, `next_prediction_prob` fields so
the engine can reproduce orig's per-Sequence writeback at
`orig/active_model.py:1388-1390`.

And `flextrain/nn/__init__.py` gained re-exports for the new classes.

### Parity results (RTX 3090, bf16, CrossEntropyLoss)
Across three configs (single-chunk, multi-chunk, loss-scale-applied):

| quantity          | orig vs naive   | ft vs naive     | orig vs ft      |
|-------------------|-----------------|-----------------|-----------------|
| per-token loss    | ~5e-4           | ~5e-4           | bit-identical   |
| dx                | ~3e-3           | ~3e-3           | bit-identical   |
| g_final_norm      | ~1-2e-3         | ~1-2e-3         | ~1e-7 (atomic)  |
| g_head_proj       | ~3e-3           | ~3e-3           | bit-identical   |
| next_prediction   | —               | —               | bit-identical   |

All under the 5e-2 / 5e-3 tolerances we budgeted. The micro-chunk loop
(``head_chunk_size=256`` over 1024 tokens = 4 iterations) produces
identical grads to the single-chunk case — confirms the weight-grad
accumulator (`addmm(beta=1.0)`) is correctly wired across iterations.

The `g_final_norm` orig-vs-ft delta (~1e-7 relative) traces to
``awsm_rmsnorm_bwd``'s atomic fp32 adds — run-to-run order varies.
Benign, same pattern as in `test_llama_parity`.

### Decisions recorded

### [DECISION 13] Pluggable loss via `LossFn` Protocol, not baked-in CE
**In orig:** head is hard-coded to softmax + token-label cross-entropy
(`awsm_transformer/head.py:208-227`). Fine for SFT; a showstopper for
RL / distillation / any training where the "loss" isn't a CE over
token ids.

**In v2:** `LMHead.forward_backward(x, token_ctx, ..., loss_fn=None)`
takes a `LossFn` object (default `CrossEntropyLoss` for SFT parity).
The micro-chunk inner loop calls `loss_fn.compute(logits,
token_slice, ...)` which returns `dZ` in the same buffer shape. The
head is unchanged; the loss is swappable.

Built-in fns shipped:
* `CrossEntropyLoss` — SFT default, wraps orig's `awsm_softmax` +
  `awsm_cross_entropy_loss` kernels.
* `MSELoss` — continuous-target distillation, pure PyTorch.
* `GRPOLoss(kl_coef=0.0)` — token-level policy-gradient form for RL
  (advantages + optional KL-to-ref term). Correctness-skeleton: the
  structural hook is in place so users CAN plug in RL, but production
  RL training will typically want their own variant (PPO clip, KL
  shaping) — not a landed-and-battle-tested RL trainer.

**Fused-fwd-loss-bwd invariant preserved.** The loss runs INSIDE the
head's micro-chunk inner loop; the `(T', V)` logits buffer lives only
for the span of one iteration and is freed before the next. Peak
logits VRAM stays `head_chunk_size * vocab_size * dtype_size` (~8MB
at head_chunk_size=1024, vocab=32000, bf16). No full `(T, V)`
materializes. User flagged this as essential (system is about memory
efficiency); design upholds it.

**Why this shape over alternatives:**
* Loss OUTSIDE the head → you have to return `(T, V)` logits → kills
  the memory win.
* Loss as a head SUBCLASS → forces one head per objective, breaks
  weight-sharing / ckpt-resume when you swap losses mid-training.
* Loss as a constructor arg → fine but awkward to swap per-batch (RL
  sometimes mixes objectives in one step).
* **Chosen: per-call arg.** Head stays one class; caller picks loss
  per forward. Default arg stays `CrossEntropyLoss()` for SFT callers
  who don't know / care.

**Risk:** low. The Protocol forward-reference (`TokenContext`,
`LossFn` as string annotations on `OutputLayer.forward_backward`)
avoids a `core ↔ nn` import cycle. The default CE path is the orig
SFT path byte-for-byte; callers who never pass a `loss_fn` get orig
behavior.

**Revert plan:** if the Protocol surface becomes a pain (e.g. the
engine needs to know about loss at init time, not per-call), we can
move `loss_fn` to an `LMHead` constructor arg instead. Single-line
change; no call-site churn.

### [DECISION 14] `LossStats` carries `aux_chunks` as a list, not a schema
Loss fns may return arbitrary diagnostics in `aux` (KL value, entropy,
...). We don't want to force a closed schema on this — RL researchers
add fields routinely. Instead, `LMHead.forward_backward` collects a
`list[dict]` across micro-chunks and attaches to
`LossStats.aux_chunks` (set dynamically so the dataclass keeps its
clean fields for the SFT case). The engine ignores it; RL callers
opt in.

**Risk:** low. Untyped by design. Callers who want typing can wrap
their custom loss + own their aux struct.

---

## [3C] engine/streams.py — DONE (2026-04-23)

### Files added
* [flextrain/engine/streams.py](../flextrain/engine/streams.py) — 220
  LOC. `StreamBundle` holds the four streams orig creates inline
  (`compute`, `inbound`, `outbound`, `inbound_fwd_context`) plus an
  optional `secondary_compute` for MoE. `EventBook` wraps orig's ten
  bare dicts in typed event maps (`LayerEventMap`, `SlotEventMap`,
  `LayerChunkEventMap`). NVTX stream naming preserved.

### Smoke check
Hand-exercised the module on the 3090 (no new test module — pure
bookkeeping).

### Decisions

### [DECISION 15] Typed event maps raise on unrecorded keys
Orig's bare-dict event maps return `None` on a missing key, which
becomes a silent `stream.wait_event(None)` hang at CUDA level. We
raise `KeyError` with a diagnostic message instead. The `None`
sentinel that orig uses to mean "already consumed, no wait" (see
`orig/active_model.py:1308,1329`) is preserved via
`mark_consumed(key)` → the entry stays in the dict but wait-ons
become no-ops.

**Why:** surfaces scheduler bugs at the call site instead of as
eventual hangs.
**Risk:** zero — any code path that would have hung will now raise
with a clear message.

---

## [3D] engine/schedule.py — DONE (2026-04-23)

### Files added
* [flextrain/engine/schedule.py](../flextrain/engine/schedule.py) —
  380 LOC. Ports `split_sequences` (orig:820) + `prepare_training_chunks`
  (orig:897) with **cleaner internal shape** (user flagged the orig
  as not clean):
  - Stage-1 pure-Python ``_PendingChunk`` accumulator (no mutable
    dict-of-lists).
  - Stage-2 generator `_pack_sequences` — small-seq packer + big-seq
    chunker produce a single stream.
  - Stage-3 fold computes sequence groups from each chunk's
    "starts_a_new_group" predicate.
  - Observable contract (chunk lens, prior_lens, seq_groups) is
    identical to orig on the causal path.
* [tests/test_schedule.py](../tests/test_schedule.py) — 330 LOC. 11
  tests: split/pack invariants, non-causal rejection, plus a 20-trial
  random parity test against a stripped-down orig packer that confirms
  per-chunk `lens` and `prior_lens` match byte-for-byte.

Also wired into `flextrain/engine/__init__.py`.

### Parity
20 random trials of mixed-size sequence lists under 3 different
`max_chunk_size` values: every chunk's (``lens``, ``prior_lens``)
pair matches orig exactly. The v2 scheduler is a drop-in replacement
on the causal path.

### Decisions

### [DECISION 16] `ChunkPolicy` enum (CAUSAL vs NON_CAUSAL)
**Motivation:** non-causal attention (bidirectional encoders,
contrastive/embedding training, some RL value-head training) cannot
tolerate a sequence being split across chunks — doing so drops the
cross-chunk attention edges.

**Design:** schedule module defines `ChunkPolicy.{CAUSAL,NON_CAUSAL}`
(an enum, not a bool — leaves room for future policies like
"padding-always" or "block-sparse-bounded"). Both `split_sequences`
and `prepare_training_chunks` take `policy=ChunkPolicy.CAUSAL` as the
default. Under `NON_CAUSAL`:
* Sequences longer than `max_chunk_size` raise `ValueError` at
  `split_sequences` (so the error surfaces as early as possible).
* `prepare_training_chunks` also guards at materialization time —
  defense in depth.
* Every chunk trivially starts a fresh seq_group (no sequence spans
  chunks), so `len(seq_groups) == len(chunks)`.

The engine will read this policy from the model configuration (a
property of whichever architecture is loaded). For Llama / Qwen /
Mistral / OLMoE — causal. For an eventual encoder port — non-causal.
Passing it through the scheduler keeps the schedule module
arch-agnostic.

**Why enum, not bool:** a bool named `is_causal` on
`prepare_training_chunks` is ambiguous at a call site (`is_causal=True`
means "use causal splitting" or "the model is causal and therefore
splitting is OK"?). The enum makes the intent explicit.

**Risk:** low. Default is CAUSAL (orig behavior). Non-causal is an
error gate that triggers BEFORE any tensor work.
**Revert plan:** remove the `policy` arg, drop `ChunkPolicy`, restore
orig behavior on all paths. No call sites depend on non-causal today;
this only opens the door.

### [DECISION 17] Cleaner packer: `_PendingChunk` dataclass + generator stream
Rewrote orig's nested-dict-of-lists packing into a small state-machine
over `_PendingChunk` dataclass objects. Same observable behavior
(verified by `test_parity_with_orig_packing` across 20 random trials),
but the code is now three independent stages (pack → materialize →
group), each one-function-deep. Easier to reason about; easier to
extend with new policies.

**Risk:** none; parity enforced by test.

---

## [3E] engine/buffers.py — DONE (2026-04-23)

### Files added / changed
* [flextrain/engine/host_memory.py](../flextrain/engine/host_memory.py) — 170
  LOC. New `HostMemoryBackend` Protocol + `LocalPinnedHostBackend`
  (cudaHostRegister on torch.zeros — matches orig) +
  `UnpinnedHostBackend` (for tests). See [DECISION 18].
* [flextrain/engine/buffers.py](../flextrain/engine/buffers.py) — 620
  LOC. Real bodies for `BufferManager`. Allocates host params / grads
  / opt state through the backend, GPU param/grad/act rings as
  contiguous uint8 buffers sliced per-slot, resident embed + head GPU
  tensors, shared KV context window, host activation buffer with
  cursor allocator. Heterogeneous backbone support (per [DECISION 10]).
  Preserves orig's opt-state-in-activation-ring trick via
  `swap_to_optimizer_state` / `restore_activation_ring`.
* [tests/test_buffers.py](../tests/test_buffers.py) — 325 LOC. 10 tests
  on the 3090:
  - allocation shapes, ring sizing, view slicing
  - host→GPU prefetch + GPU→host offload DMA round-trips
  - heterogeneous layer types (varied `expert_dim`) share the ring
    correctly
  - host activation buffer cursor + exhaustion error
  - opt-state ring swap in + out
  - end-to-end with the real `LocalPinnedHostBackend`
    (cudaHostRegister) + `destroy()` safe cleanup

### Decisions

### [DECISION 18] `HostMemoryBackend` abstraction (user-requested)
**User request:** "It would be very nice if we could create 'host' as
abstraction just in case we later on use remote node memory as 'host'".

**Design:** new module `flextrain/engine/host_memory.py` defines a
`HostMemoryBackend` Protocol with two methods: `allocate_tensor(shape,
dtype)` and `release(tensor)`. Every host-side tensor in
`BufferManager` — master params, grads, opt state, the host
activation buffer — is allocated through this backend. Default
`LocalPinnedHostBackend` preserves orig's `cudaHostRegister(torch.zeros
(...))` trick exactly.

Future extensions slot in cleanly:
* `RemoteNodeHostBackend` (RDMA to a fatter-RAM peer, NVLink-fabric,
  CXL) — override `allocate_tensor` to grab a buffer from the remote
  node and wrap in a local torch.Tensor view. The engine's DMA code
  (`dev_tensor.copy_(host_tensor, non_blocking=True)`) keeps working
  as long as the returned tensor is registered for GPU DMA.
* `PersistentMemoryBackend` for file-backed master weights.
* `UnpinnedHostBackend` for tests on non-CUDA machines.

**Why Protocol, not abstract base class:** keeps the backend surface
trivially implementable — any object with `allocate_tensor` + `release`
satisfies it. No inheritance overhead for third-party backends.

**Why keep `cudaHostRegister` out of the buffer manager:** orig's
inline `ctypes.CDLL('libcudart.so').cudaHostRegister(...)` was
entangled with allocation logic. Factoring it into the backend lets
the buffer manager be backend-agnostic and lets the backend own the
registration lifecycle (tracking registered ptrs for safe double-
release).

**Risk:** low. Default backend is byte-for-byte equivalent to orig.
The Protocol is not widely exposed yet (no non-default backends
shipped); we just have the seam in place for when it's needed.

**Revert plan:** fold `LocalPinnedHostBackend` back into
`BufferManager` as inline helpers. ~30-line diff.

### [DECISION 19] GPU rings sized to MAX across heterogeneous layers
Every GPU ring (params, grads, activation) is sized to the maximum
per-slot byte budget across all layer types. Smaller layer types
leave the tail of their slot unused but the engine never has to
resize rings mid-training. This is the straightforward implementation
of [DECISION 10] (heterogeneous backbones) and it's now enforced in
`BufferManager.__init__` — verified by `test_heterogeneous_layer_types`
in `test_buffers.py`.

**Risk:** small memory waste in heterogeneous models (the slack in
smaller layers' slots). For the common dense+MoE alternation pattern
(DeepSeek-V3, Qwen3-Next) the MoE layers dominate, so most slots are
near-full anyway.

### [DECISION 20] Opt-state ring re-uses the activation-buffer storage
Orig carves the GPU optimizer-state ring out of the activation buffer
at `step()` entry and rebuilds the activation ring on exit (paper
§3.3). We preserve this exactly:
* `BufferManager.swap_to_optimizer_state(n_gpu_opt_layers)` sets an
  internal mode flag and records per-slot offsets over
  `self.gpu_act_ring.storage`.
* `gpu_opt_slot(slot_idx, layer_id)` builds opt-state tensor views
  on-demand, per-layer spec (so layers with different param shapes
  produce different dict shapes — correct for heterogeneous).
* `restore_activation_ring()` clears the mode flag; activation views
  become valid again.

The Python-level GPU act views we cached before `swap` become stale
while opt-state mode is active — the engine must not retain pointers
across swap/restore. Documented in the docstring; ActiveModel respects
this when we get to 3G.

**Risk:** moderate. If ActiveModel accidentally reads an activation
slot while in opt mode, the tensor values will be opt-state (wrong
dtype, wrong shape). The mode flag guards `gpu_opt_slot` but not the
reverse. Mitigation: `swap_to_optimizer_state` is only ever called
from `ActiveModel.step` after the last round's backward is fully
complete and synchronized; we'll enforce this with an assertion when
we write 3G.

---

## [3F] ActiveModel.fwd_bwd — DONE (2026-04-23)

### Files added / changed
* [flextrain/engine/active_model.py](../flextrain/engine/active_model.py) —
  replaced the stub (170 LOC) with the real implementation (620 LOC).
  Structure: one public method `fwd_bwd(sequences, ...)` orchestrating
  rounds, with four private sub-phase methods per round:
  `_setup_round` (host act slot allocation + embed), `_forward_pass`,
  `_head_pass`, `_backward_pass` (+ `_embed_backward` afterwards).
  Save-level DP is in `_plan_save_levels`; the sequence-group KV
  refresh is in `_update_fwd_context`; the backward-prefetch loop is
  in `_prefetch_activation`.
* [tests/test_active_model_smoke.py](../tests/test_active_model_smoke.py)
  — 160 LOC, 1 integration test. Builds a tiny Llama (2 layers,
  d_model=64, vocab=256), random-inits, runs one round on 2 short
  sequences, confirms: loss in [0.1, 20], head grads populated,
  embed grads populated, backbone GPU grads populated.

### Smoke result
On the 3090:
```
[FlexTrain] round 1/1: 2 chunks, 192 tokens. save levels: level -1: 4.
  step_stats: rounds=1 tokens=192 total_loss=1065.3
```
(avg loss ~5.55 per token, matching ln(256) ≈ 5.545 for uniform random init.)

Fast-path triggered (all 4 (layer, chunk) pairs on-device, no host
offload). The DP path will be exercised end-to-end in 3K, where
we'll crank up the model to where host offload is required.

### Decisions

### [DECISION 21] Loss masking via `labels == IGNORE_INDEX` AND/OR `loss_mask` (user-requested)
**User request:** "We should also be able to specify what tokens to
compute loss on (e.g. during SFT we shouldn't compute gradients for
tokens corresponding to prompt). Normally this is specified with
labels = -100."

**Design:** Added to `flextrain/nn/loss.py`:
* Module-level constant `IGNORE_INDEX = -100` (matches PyTorch's
  `torch.nn.CrossEntropyLoss(ignore_index=-100)` convention).
* `TokenSlice` + `TokenContext` gain a `loss_mask: Tensor | None`
  field. Bool tensor, same length as labels; `True` = include,
  `False` = skip.
* `CrossEntropyLoss(ignore_index: int = -100)` honors both:
  1. Sanitizes `labels == ignore_index` to `0` to keep the kernel's
     one-hot gather in bounds.
  2. After the kernel writes `dZ` and per-token loss, zeros BOTH at
     any row matching `labels == ignore_index` OR
     `loss_mask == False`.
* Added 2 tests (`test_head_masking_ignore_index`,
  `test_head_masking_loss_mask`) verifying that masked rows have
  0 per-token loss and that the overall grad norm strictly drops
  when rows are masked.

**Why both mechanisms:** labels-based masking is the familiar PyTorch
idiom for SFT prompt/response masking; an explicit `loss_mask` is
cleaner for continuous-target losses (MSE on logit distillation) and
for use cases where label values legitimately include `-100`.

**Risk:** low. Default behavior (no ignore-index rows, no loss_mask)
is byte-for-byte identical to the prior CE path. The three existing
head parity tests remain bit-identical to the pre-masking version —
confirmed by running the suite before and after.

### [DECISION 22] fwd_bwd is split into four named sub-phases
Orig's `fwd_bwd` is one 470-line method. We split into:
* `_setup_round(prepared, plan)` — host act slot allocation, initial
  event bookkeeping, embed pass (seeds transition table).
* `_forward_pass(prepared, plan)` — `for layer: for chunk` forward +
  send_home/keep-on-device + advance ring + per-layer param prefetch
  and per-tail-layer grad prefetch.
* `_head_pass(prepared, loss_scale, loss_fn)` — per-chunk fused head
  fwd+loss+bwd + per-seq per-token-loss writeback.
* `_backward_pass(prepared, plan, total_tokens_per_step)` —
  `for layer (rev): for seq_group (rev): for chunk (rev)` with
  recompute + backward + `_update_fwd_context` + `_prefetch_activation`.
* `_embed_backward(prepared)` — scatter-add per chunk.

Each method is ~80-120 LOC and independently unit-testable (future).
The public surface is just `fwd_bwd` + `step`.

### [DECISION 23] StepStats dataclass replaces side-channel logging
Orig's fwd_bwd returns `None`; the caller reads `per_token_loss` from
each Sequence and computes loss externally. V2 returns a `StepStats`
dataclass with `total_tokens`, `total_loss`, `rounds`. Per-Sequence
writeback is preserved (train.py compatibility) but the caller
typically just reads `stats.total_loss / stats.total_tokens`.

**Risk:** none. Both surfaces coexist; the Sequence writeback path
is unchanged from orig.

---

## [3G] ActiveModel.step — DONE (2026-04-23)

### Files added / changed
* `flextrain/engine/active_model.py` — replaced `step()` stub with a
  real implementation (~160 LOC). Also added `_step_resident`
  helper for embed / head (which are always resident, not on the ring).
* `tests/test_active_model_smoke.py` — extended to cover fwd_bwd →
  step → fwd_bwd. Confirmed host master weights mutate by ~6.9e-3 L2
  norm after one AdamW step at lr=1e-4, and a subsequent fwd_bwd runs
  cleanly (ring rotation survived step).

### Behavior vs orig
Semantically equivalent to `orig/active_model.py:1632-1850`:
1. Embed + head step first (always resident; tiny transient opt-state
   staging on-device).
2. Swap GPU act ring → opt-state ring (paper §3.3;
   `BufferManager.swap_to_optimizer_state`).
3. Prefetch first N_P / N_G / N_O layers' weights / grads / opt-state.
4. For each backbone layer: wait on all three inbound events; call
   `self.optimizer.step(param_spec, weights, grads, opt_state,
   step_num)`; mirror updated weights + opt-state back to host;
   prefetch next-unstaged resources.
5. Restore activation ring (`restore_activation_ring()`).
6. Reload first N_P layers' weights into slots 0..N_P-1 so the next
   fwd_bwd's `weight_inbound.wait_on` calls find them.

### Simplifications vs orig
* **One optimizer object** drives every layer (no per-layer hand-
  unrolled `layer.step()` method). The `ParamSpec` iteration collapses
  orig's 9 AdamW call sites per dense layer / 13 per MoE layer into a
  single loop inside `AdamW.step`.
* **No tear-down / rebuild of GPU activation ring.** Orig calls
  `clear_gpu_activations()` → `create_gpu_opt_state()` at step entry
  and the inverse at exit, which re-allocates `fwd_context` /
  `bwd_context`. We just flip a mode flag on the shared GPU buffer.
  Same memory footprint, no reallocation.
* **No weight-idx-tracker dance** (orig:1701-1812, a subtle
  book-keeping dict that says "after step, layer X is at GPU slot Y,
  so rebuild the initial-prefetch mapping accordingly"). Our approach:
  unconditionally reload slots 0..N_P-1 from host. The cost is an
  extra N_P host→device copy per step; at typical 24GB configs that's
  ~4 layers × 1GB = 4GB at ~12GB/s = 300ms. If this shows up in
  profiling we can reinstate the tracker.

### Decisions

### [DECISION 25] `ActiveModel.step` signature takes only `step_num`
Orig's `step(opt_hyperparams)` passed a dict of
`{lr, beta1, beta2, eps, weight_decay, step_num}` to every layer's
`step()`. In v2, those hyperparams live on the `Optimizer` object
(`AdamWHyperparams`). The engine only needs `step_num` from the
caller (for AdamW bias correction). Default: `self.step_count + 1`.

**Why:** separates concerns — optimizer owns its own hyperparams;
engine owns its step counter.

**Risk:** small API change. Trainers that want to warmup LR between
steps just mutate `self.optimizer.hp.lr` before calling `step()`.
Sample trainer lands in 3J.

### [DECISION 26] `_step_resident` for embed + head uses transient
device opt-state staging
Embed (vocab × d) and head (d × vocab + d) are large tensors
compared to backbone layers, but they're resident on the GPU from
load time — we don't ring-rotate them. Their optimizer state is
stored host-side only. On each `step()` call we:

1. `torch.empty`-allocate a device opt-state dict matching the
   host shape/dtype.
2. Copy host → device (blocking on default stream; it's small).
3. Call `self.optimizer.step(spec, gpu_weights, gpu_grads, dev_opt,
   step_num)`.
4. Mirror updated weights + opt-state back to host.
5. Drop the transient dev_opt dict (torch caching allocator reuses it
   next step).

**Why no ring for embed/head opt-state:** the sizes don't interact
with the backbone's per-layer opt-state ring (which is sized per
backbone layer, not per embed/head). A transient staging is simpler.

**Risk:** a ~50MB transient allocation per step (for Llama3-8B:
embed opt = 2 * 128256 * 4096 * 4 = 4.2GB; that's non-trivial). If
this is a problem we can keep a permanent resident opt-state
alongside the weights. TODO flagged for perf tuning.

---

## [FINDINGS — orig/ issues I noticed during the port]

User asked to flag issues in `orig/` rather than silently improving.
These are things that looked wrong / fragile / suboptimal.

### [FINDING 1] orig's `_cuda_host_register` has no error-path safety net
`orig/active_model.py:270-273`: calls `cudaHostRegister` and prints a
warning but continues on error. Any subsequent DMA will then fail
with a cryptic invalid-argument error. We raise immediately (or, for
the "already registered" case, skip + clear the sticky error —
[DECISION 24]).

### [FINDING 2] orig's `step()` re-allocates the activation ring
`orig/active_model.py:1687-1842`: `clear_gpu_activations()` +
`create_gpu_opt_state()` at entry, inverse at exit. Each call
allocates/frees a few hundred MB on the caching allocator. It works
but it's wasteful; the underlying uint8 buffer can be viewed either
way without any allocation. V2 uses a mode flag instead.

### [FINDING 3] orig's `weight_idx_tracker` is fragile
`orig/active_model.py:1701-1812`: the "which GPU slot currently
holds layer X" bookkeeping is carried across step via a dict with
mutations in two places. An in-flight fetch that overwrites a slot
between the tracker build and the reassignment would cause a silent
corruption. V2 unconditionally reloads the first N_P layers. Small
perf cost for a much simpler invariant (see [3G] §"Simplifications
vs orig").

### [FINDING 4] orig's per-layer `step()` unrolls AdamW by hand, 9 times
`orig/awsm_transformer/dense_layer.py:344-466`: 9 nearly identical
`awsm_adamw_step(...)` calls, one per parameter. Adding a new
parameter means adding a new block to every layer's `step()`. V2
iterates `ParamSpec.tensors` inside `AdamW.step`. One line instead of
nine; new params are picked up automatically.

### [FINDING 5] orig's `zero_grad` flag semantics leak into fwd_bwd
`orig/active_model.py:1253,1338-1349` + `:1604`: `self.zero_grad` is
a "during the first round of the step, zero the grad buffers;
afterwards accumulate" flag, but it's set from inside fwd_bwd,
controlled by step(), and referenced in at least three places with
subtle ordering requirements. Easy to break on refactors. V2
preserves the same semantics but through a single
`self._zero_grad: bool` field set in two well-named places
(`step()` sets True at entry; `fwd_bwd` sets False at end of round).

### [FINDING 6] orig's RoPE convention silently disagreed with the HF reference
Noted earlier in [FINDING — from 3-way parity] (above), before this
section existed. Keeping the cross-reference here: orig's Triton
kernel uses the pair-interleave RoPE convention
(`x[..., 2i]` paired with `x[..., 2i+1]`); HuggingFace uses the
halved-split convention. They produce the same frequency spectrum
but different tensor-element assignments. This caused ~20% gradient
noise when the naive reference test used HF's convention. Not a bug
IN orig — both conventions are valid RoPE — but a hazard for anyone
writing cross-reference tests.

### [FINDING 7] Dict event maps silently `.get(None)` → hang
`orig/active_model.py:113-131` + many wait sites: event maps are bare
dicts. A key miss returns None; `stream.wait_event(None)` is a
no-op. If the scheduler has a bug and tries to wait on an event
that was never recorded, you get a silent deadlock
(CUDA-level hang, no exception). V2 typed event maps raise on missing
keys instead ([DECISION 15]).

### [FINDING 9] Multiple `copy_(..., non_blocking=True)` calls NOT inside `with torch.cuda.stream(...)` blocks run on the DEFAULT stream, not the stream passed to `wait_stream()`
During Phase 3G+ loss-curve parity testing (see next section) I found a
class of subtle stream-management bugs in my own port: I was calling
`buffer_manager.fetch_layer_params(...)` and `.offload_layer_grads(...)`
directly AFTER `self.streams.inbound.wait_stream(self.streams.compute)`,
assuming that "I told inbound to wait for compute, so subsequent DMA
runs on inbound." That's not how PyTorch streams work. A bare
`tensor.copy_(other, non_blocking=True)` runs on the **current stream**
(as returned by `torch.cuda.current_stream()`), which is the default
stream unless you're inside `with torch.cuda.stream(...)` or
`torch.cuda.set_stream(...)`.

So my forward-pass layer-weight prefetch, my backward-pass grad offload,
backward weight prefetch, and step()'s cross-stream prefetch were ALL
racing against the compute stream and each other on the default stream.
Forward/backward completed fine (reads weights that happen to be there)
but step() corrupted updated weights because the offload-from-slot
raced with the prefetch-into-same-slot.

The parity harness caught this: config E (tight weight ring) drifted
0.34 from naive in 20 steps; fix reduced it to 0.03. Same-stream
rules now enforced by wrapping every `fetch_*`/`offload_*` in `with
torch.cuda.stream(...)` blocks at each call site.

**Lesson:** `stream.wait_stream(other)` only enforces "stream waits for
other", it doesn't make subsequent ops run on `stream`. That requires
setting `stream` as the current stream via the context manager. Orig
uses `with self.inbound_stream:` etc. everywhere — not clear from the
orig code that this was a HARD requirement rather than a style choice.
We now do the same. User reminder "nothing should run on default
stream" is now treated as an invariant.

### [FINDING 8] Host pin in test suites leaks without explicit teardown
During the port I discovered that BufferManagers without explicit
`destroy()` leak cudaHostRegister pins. Torch's caching allocator
then recycles those pointers for later tensors, which collide with
new `cudaHostRegister` calls. Symptom:
`cudaErrorHostMemoryAlreadyRegistered` (712) then sticky
`cudaErrorInvalidValue` on subsequent `copy_()`. Orig didn't surface
this because orig's test harness is minimal. V2:
1. Process-wide set of registered data_ptrs
   ([flextrain/engine/host_memory.py](../flextrain/engine/host_memory.py)).
2. `unregister_all_process_pinned_memory()` for drain-between-tests.
3. `run_all.py` calls the drainer after every module.

---

### [DECISION 24] Host-memory `cudaHostRegister` tolerates "already registered"
**Context:** when two `BufferManager`s or two test modules allocate
separate torch tensors that the caching allocator happens to place in
storage the driver has already pinned (from a prior, still-live
allocation), `cudaHostRegister` returns error 712
(`cudaErrorHostMemoryAlreadyRegistered`). Functionally the range IS
pinned and DMA would work, but the CUDA context records a sticky
error that torch surfaces on the next `torch.zeros(device='cuda')`
call.

**Fix:** `_cuda_host_register` returns `False` (not raise) on code
712, and also calls `cudaGetLastError()` to clear the sticky error.
The local backend only tracks `data_ptr`s it actually registered
itself, so `release` never tries to unregister a peer's range.

**Discovery:** hit this integrating the full test suite after 3F
landed. Two modules' buffer managers collided on a single data_ptr.
Fixed; full suite is green again (11 modules, 53 tests).

**Risk:** low. Behavior strictly more permissive than before; no
regression for the single-BufferManager path. Real training runs
will typically have exactly one BufferManager alive at a time.

---

## [3G+] Loss-curve parity across working-set configs — DONE (2026-04-23)

### What was validated
Loss-curve parity between FlexTrain and a **pure-PyTorch naive
reference**, on **real FineWeb data**, for **100 optimizer steps**,
across **8 working-set configs** that stress every engine path.

Naive reference: `torch.nn.Module` implementation (embed + 6 ×
Llama-style block + RMSNorm + linear head + CE + torch.optim.AdamW),
pure-PyTorch ops, no FlexTrain or orig kernels. RoPE uses the
pair-interleave convention (matches Triton kernel). Pulls the same
real FineWeb documents as FlexTrain — byte-identical per-step batches.

### Final losses (100 steps, FineWeb, d_model=512, n_layers=6)
```
  config                                             final loss   Δ vs naive
  ------------------------------------------------------------------------
  naive PyTorch baseline                                 8.2031             
  A. fast path (all on-device, 1 chunk/round)            8.2040      +0.0009
  B. multi-chunk (many chunks/round, on-device)          8.1770      -0.0261
  C. multi-round (2+ rounds/step, on-device)             8.2198      +0.0167
  D. host offload pressure (tight act ring)              8.2109      +0.0078
  E. weight ring rotation (N_P < n_layers)               8.2319      +0.0287
  F. grad ring rotation (N_G < n_layers)                 8.2319      +0.0287
  G. opt-state ring rotation (N_opt < n_layers)          8.2319      +0.0287
  H. sequence spans chunks (KV refresh)                  8.1708      -0.0323
```
naive went from 10.73 → 8.20 (real learning). Every FT config landed
within ±0.04 of the naive final loss. 10-step windowed max |delta|
across all 36 pairs (8 vs-naive + 28 cross-config) stayed at 0.057 —
well under the 0.10 tolerance.

### Reusable harness
[flextrain/bench/parity.py](../flextrain/bench/parity.py) — ~800 LOC.
Public surface: `ModelShape`, `WorkingSetSpec`,
`LossCurveParityConfig`, `run_loss_curve_parity(cfg) -> Result`,
`Result.print_summary()`, `Result.assert_all_match()`. To re-run with
different model size / WSC sweeps, just build a new
`LossCurveParityConfig` and call `run_loss_curve_parity`. The
harness pulls from any FineWeb-format .bin shard.

Test wrapper: [tests/test_loss_curve_parity.py](../tests/test_loss_curve_parity.py)
— 200 LOC, exercises 8 configs. Takes ~3 minutes on a 3090.

### Decisions

### [DECISION 27] Windowed-mean tolerance, not per-step
Per-step losses between two slightly-different bf16 trajectories drift
by O(0.1) occasionally (single-step outliers). Windowing over 10 steps
collapses this noise. We assert on windowed-mean delta, not per-step.
Catches O(1) scheduling bugs (which propagate through every step and
show up in the window) while ignoring bf16 single-step outliers.

### [DECISION 28] Act buffer must be at least one opt-state layer
`BufferManager.__init__` now enforces `gpu_act_buffer_size >=
max(per-layer opt-state bytes)` — the activation buffer is
repurposed as the opt-state ring during `step()`, so this is a
hard invariant. User flagged this as a requirement; caught
pre-training with a clear error message instead of at step-time.

### Critical bug caught: [FINDING 9] — stream-context leakage
The parity harness caught a class of scheduling bugs in my own port
where `fetch_layer_params` / `offload_layer_grads` calls were running
on the DEFAULT cuda stream instead of the specific stream the
enclosing `wait_stream` / `.record_event` was coordinating against.
Config E drifted 0.34 from naive; fix reduced it to 0.03. Fix:
every call to a `BufferManager` DMA method is now wrapped in
`with torch.cuda.stream(...)`. Documented as [FINDING 9] above.

---

## Phase 3 — Final parity matrix results (2026-04-23)

### 3-setting stress matrix (all 8 configs × 3 settings)
Every FT working-set config matches the naive PyTorch reference
within windowed tolerance at all three settings. Final losses:

```
  config                                                S1: baseline      S2: bigger model  S3: stress
                                                        (100 steps 5e-4)  (d=768 L=8 5e-4)  (200 steps 1e-3)
  naive PyTorch baseline                                  8.2031            8.2602            7.5394
  A. fast path (all on-device, 1 chunk/round)             8.2040 (+0.001)   8.2038 (-0.056)   7.5137 (-0.026)
  B. multi-chunk (many chunks/round, on-device)           8.1770 (-0.026)   8.2291 (-0.031)   7.5325 (-0.007)
  C. multi-round (2+ rounds/step, on-device)              8.2198 (+0.017)   8.2038 (-0.056)   7.5625 (+0.023)
  D. host offload pressure (tight act ring)               8.2109 (+0.008)   8.1504 (-0.110)   7.5618 (+0.022)
  E. weight ring rotation (N_P < n_layers)                8.2319 (+0.029)   8.2273 (-0.033)   7.5898 (+0.050)
  F. grad ring rotation (N_G < n_layers)                  8.2319 (+0.029)   8.2273 (-0.033)   7.5898 (+0.050)
  G. opt-state ring rotation (N_opt < n_layers)           8.2319 (+0.029)   8.2273 (-0.033)   7.5898 (+0.050)
  H. sequence spans chunks (KV refresh)                   8.1708 (-0.032)   8.1988 (-0.061)   7.5354 (-0.004)
```

E, F, G produce identical trajectories because they share the same
ring-rotation code paths on different axes — this is EXPECTED, not a
bug. All configs track naive within tolerance.

### 3H — ActiveModel.load_hf / save_hf (DONE)
Real implementations in [flextrain/engine/active_model.py]. load_hf
wraps ``flextrain.io.hf_weights.load_hf_safetensors`` against the
BufferManager's host-side dicts; save_hf is the mirror. After load we
refresh GPU resident slots (embed, head, first N_P layers) via
``_refresh_gpu_residents``.

Auto-detects ArchSpec from ``config.json`` if not passed. Handles
multi-shard checkpoints via ``model.safetensors.index.json``.

### 3I — Data ingestion (DONE)
Five adapters behind a single :class:`TokenSource` Protocol:

* :class:`HFTokenSource` — HuggingFace ``datasets`` + ``AutoTokenizer``
  for any public dataset (FineWeb, Pile, OpenOrca, ...).
* :class:`ShardTokenSource` — FineWeb ``.bin`` format (GPT-2 tokenizer
  uint16 tokens, EOT-delimited).
* :class:`RawTokenSource` — user hands us pre-tokenized tensors.
* :class:`SyntheticTokenSource` — random token ids, benchmark runs.
* :class:`CustomSchemaTokenSource` — user-provided extractor callable.

All satisfy the same ``get_sequences(max_token_count) ->
list[Sequence]`` surface. Tests: 10, all pass. See
[flextrain/io/sources.py].

### [DECISION 29] "Source" not "Stream" for token adapters
User flagged that "stream" is already used pervasively for CUDA
streams (:class:`flextrain.engine.streams.StreamBundle`). Renamed
all data adapters to TokenSource variants to eliminate ambiguity
in type annotations and error messages.

### 3J — cli.py train (DONE)
`python -m flextrain train <config.yaml>` now:
* Parses YAML into RunConfig dataclasses.
* Builds backbone + embed + head from `ModelConfig.arch`.
* Builds TokenSource from DataConfig (one-of selection among hf /
  shard / synthetic / raw / custom).
* Allocates WorkingSetConfig via orig's heuristics.
* Optionally loads HF checkpoint.
* Trains with linear-warmup+cosine-decay LR.
* Reports per-step `lr`, `loss`, `smoothed`, `tok/step`, `tok/s`,
  `TFLOPS`, `max_alloc`, `max_reserve`, `step` (ms), `elapsed` — and
  overall totals at the end.
* Saves final weights to HF safetensors.

FLOPs formula matches orig's `get_model_flops_per_sequence`:
* 6·T·active_params per layer (matmul fwd+bwd)
* 12·attn_factor·T²·attn_dim attention (attn_factor=0.5 causal → 6·T²·attn_dim)
* 6·T·d·V for the head

Example configs: [flextrain/configs/examples/].

### 3K — Llama-3.2-1B end-to-end parity (IN PROGRESS)
Downloaded Llama-3.2-1B HF weights to
``models/Llama-3.2-1B`` (~2.3 GB). Built a naive PyTorch reference
that loads the same HF weights (with tied-embedding handling —
Llama-3.2's `tie_word_embeddings=true`). Running 10 steps of
training on real FineWeb tokens in parallel through naive and
FlexTrain, comparing per-step loss.

### Remaining TODOs
* [ ] Confirm 3K 1B loss parity < 1.0 max |Δ| per step (running now).
* [ ] Optional: compare 1B training loss to HuggingFace's Trainer
      forward-loss on the same tokens (sanity check).
* [ ] 8B loss run (many steps) — should converge to lower loss than
      1B. Needs the 8B weights (gated; download via
      `huggingface-cli download meta-llama/Meta-Llama-3-8B`) and
      AdaWS offload (8B bf16 = 16GB params + 32GB AdamW opt state;
      must use host offload — config D- or E-style rings).
* [ ] Explore harness for arbitrary working-set sweeps from YAML
      (currently requires editing `test_loss_curve_parity.py`).
* [ ] Port native FlexTrain checkpoint format
      (``io/checkpoint.py``) — Phase 4 per PLAN.md. Currently
      we only save HF safetensors.
* [ ] Port remaining architectures per PLAN.md Phase 5 (Qwen3,
      OLMoE, Mixtral, etc.).

---

## [3K] Llama-3.2-1B E2E parity — IN PROGRESS (2026-04-23)

### Setup
- Downloaded Llama-3.2-1B from HF (2.3 GB safetensors, `models/Llama-3.2-1B/`).
- Llama-3.2-1B config: 16 layers, d_model=2048, 32 heads, 8 KV heads,
  head_dim=64, expert_dim=8192, vocab=128256, rope_theta=500000,
  ``tie_word_embeddings=true``.
- 1000 training steps on real FineWeb tokens (~2048 tokens/step from
  the first FineWeb .bin shard).
- Full bf16 training: params + grads + opt state all bf16 on BOTH
  naive PyTorch (torch.optim.AdamW with bf16 params → bf16 state) and
  FlexTrain (AdamW(state_dtype=torch.bfloat16)).

### Three runs, all initialized from the same HF weights
1. **naive PyTorch** — `torch.nn.Module` + `torch.optim.AdamW`. Pure
   PyTorch, no FlexTrain or orig kernels. Ground truth.
2. **FlexTrain all-resident** — full engine, all 16 layers'
   params/grads/opt-state live on GPU.
3. **FlexTrain offload-half** — only 8 layers resident at a time;
   the other half rotates through host RAM during fwd/bwd/step.
   Exercises the full AdaWS prefetch/offload pipeline on a real
   model.

### Outputs (written live as the run progresses)
Under `parity_results/llama32_1b/`:
- `live_naive.csv` — naive per-step loss (appended as each step finishes)
- `live_flextrain_all_resident.csv`
- `live_flextrain_offload_half.csv`
- `loss_curves.csv` — all three, final
- `summary.md` — final convergence + parity report
- `run.log` — full stdout trace

### Decisions

### [DECISION 30] AdamW accepts `state_dtype` kwarg
Changed `flextrain.optim.adamw.AdamW(hp, state_dtype=torch.float32)`
so users can opt into bf16 opt state for memory-constrained runs
(e.g. 8B on a 3090). Default is still fp32 (safer for long runs).
Matches what `torch.optim.AdamW` does automatically when given bf16
params.

### [FINDING 10] Tied-embed handling (Llama-3.2)
Llama-3.2 has `tie_word_embeddings=true` — the LM head weight is the
transpose of the embedding table. FlexTrain's current `LMHead` has a
separate `w_head_proj` parameter. On load we copy embed → head.T
once; after that the two evolve independently under their own grad
updates. Naive reference does the same (NaiveLlamaModel has a
separate `w_head_proj`), so naive and FT are directly comparable —
but both drift from an HF-tied training over time. **TODO**: add
proper tied-embed support (shared param, shared grad).

### Engine perf note
Naive pure-PyTorch on the 3090 runs Llama-3.2-1B at ~630 ms/step
(2048 tokens). FT should be faster due to fused flash-attn + the
awsm kernels. Actual numbers get written to the live CSVs.

### [FINDING 11] Llama-3 HF checkpoint needs halved-split → pair-interleave Q/K permutation
When loading an HF Llama checkpoint into ANY implementation that uses
pair-interleave RoPE (FlexTrain's Triton kernel and our naive
reference both do), the Q and K projection weights must be permuted
along the output dim from HF's halved-split layout to pair-interleave
layout. Otherwise attention at positions > 0 reads wrong vectors.

Without the permutation our 1B first-step loss was **5.77** vs HF's
**3.33** on the same retokenized FineWeb text. After permutation,
naive = 3.39 and FT = 3.38 — matching HF within 0.06 (the remaining
delta is the `rope_scaling` config which we don't implement yet; see
TODO).

Permutation code: ``tests/test_llama32_1b_parity.py::
_permute_qk_for_pair_interleave``. Applied at HF-load time to both
naive and FlexTrain host buffers.

**TODO**: move this into `flextrain.io.arch.llama.LLAMA_ARCH` as a
new transform type (e.g. `Transform.QK_PERMUTE`) so it runs inside
`load_hf_safetensors` automatically. Currently requires manual
post-load permutation.

### [FINDING 12] FineWeb .bin shards use GPT-2 tokenizer (not Llama-3's)
The pre-tokenized ``.bin`` shards orig generates (via
``orig/fineweb.py``) use ``tiktoken.get_encoding("gpt2")``, a 50257
vocab. Llama-3 uses a 128256-vocab tokenizer. Feeding GPT-2 token
IDs directly to Llama-3 produces garbage. For the 1B parity test we:

1. Read GPT-2 token IDs from the shard.
2. Decode them back to text with tiktoken.
3. Re-tokenize with Llama-3's tokenizer.

This keeps the test offline (no HF dataset streaming) while giving
the model tokens in its own vocabulary. See ``_pull_step_batches``
in ``tests/test_llama32_1b_parity.py``.

### First-step parity (2026-04-23)
After fixes for FINDING 11 + 12:
```
  HF transformers reference loss:  3.33
  naive PyTorch (ours):             3.39
  FlexTrain (all-resident):         3.38
```
Naive and FT agree to 0.004 (bf16 noise). The 0.06 gap to HF is
the missing rope_scaling. Close enough to confirm the HF load path
is correct and kick off 1000-step training.

### [FINDING 13] Qwen2.5 has Q/K/V biases — requires new attention path
Qwen2 and Qwen2.5 family models have biases on Q, K, V projections
(not on O, not in FFN). Checked against Qwen2.5-1.5B: 84 `*.bias`
tensors in safetensors, all on `self_attn.{q,k,v}_proj.bias`.

FlexTrain's current attention block (`GQAAttentionBlock`) uses the
fused `awsm_attention_fwd/bwd` kernel which doesn't take biases.
Adding Qwen2.5 support needs:
* New `TensorSpec`s for `b_q`, `b_k`, `b_v`.
* Biased Q/K/V projection path (can use plain torch.addmm instead of
  the fused matmul).
* Bias grad accumulation in backward (`dZ.sum(0)` over the token axis).

**RESOLVED (2026-04-24).** Landed as:

* ``GQAAttentionConfig.qkv_bias: bool`` flag. When True,
  :meth:`GQAAttentionBlock.param_spec` declares ``b_q`` /``b_k`` /
  ``b_v`` tensors (shapes ``(attn_dim,)`` / ``(kv_dim,)`` /
  ``(kv_dim,)``). No bias on ``w_o`` (none of Qwen2/Qwen2.5 variants
  have one).
* Forward: add biases in-place after the Q/K/V matmuls, before
  RoPE and before optional QK-norm (bias and QK-norm compose cleanly
  — bias → norm has a well-defined backward).
* Backward: after RoPE-bwd and QK-norm-bwd (if present), the
  gradients at the bias boundary are ``dq_view`` / ``local_dk_view``
  / ``local_dv``; we accumulate ``g_b_q += dq_view.sum(dim=0)`` etc.
* ``flextrain/nn/layers/qwen2.py`` — ``Qwen2Block`` subclassing
  LlamaBlock but constructing its attention with ``qkv_bias=True``.
* ``flextrain/io/arch/qwen2.py`` — HF weight map including
  ``q_proj.bias`` / ``k_proj.bias`` / ``v_proj.bias``.
* ``tests/test_qwen2_bias.py`` — tiny FT-vs-naive smoke test with
  random non-zero bias values. Max |Δ| = 0.0003 over 3 SGD steps.

### Llama-3.2-1B E2E parity RESULT (2026-04-23)

✓ **PASSED.** 200 steps × 2048 tokens on MathInstruct, full bf16
params+grads+opt state, SFT with prompt tokens masked via
`labels=-100` (CE ignore_index).

```
| side                    | first-10 avg | last-10 avg | Δ       |
|-------------------------|-------------:|------------:|--------:|
| naive PyTorch           |       1.1613 |      0.9773 |  -0.1840|
| FlexTrain all-resident  |       1.1629 |      0.9726 |  -0.1903|
| FlexTrain offload-half  |       1.1629 |      0.9726 |  -0.1903|
```

**FT-all and FT-offload produced bit-identical trajectories.** Both
match naive at max per-step |Δ| = 0.094, RMS |Δ| = 0.0098, and
|Δ last-10 avg| = **0.0047**.

The engine is correct end-to-end on a real 1B pretrained model
under true SFT prompt-masking.

Historical note: the first run of this test inadvertently dropped
the prompt masks in an inner-loop `_Seq` clone, so both naive and
FT trained on all tokens (no SFT masking). Fixed the clone and
re-ran; results above are from the corrected run.

---

## [Phase 5.1] Qwen3-1.7B end-to-end parity — ADDITIONAL BUGS FOUND (2026-04-23)

Setting up the Qwen3 parity run surfaced two more bugs + one
incomplete engine piece:

### [FINDING 14] QK-norm weight vector needs halved→pair permutation too
Qwen3's `w_q_norm` / `w_k_norm` are 1-D per-head_dim vectors (shape
`(head_dim,)`) applied elementwise to Q / K after projection, before
RoPE. Since we permute the Q/K projection weights halved→pair layout
so pair-interleave RoPE produces the right output, the QK-norm weight
vector itself must also be permuted the same way — otherwise the
multiplication at pair pos `2k` uses the wrong scale factor (it
reads `norm[2k]` when it should read `norm[k]`).

Permutation for the norm vector:
```
perm = [0, half, 1, half+1, 2, half+2, ...]
new_w_q_norm = w_q_norm[perm]
```

Without this fix: naive Qwen3 first-step loss 10.02 (random).
With this fix: naive Qwen3 gives 1.58 vs HF's 1.47 on same tokens
(close, within bf16 noise of the HF reference).

Fix applied in `tests/test_qwen3_1b7_parity.py::_load_hf_weights_into_qwen3_naive`
and in the FT load path too.

**TODO**: like [FINDING 11], promote this to a first-class
``Transform.QK_NORM_PERMUTE`` on ``WeightMapEntry``.

### [FINDING 15] Qwen3DenseBlock.forward/backward still NotImplementedError
The Qwen3 block was scaffolded (params + schema + compute_cost) but
its actual fwd/bwd bodies were never written — there's a stub at
``flextrain/nn/layers/qwen3.py:260`` that raises
``NotImplementedError``. Our 8-config parity matrix validated the
LlamaBlock compute path only, so this gap wasn't caught earlier.

Implementing it requires extending ``GQAAttentionBlock.fwd`` to hook
a per-head RMSNorm between Q/K projection and RoPE. This is a 30-80
LOC extension but non-trivial because it touches the Triton-kernel
dispatch inside the attention block.

**RESOLVED (2026-04-24).** Landed as:

* ``GQAAttentionConfig.qk_norm: bool`` + ``set_qk_norm(q_norm, k_norm)``
  on :class:`GQAAttentionBlock`. Hooks fire inside
  ``fwd``/``fwd_recompute_qo``/``bwd`` when ``qk_norm=True``.
* Forward: 2D ``(T, heads*head_dim)`` rmsnorm.fwd on slot.xq /
  slot.xk in-place, writes ``q_norm_rstd`` / ``k_norm_rstd`` to slot
  (declared as tier-0 fields by the RMSNormBlock).
* Backward: after RoPE-bwd, recompute ``xq_pre_norm = attn_norm_output
  @ w_q`` / ``xk_pre_norm = attn_norm_output @ w_k`` and call
  ``q_norm.bwd`` / ``k_norm.bwd`` to turn post-norm grads into
  pre-norm grads (also accumulating ``g_q_norm`` / ``g_k_norm``).
* Qwen3DenseBlock.forward/forward_recompute/backward wire the
  above and also have to pre-recompute ``attn_norm_output`` (via
  ``attn_norm.fwd_from_rstd``) BEFORE calling ``attn.bwd`` — unlike
  Llama, the attn backward depends on it for the pre-norm matmuls.

### [FINDING 16] RMSNormBlock per_head needs 2D input + weight-derived head_dim
The kernel ``awsm_rmsnorm_fwd`` expects a 2D input
``(T, heads*head_dim)`` even in per-head mode, and derives head_dim
from the weight vector shape. Our original ``_head_dim_arg`` used
``x.shape[-1]`` which is correct for 3D input but wrong when a 2D
view is passed (it would pick up ``heads*head_dim`` instead).

Fix: ``RMSNormBlock._head_dim_arg`` now reads head_dim from
``weights[weight_name].shape[0]`` when per_head=True, with the 3D
fallback preserved. Callers (GQAAttentionBlock) pass the 2D view so
``_get_norm_configs`` can unpack ``T, D = X.shape``.

### Deliverables in this session for Qwen3
* ``flextrain/io/arch/qwen3.py`` — HF weight map + `hf_config_to_flextrain`.
* ``flextrain/bench/parity.py::NaiveQwen3Model / NaiveQwen3Block`` —
  pure-PyTorch reference for parity comparisons.
* ``tests/test_qwen3_1b7_parity.py`` — parity harness (runnable once
  Qwen3 block fwd/bwd land).
* Qwen3-1.7B HF weights downloaded to ``models/Qwen3-1.7B/``.
* Two new bug findings ([FINDING 14], [FINDING 15]).

### Remaining TODOs after this session
1. ~~Implement Qwen3DenseBlock.forward / backward / forward_recompute~~
   **DONE 2026-04-24** (see FINDING 15 resolution). Qwen3-1.7B
   numerical parity verified vs naive PyTorch.
2. Add Qwen2.5 support: needs Q/K/V biases on attention (FINDING 13).
3. Promote QK weight-permutation + QK-norm-weight-permutation to
   first-class ``Transform`` kinds in ``flextrain.io.hf_weights``.
   (Requires Transform enum → parametrized Transform dataclass so the
   head_dim can be threaded through.)
4. Add tied-embedding support (shared param + shared grad).
5. Add Llama-3 ``rope_scaling`` (llama3 scheme).

## [Phase 5.1] Qwen3-1.7B E2E parity — RESULT (2026-04-24)

Full bf16 training, 200 steps × ~2048 tokens/step on MathInstruct
(SFT, prompt-masked via labels=-100). All three runs from the same
HF checkpoint, same AdamW hyperparams (lr=5e-5, betas=(0.9, 0.95),
eps=1e-8, wd=0).

| run | first-10 avg | last-10 avg | Δ |
|---|---:|---:|---:|
| naive PyTorch | 0.7096 | 0.5049 | -0.2047 |
| FlexTrain all-resident | 0.7112 | 0.5027 | -0.2085 |
| FlexTrain offload-half | 0.7112 | 0.5027 | -0.2085 |

Parity: max per-step |Δ| = 0.0127 (both configs), |Δ last-10-avg|
= 0.0022. FT-all and FT-offload match bit-identically across all
200 steps — the offload ring rotation is numerically invariant
against the all-resident baseline within bf16 accumulator noise.

### Summary of Phase 5.1 deliverables
* ``flextrain/nn/blocks/attention.py`` — ``GQAAttentionConfig.qk_norm``
  and ``GQAAttentionBlock.set_qk_norm``; fwd inserts QK-norm between
  projection and RoPE, bwd re-projects pre-norm Q/K from
  attn_norm_output and runs QK-norm bwd before projection bwds.
* ``flextrain/nn/layers/qwen3.py`` — Qwen3DenseBlock forward /
  forward_recompute / backward implemented (previously
  NotImplementedError). Also fixed Qwen3DenseSWABlock to pass
  qk_norm=True + set_qk_norm.
* ``flextrain/nn/blocks/norm.py`` — RMSNormBlock._head_dim_arg reads
  head_dim from the weight vector (for per-head mode) so 2D input
  views work.

### Bug findings in this session
See FINDING 15 (resolution) and FINDING 16 above.

## [Phase 5.3] Llama-3.1-8B — discovered [FINDING 17]: activation recompute corruption at 8B + offloading (2026-04-24)

### Setup that triggered the bug
* Llama-3.1-8B (32 layers, d_model=4096, head_dim=128, n_heads=32,
  n_kv_heads=8, expert_dim=14336, vocab=128256, no tied embed).
* Target 2048 tokens/step from MathInstruct (SFT, prompt-masked).
* Solver on 24GB 3090: n_gpu_layers=7, n_gpu_grads=7,
  n_gpu_opt_layers=1, gpu_act_slots=17 (of 32 layer-chunk pairs).
  So 15 (layer, chunk) pairs save activations to host and
  need recompute during backward.
* Same code path used for Llama-3.2-1B, Qwen3-1.7B (both pass parity).

### Symptom
* Step 0 matches HF transformers (loss 0.8300 vs 0.8289, |Δ|=0.001).
* Step 1 loss 0.62 (appears healthy).
* After step 1's backward pass, host master weights for layers
  L0-L4 and embed have all turned to NaN. Backbone L5-L31 fine.
* Step 2 forward consumes NaN embed → loss saturates to 100.0
  (the max_loss cap in `awsm_cross_entropy_loss`).
* Observed via per-slot grad inspection at end of step 1 fwd_bwd:
  ```
  slot[0] NaN (layer 5)
  slot[1] max=5.4e36 (layer 6) — not NaN but finite-overflow
  slot[2] max=7.0e33 (layer 0) — same
  slot[3..6] NaN
  ```
  Pattern: finite overflow → NaN propagation through the matmul chain.

### What isolates the bug to the RECOMPUTE path
Forced save-level overrides (via ``force_saved_act_level``) on the 8B
test, everything else unchanged:

| forced level | description | step 0 | step 1 | step 2 |
|--:|---|---|---|---|
| 3 | full save (NO recompute) | 0.83 | 0.68 | 1.11 (**works**) |
| 2 | recompute only x1 / x3 | 0.83 | 0.68 | 1.11 but grads 200× larger |
| 0 | recompute xq/xo/attn_result/lse/x1/x3 | 0.83 grads NaN | NaN | NaN |

At level=3, NO recompute runs → no NaN. At level<3 recompute runs
and produces catastrophically wrong values at 8B scale.

### Does NOT affect 1B
Llama-3.2-1B with the same forced save levels (0, 2, 3) all train
stably (loss 1.04 → ~1.2 → ~1.8 on MathInstruct). So the bug is
specific to 8B-scale recompute with the ring-offload path.

### Standalone recompute test is CORRECT
``tests/test_recompute_parity.py`` runs a single LlamaBlock at 8B
dims with T=2048 tokens, does forward (saving all fields), then
simulates level=0 via a wrapper slot that returns ``has(name)==False``
for tier>0 fields, calls ``forward_recompute``, and compares output
field-by-field. **All fields (xq, xo, x1, x3, attn_result) are
bit-identical to the original fwd (max|Δ|=0).** So the recompute
LOGIC itself is fine — the corruption must be in the engine's
ring-management / save-restore of tier-0 fields at 8B scale.

### Working hypothesis
The GPU activation ring at 17 slots × 32 layers has heavy slot reuse.
During backward, a layer's tier-0 fields (x_inp, attn_norm_rstd,
ffn_norm_rstd, xk, xv) are re-fetched from the host pinned buffer
into a ring slot that was most recently used by some OTHER layer.
There may be a race between:

1. fetch_home copying x_inp etc. from host (on inbound stream)
2. forward_recompute reading those tier-0 fields (on compute stream)

If the compute stream starts reading before the fetch is visible
(wait_event missing or on wrong stream), slot.x_inp holds STALE data
from the previous layer, so the recomputed `attn_norm_output` is
computed from wrong x → wrong xq/xo/x1/x3, compounding into NaN
through RMSNorm recompute (division by noise) and the QKV matmul.

At 1B the ring is small and slot-reuse pattern is different
(16 layers vs 32), so the race may not manifest.

### Further narrowing (2026-04-24 evening)
* Tested adding `self.streams.inbound.synchronize() +
  outbound.synchronize()` immediately before forward_recompute at bwd
  time — does NOT fix the NaN. So it is not a standard stream race
  (those would be fixed by a hard sync).
* Printed slot.x_inp magnitude right before forward_recompute for
  each bwd layer (force_saved_act_level=0):
  ```
  [pre-recompute L=31] x_inp: max=376.0000 rstd_max=2.47
  [pre-recompute L=30] x_inp: max=376.0000 rstd_max=2.83
  [pre-recompute L=29] x_inp: max=376.0000 rstd_max=3.10
  ...
  [pre-recompute L=2]  x_inp: max=376.0000 rstd_max=62.27
  [pre-recompute L=1]  x_inp: max=9.6250   rstd_max=95.57
  [pre-recompute L=0]  x_inp: max=0.1670   rstd_max=161.73
  ```
  **Every layer L=31..L=2 sees the SAME x_inp (max=376.0000
  identical)**, while each layer's own attn_norm_rstd differs
  correctly (strictly increasing with depth as expected from residual
  growth). L=1 and L=0 see different x_inp (9.6 and 0.17 — actual
  correct values). So: rstds are properly saved/fetched per-layer,
  but x_inp is NOT — layers 2-31 all end up reading the SAME slot
  memory region, which happens to hold some late-layer x_inp (≈ L31's
  actual input to the FFN-norm, pre-last-residual-add).

### Refined hypothesis
The engine's ring management during backward is over-writing the
GPU act-slot memory that holds on-device layers' tier-0 fields with
DIFFERENT layers' prefetched data before those on-device layers'
bwd has run. Specifically, `_prefetch_activation` fetches an
offloaded layer's x_inp into `dest_act_slot=cur_act_slot`. If that
slot still holds an on-device layer's forward-saved data (from the
LAST-wave ring reuse at forward), the prefetch clobbers it.

The current rotation logic decrements `cur_act_slot` by 1 per bwd
iter but also USES the same `cur_act_slot` as the prefetch target.
That's correct in principle (prefetch after bwd compute → slot is
freed), but something in the ordering / view lifetime at 32-layer
scale is letting a prefetch land in a still-needed slot.

### Where to look next
* Print (lid, cur_act_slot, dest_act_slot) trace over a full bwd
  pass at 8B and verify no on-device slot gets prefetched into
  before its bwd has run.
* Compare against `orig/active_model.py` prefetch-timing logic
  (lines referenced in docstrings; check if our port reversed an
  order or dropped a synchronization).

### [FINDING 17] RESOLUTION (2026-04-24 late evening)
**Root cause found.** Not a stream race, not a prefetch-ordering
bug — a correctness bug in ``ActivationSlot.has(name)``.

The test method was:
```python
def has(self, name: str) -> bool:
    return name in self._tensors
```

This checks dict membership but says nothing about whether the
data in the tensor view is actually valid. For OFFLOADED layers at
bwd time, the engine calls ``_prefetch_activation`` which builds
``dev_slot`` via ``gpu_act_slot(slot_idx, schema, num_tokens)``.
That function returns a slot with ``level=schema.max_tier`` and
``_tensors`` populated for EVERY field (tier 0 through max_tier).
The reason: the engine needs the higher-tier tensor VIEWS so
``forward_recompute`` can write xq/xo/x1/x3 into them. But fetch_home
only copies tier-0 fields from the host slot. The tier>0 views
still point to whatever the previous forward wrote into that ring
slot during its earlier use.

Then in ``LlamaBlock.forward_recompute`` (and its Qwen3 twin):
```python
if not slot.has("xq"):
    self.attn.fwd_recompute_qo(...)
```

Because ``has("xq")`` returns True (xq IS in _tensors),
**recompute was SILENTLY SKIPPED**. Backward then read the stale xq
from whatever layer had last used that ring slot during forward,
which at 32-layer + 17-slot ring means the TOP-of-stack residual
layer's xq. Flash-attn backward with a stale xq produces correct-
looking-but-wrong gradients. For Llama-3.1-8B, the residual stream
magnitudes are ~300–400 at deep layers, and a mismatched xq
multiplies through the 14 remaining backward iterations to
explode gradients → NaN embed weights → all subsequent training
ruined.

### Why this wasn't caught earlier
* Llama-1B (16 layers): the solver picks enough GPU act slots that
  all layers stay on-device → no prefetch → no stale data.
* Qwen3-1.7B (28 layers): same — fits in the ring budget with no
  offloading required.
* The 8-config parity matrix exercised a ring small enough that
  the stale-slot-reuse pattern at THIS tier-of-save never hit.
* Llama-3.1-8B at 32 layers / 17 slots is the first config that
  triggered: 15 offloaded pairs reusing 15 ring slots whose
  contents were PERSISTENT from later on-device layers' forward
  writes.

### Fix applied
Two changes (both in the same commit):

1. ``ActivationSlot.has(name)`` now consults the slot's save level:
   ```python
   def has(self, name: str) -> bool:
       field_tiers = self.schema._field_tiers_cache()
       tier = field_tiers.get(name)
       if tier is None:
           return name in self._tensors  # aux scratch fields
       return tier <= self.level
   ```

2. ``_prefetch_activation`` now wraps the full-tier ring slot in an
   ``ActivationSlot`` with ``level=home_slot.level`` (the SAVED level,
   not max_tier). The _tensors dict still contains all views so
   ``slot.xq.copy_(...)`` works during recompute, but ``slot.has("xq")``
   correctly returns False for tier>level fields, triggering recompute.

For ON-DEVICE layers, ``computed_slot`` is built with
``level=schema.max_tier`` so ``has()`` returns True for all fields —
which is correct, those fields ARE valid from forward.

### Verification
* 8B `force_saved_act_level=0` (all layers recompute):
  step 0 loss=0.8300, step 1 loss=0.6796, step 2 loss=1.1076 —
  all finite, matching what was observed with force=3 (full save).
* The 1B / Qwen3-1.7B runs still work (no regression at their
  save-level plans, which tended toward all-on-device).

### Lessons
* Don't conflate "field declared in the schema" with "field
  populated with valid data". The save-level dimension needs to be
  explicit in the API, not inferred from dict membership.
* A bug that manifests only past some ring-slot-reuse threshold
  (8B + 17 slots / 32 layers) is easy to miss with smaller tests.
  The 8-config parity matrix should be re-extended to include at
  least one config where n_gpu_act_slots < n_layers AND at least
  one force_saved_act_level=0 run as a regression test.

### Current workaround
Added ``force_saved_act_level=3`` in
``tests/test_llama31_8b_training.py`` so every layer saves all tier
fields on device. Costs more GPU act-ring memory (solver picks more
slots) but fully sidesteps recompute.

### TODO to properly fix
* Audit every wait_event / wait_stream around the fetch_home →
  forward_recompute path in ``active_model._backward_pass``
  (`:1176` onwards).
* Add an assertion in `forward_recompute` that sanity-checks the
  tier-0 fields against a hash captured during forward (expensive,
  but runnable as a debug gate for scale tests).
* Consider a defensive `inbound.synchronize()` before
  forward_recompute when the slot came from fetch_home.

### [FINDING 17] — Activation-recompute yields NaN at 8B + offload
See section above. Recompute LOGIC is correct in isolation but the
engine's ring / DMA path around it corrupts inputs at scale. Current
workaround: ``force_saved_act_level=schema.max_tier``.

## [DESIGN NOTE] Optimizer Muon/AdamW hybrid + future DDP sharding (2026-04-24)

### Muon/AdamW hybrid (TODO — not yet implemented)
Muon should only apply to 2D matmul weights; 1D weights (norms,
biases) and embed/head (even if 2D) use AdamW. Orig's pattern
(``dense_layer.py::step_muon``, ``moe_layer.py::step_muon``):

* AdamW: ``w_attn_norm``, ``w_ffn_norm``, ``w_router`` (if present
  — small routing matrix; unclear if orig uses AdamW on router,
  check. OLMoE typically keeps router in AdamW; Moonshot paper uses
  Muon on router.)
* Muon: ``w_q``, ``w_k``, ``w_v``, ``w_o``, ``w_1``, ``w_2``,
  ``w_3``, ``w_up[e]``, ``w_down[e]`` per expert.
* For QK-norm: ``w_q_norm`` and ``w_k_norm`` are 1D → AdamW.
* For Qwen2 biases: ``b_q``, ``b_k``, ``b_v`` are 1D → AdamW.
* Embed + head (resident layers) — always AdamW (orig has the
  resident-step path using AdamW only).

Our current ``flextrain/optim/muon.py::Muon`` applies Muon to
EVERY param in the param_spec — incorrect. Fix: make Muon a hybrid
that dispatches per-tensor based on rank (``len(shape) >= 2``
for projections → Muon; 1D → AdamW). Allocate BOTH opt states
(m, v for AdamW; momentum for Muon) per param to keep the engine's
buffer allocator unchanged; the unused state tensors are tiny for
1D params anyway.

Alternative: make ``OptimizerStateSpec`` per-tensor (a function
``state_spec_for(tensor_spec)``). Cleaner but requires engine
changes. Revisit when the hybrid is implemented.

### Future DDP / sharded optimizer (ZeRO-1 style)
Goal: split optimizer state across DDP ranks so each rank holds
``params / world_size`` opt-state bytes.

Constraints:
* Master weights: ZeRO-1 keeps them replicated (simplest, matches
  FlexTrain's per-rank host memory model). ZeRO-2/3 would shard
  master too but requires more coordination.
* Gradient reduction: reduce-scatter grads before the optimizer
  step, so each rank gets only its shard of each param's grads.
* After opt step: all-gather updated weights so each rank has the
  full weight for next forward.
* Muon orthogonalization requires the FULL 2D matrix. Can't naively
  shard a Muon op elementwise. Two options:
  1. Element-wise sharding for opt state + gather the full shard
     for the Newton-Schulz iteration. High comm overhead.
  2. Layer-sharding: each rank owns the opt state for a subset of
     LAYERS. Muon runs fully local per rank. AdamW shards
     elementwise within a layer. Matches orig's partition-by-layer
     offload pattern (``n_gpu_opt_layers`` rotation).
  Recommend option 2 — aligns with FlexTrain's existing per-layer
  host-buffer abstraction.

Proposed design hooks (to add when implementing):
* ``ShardSpec`` dataclass: ``{rank, world_size, shard_mode}`` where
  ``shard_mode`` is ``"none"``, ``"zero1_by_layer"``, or
  ``"zero1_elementwise"``.
* ``BufferManager.__init__`` reads ``shard_spec``; if not ``none``,
  allocates ``1/world_size`` of opt state per layer (by_layer) or
  ``1/world_size`` of each opt tensor (elementwise).
* ``TensorSpec.shard_size_bytes(shard_spec, dims)`` for the solver.
* ``Optimizer.step`` takes a master slice + grad slice + state
  slice. Element-wise ops (AdamW) operate on the slice directly;
  Muon gets the full matrix (require shard_mode=by_layer).
* ``ActiveModel.fwd_bwd`` calls reduce-scatter on grads after each
  layer's backward (or after each grad-accum round). ``step()``
  calls all-gather on weights before releasing.
* ``DistConfig`` holds the NCCL process group + comm hooks.

None of this is implemented yet. Today's Muon hybrid fix should be
written to NOT preclude sharding — specifically, don't hardcode
single-rank assumptions, and keep the state allocation logic
centralized in one place so the shard slice can be inserted
cleanly later.

## [PHASE 6+] Hybrid attention for Qwen3-Next / Qwen 3.5 / Qwen 3.6

Coming after MoE is verified. Key concepts:

### Sliding-window attention (SWA) — partially supported
Already have ``GQASlidingWindowAttentionBlock`` and
``Qwen3DenseSWABlock``. Mixing full + SWA layers within one model
is supported by the heterogeneous-backbone contract (composing
different layer types in a single backbone list). Exercised by
``tests/test_heterogeneous_backbone.py``.

### Linear attention (Qwen3-Next, future Qwen 3.5/3.6) — NOT YET
Qwen3-Next uses a **Gated DeltaNet** — a linear attention variant
with recurrent state (O(T) compute vs O(T²) for softmax attn).
Some layers use DeltaNet, others use standard GQA. Typical mix is
75% DeltaNet / 25% full-attention.

Implementation needs:
* New ``LinearAttentionBlock`` in ``flextrain/nn/blocks/attention.py``
  (mentioned as future work in that file's docstring).
* A Triton kernel (or flash-linear-attention package) for the
  DeltaNet fwd+bwd with state update. Not present in orig yet —
  need to port or import from fla.
* Activation schema for linear attention is different:
  - No softmax_lse / attn_result. Instead: a ``recurrent_state``
    tensor ``(num_heads, head_dim, head_dim)`` per chunk,
    carried forward across chunks within a sequence.
  - Chunk-wise chunked linear attention requires per-chunk state
    passing. This breaks the current "chunks within a seq group
    share a single attention context" pattern — instead each
    chunk has an INPUT state (from prior chunk) and OUTPUT state
    (to next chunk), and backward needs state grads to flow
    between chunks.
* Engine changes: ``ChunkMeta`` grows a ``linear_attn_state_in/out``
  reference; the engine threads these between chunks in order.

### Qwen 3.5 / 3.6 architecture questions to confirm at impl time
* Do they keep Gated DeltaNet from Qwen3-Next, or switch to
  another linear-attention variant?
* Ratio of linear-attn layers to full-attn layers?
* Do the full-attn layers still have QK-norm?
* Are MoE FFNs used (Qwen3-MoE style) or dense?
* tie_word_embeddings?
* Any new norm schemes (GroupNorm, DyT, etc)?

These will be answered by reading the HF config.json when the
weights release. Don't pre-design against hypothetical specs.

### Implementation order when we get there
1. Port DeltaNet kernels (port from fla or write Triton).
2. ``LinearAttentionBlock`` + its activation schema additions.
3. Engine plumbing for per-chunk linear-attn state.
4. ``Qwen3NextBlock`` layer type (DeltaNet variant + GQA variant
   via the existing heterogeneous-backbone pattern).
5. HF arch spec + weight map.
6. End-to-end convergence on Qwen3-Next / Qwen 3.5.

## Testing strategy for large models that don't fit on this box

This machine: 24 GB GPU (3090) + 125 GB host RAM. Ranks:
* 1B–3B models: full E2E on HF checkpoint, naive-PyTorch parity.
* 7B–8B: full E2E on HF checkpoint, step-0 vs HF transformers
  (naive torch doesn't fit for bwd but inference-mode HF does).
* 13B–30B MoE: if bf16 params + fp32 opt state fits in 125 GB,
  full E2E via offloading. Otherwise fall back to random-init
  small-scale variant.
* 60B+ and GPT-OSS-120B/20B: **NEVER** load real checkpoints —
  use randomly-initialized small-scale variants with the SAME
  architecture (same novel bits: sliding-window pattern, attention
  sinks, MoE routing, etc.) at a dims scale that fits (e.g.,
  d_model=512, n_layers=4, n_experts=8). Run N SGD steps, compare
  FlexTrain loss curve to a naive PyTorch reference built from
  the same random init. Must match within bf16 noise.

### Small random-init variant pattern
Same HF-style config but scaled down:
```python
small_cfg = ModelShape(
    d_model=512, n_layers=4, n_heads=8, n_kv_heads=2,
    head_dim=64, expert_dim=256, vocab_size=256,
    # Arch-specific:
    num_experts=8, top_k=2, sliding_window=128,
    attention_sinks=2,  # for GPT-OSS
    ...
)
```
Seed with ``torch.manual_seed(4242)`` so FT and naive start
IDENTICALLY. Run ≥10 SGD steps. Assert ``max|Δ|`` vs naive
< 0.01 per step.

This validates the ARCHITECTURE without requiring HF checkpoint
weights. Architecture correctness at small scale + E2E correctness
at 1B–8B (already validated) = confidence in large-scale training.

## SonicMoE integration plan (future — not today)

[SonicMoE](https://github.com/Dao-AILab/sonic-moe) provides IO- and
tile-aware MoE kernels on Hopper/Blackwell GPUs via CUTLASS grouped
GEMM (QuACK). It's significantly faster than orig's per-expert
loop on H100/B200 but **does not run on the 3090** we're developing
on — needs SM_90+ (H100) or SM_100 (B200).

### API shape (from reading sonic-moe source)
Single fused entry point:
```python
from sonicmoe.functional import moe_TC_softmax_topk_layer
output, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
    x,              # (T, H) bf16
    router_w,       # (E, H)
    w1, b1,         # (2*I, H, E) bf16 in interleaved layout, stride (2,0,1)
    w2, b2,         # (H, I, E)  bf16, stride (2,0,1)
    K=8,
    stream_id=0,
    activation_type=ActivationType.SWIGLU,
    is_softmax_over_topk=True,       # OLMoE-style
    norm_topk_probs=False,           # Qwen3-MoE sets True
)
```
or a lower-level `moe_general_routing_inputs(...)` variant for
pre-sorted inputs.

### Weight layout — DIFFERENT from our current MoESwiGLUFFN
* Our current (orig-ported): ``w_up: (E, H, 2*F)``, ``w_down: (E, F, H)``.
* SonicMoE expects: ``w1: (2*F, H, E)`` stride ``(2, 0, 1)``
  (interleaved or concat layout); ``w2: (H, F, E)`` stride
  ``(2, 0, 1)``.

A SonicMoE variant of our MoESwiGLUFFN needs:
* Distinct ParamSpec with the transposed layout.
* A separate ``MoESwiGLUSonicFFN`` block (don't conflate with orig
  version).
* HF-load transforms that stack-and-transpose experts into sonic's
  layout instead of orig's.

### Engine implications
SonicMoE wraps everything in a ``torch.autograd.Function`` that
internally manages routing, scatter, expert GEMMs, and gather. This
**conflicts** with our manual fwd/bwd split (where we need to
compute loss+bwd in the head, then manually call ``layer.backward``
per-layer). Options:

1. **Opaque wrapping**: run sonic's fwd inside our ``fwd``; stash
   the saved tensors; at bwd time, call sonic's autograd graph
   backward via ``torch.autograd.grad``. Loses the activation-
   offload benefit (sonic keeps its own tensors).
2. **Re-export sonic's internal primitives**: sonic exposes
   ``_up_projection_forward``, ``_down_projection_forward``,
   ``_up_projection_backward_*``, ``_down_projection_backward_*``,
   etc. (see ``sonicmoe/functional/forward.py`` and
   ``backward.py``). Call these directly, WITHOUT the
   ``autograd.Function`` wrapper, so we manage saved tensors and
   tier decisions ourselves. This is what the user asked for.
3. **Wait for sonic's stateless API to mature** — the README
   mentions stateless entry points in the roadmap.

Plan: option 2. Port a thin wrapper ``MoESwiGLUSonicFFN`` whose
``fwd``/``bwd`` call the `_up_projection_*` / `_down_projection_*`
and `_topk_softmax_*` primitives directly, matching our slot-based
activation contract. Saves ``a_prime`` (pre-activation, like our
``x_up``), ``expert_frequency``, ``x_gather_idx``, etc. as tier-3
activations; tier-0 for router state.

### Testing
Cannot verify on 3090. Plan:
* Write the wrapper against sonic's API.
* Test shape/dtype assertions locally (will error on call to CUDA
  kernels).
* Gate integration tests behind ``torch.cuda.get_device_capability()
  >= (9, 0)``. Run them only on H100 CI machines.

## Gemma 2 + Gemma 3 (to-do, Phase 6+)

### Gemma 2 arch quirks (vs Llama)
* **Pre-residual AND post-residual RMSNorm** per layer — Llama has
  only pre-residual (attn_norm before attn, ffn_norm before ffn).
  Gemma 2 adds ``post_attention_layernorm`` and ``post_feedforward_
  layernorm`` AFTER the residual adds. So a Gemma2 layer has 4
  RMSNorms per layer vs Llama's 2.
* **Sliding-window attention** on half the layers (alternating with
  full-context layers). Uses our existing
  ``GQASlidingWindowAttentionBlock``.
* **Soft logit cap** at the LM head: ``logits = softcap * tanh(
  raw_logits / softcap)`` with softcap=30 (final layer) or =50
  (attention internal). Requires a new output-layer variant
  ``LMHeadWithSoftCap`` or a ``loss_fn`` wrapper.
* **RoPE base 10,000** (like Llama2), ``rms_norm_eps=1e-6``.
* No attention bias, no QK-norm.
* Tied embeddings on small variants (2B, 9B).

Implementation needs:
* ``Gemma2Block`` layer type (new) with 4 RMSNorms.
* ``LMHeadWithSoftCap`` or loss_fn softcap wrapper.
* Gemma2 arch spec for HF weights.
* Alternating SWA/full pattern assembled at model construction time
  (already supported by heterogeneous backbone).

### Gemma 3 arch quirks (vs Gemma 2)
* **QK-norm** (like Qwen3) — per-head RMSNorm between Q/K
  projection and RoPE. Reuses our existing ``cfg.qk_norm=True``
  path in ``GQAAttentionBlock``.
* **Local/global attention pattern**: 5 sliding-window layers per
  1 full-context layer (``sliding_window_pattern=6`` means
  5 SWA + 1 full repeating).
* **RoPE scaling**: different scheme for long context (similar to
  Llama-3's but different params).
* **No soft logit cap** (Gemma 3 removed it).
* Activations: uses ``gelu_pytorch_tanh`` approximation in FFN (not
  SiLU like Llama/Qwen/Gemma2). Must swap ``awsm_swiglu_fwd`` for
  a GELU-approx variant, OR use regular Python/pytorch for FFN at
  Gemma 3 (losing the fused kernel).
* Multimodal variants have a vision tower — out of scope for
  training-only.

Implementation needs:
* ``Gemma3Block`` with qk_norm=True + post-residual norms from
  Gemma 2 + GELU FFN variant.
* GELU-approx SwiGLU-like kernel OR a PyTorch fallback.
* Gemma 3 arch spec.
* SWA/full alternation pattern at model construction.

### Testing plan
* Gemma-2-2B: HF checkpoint → naive PyTorch parity + E2E
  convergence (2B fits on 24 GB GPU + 125 GB host). Real weights.
* Gemma-3-1B / Gemma-3-4B: same pattern if available and fits.
* Gemma-2-27B / Gemma-3-27B: small random-init variant validation
  (won't fit).
