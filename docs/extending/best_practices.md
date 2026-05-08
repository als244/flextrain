# Best practices

Reference for writing blocks / layers / archs that look like the rest
of FlexTrain. Read alongside the contract pages — those define
**what's required**; this page covers **how to do it consistently and
correctly**.

Three concerns this doc covers:

1. **Style** — naming and class shape, so a reader of your block
   can navigate it the same way they navigate Llama / Qwen.
2. **Activation schemas** — what to declare as a field vs. scratch
   vs. aux, and how to choose tiers.
3. **User-responsibility contracts** — the working-set planner
   trusts your memory numbers and the DP solver trusts your compute
   cost numbers. Both are loadbearing; both are easy to get wrong.

## 1. Style conventions

### Naming

| Thing | Convention | Example |
|---|---|---|
| Parameter | `w_<role>` | `w_q`, `w_o`, `w_1`, `w_attn_norm`, `w_router` |
| Gradient | `g_<role>` (auto-derived from param name by stripping `w_`) | `g_q`, `g_attn_norm` |
| Optimizer state | engine-managed under the param name | n/a |
| Activation tensor field | `x<role>` for projected tensors | `xq`, `xk`, `xv`, `xo`, `x1`, `x3`, `x_up`, `x_inp` |
| RMSNorm rstd field | `<prefix>_rstd` (built from `RMSNormBlock(prefix=...)`) | `attn_norm_rstd`, `ffn_norm_rstd`, `q_norm_rstd` |
| Other named fields | `<role>` (descriptive) | `attn_result`, `softmax_lse`, `router_weights`, `chosen_experts` |

Rules of thumb:

* If the engine needs to auto-derive a name (param → grad), the
  prefix matters — start params with `w_`.
* `RMSNormBlock(prefix=...)` is the right way to introduce a new
  norm; the block constructs `w_{prefix}` and `{prefix}_rstd` for
  you. Don't re-implement RMSNorm.
* Activation field names are bare strings (no `w_` prefix). Engines
  expose them via `slot.<name>`.

### Class shape

**Block** = frozen `<Name>Config` dataclass + class with exactly five
methods, in this order:

```python
@dataclass(frozen=True)
class MyBlockConfig:
    # shape params
    d_model: int
    expert_dim: int
    # dtype overrides — match the convention from existing blocks
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None        # None → falls back to compute_dtype
    grad_dtype: torch.dtype | None = None          # None → falls back to compute_dtype


class MyBlock:
    def __init__(self, cfg: MyBlockConfig) -> None: ...
    def fields(self) -> tuple[ActivationField, ...]: ...
    def param_spec(self) -> ParamSpec: ...
    def fwd(self, ...) -> torch.Tensor: ...
    def bwd(self, ...) -> torch.Tensor: ...
    def compute_cost(self, chunk: ChunkMeta, max_tier: int) -> ComputeCost: ...
```

Don't allocate buffers in `__init__` — just store `cfg`. The engine
allocates everything. See `flextrain/nn/blocks/ffn_dense.py` for the
template.

**Layer** = frozen `<Family>BlockConfig` dataclass with a `dims()`
helper + `<Family>Block` class:

```python
@dataclass(frozen=True)
class MyArchBlockConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int
    rms_norm_eps: float = 1e-5
    rope_base: float = 10_000.0
    is_causal: bool = True
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None
    norm_grad_dtype: torch.dtype = torch.float32      # convention; cheap, possibly precision-helpful
    norm_master_dtype: torch.dtype = torch.float32

    def dims(self) -> dict[str, int]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attn_dim": self.n_heads * self.head_dim,
            "kv_dim": self.n_kv_heads * self.head_dim,
            "expert_dim": self.expert_dim,
        }


class MyArchBlock:
    def __init__(self, layer_id: int, cfg: MyArchBlockConfig):
        self.layer_id = layer_id
        self.cfg = cfg
        self._dims = cfg.dims()
        # Compose blocks (no allocation):
        self.attn_norm = RMSNormBlock(prefix="attn_norm", eps=cfg.rms_norm_eps, ...)
        self.attn = GQAAttentionBlock(...)
        self.ffn_norm = RMSNormBlock(prefix="ffn_norm", eps=cfg.rms_norm_eps, ...)
        self.ffn = ...
        # Build schema and merged param spec from blocks:
        x_inp = ActivationField("x_inp", lambda n, d: (n, d["d_model"]), cfg.compute_dtype, tier=0)
        self.schema = ActivationSchema(
            fields=concat_fields([(x_inp,), self.attn_norm.fields(), self.attn.fields(), ...]),
            max_tier=3,
        )
        self.param_spec = ParamSpec.merge([blk.param_spec() for blk in (self.attn_norm, self.attn, ...)])

    def forward(self, x, chunk, weights, slot, ctx) -> torch.Tensor: ...
    def forward_recompute(self, slot, chunk, weights, ctx) -> None: ...
    def backward(self, dx, chunk, weights, grads, slot, ctx) -> torch.Tensor:
        upstream_dx, inter = self.backward_dgrad(dx, chunk, weights, grads, slot, ctx)
        self.backward_wgrad(inter, weights, grads, slot, ctx)
        return upstream_dx
    def backward_dgrad(...) -> tuple[torch.Tensor, BackwardIntermediates]: ...
    def backward_wgrad(...) -> None: ...
    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        max_tier = self.schema.max_tier
        return ComputeCost.sum([blk.compute_cost(chunk, max_tier) for blk in ...], max_tier=max_tier)
```

`backward` is always a delegating shim over `backward_dgrad` /
`backward_wgrad`. Don't write a monolithic `backward` and skip the
split — LoRA's fast path needs it (see
[`layer_contract.md`](layer_contract.md#optional-split-backward-into-backward_dgrad--backward_wgrad)).

### Dtype defaults

| Role | Default | Rationale |
|---|---|---|
| `compute_dtype` | bf16 | What kernels see. |
| `master_dtype` | bf16 (= compute_dtype) | Override to fp32 for higher-precision updates at 2× param memory. |
| `grad_dtype` | bf16 | Halves grad memory vs fp32, fine on most ops. |
| `opt_state_dtype` (matmul params) | bf16 | Engine default. fp32 if cold-start training at scale (numerical updates need full precision). |
| `norm_master_dtype` / `norm_grad_dtype` | fp32 (convention) | RMSNorm weights are 1-D so the fp32 vs bf16 byte cost is negligible. The intuition is that fp32 may help precision on small-LR updates of small tensors, but this hasn't been measured rigorously — it's a low-cost convention, not a correctness requirement. |
| LoRA adapter dtypes | bf16 across the board (matches `from_pretrained` default) | HF PEFT's documented convention is fp32 for adapter master / grad / opt-state at scale; FT's default is bf16 for memory. Override per training recipe. |

The `BuildContext` defaults that `from_pretrained` uses are bf16 /
bf16 / bf16 / fp32-norm-grad — match those when picking your block
config defaults so the arch builder doesn't need a special case.

## 2. Activation schemas — what to declare and at what tier

The schema is your block's most consequential public surface: the
working-set planner reads it to size buffers, and the DP solver
uses tiers + your `compute_cost` to choose what to save where.

### Where tiers live: blocks declare, layers aggregate

Tiers are a per-FIELD property, set by the BLOCK that owns the field
in its `fields()` method. Layers do not change tiers — they just
concatenate their blocks' field tuples:

```python
# In the layer's __init__:
x_inp = ActivationField("x_inp", lambda n, d: (n, d["d_model"]),
                        cfg.compute_dtype, tier=0)
self.schema = ActivationSchema(
    fields=concat_fields([
        (x_inp,),                # layer-owned field (typically tier 0)
        self.attn_norm.fields(),
        self.attn.fields(),
        self.ffn_norm.fields(),
        self.ffn.fields(),
    ]),
    max_tier=3,                  # max across all fields above
)
self.param_spec = ParamSpec.merge([
    blk.param_spec() for blk in (self.attn_norm, self.attn,
                                 self.ffn_norm, self.ffn)
])
```

The layer typically owns one extra field of its own (`x_inp` for
the residual-stream input) and aggregates everything else from
blocks. `compute_cost` follows the same pattern at runtime:
`ComputeCost.sum([blk.compute_cost(chunk, max_tier) for blk in ...],
max_tier=max_tier)`.

The save level chosen by the DP solver is per-(layer, chunk), not
per-field. At runtime, every field with `tier <= slot.level` is
guaranteed valid; every field with `tier > slot.level` is unset and
your `forward_recompute` must produce it before bwd reads it.

### Save levels and tiers — what the numbers mean

A save level `L` for a (layer, chunk) means: save every field with
`tier <= L`; recompute the rest. So:

| `slot.level` | Saved at fwd | Recomputed in bwd | Memory | Compute |
|---|---|---|---|---|
| 0 | only tier-0 fields | every tier ≥ 1 field | smallest | most recompute |
| 1 | tier-0 + tier-1 | every tier ≥ 2 field | more | less recompute |
| 2 | tier 0/1/2 | every tier-3 field | even more | only tier-3 recomputed |
| `max_tier` | every field | nothing | largest | zero recompute |

So tier number = "how much memory budget is needed to keep this
saved." Higher tier = needs more budget. The DP solver picks `L`
per (layer, chunk) to fit the activation budget while minimizing
compute_time + transfer_time.

When picking the tier for a NEW field, ask: "if memory is the
tightest, which fields am I willing to drop and recompute?" Those
are the high-tier ones. Fields you absolutely want pinned in
memory regardless of pressure go at tier 0.

### What to declare as a field (vs. scratch / aux / chunk.extra)

| Use this | When | Example |
|---|---|---|
| `ActivationField` (lives in `slot`) | Tensor needed in BOTH `fwd` and `bwd` of the **same** block | `xq` (Q projection used in fwd attn AND bwd attn) |
| `ctx.scratch(shape, dtype)` | Tensor used only within one method, ephemeral | `attn_norm_output` between `RMSNormBlock.fwd` and `GQAAttentionBlock.fwd` (one call's worth) |
| `slot.aux[name]` | Tensor passed between blocks of the **same** layer call | recomputed RMSNorm output ferried from layer's `forward_recompute` to `attn.bwd` |
| `chunk.extra[name]` | Tensor spanning a block's fwd → its own bwd of the **same** chunk | MoE routing state (`index_mapping`, `expert_counts`) |

The wrong choice has consequences:

* Declaring scratch as a field → wastes activation-buffer memory
  every chunk, every layer, even though only one method needs it.
* Using `ctx.scratch` for state that needs to survive into bwd →
  the scratch pool reclaims it; bwd reads garbage.
* Storing on `self` to "be safe" → engine swaps the layer between
  GPU and host between fwd and bwd; the reference is stranded.

### Choosing the tier

Tier semantics: a save level `L` persists every field with
`tier <= L`. Higher tiers = fields that the planner can drop and
recompute when memory is tight.

Real conventions extracted from in-tree blocks:

| Tier | Use for | In-tree examples |
|---|---|---|
| **0** | Always saved. Tiny, expensive (or impossible) to recompute. The engine NEVER drops these regardless of memory pressure. | RMSNorm `*_rstd`, `xk` / `xv` (KV-cache fillers), MoE router state (`x_router`, `router_weights`, `chosen_experts`), `x_inp` (residual-stream input) |
| **1** | Saved when budget allows level ≥ 1. Mid-sized state from flash-attention's fwd that's moderately costly to redo. | `attn_result`, `softmax_lse` |
| **2** | Saved when budget allows level ≥ 2. Large pre-projection tensors; saving them avoids a matmul in bwd. | `xq`, `xo` |
| **3** | Saved when budget allows level ≥ 3 (= maximum). Largest fwd intermediates; only saved when memory is loose. When dropped, they're recomputed via the layer's `forward_recompute`. | `x1`, `x3` (SwiGLU gate/up outputs), `x_up` (MoE pre-SwiGLU expert inputs) |

Heuristics for a NEW field:

* "Tiny, must always be saved (or recompute is impossible / very
  costly relative to size)" → tier 0. Examples: rstds, small
  int32 routing tables.
* "Saves a matmul in bwd; medium-to-large; we'd drop it first
  under memory pressure" → tier 2 or 3 depending on size relative
  to the others in the layer (largest goes to tier 3).
* "The largest fwd intermediates — only saved when memory is
  abundant; otherwise recomputed by `forward_recompute`" → tier 3.

### `token_axis` — easy to get wrong

`token_axis` tells the engine which dim of `shape_fn`'s output
scales with `num_tokens`. The engine narrows that axis to the
chunk size at runtime. Three patterns from real blocks:

```python
# Standard: T is the leading dim.
ActivationField("xq", lambda n, d: (n, n_heads, head_dim), bf, tier=2, token_axis=0)

# softmax_lse is (n_heads, T) — T is on dim 1.
ActivationField("softmax_lse", lambda n, d: (n_heads, n), torch.float32,
                tier=1, token_axis=1)

# MoE x_up is (T*top_k, 2*expert_dim) — T scales the leading dim
# but with a multiplier; flag this with token_axis=None so the
# engine doesn't try to narrow naively.
ActivationField("x_up", lambda n, d: (n * top_k, 2 * expert_dim), bf,
                tier=3, token_axis=None)
```

If `token_axis` doesn't match where `n` actually appears in
`shape_fn`'s return, the engine narrows the wrong axis at runtime
— the block will read off-the-end memory or get the wrong T.

### `offload` and `persist` — when to override

Both default to `True` and you almost never need to touch them.

* `offload=False` — set when the field is device-local and never
  worth shipping to host (e.g. a tiny lookup table that the bwd
  recomputes anyway). MoE routing tables are reasonable
  candidates; in practice the in-tree blocks leave them at the
  default and rely on tier-0 to keep them resident.
* `persist=False` — set when the engine should treat the field as
  scratch shared across chunks (no per-chunk home slot). Used
  rarely; the default `True` is the safe choice.

When in doubt: leave both at `True`.

## 3. User-responsibility contracts

### Memory contract (working-set planner)

**You declare; the engine trusts.** The planner uses your
`ActivationField.shape_fn(num_tokens, dims) * dtype.itemsize` to size
per-(layer, chunk) activation slots, and your `ParamSpec` shapes /
dtypes to size parameter / master / grad / opt-state buffers per
layer. These are not estimates; they're the ground truth.

Concrete consequences when memory numbers are off:

| Mistake | What happens |
|---|---|
| `shape_fn` returns smaller tensor than `fwd` actually writes | Slot is too small. Best case: `RuntimeError` at the slot write. Worst case: silent overrun into a neighboring tensor → corrupt activations, NaNs that the loss curve eventually surfaces. |
| `shape_fn` returns larger than needed | Wasted activation-buffer bytes. The planner picks a smaller chunk size than necessary; throughput drops. |
| Wrong `dtype` (declared bf16, write fp32) | Slot bytes are sized for bf16; the fp32 write either errors or aliases two bf16 fields. |
| Wrong `token_axis` | At runtime the engine narrows the wrong axis to chunk size; downstream blocks see the wrong shape. |
| `ParamSpec` shape doesn't match what `fwd` reads | Either a shape error or — if the wrong shape happens to be valid — silently wrong math. |

How to verify before you trust the numbers:

* Build the layer once with realistic dims, allocate one slot, and
  compare `slot.<name>.shape` against your block's `fwd` output for
  the same field. Shapes must match exactly.
* `tests/test_arch_parity.py` exercises real HF weights through the
  schema; if shapes are wrong it surfaces immediately at load time.
* When the `from_pretrained` log prints "n_gpu_layers=K/N", compare
  K against your back-of-envelope (params + activations / 24 GB).
  If K is much smaller than expected, either your tensors are
  bigger than the planner thinks or your `compute_cost` is forcing
  it to a bad save tier.

### Compute contract (DP solver)

The save-level DP solver picks the tier per (layer, chunk) by
minimizing total step time = `compute_time` + `memory_transfer_time`,
subject to the activation memory budget. Your `compute_cost` feeds
the `compute_time` side of that minimization.

For the full DP-solver mechanics — what the solver minimizes, what
inputs it consumes (your `compute_cost` + your fields' byte sizes +
the hardware probe's TFLOPs / PCIe BW), and how to inspect or
override what it picked — see
[`../working_set.md`](../working_set.md#how-the-dp-solver-picks-save-levels).

```python
@dataclass(frozen=True)
class ComputeCost:
    total_fwd_flops: int                     # full forward pass
    avoided_recompute_flops: tuple[int, ...] # length max_tier+1; element L = FLOPs the engine
                                             # can SKIP in bwd if save tier L was chosen
```

`avoided_recompute_flops` MUST be monotone non-decreasing
(`__post_init__` raises if not), and the last entry can't exceed
`total_fwd_flops`. The intuition: saving more activations skips
more recompute, never less.

Consequences when compute numbers are off:

| Mistake | What happens |
|---|---|
| Under-report `total_fwd_flops` or `avoided_recompute_flops` | Solver thinks recompute is cheap → picks lower save tier → wall-clock balloons (recompute is actually expensive). |
| Over-report | Solver thinks recompute is costly → picks higher save tier → activation memory inflated → OOM, or planner falls back to a smaller chunk size. |
| `avoided_recompute_flops` not monotone | `__post_init__` raises at block construction. |
| Forgot to scale by `seq_len` per packed sequence | Solver sees per-chunk numbers that don't track real per-step compute; tier choice is wrong for some chunk sizes but not others. |

How to count FLOPs accurately:

* **Matmuls dominate; count them precisely.** A `(M, K) @ (K, N) →
  (M, N)` matmul is `2 * M * N * K` FLOPs (multiply-add as 2 ops).
* **Norms / element-wise are negligible.** `RMSNormBlock.compute_cost`
  literally returns `total_fwd_flops=0` — and that's correct. Skip
  them; the solver won't make different choices because of a 1%
  contribution.
* **Loop over `chunk.seq_lens_host`** (and `prior_seq_lens_host` for
  attention) so the cost reflects this chunk's actual packing, not
  a uniform-T approximation. Real attention example:
  `attn_prior = 4 * seq_len * prior_len * cfg.attn_dim`.
* **Attribute each matmul to the highest tier whose save would skip
  it.** Pattern from `GQAAttentionBlock.compute_cost`:

  ```python
  qo = 2 * (2 * seq_len * cfg.d_model * cfg.attn_dim)   # Q + O proj FLOPs
  total += qo
  if max_tier >= 2:
      for L in range(2, max_tier + 1):
          avoided[L] += qo
  ```

  Q and O are tier-2 fields; saving at level ≥ 2 skips re-projecting
  them in bwd, so add `qo` to `avoided[L]` for every `L >= 2`.

### Honor the engine's runtime assumptions

The contracts above are static. The runtime assumptions in
[`flow.md`](flow.md#engine-assumptions-dont-break-these) are
equally loadbearing: layers don't carry state on `self` between
fwd and bwd, `forward_recompute` produces output identical to
`forward`, layers don't allocate scratch via `torch.empty`. Re-read
that section before claiming a new block / layer is done.

## 4. Symptom → cause table

Use this when something's not working after writing a new block,
layer, or arch:

| Symptom | Most likely cause |
|---|---|
| `RuntimeError` writing into a slot field at fwd | `shape_fn` smaller than what `fwd` writes, or wrong `dtype` |
| Step-0 logit max\|Δ\| vs HF much larger than the bf16 noise floor (~0.5) and grows with depth | RoPE convention mismatch — Q/K halved → pair-interleave permutation didn't run (see [`tutorial_phi3.md`](tutorial_phi3.md) Step 5) |
| Step-0 logit max\|Δ\| acceptable but loss curves diverge after a few steps | A param's gradient isn't being accumulated, or `g_<name>` is misnamed (engine can't find the slot) |
| Loss-curve agreement is fine at save_level=max but breaks at lower levels | `forward_recompute` produces non-identical output to `forward` (save-tier non-invariance) |
| OOM at first chunk allocation | `shape_fn` returns larger than budget allowed, OR `compute_cost` over-reports recompute and the solver picks too-high a tier |
| Step time worse than expected, no OOM | `compute_cost` under-reports `avoided_recompute_flops` — solver picked too-low a tier; bwd is doing redundant work |
| `__post_init__` raises "avoided_recompute_flops must be monotone" | Tier accounting in `compute_cost` is wrong — saving more must avoid at least as much |
| `slot.<name>` raises `AttributeError` at bwd | Field is declared at a tier > slot.level and `forward_recompute` didn't `slot.set(name, ...)` for it |
| `forward_recompute` runs but bwd still fails | Recomputed tensor doesn't match the tier's `shape_fn` / `dtype` — check both |
| Numerical drift in only one MoE layer | `chunk.extra` write got reused across blocks (key collision), or the routing-state tensor was held by reference and the next block's fwd mutated it |

## See also

* [`flow.md`](flow.md) — engine runtime assumptions and step
  lifecycle. Re-read after writing a new block.
* [`block_contract.md`](block_contract.md) — formal Block convention
  + in-tree catalog with real signatures.
* [`layer_contract.md`](layer_contract.md) — Layer Protocol + the
  Activation / ParamSpec types.
* [`chunk_contract.md`](chunk_contract.md) — runtime values handed
  to every protocol call.
* [`tutorial.md`](tutorial.md) and [`tutorial_phi3.md`](tutorial_phi3.md)
  — worked examples that exercise these conventions end-to-end.
