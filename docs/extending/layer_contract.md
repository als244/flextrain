# Layer contract

The **layer** is the engine-visible unit. The engine treats a model
as `embed: InputLayer + backbone: list[Layer] + head: OutputLayer`,
and calls each layer's protocol methods at well-defined points in
the training step (see [`flow.md`](flow.md)).

This page is the formal reference for those Protocols and the
supporting types (`ActivationField` / `ActivationSchema` /
`ParamSpec` / `TensorSpec`). For runtime values handed into these
methods (`ChunkMeta`, `LayerContext`, `ActivationSlot`), see
[`chunk_contract.md`](chunk_contract.md). For style conventions,
tier choices, and the memory / compute-cost contracts the engine
relies on, read [`best_practices.md`](best_practices.md) alongside
this page.

## The `Layer` Protocol

```python
class Layer(Protocol):
    layer_id: int
    schema: ActivationSchema
    param_spec: ParamSpec

    def forward(
        self, x, chunk: ChunkMeta, weights, slot: ActivationSlot,
        ctx: LayerContext,
    ) -> torch.Tensor: ...

    def forward_recompute(
        self, slot, chunk, weights, ctx,
    ) -> None: ...

    def backward(
        self, dx, chunk, weights, grads, slot, ctx,
    ) -> torch.Tensor: ...

    # Optional split. Layers MAY also implement these two methods so
    # the engine can call dgrad alone (LoRA fast path -- skip the
    # per-projection Wgrad matmul on frozen base weights). When
    # implemented, ``backward`` is a delegating shim over them.
    def backward_dgrad(
        self, dx, chunk, weights, grads, slot, ctx, *,
        skip_target_names: frozenset[str] = frozenset(),
    ) -> tuple[torch.Tensor, BackwardIntermediates]: ...

    def backward_wgrad(
        self, intermediates, weights, grads, slot, ctx, *,
        skip_target_names: frozenset[str] = frozenset(),
    ) -> None: ...

    def compute_cost(self, chunk) -> ComputeCost: ...
```

`schema` and `param_spec` MUST be set in `__init__`; the engine reads
them before the first forward to size the activation buffer and the
parameter / gradient / optimizer-state buffers.

### `forward(x, chunk, weights, slot, ctx)`

Inputs:
* `x` — `(num_tokens, d_model)` residual stream input.
* `chunk` — sequence-packing info; see
  [`chunk_contract.md`](chunk_contract.md).
* `weights` — `dict[str, Tensor]` keyed by `param_spec` names
  (`w_q`, `w_attn_norm`, …). All on-device.
* `slot` — `ActivationSlot` with pre-allocated tensors for every
  field in your `schema`. Layers MUST `slot.x_inp.copy_(x)` if
  their schema declares an `x_inp` field, and write all required-
  tier-0 activations. They may write higher-tier activations if
  the slot's level allows (`slot.level >= field.tier`).
* `ctx` — per-call runtime context; see
  [`chunk_contract.md`](chunk_contract.md).

Returns the residual-added output `(num_tokens, d_model)`.

### `forward_recompute(slot, chunk, weights, ctx)`

Called during backward when `slot.level < schema.max_tier` —
fields with `tier > slot.level` were not saved at forward time and
must be reconstructed before `backward` is called.

Layers check `slot.has("xq")` (etc.) to decide what to recompute.
Save-tier semantics live in [`../working_set.md`](../working_set.md).

### `backward(dx, chunk, weights, grads, slot, ctx)`

Inputs: upstream gradient `dx` (same shape as forward's return),
chunk meta, weights, mutable `grads` dict (engine-zeroed once per
training step), the `slot` with all activations now valid (post-
`forward_recompute` if it was needed), and `ctx`.

Returns: `dx` for the preceding layer.
Side effect: accumulates weight grads into `grads[g_<param_name>]`
in-place. Naming convention: `g_<name without leading w_>`,
matching `state_key` / `flextrain.optim.base`.

### Optional: split backward into `backward_dgrad` / `backward_wgrad`

Layers MAY implement an optional split form of backward that lets
the engine call **dgrad alone** for layers whose Wgrad would be
wasted work (LoRA's frozen base weights). When the split is
implemented, the canonical pattern is for the monolithic
`backward(...)` to be a delegating shim:

```python
def backward(self, dx, chunk, weights, grads, slot, ctx):
    upstream_dx, inter = self.backward_dgrad(dx, chunk, weights, grads, slot, ctx)
    self.backward_wgrad(inter, weights, grads, slot, ctx)
    return upstream_dx
```

This guarantees zero-behavior-change for any caller that doesn't
opt into the split — engine, parity benches, tests all keep going
through `backward(...)` exactly as before. See
`flextrain/nn/layers/llama.py` for a working example.

`backward_dgrad` returns `(upstream_dx, BackwardIntermediates)` and
ALSO writes inline Wgrads into `grads` for projections that don't
need a recomputed-RMSNorm operand (e.g. `g_o`, `g_2`, attention
biases, RMSNorm gains). It takes `grads` precisely so it can do
this. The 2-D matmul Wgrads that DO need the recomputed RMSNorm
output (`g_q`, `g_k`, `g_v`, `g_1`, `g_3` / `g_up`) are deferred
until `backward_wgrad`.

For projection names listed in `skip_target_names`, the inline
addmm is skipped and the operand pair is stashed for the LoRA
wrapper:

```python
@dataclass
class BackwardIntermediates:
    proj_inputs_and_grads: dict[str, tuple[torch.Tensor, torch.Tensor]]
    aux: dict[str, Any]
```

* `proj_inputs_and_grads[name] = (X, dY)` for each LoRA-targetable
  projection — same `(X, dY)` the layer would have used for its
  `dW = X^T @ dY` accumulation. Names match `TensorSpec.name`
  (`"w_q"`, `"w_o"`, `"w_1"`, `"w_up"`, …). For MoE projections
  the tensors are 3-D `(num_experts, T_e, dim)`.
* `aux` — opaque to LoRA, used by the layer to ferry layer-internal
  state (e.g. recomputed RMSNorm outputs) from dgrad to wgrad.

`backward_wgrad(intermediates, weights, grads, slot, ctx, *, skip_target_names=frozenset())`
accumulates the deferred `dL/dW` into `grads`, **except** for
projections whose name is in `skip_target_names`. The engine
passes a non-empty skip set when the layer is wrapped by
`LoRAWrapperLayer`; LoRA then takes the same `(X, dY)` from the
intermediates and runs its rank-r matmuls directly into the LoRA
`g_a, g_b` accumulators, never materializing `dW` for the frozen
base.

When NOT to bother: small layers (norms, linear-attention internal
state) where Wgrad is already cheap and not LoRA-targeted. Just
keep the monolithic `backward(...)`.

The migration is staged across phases (see
[`../internal/lora_fast_backward.md`](../internal/lora_fast_backward.md)).

### `compute_cost(chunk)`

Returns `ComputeCost(total_fwd_flops, avoided_recompute_flops)`
where `avoided_recompute_flops` is a tuple of length
`schema.max_tier + 1`: element `L` is the FLOPs the engine
**doesn't** have to redo in backward if the chunk was saved at
tier `L`. Must be monotone non-decreasing; the last entry can't
exceed `total_fwd_flops`.

Each block reports its own `ComputeCost`; the layer sums them via
`ComputeCost.sum(parts, max_tier=...)`. The save-level DP solver
consumes these numbers — see
[`../working_set.md`](../working_set.md#how-the-dp-solver-picks-save-levels)
for what the solver minimizes and how your numbers shape its
decisions, and
[`best_practices.md`](best_practices.md#compute-contract-dp-solver)
for the FLOP-counting playbook.

## `ActivationField` / `ActivationSchema`

A field is one piece of saved-or-recomputable state:

```python
ActivationField(
    name="xq",
    shape_fn=lambda n, d: (n, d["n_heads"], d["head_dim"]),
    dtype=torch.bfloat16,
    tier=2,        # higher = saved at higher save levels only
    token_axis=0,  # which dim is num_tokens (engine narrows this
                   # axis to chunk size at runtime). None = doesn't
                   # scale with T.
    offload=True,  # set False for device-only fields (e.g. MoE
                   # router metadata trivially recomputable)
    persist=True,  # set False for engine-owned scratch reused across
                   # chunks (no per-chunk home slot)
)
```

A schema is a tuple of fields plus a `max_tier`:

```python
ActivationSchema(
    fields=concat_fields([
        attn_norm.fields(),
        attn.fields(),
        ffn_norm.fields(),
        ffn.fields(),
    ]),
    max_tier=3,
)
```

### Save tier conventions (loose)

A field with `tier=N` is saved when the engine picks save level
`L >= N` for this (layer, chunk), and recomputed when `L < N`. Save
level 0 means save only tier-0; save level `max_tier` means save
everything. So tier number = "how much memory budget is needed to
keep this saved" — higher tier is dropped first under pressure.

* **Tier 0** — always saved (engine never drops these regardless of
  memory pressure). Tiny; recompute is impossible or wasteful.
  Examples: RMSNorm rstd, MoE expert counts, router weights, `xk` /
  `xv`, `x_inp`.
* **Tier 1** — saved when budget permits level ≥ 1. Mid-sized
  flash-attention state (`attn_result`, `softmax_lse`).
* **Tier 2** — saved when budget permits level ≥ 2. Large
  pre-projection tensors (`xq`, `xo`).
* **Tier 3** — saved when budget permits level = max. Largest fwd
  intermediates (`x1`, `x3`, `x_up`); recomputed via
  `forward_recompute` when memory is tight.

Tiers are declared per-FIELD by the BLOCK that owns the field.
Layers aggregate via `concat_fields([block.fields() for block in ...])`
and don't change tiers. See [`best_practices.md`](best_practices.md)
for the assembly pattern.

The save-level solver picks the lowest tier that fits the working-
set budget while balancing the recompute FLOPs you reported in
`compute_cost`. You declare tiers; you don't choose save level.

## `ParamSpec` / `TensorSpec`

```python
ParamSpec(tensors=(
    TensorSpec(
        name="w_q",
        shape_fn=lambda d: (d["d_model"], d["attn_dim"]),
        compute_dtype=torch.bfloat16,
        master_dtype=torch.bfloat16,    # default = compute_dtype
        grad_dtype=torch.bfloat16,      # default = compute_dtype
        opt_state_dtype=torch.float32,  # default = bf16
        optimizer="muon",               # default = None (auto-infer)
    ),
    # ...
))
```

* `name` — must start with `w_` for parameters managed by the
  optimizer. Special-purpose layers may use other prefixes.
* `shape_fn(dims) -> tuple[int, ...]` — the shape, computed from a
  `dims: dict[str, int]` map. Standard dim names: `d_model`,
  `n_heads`, `n_kv_heads`, `head_dim`, `attn_dim`, `kv_dim`,
  `expert_dim`, `vocab_size`, `num_experts`, `top_k`. Add custom
  ones — your dims map controls everything.
* Per-role dtypes (compute / master / grad / opt-state) are honored
  per-tensor by the engine. See [`../dtypes.md`](../dtypes.md).
* `optimizer` — hint for hybrid optimizers like `HybridMuonAdamW`.
  `None` triggers auto-classification (2-D / 3-D → Muon, others →
  AdamW).

## `InputLayer` (embedding)

```python
class InputLayer(Protocol):
    schema: ActivationSchema
    param_spec: ParamSpec

    def forward(
        self, token_ids, chunk, weights, ctx,
    ) -> torch.Tensor: ...

    def backward(
        self, dx, token_ids, chunk, weights, grads, ctx,
    ) -> None: ...

    def compute_cost(self, chunk) -> ComputeCost: ...
```

No `slot` — the input "activation" is the `token_ids` tensor,
already owned by `ChunkMeta`. The schema is still present for
symmetry but typically has `max_tier=0` and no fields.

## `OutputLayer` (LM head)

Single fused method:

```python
def forward_backward(
    self, x, token_ctx, chunk, weights, grads, ctx,
    *, loss_scale: float = 1.0, loss_fn: LossFn | None = None,
) -> tuple[torch.Tensor, LossStats]:
    ...
```

Fuses head projection + loss + backward in one step, micro-chunked
along the token axis so no full `(num_tokens, vocab_size)` logits
tensor is ever materialized (peak logits VRAM ≤
`head_chunk_size * vocab_size`).

The loss is pluggable via `loss_fn` (see `flextrain.nn.loss`): the
head calls `loss_fn` with each `(T', V)` logits slice and gets back
`dZ`. This keeps the head arch-specific but loss-agnostic — same
head drives SFT cross-entropy, RL (GRPO/PPO/DPO), distillation
(MSE), etc.

Returns `(dx, LossStats)`:

```python
@dataclass
class LossStats:
    per_token_loss: torch.Tensor       # (T,) fp32
    next_prediction: torch.Tensor      # (T,) int64 — argmax(softmax)
    next_prediction_prob: torch.Tensor # (T,) fp32
    token_count: int
```

`dx` shape `(num_tokens, d_model)` is what the last backbone
layer's `backward` consumes.
