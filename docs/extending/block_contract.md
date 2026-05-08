# Block contract

A **block** is an algorithmic unit — attention, FFN, MoE, RMSNorm,
RoPE, etc. Blocks are NOT part of the engine protocol; they're a
convention shared by everything under `flextrain/nn/blocks/`. Layers
compose blocks; the engine sees the layer.

This page is the formal contract. For naming, dtype, and tier
conventions to follow when writing a new block — and the
user-responsibility contracts your `compute_cost` and `fields()`
must honor — read [`best_practices.md`](best_practices.md) alongside.

## What every block must provide

Three required methods, each with the same signature across all
in-tree blocks:

```python
def fields(self) -> tuple[ActivationField, ...]: ...
def param_spec(self) -> ParamSpec: ...
def compute_cost(self, chunk: ChunkMeta, max_tier: int) -> ComputeCost: ...
```

* `fields()` — what activation fields this block contributes to the
  layer's `ActivationSchema`. The layer sums them with
  `concat_fields([attn.fields(), ffn.fields(), ...])`.
* `param_spec()` — what tensors this block owns. The layer sums them
  with `ParamSpec.merge([attn.param_spec(), ffn.param_spec(), ...])`.
* `compute_cost(chunk, max_tier)` — block-level FLOPs and per-tier
  avoided-recompute FLOPs. The layer sums them with
  `ComputeCost.sum(parts, max_tier=max_tier)`.

See [`layer_contract.md`](layer_contract.md) for `ActivationField`,
`ParamSpec`, and `ComputeCost` definitions.

## What `fwd` / `bwd` look like

The shape of `fwd` and `bwd` varies per block — each takes whatever
operands and produces whatever return that algorithm naturally
requires. The layer is responsible for slicing weights / grads /
slot and passing the right pieces in, and for chaining outputs to
the next block.

There is no shared base class. Treat the in-tree blocks as the
spec — pick the closest analog and follow its signature shape.

## Per-chunk scratch

If a block needs state that spans its `fwd` → `bwd` of the **same
chunk** (e.g. MoE routing decisions: `index_mapping`, `expert_counts`,
the gathered expert inputs), use the `chunk.extra` mapping on
`ChunkMeta`:

```python
# block fwd
chunk.extra["myblock_state"] = (index_mapping, expert_counts, ...)

# block bwd (same chunk, later in the step)
state = chunk.extra["myblock_state"]
```

`flextrain/nn/blocks/ffn_moe.py` does this for routed MoE. The engine
does NOT pre-allocate or register chunk-extra; it's just a dict the
block reads and writes. See [`chunk_contract.md`](chunk_contract.md)
for the `chunk.extra` vs `slot.aux` distinction.

## In-tree block catalog

Every block in `flextrain/nn/blocks/`. When picking blocks for a new
arch, browse this list first — most architectures compose entirely
from existing blocks, especially attention + FFN + norm.

### Attention

| Block | File | Used by |
|---|---|---|
| `GQAAttentionBlock` | `attention.py` | Llama, Qwen2, Qwen3-dense, OLMoE, Qwen3-MoE, Qwen3.5 (full attention layers) |
| `GQASlidingWindowAttentionBlock` | `attention.py` | Mistral, Gemma 2 (sliding-window layers) |
| `GQAAttentionGatedBlock` | `attention_gated.py` | Qwen3-Next (full attention layers — output gate variant) |

`GQAAttentionBlock.fwd` signature:

```python
def fwd(
    self,
    x_resid: torch.Tensor,            # (T, d_model) residual stream
    attn_norm_output: torch.Tensor,   # (T, d_model) RMSNorm output
    chunk: ChunkMeta,
    weights: Mapping[str, torch.Tensor],
    slot,                             # ActivationSlot
    ctx: LayerContext,
) -> torch.Tensor:
    """Returns (T, d_model) — input residual + attn_result @ W_O."""
```

`GQAAttentionBlock.bwd` signature:

```python
def bwd(
    self,
    dx_resid: torch.Tensor,
    chunk: ChunkMeta,
    weights: Mapping[str, torch.Tensor],
    grads: MutableMapping[str, torch.Tensor],
    slot,
    ctx: LayerContext,
    attn_norm_output: torch.Tensor,
    *,
    skip_grads: frozenset[str] = frozenset(),
    capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Returns dx w.r.t. attn_norm_output. Accumulates g_q/g_k/g_v/g_o
    into grads. skip_grads/capture_xy are the LoRA fast-path knobs."""
```

`GQASlidingWindowAttentionBlock` and `GQAAttentionGatedBlock` follow
the same shape, with extra config (window size, gate projection).

### Feed-forward (dense)

| Block | File | Used by |
|---|---|---|
| `SwiGLUFFN` | `ffn_dense.py` | Llama, Qwen2, Qwen3-dense, Mistral, Qwen3.5-dense |

`SwiGLUFFN.fwd` signature:

```python
def fwd(
    self,
    ffn_norm_output: torch.Tensor,           # (T, d_model)
    weights: Mapping[str, torch.Tensor],
    attn_output_with_residual: torch.Tensor, # (T, d_model) — added inline
    out_tensor: torch.Tensor,                # write target (T, d_model)
    slot,
    ctx: LayerContext,
) -> torch.Tensor:
    """Returns out_tensor = attn_output_with_residual + ffn(ffn_norm_output).
    Writes intermediates x1 / x3 / x_up into slot."""
```

### Feed-forward (mixture-of-experts)

| Block | File | Used by |
|---|---|---|
| `MoESwiGLUFFN` | `ffn_moe.py` | OLMoE, Qwen3-MoE, Qwen3.5-MoE |
| `MoESwiGLUSharedExpertFFN` | `ffn_moe_shared.py` | Qwen3.5-MoE (256 routed + 1 shared expert) |

MoE blocks use the same `fwd` shape as `SwiGLUFFN` plus the routing
machinery — they own `w_router` and dispatch tokens to per-expert
weights stored as 3-D stacked tensors. See `flextrain/nn/blocks/ffn_moe.py`
for the canonical pattern, including the `chunk.extra` use for
routing state.

### Norm

| Block | File | Used by |
|---|---|---|
| `RMSNormBlock` | `norm.py` | every layer in every arch |

`RMSNormBlock.fwd` signature:

```python
def fwd(
    self,
    x: torch.Tensor,                  # (T, d_model)  (or per-head)
    weights: Mapping[str, torch.Tensor],
    rstd_out: torch.Tensor,           # (T, 1) — fp32 reciprocal-stdev cache
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns y = rmsnorm(x) * w. Writes 1/sqrt(mean(x^2)+eps) into rstd_out."""
```

`RMSNormBlock` supports both whole-row norm (`per_head=False`) and
per-head norm (`per_head=True`) — the QK-norm variant in Qwen3 /
OLMoE / Gemma 3.

### Linear attention

| Block | File | Used by |
|---|---|---|
| `GatedDeltaNetBlock` | `linear_attn.py` | Qwen3-Next (linear-attn layers), Qwen3.5 (linear-attn layers) |

Linear-attention blocks have a substantially different `fwd`/`bwd`
shape than full attention because they carry a per-sequence
recurrent state across chunks. See [`flow.md`](flow.md) for how the
engine plumbs this state, and `flextrain/nn/blocks/linear_attn.py`
for the full implementation.

### RoPE helpers (functions, not classes)

| Helper | File | Used by |
|---|---|---|
| `apply_rope_fwd` / `apply_rope_bwd` | `rope.py` | every attention block |
| `apply_rope_partial_fwd` / `apply_rope_partial_bwd` | `rope.py` | Qwen3-Next (partial RoPE on D/2 only) |

These are free functions, not block classes — they don't declare
fields, params, or compute cost. Attention blocks call them inside
their `fwd` / `bwd`.

## When you actually need to write a new block

Most architectures compose entirely from the catalog above. Write a
new block only when:

* The architecture uses an attention or FFN variant that genuinely
  doesn't appear in the catalog (e.g. ALiBi-positioned attention,
  GeGLU instead of SwiGLU, attention-sink for GPT-OSS).
* You're prototyping a new training-time technique that needs a
  different fwd/bwd algorithm.

If you can express the new behavior as a different `Config` for an
existing block (e.g. window size, scaling factor, RoPE base), prefer
extending the config over forking the block.

When writing a new block, mirror the closest existing analog:

* New attention variant → start from `GQAAttentionBlock` in
  `attention.py`.
* New FFN variant → start from `SwiGLUFFN` in `ffn_dense.py`.
* New routing scheme → start from `MoESwiGLUFFN` in `ffn_moe.py`.

The [tutorial](tutorial.md) walks an end-to-end example.
