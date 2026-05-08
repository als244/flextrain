# Chunk contract — runtime values

The Layer protocol calls (`forward`, `forward_recompute`, `backward`,
`backward_dgrad`, `backward_wgrad`) all receive three runtime values
the engine constructs and owns: `ChunkMeta`, `LayerContext`,
`ActivationSlot`. This page is the formal reference for those.

For the static schema/protocol types (`ActivationField`,
`ActivationSchema`, the `Layer` Protocol itself), see
[`layer_contract.md`](layer_contract.md).

## `ChunkMeta` — what's in this chunk

Constructed once per chunk, lives across all layers' fwd then all
layers' bwd of that chunk. Carries sequence-packing info for
flash-attn varlen and per-chunk scratch shared between blocks of the
same layer:

| Field | Type | What it is |
|---|---|---|
| `total_q` | `int` | "New" tokens in this chunk |
| `total_k` | `int` | K-side tokens visible to attention (new + prior from KV cache) |
| `seq_lens_host` | `Sequence[int]` | Per-sequence new lengths (host-side) |
| `seq_positions` | `Tensor` (int32) | `(total_q, 1)` per-token positions within each sequence |
| `q_seq_offsets`, `k_seq_offsets` | `Tensor` (int32) | cu_seqlens-style offsets for flash-attn varlen |
| `q_seq_lens`, `k_seq_lens` | `Tensor` (int32) | per-sequence lengths, device-resident |
| `q_seq_offsets_i64` | `Tensor` (int64) | int64 mirror of `q_seq_offsets`, identity-stable for FLA's tensor cache |
| `fla_chunk_indices_64` | `Tensor` (int64) | precomputed FLA chunk-index table (chunk_size=64); avoids a D→H sync inside linear-attn fwd |
| `max_seqlen_q`, `max_seqlen_k` | `int` | longest seq in this chunk |
| `prior_seq_lens_host`, `prior_seq_offsets_host` | `Sequence[int]` | per-seq prior context lengths (host) |
| `has_more_chunks_host` | `Sequence[bool]` | per-packed-seq flag: True iff later chunks of this seq exist (used by attn bwd to decide dK/dV scratch routing) |
| `extra` | `Mapping[str, Any]` | per-chunk scratch shared across blocks (see below) |

Layers read `ChunkMeta` but do not modify it, with one exception:
the `extra` dict is mutable per-block scratch.

### `chunk.extra` — per-chunk per-block scratch

When a block needs state spanning its own `fwd` → `bwd` of the **same
chunk** (most commonly MoE routing decisions: `index_mapping`,
`expert_counts`, gathered expert inputs), use `chunk.extra`:

```python
# block fwd
chunk.extra["myblock_state"] = (index_mapping, expert_counts, ...)

# block bwd (same chunk, later in the step)
state = chunk.extra["myblock_state"]
```

`flextrain/nn/blocks/ffn_moe.py` does this for routed MoE. The engine
does NOT pre-allocate or register chunk-extra; it's just a dict the
block reads and writes.

## `LayerContext` — per-call runtime resources

Handed in fresh on every Layer protocol call. Carries the engine
allocators / streams the layer needs:

| Field | Type | Purpose |
|---|---|---|
| `scratch` | `(shape, dtype) -> Tensor` | ephemeral workspace allocator. Engine owns the pool and frees on context exit — layers MUST NOT call `torch.empty(...)` themselves |
| `kv_cache` | `KVContextWindow` (opaque) | the attention K/V ring window; attention blocks cast to the concrete type from `flextrain/engine/buffers.py` |
| `stream` | `torch.cuda.Stream` | primary compute stream for this call |
| `secondary_stream` | `torch.cuda.Stream \| None` | optional second stream — used by MoE to overlap shared-expert with routed-expert work |
| `total_tokens_per_step` | `int \| None` | total active tokens across all rounds/chunks in this gradient-accumulation step. MoE layers use this to scale the load-balance auxiliary loss. `None` for non-MoE workloads |

## `ActivationSlot` — saved + recomputable activations

One slot per `(chunk, layer)` pair. The engine constructs the slot,
fills it with pre-allocated tensors for every field at the chosen
save level, and hands it to the layer. Layers populate their
schema fields during `forward`, read them during `backward` (after
`forward_recompute` if the slot's level was below their max tier).

### Field access

Layers access fields by attribute:

```python
slot.x_inp           # the (T, d_model) saved residual input
slot.attn_result     # flash-attn output
slot.softmax_lse     # flash-attn LSE
```

`__getattr__` looks the name up in the slot's tensor dict. If the
field isn't present at this slot's level, raises `AttributeError`
listing the available fields.

### `slot.has(name)` — present at this level?

Returns True iff the field's `tier <= slot.level`. Use this in
`forward_recompute` to decide what to recompute:

```python
def forward_recompute(self, slot, chunk, weights, ctx):
    if not slot.has("xq"):
        # tier-2 not saved — recompute Q from x_inp
        ...
    if not slot.has("attn_result"):
        ...
```

A subtle point: the engine may hand out a slot whose underlying
tensor dict CONTAINS higher-tier views (e.g. it gave you a max-tier
GPU ring slot but only populated lower-tier fields from host).
`slot.has(name)` checks the declared tier, not dict membership, so
it correctly reports those higher-tier views as absent — your
`forward_recompute` will overwrite them with valid values via
`slot.set(name, tensor)`.

### `slot.set(name, tensor)` — write a recomputed field

`forward_recompute` calls `slot.set("xq", recomputed_xq)` after
producing a higher-tier value. `forward` also uses `set` when
writing into engine-provided output buffers.

### `slot.aux` — per-call mutable stash

A regular `dict[str, Tensor]` on the slot. Algorithmic blocks use it
to pass tensors between fwd/bwd helper methods within ONE layer
(e.g., an attention block stashing the local `dQ`/`dK`/`dV` from
flash-attn bwd so the RMSNorm-bwd call downstream can hand them off
for the weight-grad matmuls).

The engine clears `aux` between layers — it's strictly intra-layer
state.

## `chunk.extra` vs `slot.aux` — pick the right one

| | `chunk.extra` | `slot.aux` |
|---|---|---|
| **Lifetime** | one chunk, all layers | one (chunk, layer), one fwd→bwd pair |
| **Shared across blocks?** | yes — the layer's blocks all see the same dict | yes — but only blocks within ONE layer call |
| **Cleared by engine?** | between chunks | between layers |
| **Use for** | MoE routing state used in fwd then in bwd of the same chunk | Attention's `dQ/dK/dV` ferry, recomputed RMSNorm output reused in wgrad |

Rule of thumb: if the state needs to survive `forward_recompute`,
use `chunk.extra` (`slot.aux` is fresh per protocol call).

## Engine assumptions about layers — see also

The engine relies on layers honoring the schema/slot contract.
Things that LOOK like they'd work but break the engine — e.g.
allocating your own scratch with `torch.empty`, retaining state on
`self` across `forward → backward`, mutating `chunk.extra` after the
chunk's bwd has run — are listed in [`flow.md`](flow.md).
