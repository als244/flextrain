# Flow — how it all fits together at runtime

This doc is the mental model. Read it before the contract refs if
you want to understand WHY each contract is shaped the way it is.
After this, jump to whichever contract page covers what you need
exact signatures for.

## The four levels (recap)

```
Block      algorithmic unit (attention, FFN, MoE, RMSNorm, RoPE, ...)
  └─→ Layer      one decoder layer (composes blocks; engine sees this)
        └─→ Arch+Builder    HF-loadable family (config + weight map + builder fn)
              └─→ Model     ActiveModel = embed + backbone + head + optimizer
```

Block is a convention; Layer is a Protocol the engine relies on.
Arch+Builder is the seam between an HF checkpoint and a list of
Layers. Model is the runtime object you call `fwd_bwd` / `step` on.

* Block details → [`block_contract.md`](block_contract.md)
* Layer details → [`layer_contract.md`](layer_contract.md)
* Arch / builder / `ActiveModel` API → [`model_contract.md`](model_contract.md)
* Per-call runtime values (`ChunkMeta`, `LayerContext`,
  `ActivationSlot`) → [`chunk_contract.md`](chunk_contract.md)

## One training step, narrated

`am.fwd_bwd(batch)` followed by `am.step()`. Pseudo-pseudocode of
what the engine does:

### 1. Build chunk metadata

The data source produces a packed batch of token sequences. The
engine splits them into fixed-size **chunks** (chunk size chosen by
the working-set planner; see [`../working_set.md`](../working_set.md))
and constructs one `ChunkMeta` per chunk. `ChunkMeta` carries
sequence offsets, positions, KV-cache continuation info, and an
empty `extra: dict` for blocks to use as scratch.

### 2. Forward pass

```
x = embed.forward(token_ids, chunk, weights, ctx)

for layer in backbone:                          # outer: layers in order
    for chunk in chunks:                        # inner: chunks in order
        slot = engine_allocate_slot(layer, chunk)   # at chosen save tier
        x = layer.forward(x, chunk, weights, slot, ctx)
        # layer wrote x_inp + tier-≤slot.level fields into slot
```

Two things to internalize:

* The traversal is **layer-major, chunk-minor**. All chunks of layer
  L finish before layer L+1 starts. This is how the working-set
  solver bounds activation memory: only one layer's worth of slots
  needs to coexist with one layer's worth of weights.
* Each `(layer, chunk)` pair gets its own `ActivationSlot`. The
  layer writes whatever fields its `schema` declares for `tier <=
  slot.level`. Higher-tier fields go unwritten — they'll be
  recomputed in backward.

### 3. Head: fused forward + loss + backward

```
dx, loss_stats = head.forward_backward(
    x, token_ctx, chunk, weights, grads, ctx,
    loss_fn=cross_entropy_or_whatever,
)
```

The head is fused for memory reasons — it never materializes the
full `(num_tokens, vocab_size)` logits tensor. It micro-chunks
along the token axis, calls `loss_fn` on each `(T', V)` slice, and
accumulates head gradients into `grads` while producing `dx` for
the backbone. See [`layer_contract.md`](layer_contract.md#outputlayer-lm-head).

### 4. Backward pass

```
for layer in reversed(backbone):                # outer: layers reverse
    for chunk in reversed(chunks):              # inner: chunks reverse
        slot = engine_fetch_slot(layer, chunk)
        if slot.level < layer.schema.max_tier:
            # tier-N>slot.level fields were not saved at fwd time
            layer.forward_recompute(slot, chunk, weights, ctx)
        dx = layer.backward(dx, chunk, weights, grads, slot, ctx)
        # grads now has dL/d(layer's params) accumulated in-place

embed.backward(dx, token_ids, chunk, weights, grads, ctx)
```

Two things to internalize:

* Backward order mirrors forward order in reverse, layer-major.
* `forward_recompute` runs only when needed (`slot.level <
  schema.max_tier`). The engine controls this; layers just have to
  produce the missing fields when called.

### 5. Optimizer step

```
am.step()
# advances LR schedule, runs optimizer per-tensor (AdamW / Muon /
# HybridMuonAdamW), handles offload of master weights / opt state,
# zeros grads.
```

## Object lifecycle

| Object | Created | Lives | Cleared / replaced |
|---|---|---|---|
| `ChunkMeta` | once per chunk, before fwd | from fwd through bwd of that chunk | next chunk gets a fresh one |
| `ActivationSlot` | once per (layer, chunk), before fwd | from fwd through bwd of that (layer, chunk) | reused on next step |
| `LayerContext` | fresh on every protocol call | one call only | next call gets a fresh one |
| `BackwardIntermediates` | start of `backward_dgrad` | passed to `backward_wgrad` | dropped after `backward` returns |
| `slot.aux` (dict) | fresh when slot is allocated | one layer's fwd → bwd | engine clears between layers |
| `chunk.extra` (dict) | empty when chunk is built | the whole chunk's fwd + bwd | engine clears between chunks |

## Engine assumptions (don't break these)

The engine relies on each of these. Violations don't always blow up
loudly — some produce silent numerical bugs.

### Traversal order

* Forward order is `for layer: for chunk`; backward is the exact
  reverse.
* The engine prefetches weights / activations for the next
  (layer, chunk) pair while the current one computes. If a layer
  reorders chunks or skips one, prefetch falls out of sync.

### Layer state

* Layers do NOT carry state across `forward → backward` except via:
  * fields they declared in `schema` (lives in `slot`),
  * `slot.aux` (intra-layer, cleared between layers),
  * `chunk.extra` (intra-chunk, cleared between chunks).
* Stashing data on `self` between fwd and bwd works in single-
  process toy training, but the engine swaps layers between GPU
  and host memory — `self`-stashed tensors get stranded on the
  wrong device, or get freed mid-step. Use the slot/aux/extra
  mechanisms.
* `self.layer_id` is fine — it's set at construction.

### Slot writes

* Layers MUST `slot.x_inp.copy_(x)` (or equivalent) if their schema
  declares `x_inp` — the engine assumes it's there. RMSNorm-bwd in
  the next layer's call reads it back as the pre-norm input.
* Layers MUST write every tier-0 field of their schema in `forward`
  unconditionally. Higher tiers are conditional on
  `slot.level >= field.tier`.
* `slot.set(name, tensor)` is the only way to install a recomputed
  tensor for a higher-tier field. Don't write `slot.<name> =
  tensor` — the slot's `__setattr__` is locked down by `__slots__`.

### Save-level invariance

* Engine guarantees identical loss across save tiers. Your layer
  must therefore produce identical fwd-output regardless of whether
  tier-N fields were saved at fwd time or recomputed in
  `forward_recompute`. Stochastic ops in fwd (dropout, sampled
  routing) break this — at minimum they need to be re-seeded
  identically in `forward_recompute`.
* The save level itself is chosen per (layer, chunk) by a DP
  solver that consumes your `compute_cost` numbers and the
  fields' byte sizes. See
  [`../working_set.md`](../working_set.md#how-the-dp-solver-picks-save-levels)
  for the solver's objective and how to inspect / override its
  choices.

### Scratch allocations

* Layers and blocks MUST NOT call `torch.empty(...)` for scratch.
  Use `ctx.scratch(shape, dtype)`. The engine owns a pool, sizes it
  from the schema, and frees on context exit. Bypassing it leaks.
* `chunk.extra[k] = tensor` is fine for routing-state-style scratch
  (allocated by the block, owned by the dict, freed when the chunk
  is reclaimed). Don't put GPU scratch buffers there if they could
  be reused via `ctx.scratch`.

### KV cache

* The KV cache lives on `ctx.kv_cache`. Attention blocks update it
  inside their `fwd` (extending the K/V window with this chunk's
  K/V projections). The engine does not snapshot or clone it.
* For sequences that span multiple chunks (`prior_seq_lens_host[i]
  > 0`), the K/V from earlier chunks lives in `kv_cache` and gets
  read alongside the current chunk's K/V during attention. The
  attention block's bwd uses `chunk.has_more_chunks_host[i]` to
  decide whether dK/dV for that seq position should overwrite or
  add into the cross-chunk gradient buffer.

### LoRA fast-path split

* Layers that implement `backward_dgrad` / `backward_wgrad` MUST
  also have `backward(...)` delegate through them (see
  [`layer_contract.md`](layer_contract.md#optional-split-backward-into-backward_dgrad--backward_wgrad)).
  Otherwise behavior diverges between full-FT (calls `backward`)
  and LoRA (calls dgrad + wgrad with skip set).
* The skip mechanism is per-projection-name. If a projection's name
  appears in `skip_target_names`, the inline addmm is suppressed
  and the `(X, dY)` operand pair is stashed in
  `intermediates.proj_inputs_and_grads[name]` for the LoRA wrapper.

## Reading order from here

Pick whichever matches what you need:

* Want to add a new arch end-to-end → [`tutorial.md`](tutorial.md).
* Need exact `forward` / `backward_dgrad` signatures →
  [`layer_contract.md`](layer_contract.md).
* Picking which existing blocks to compose →
  [`block_contract.md`](block_contract.md).
* Looking up `ChunkMeta` / `LayerContext` / `ActivationSlot` fields
  → [`chunk_contract.md`](chunk_contract.md).
* Wiring the arch into `from_pretrained` →
  [`model_contract.md`](model_contract.md).
