# Multi-chunk sequence handling — dense and hybrid attention

How FlexTrain handles sequences longer than `max_chunk_size`. Two
mechanisms — one for **dense (softmax) attention**, one for **linear
attention (Gated DeltaNet)** in hybrid backbones (Qwen3-Next,
Qwen3.5*, Qwen3.6*) — share infrastructure but differ in specifics.

## Why multi-chunk seqs need special handling

`prepare_training_chunks` produces consecutive single-sequence chunks
for any seq longer than `max_chunk_size`. Both attention variants are
**stateful across the seq's tokens**:

* Dense: token at chunk-position `t` attends to all earlier tokens,
  including those in earlier chunks. The K/V of those earlier tokens
  must be visible to chunk N's attention call.
* Linear: chunk N's tokens are processed via a recurrent state
  machine. The state at chunk-N-start is the state produced by
  chunk N-1's computation.

Without cross-chunk plumbing, chunk N would compute as if it were
chunk 0 of a fresh sequence — wrong for both.

---

## Shared infrastructure

A single per-layer global window mechanism, mirrored from the existing
KV-context implementation:

* **One stream**: `streams.inbound_fwd_context`. Carries refresh
  copies from saved slot fields into the layer-current window during
  bwd. Reused for both KV and lin-state refreshes since a given layer
  is either dense or linear (never both), so the refreshes are
  naturally sequential per-layer.

* **One dispatcher** (`_update_fwd_context`): inspects the next
  reverse iteration's target layer schema and dispatches to the
  appropriate refresh helper:
  * `_refresh_kv_window(src_lid, src_chunk_id)` for layers with `xk`
  * `_refresh_lin_state_window(src_lid, src_chunk_id)` for layers
    with `lin_final_state`

  Both helpers handle the device-resident vs host-resident source
  via the same `inbound_act_slot_ready` / `home_act_slot_available`
  event tables. Different layer types differ only in source-slot
  identification (see "Source-slot rules" below).

* **WAR protection**: `inbound_fwd_context.wait_stream(compute)`
  after each chunk's bwd, before refresh writes. Same pattern as
  today's KV.

* **Group-entry sync**: `compute.wait_stream(inbound_fwd_context)`
  at every seq_group iteration entry, ensures prior iteration's
  refresh is visible to compute.

---

## Dense attention

### Window

`KVContextWindow` in `flextrain/engine/buffers.py`: `(max_context_tokens,
n_kv_heads, head_dim)` bf16, four tensors (k, v, dk, dv). Allocated
once at engine init.

### Forward

Per chunk N: layer's `forward` writes K/V into the window at varlen
positions `[prior_seq_offsets + prior_seq_lens, ...)`. Slots save
`xk`/`xv` (tier 0) for bwd. **No engine hop during fwd.**

### Backward

Per chunk N's bwd:
* Window must hold K/V for L's chunks 0..N at the right positions.
* Refresh source: `slot[L, target_chunk_id].xk/xv` (the layer's OWN
  saved K/V at the target chunk).
* Refresh runs on `inbound_fwd_context` after the previous reverse
  iteration's bwd; `_update_fwd_context_dispatch` selects the KV
  branch.
* `dk`/`dv` accumulate into `kv_bwd_dk`/`kv_bwd_dv` (alias of
  `kv_fwd.dk`/`kv_fwd.dv`).

### Layer-entry init

Not needed: at the end of fwd the window already holds K/V for the
top layer's last seq_group (which is the first iteration of bwd).

### Memory

Per-layer global window: `max_context_tokens × n_kv_heads × head_dim
× 2 × 4`. Independent of chunk count. Per-chunk slot fields
`xk`/`xv` go through standard offload/prefetch.

---

## Linear attention

### Slot field

`lin_final_state: (HV, K, V) fp32`, tier 0, ALWAYS allocated for
linear-attn schemas. Single tensor per slot — no row indexing —
because `_pack_sequences` guarantees that any chunk that participates
in a multi-chunk seq is a dedicated single-seq chunk (per
`flextrain/engine/schedule.py:_emit_large`).

Population rule: chunk N's fwd writes `slot[L, N].lin_final_state`
iff chunk N has more chunks ahead in its seq. Otherwise the field is
allocated but unused.

### Windows

`LinAttnStateWindow` (new, in `flextrain/engine/buffers.py`):

* `fwd: (HV, K, V) fp32` — input `initial_state` for FLA fwd /
  FLA bwd. Filled by:
  * fwd: chunk N's `_fwd_fla` writes `final_state` into it after FLA
    returns (when chunk N has more chunks ahead).
  * bwd: `_refresh_lin_state_window` pulls `slot[L, src_chunk].lin_final_state`
    into it before chunk's bwd consumes it.

* `bwd: (HV, K, V) fp32` — accumulates `dh0` (FLA bwd output) so
  chunk N's bwd reads it as its `dht` argument.

Both are allocated once at engine init if any backbone layer has
`lin_final_state` in its schema.

### Forward

```
for layer L (forward order):
  zero(lin_state_window) at layer entry
  for chunk N (forward order):
    initial_state = lin_state_window.fwd if continuation else zero
    final_state = layer.fwd(chunk N, initial_state, output_final=…)
    if chunk N has more chunks:
      lin_state_window.fwd.copy_(final_state)
      slot[L, N].lin_final_state.copy_(final_state)
```

No engine refresh hop during fwd: chunks within a layer process in
order, and the window is incrementally updated.

### Backward

```
At _backward_pass entry (before layer loop):
  if backbone[-1].schema has lin_final_state:
    # Pre-bwd init for the top layer's first reverse iteration
    last_group = prepared.seq_groups[-1]
    first_rev_chunk = last_group[-1]
    if first_rev_chunk has prior chunk in same seq:
      _refresh_lin_state_window(top_layer, first_rev_chunk.id - 1)
    # On inbound_fwd_context stream; first iter's compute waits.

for layer L (reverse order):
  if L.schema has lin_final_state:
    zero(lin_state_window.bwd) at layer entry
  for seq_group reverse:
    compute.wait_stream(inbound_fwd_context)  # group entry
    for chunk_in_group reverse:
      slot[L, chunk.id] resident wait
      with compute:
        layer.forward_recompute(...)
        layer.backward(...)
          # bwd reads lin_state_window.fwd as initial_state
          # bwd reads lin_state_window.bwd as dht
          # bwd writes lin_state_window.bwd <- dh0 (compute stream, no event needed)
      inbound_fwd_context.wait_stream(compute)  # WAR
      _update_fwd_context_dispatch(target_iter):
        # See dispatcher logic below
```

### Source-slot rules (post-bwd dispatcher)

The dispatcher determines `(target_layer, target_chunk_id)` for the
NEXT reverse iteration:

```
if chunk_in_group_ind > 0:
    # Within-group: next iter is same layer, chunk_in_group - 1
    target_layer = self.backbone[layer_ind]
    target_chunk_id = group[chunk_in_group_ind - 1].id
elif seq_group_ind > 0:
    # Cross-group within layer: next iter is prior group's last chunk
    target_layer = self.backbone[layer_ind]
    target_chunk_id = prepared.seq_groups[seq_group_ind - 1][-1].id
elif layer_ind > 0:
    # Cross-layer: next iter is prior layer's last group's last chunk
    target_layer = self.backbone[layer_ind - 1]
    target_chunk_id = prepared.seq_groups[-1][-1].id
else:
    return  # End of bwd; nothing to prefetch.
```

Then dispatch to the right helper based on target_layer.schema:

```
schema = target_layer.schema
if schema.has_field("xk"):
    # KV: source is target_layer's OWN slot at target_chunk_id
    self._refresh_kv_window(target_layer.layer_id, target_chunk_id)
if schema.has_field("lin_final_state"):
    # Lin-state: source is target_layer's slot at target_chunk_id - 1
    target_packed_seq_info = self._lin_attn_plan.per_chunk[target_chunk_id]
    if any(info.has_prior_chunks for info in target_packed_seq_info):
        self._refresh_lin_state_window(target_layer.layer_id, target_chunk_id - 1)
    # else: no refresh; receiver chunk is start-of-seq, won't read window
```

The `target_chunk_id - 1` for lin-state vs `target_chunk_id` for KV
is the key semantic difference: KV is per-chunk's-tokens, lin_final_state
is per-chunk-INPUT-state which lives at chunk N-1's output.

### Memory cost

* `lin_state_window`: 2 × HV × K × V × 4 = 4 MiB on Qwen3.5.
* `lin_final_state` slot field: HV × K × V × 4 = 2 MiB per slot.
  Across n_gpu_act_slots × n_linear_layers, on Qwen3.5-MoE-35B:
  ~480 MiB of slot ring growth, similar host buffer.

### Why a slot field (not just a window-only)

Bwd of chunk N needs `initial_state` (= state at chunk N's input).
That's the FINAL state of chunk N-1, which was computed during fwd.
Without a slot field, bwd would need to re-run fwd of chunk N-1
end-to-end to reproduce it (an O(T) recompute). The 2 MiB slot field
amortizes this away — exactly the same logic as why KV is saved
rather than recomputed.

### The off-by-one source-chunk asymmetry vs dense

Dense uses `target_chunk_id` for source slot lookup; linear uses
`target_chunk_id - 1`. This is not arbitrary — it falls out of what
each saved field represents:

* `slot[L, N].xk/xv` holds the K/V values OF chunk N's tokens at
  layer L. Chunk N's bwd reads these for its own attention recompute,
  so target chunk N's source slot is L's slot at chunk N (its OWN
  slot).

* `slot[L, N].lin_final_state` holds the recurrent state AFTER
  chunk N's tokens at layer L (a boundary value, FLA's `final_state`).
  Chunk N's bwd reads the state BEFORE its tokens (= chunk N-1's
  final state), so target chunk N's source slot is L's slot at
  chunk N-1 (its PREDECESSOR's slot).

K/V is per-token data; recurrent state is a boundary value. The
naming convention (save final at slot N) is FLA-aligned and
keeps fwd writes self-contained (each chunk only writes its own
slot). The off-by-one lives in the dispatcher, not in the helpers.

### Why pre-bwd init is needed for linear but not dense

End of fwd state of each window:

* KV window: holds K/V values of the LAST layer (L_top) for ALL
  chunks. Fwd's chunk-by-chunk writes accumulated into the global
  positions; the last layer's writes overwrote earlier layers'.
* Lin window: holds `state[N_last]` of L_top — the state AFTER the
  last chunk's tokens.

Bwd's first iteration runs at L_top's last chunk:

* Dense bwd of chunk N_last reads K/V for tokens [0, N_last_end),
  including N_last's own tokens. Window already has all of L_top's
  K/V from fwd. **Match.**
* Linear bwd of chunk N_last reads state[N_last - 1] (state at
  N_last's INPUT). Window has state[N_last] (state at N_last's
  OUTPUT). **Mismatch — needs explicit refresh.**

So pre-bwd init is a one-shot dispatcher fire at `_backward_pass`
entry to populate `lin_state_window.fwd` from `slot[L_top, N_last - 1]
.lin_final_state`. KV's branch fires too (redundantly) for dense top
layers; the redundant copy is harmless and keeps the dispatcher
unified across types.

### Pre-bwd init dependency check (same as regular post-bwd refresh)

Pre-bwd is structurally identical to a regular dispatcher fire — same
stream (`inbound_fwd_context`), same WAR protection, same source-slot
event waits:

```
src_key = (L_top.layer_id, N_last - 1)  # for lin; (L_top, N_last) for KV
if src_key in events.inbound_act_slot_ready:
    inbound_fwd_context.wait_event(events.inbound_act_slot_ready[src_key])
    src_slot = events.dev_act_slot_mapping[src_key]
    with stream(inbound_fwd_context):
        lin_state_window.fwd.copy_(src_slot.lin_final_state)
else:
    avail = events.home_act_slot_available.get(src_key)
    if avail: inbound_fwd_context.wait_event(avail)
    home_slot = self._host_act_slots[src_key]
    with stream(inbound_fwd_context):
        lin_state_window.fwd.copy_(home_slot.lin_final_state)
```

The first iteration's `compute.wait_stream(inbound_fwd_context)` at
seq_group entry waits for the pre-bwd init to land, before compute
reads the window.

### Prefetch chain during bwd (steady state)

After every chunk's bwd, the dispatcher fires for the NEXT reverse
iteration. The chain is:

1. Iteration K's bwd completes on compute stream.
2. `inbound_fwd_context.wait_stream(compute)` — WAR protection so
   refresh doesn't overwrite window before compute is done reading.
3. Dispatcher determines target = iteration K+1's (layer, chunk_id).
   Branches by target_layer's schema (KV / lin / both / neither).
4. Each fired branch issues its source-slot copy on
   `inbound_fwd_context`, after waiting on the source-slot-resident
   event (`inbound_act_slot_ready` for device, `home_act_slot_available`
   for host).
5. Iteration K+1 starts; at seq_group entry it calls
   `compute.wait_stream(inbound_fwd_context)` — ensures the refresh
   landed. Within-group chunks reuse the wait state from the group's
   first iteration.

The dispatcher only checks dependencies; it doesn't compute fresh
state. The state was computed during fwd and saved to slot fields.

### Multi-seq-group robustness

A round can have multiple seq_groups in any order with any chunk
counts. Example layout:

* G0: small-seq packed chunk (1 chunk, no multi-chunk seqs)
* G1: long-seq-A (4 chunks: 1, 2, 3, 4)
* G2: small-seq packed chunk (1 chunk: 5)
* G3: long-seq-B (3 chunks: 6, 7, 8)

Bwd reverse order: 8, 7, 6, 5, 4, 3, 2, 1, 0.

Walking through the lin window content right before each chunk's
bwd, with the dispatcher firing at each transition:

| chunk | needs initial_state | source slot | window before bwd | reads window? |
|---|---|---|---|---|
| 8 | state[7] of B | slot[L, 7] | state[7] (pre-bwd init) | yes |
| 7 | state[6] of B | slot[L, 6] | state[6] (refresh after 8) | yes |
| 6 | none (B starts fresh) | — | state[5] (refresh after 7) | NO — info.has_prior=False |
| 5 | none (small-seq) | — | state[5] (no refresh: has_prior=False) | NO |
| 4 | state[3] of A | slot[L, 3] | state[3] (Path B refresh after 5) | yes |
| 3 | state[2] of A | slot[L, 2] | state[2] | yes |
| 2 | state[1] of A | slot[L, 1] | state[1] | yes |
| 1 | none (A starts fresh) | — | state[0] (refresh from prior would-be) | NO |
| 0 | none (small-seq) | — | (state[0]) | NO |

The dispatcher fires refreshes greedily based on `has_prior_chunks`
of the target chunk. The LAYER's bwd then independently checks
`info.has_prior_chunks` and passes `initial_state=None` when False,
ignoring window content.

### Invariants that make the design robust

The design works for arbitrary D/L layer patterns and arbitrary
seq_group orderings/sizes BECAUSE of these invariants:

1. **Contiguous chunk_ids per long seq.** `_pack_sequences`
   (`flextrain/engine/schedule.py:_emit_large`) emits all chunks of a
   large seq consecutively. So for a continuation chunk K of a long
   seq, chunk K-1 is always the prior chunk of the SAME long seq.
   `target_chunk_id - 1` source-slot lookup is therefore valid.

2. **Source slot is populated iff target needs it.**
   `slot[L, K-1].lin_final_state` is populated by fwd iff chunk K-1
   has `has_more_chunks=True`. Target K needs the window populated iff
   `has_prior_chunks=True`. By construction (chunks of the same
   long seq are contiguous), K's `has_prior=True` ⟺ K-1 is the prior
   chunk of the same seq, with K-1's `has_more=True`. So the
   dispatcher's refresh source is always populated when target
   needs it.

3. **Layer-side safety net.** The layer's `bwd` (and `fwd`) reads
   `chunk_seq_infos[i]` for each packed-seq i. If
   `info.has_prior_chunks=False`, layer passes `initial_state=None`
   (or zero) to FLA, regardless of window content. So even if
   stale data lingers in the window across transitions, the layer
   doesn't read it for chunks that don't need it.

4. **Different layers' state are independent.** Each layer's bwd
   loop processes its own chunks against its own slot fields
   (`slot[L, N].lin_final_state`, indexed by L). Different L values
   key into different slots. Layer-boundary refresh transitions the
   window from "stale data from last linear layer" to "fresh data
   for the new linear layer."

5. **Different long seqs are state-independent.** Two different long
   seqs A and B have different chunk_id ranges. A's chunks all live
   in `[A_start, A_end]`, B's in `[B_start, B_end]`. A's state never
   bleeds into B's reads because their source slots have different
   keys, and the layer's safety net (invariant 3) zeros window reads
   for B's first chunk.

6. **Group ordering doesn't matter.** Path A (within-group: target =
   group[chunk_in_group - 1]) and Path B (cross-group: target = prior
   group's last chunk) both resolve to the right chunk_id regardless
   of which long seq is in which group. Source-slot lookup uses
   chunk_id, which is global.

These invariants together mean the design handles:
* Pure dense or pure linear backbones.
* Arbitrary D/L interleaving (D-D-D-L-D-L-D, D-L-D-L-D, etc.).
* Multi-seq-group rounds with non-uniform group sizes.
* Mixed multi-chunk seqs interleaved with packed-small-seq chunks.

---

## Comparison: dense vs linear, side-by-side

| | Dense attention | Linear attention |
|---|---|---|
| Per-layer global structure | KV window (k, v, dk, dv) | LinAttnStateWindow (fwd, bwd) |
| Window shape | `(max_context_tokens, n_kv_heads, head_dim)` × 4 | `(HV, K, V)` × 2 |
| Window populated during fwd by | chunk's fwd writing K/V at varlen offsets | chunk's fwd writing final_state when continuing |
| Window populated during bwd by | `_refresh_kv_window` (slot[L, target].xk/xv → window) | `_refresh_lin_state_window` (slot[L, target-1].lin_final_state → window) |
| Source slot field | `xk`, `xv` (tier 0) | `lin_final_state` (tier 0, NEW) |
| Bwd accumulator | `dk`, `dv` in window | `dh0` in bwd window, written by compute stream after each chunk's bwd |
| Refresh stream | `inbound_fwd_context` | `inbound_fwd_context` (SAME stream) |
| Layer-entry pre-init in bwd | Not needed (fwd's last write covered) | Top layer needs pre-bwd init (fwd's last write is state[N], bwd needs state[N-1]) |
| Layer-boundary action in bwd | window holds prior-layer's K/V briefly; dispatcher refreshes for new layer at prior iter's tail | Path B at prior iter's tail populates new layer's window; bwd window zeroed at new layer's entry |
| Seq-group-boundary action | Path A handles cross-group within layer | Same dispatcher logic; lin-state Path A pulls from slot[L, prior_group.last] |
| Memory cost (Qwen3.5-MoE flagship) | KV window ~10s of MiB, slot fields per-chunk ~hundreds of MiB | windows ~4 MiB, slot field ~480 MiB across ring |

---

## Multi-step / multi-round

Both mechanisms are per-round:

* KV: window reused across rounds (zero on entry).
* Linear: same. Both windows zeroed at round entry; slot field
  written/read within the round.

No cross-round state.

---

## Appendix: round plan (`linear_attn_state.py`)

`LinearAttnRoundPlan` holds `per_chunk: list[list[MultiChunkSeqInfo]]`
indexed by `chunk_id`. Each `MultiChunkSeqInfo`:

* `seq_id`: stable Sequence.seq_id
* `chunk_in_seq_idx`: 0-based index of the chunk within seq's chunk run
* `has_prior_chunks`: True iff chunk_in_seq_idx > 0
* `has_more_chunks`: True iff later chunks exist for the same seq

For dedicated single-seq chunks (the only kind that participates in
multi-chunk math), `per_chunk[chunk_id]` has exactly one entry.

The plan is informational. The dispatcher reads it to decide whether
to fire the lin-state refresh; the layer's `_fwd_fla` reads it to
decide whether to pass `initial_state=zero` vs `initial_state=window.fwd`.
