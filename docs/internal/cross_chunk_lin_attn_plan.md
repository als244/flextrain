# Cross-chunk linear-attn implementation plan (Item 3c)

Concrete, sequenced plan for plumbing FLA `initial_state` /
`output_final_state` / `dh0` / `dht` through FT's linear-attn path so
sequences spanning multiple chunks compute correctly.

**Reference doc**: `docs/multi_chunk_seq_handling.md` — mechanism,
memory, dependencies, ordering, invariants. This file is the
executable plan keyed to that doc.

## Pre-work invariants (don't break)

1. Dense-attn KV-context machinery (`_update_fwd_context`,
   `KVContextWindow`, `inbound_fwd_context` stream) must remain
   bit-identical for dense-only models. Stage 3a baseline (Llama-3.2-1B,
   Qwen3-1.7B) MUST still produce bit-identical logits to current
   master (`3c3b54d`) after the dispatcher refactor. This is the
   first thing verified.
2. Single-chunk seqs (the common case) must hit the existing
   `_fwd_fla` path — `chunk_gated_delta_rule_fwd(initial_state=None,
   output_final_state=False)` — so existing parity baselines
   (test_arch_lora_e2e, Qwen3.5-9B / 35B-MoE LoRA loss curves) are
   unchanged.
3. No layer signature changes that break heterogeneous backbones.
   Cross-chunk plumbing is additive via optional kwargs / context fields.

## Design summary (re-derived; see reference doc for full analysis)

* **One slot field**: `lin_final_state: (HV, K, V) fp32`, tier 0,
  always allocated for linear-attn schemas. Single tensor (not
  row-indexed) because `_pack_sequences` produces dedicated single-
  packed-seq chunks for any chunk participating in a multi-chunk seq.
* **Two windows**: `lin_state_window.fwd` and `.bwd`, both
  `(HV, K, V) fp32`. Allocated once at engine init if any backbone
  layer has `lin_final_state` in its schema.
* **One stream**: reuse `inbound_fwd_context` (no new stream).
* **Unified dispatcher**: `_update_fwd_context` becomes a target-
  iteration-aware dispatcher that branches by target_layer's schema.
  Dense (`xk` field) and linear (`lin_final_state` field) branches
  are independent helpers; both, either, or neither can fire for a
  given target.
* **Source-slot rule**: KV uses `target_chunk_id`; lin-state uses
  `target_chunk_id - 1` (off-by-one due to boundary-vs-per-token
  semantics, see reference doc).
* **Pre-bwd init**: at `_backward_pass` entry, fire the dispatcher
  once for the top layer's first reverse iteration. Populates the
  window correctly for the first compute iteration.
* **Layer safety net**: layer's `fwd` and `bwd` check `info.has_prior_chunks`
  per packed-seq and pass `initial_state=None` when False, ignoring
  window content. Makes the design robust to stale window data
  across transitions.

## File-by-file plan

### 1. `flextrain/engine/linear_attn_state.py` — round plan only

Replace the existing bank classes with just the round plan:

* `MultiChunkSeqInfo(seq_id, chunk_in_seq_idx, has_prior_chunks,
  has_more_chunks)` (existing)
* `LinearAttnRoundPlan(per_chunk: list[list[MultiChunkSeqInfo]])`
* `build_linear_attn_round_plan(prepared) -> LinearAttnRoundPlan`

Drop:
* `LinearAttnStateBank` / `LinearAttnDStateBank` (replaced by
  `LinAttnStateWindow` + slot field).

### 2. `flextrain/engine/buffers.py` — `LinAttnStateWindow`

Mirror of `KVContextWindow`:

```python
@dataclass
class LinAttnStateWindow:
    fwd: torch.Tensor   # (HV, K, V) fp32 — initial_state for FLA fwd / bwd
    bwd: torch.Tensor   # (HV, K, V) fp32 — dh0 chain accumulator

    @classmethod
    def create(cls, hv, k_d, v_d, device):
        shape = (hv, k_d, v_d)
        return cls(
            fwd=torch.zeros(shape, dtype=torch.float32, device=device),
            bwd=torch.zeros(shape, dtype=torch.float32, device=device),
        )

    def zero_(self):
        self.fwd.zero_()
        self.bwd.zero_()
```

`BufferManager.__init__` allocates the window if any layer schema
has `lin_final_state`. Stash on `self.lin_state_window`.

### 3. `flextrain/nn/blocks/linear_attn.py` — schema + fwd + bwd

#### a. Schema field

Add to `LinearAttnSchema.fields()`:

```python
ActivationField(
    "lin_final_state",
    lambda n, d: (cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim),
    torch.float32, tier=0,
),
```

Shape is independent of `n` (per-chunk constant). `byte_size`
returns `HV*K*V*4` regardless of token count.

#### b. `_fwd_fla` — accept cross-chunk inputs

Read cross-chunk parameters from `ctx`:

```python
def _fwd_fla(
    self, q_n, k_n, v_h, a, beta, A_log, dt_bias, slot,
    cu_seqlens=None, chunk_indices=None,
    *, ctx=None,                              # NEW: thread ctx through
) -> torch.Tensor:
    ...
    infos = ctx.lin_attn_chunk_seq_infos if ctx else None
    fwd_window = ctx.lin_attn_fwd_window if ctx else None
    needs_state = (
        infos is not None and fwd_window is not None
        and any(i.has_prior_chunks or i.has_more_chunks for i in infos)
    )
    if not needs_state:
        # Today's path: bit-identical to current behavior.
        g_post, o, A_int, _, _, _ = chunk_gated_delta_rule_fwd(
            ..., initial_state=None, output_final_state=False, ...)
        ...
        return o

    # Cross-chunk path. Build (N_packed, HV, K, V) initial_state.
    # For dedicated single-seq chunks (the only kind that participates
    # in multi-chunk math), N_packed=1 and the row maps directly.
    ...
```

#### c. `bwd` — accept cross-chunk inputs

Same pattern via `ctx`. Build `initial_state` and `dht` from windows
when `infos` indicates need; else pass None.

After FLA bwd returns `dh0`, write into `ctx.lin_attn_bwd_window`
on compute stream (no event needed, same stream).

### 4. `flextrain/core/layer.py` — `LayerContext` extension

Already partially done. Final fields:

```python
current_chunk_id: int | None = None
current_layer_id: int | None = None

# Cross-chunk linear-attn round plan + windows. Set by engine
# per-(chunk, layer) iteration during fwd/bwd. Layers reading
# any of these MUST tolerate None (unit tests outside the engine).
lin_attn_chunk_seq_infos: Any = None        # list[MultiChunkSeqInfo] | None
lin_attn_fwd_window: torch.Tensor | None = None
lin_attn_bwd_window: torch.Tensor | None = None
```

Drop the bank fields I added in the earlier sketch.

### 5. `flextrain/engine/active_model.py` — engine wiring

#### a. Round setup

In `_setup_round`:

```python
self._lin_attn_plan = build_linear_attn_round_plan(prepared)
self.buffers.lin_state_window.zero_() if self.buffers.lin_state_window else None
```

#### b. Forward pass

Per chunk N at layer L:

```python
ctx.current_chunk_id = chunk.id
ctx.current_layer_id = lid
if self.buffers.lin_state_window is not None:
    ctx.lin_attn_chunk_seq_infos = self._lin_attn_plan.per_chunk[chunk.id]
    ctx.lin_attn_fwd_window = self.buffers.lin_state_window.fwd
    ctx.lin_attn_bwd_window = self.buffers.lin_state_window.bwd
```

Layer entry zero (between layers in fwd outer loop):

```python
if self.buffers.lin_state_window is not None:
    self.buffers.lin_state_window.zero_()
```

Inside `_fwd_fla`'s cross-chunk path, after FLA returns `final_state`:
* Write to slot: `slot.lin_final_state.copy_(final_state.squeeze())`.
* Write to window: `lin_attn_fwd_window.copy_(final_state.squeeze())`.
* Conditional on `info.has_more_chunks` for the (single) packed-seq
  in this chunk.

#### c. Backward pass

Refactor `_update_fwd_context` into a target-iteration dispatcher:

```python
def _update_fwd_context(
    self, *, seq_group_ind, chunk_in_group_ind, layer_ind, prepared
):
    """Determine the next reverse iteration's target and refresh
    the appropriate windows. Dispatches by target_layer's schema."""
    target = self._next_reverse_iteration_target(
        seq_group_ind, chunk_in_group_ind, layer_ind, prepared
    )
    if target is None:
        return
    target_layer, target_chunk_id = target
    schema = target_layer.schema
    if schema.has_field("xk"):
        self._refresh_kv_window(
            target_layer.layer_id, target_chunk_id, prepared
        )
    if schema.has_field("lin_final_state"):
        target_infos = self._lin_attn_plan.per_chunk[target_chunk_id]
        if any(info.has_prior_chunks for info in target_infos):
            src_chunk_id = target_chunk_id - 1
            self._refresh_lin_state_window(
                target_layer.layer_id, src_chunk_id, prepared
            )
```

`_next_reverse_iteration_target` consolidates the Path A / Path B
logic:

```python
def _next_reverse_iteration_target(
    self, seq_group_ind, chunk_in_group_ind, layer_ind, prepared
):
    # Path A: same layer, same group, prior chunk-in-group
    if chunk_in_group_ind > 0:
        return (
            self.backbone[layer_ind],
            prepared.seq_groups[seq_group_ind][chunk_in_group_ind - 1].id,
        )
    # Path A': same layer, prior group, last chunk-in-group
    if seq_group_ind > 0:
        return (
            self.backbone[layer_ind],
            prepared.seq_groups[seq_group_ind - 1][-1].id,
        )
    # Path B: prior layer, last group, last chunk-in-group
    if layer_ind > 0:
        return (
            self.backbone[layer_ind - 1],
            prepared.seq_groups[-1][-1].id,
        )
    return None
```

`_refresh_kv_window(lid, chunk_id, prepared)` extracts today's
`_update_fwd_context` body but parameterized by source slot key.
Same Path-A/Path-B logic that copies `xk`/`xv` into `kv_fwd` window.

`_refresh_lin_state_window(lid, chunk_id, prepared)` is the
analog: copy `lin_final_state` from slot[lid, chunk_id] into
`lin_state_window.fwd` on `inbound_fwd_context`. Source-on-device
vs source-on-host logic mirrors `_refresh_kv_window`.

#### d. Pre-bwd init

At `_backward_pass` entry, before the layer loop:

```python
def _backward_pass_pre_init(self, prepared):
    """One-shot dispatcher fire for the top layer's first reverse
    iteration. Pre-populates windows so first chunk's compute reads
    correct data."""
    top_layer = self.backbone[-1]
    target_chunk_id = prepared.seq_groups[-1][-1].id
    schema = top_layer.schema
    if schema.has_field("xk"):
        # KV is naturally correct from fwd's last write, but firing
        # the refresh is a redundant copy that's harmless.
        self._refresh_kv_window(top_layer.layer_id, target_chunk_id, prepared)
    if schema.has_field("lin_final_state"):
        target_infos = self._lin_attn_plan.per_chunk[target_chunk_id]
        if any(info.has_prior_chunks for info in target_infos):
            self._refresh_lin_state_window(
                top_layer.layer_id, target_chunk_id - 1, prepared
            )
```

Actually for KV the redundant refresh might MOVE data unnecessarily.
Consider a check: only fire if window doesn't already hold target's
content. For Stage 3c's commit C0 (refactor only), preserve current
behavior — just split the dense path into a helper without firing
extra refreshes. Pre-bwd init is added only in Commit C5 with the
linear branch.

#### e. Layer-entry bwd window zero

At the start of each linear-attn layer's bwd loop:

```python
if cur_layer.schema.has_field("lin_final_state"):
    self.buffers.lin_state_window.bwd.zero_()
```

Zero only `bwd` — `fwd` is correctly populated by the prior
iteration's dispatcher fire (or pre-bwd init for the top layer).

### 6. `flextrain/core/working_set.py` — accounting

Add to `_baseline_gpu_activation_memory`:

```python
# Linear-attn cross-chunk state windows (fwd + bwd), if backbone
# has linear-attn layers.
if num_v_heads and head_v_dim and head_k_dim:
    lin_state_window_bytes = (
        2  # fwd + bwd windows
        * num_v_heads * head_k_dim * head_v_dim * 4  # fp32
    )
    bytes_used += lin_state_window_bytes
```

The per-slot `lin_final_state` field is captured automatically by
the schema-driven `home_size_bytes(...)` /
`device_size_bytes(...)` paths since we added it to the schema.

## Implementation order

1. **Commit C0**: Refactor `_update_fwd_context` into the dispatcher
   form with KV-only branch. Pure code reorganization; dense math
   unchanged. Verify Stage 3a runs produce bit-identical logits to
   master `3c3b54d`.
2. **Commit C1**: Add `lin_final_state` schema field. Slot field
   allocated but never written/read. Verify single-chunk parity
   tests still pass (the field is a tier-0 zero-initialized buffer
   that nothing touches).
3. **Commit C2**: Add `LinAttnStateWindow` + `BufferManager`
   allocation. Verify single-chunk parity (window allocated but
   never written/read).
4. **Commit C3**: `_fwd_fla` + `bwd` cross-chunk plumbing via
   `ctx`. Single-chunk path falls through unchanged. Verify
   single-chunk parity bit-identical.
5. **Commit C4**: Engine fwd-pass cross-chunk wiring (round plan,
   ctx threading, layer-boundary zero, slot writes). Verify
   single-chunk parity bit-identical AND Stage 3b fwd parity
   improves on Qwen3.5-2B (no longer reads garbage initial_state).
6. **Commit C5**: Engine bwd-pass cross-chunk wiring — dispatcher's
   lin branch + pre-bwd init + layer-entry zero. Run Stage 3a (dense
   regression — must still pass), Stage 3b (Qwen3.5-2B should now
   pass with uniform per-chunk argmax).
7. **Commit C6**: working_set planner accounting.
8. **Commit C7**: README updates + final regression runs +
   memory-file update.

Each commit pushed independently so any regression is bisectable.

## Acceptance bar

After C5: `tests/multi_chunk_dense_parity/run_e2e.py` should produce:
* Llama-3.2-1B: PASS unchanged (rel ~1.8e-5, per-chunk argmax ~98%
  uniformly).
* Qwen3-1.7B: PASS unchanged.
* Qwen3.5-2B: flips from FAIL to PASS. Per-chunk argmax uniform
  ≥95%, ΔCE per chunk near zero.

Plus `tests/test_arch_lora_e2e.py` for Qwen3.5-2B / 9B / MoE-35B
must still pass with single-chunk seqs unchanged.

## Risks / things to verify

1. **Slot-field size in dims schema**: `lin_final_state` is constant-
   shape, doesn't depend on `num_tokens`. Slot allocator must handle
   this correctly. Test: `byte_size` returns `HV*K*V*4` regardless
   of `n`.
2. **Mixed backbones at cross-layer transitions**: tested by example
   walk-throughs in the reference doc. Dispatcher correctly fires
   the right branch based on target_layer's schema.
3. **Multi-seq-group rounds with non-uniform sizes**: tested by
   walk-through. Layer's `has_prior_chunks` safety net catches any
   stale-window-data case.
4. **Top layer pre-bwd init source slot might be evicted from device
   ring**: handle host fallback like `_refresh_kv_window`'s host
   path.
5. **`output_final_state=True` cost**: FLA's fwd allocates a final
   state tensor when this flag is true. Worst case ~2 MiB per FLA
   call — negligible.
6. **Slot allocator behavior with constant-shape field**: must verify
   the per-slot device tensor allocates at the correct fixed size.
7. **Tier-0 always-saved cost**: `lin_final_state` is always saved
   to host on offload. For Qwen3.5-MoE-35B with many chunks, host
   buffer grows linearly. Offload bandwidth: 2 MiB per (chunk,
   layer) — small, fits in existing offload stream budget.
