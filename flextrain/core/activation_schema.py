"""Activation-slot abstractions.

The seam this replaces
----------------------
In ``orig/awsm_transformer/dense_layer.py`` and ``moe_layer.py``, each layer
type maintains FOUR parallel code paths that must agree name-by-name,
dtype-by-dtype, offset-by-offset, and tier-by-tier:

1. ``make_act_slot(num_tokens, saved_level, buffer=None)`` -- allocates fresh
   tensors OR slices out of a flat uint8 buffer with explicit offset math.
2. ``get_act_slot_size(num_tokens, saved_level)`` -- byte size for the DP
   solver; must match ``make_act_slot`` exactly.
3. ``send_activations_home(home, computed, level)`` -- hardcoded
   ``save_level_mapping`` dict naming which key lives at which tier.
4. ``fetch_activations(...)`` -- reverse of (3).

Adding a new activation tensor means touching >= 5 places (the four above plus
the per-level FLOP bookkeeping in ``get_fwd_flops``), all in lockstep. The MoE
file has an explicit TODO: "clean this up and have systematic way of handling
act slots!!!!".

What this module gives instead
------------------------------
A layer declares its activation tensors ONCE as a tuple of :class:`ActivationField`.
The :class:`ActivationSchema` collects those fields and derives size, layout,
send-home, and fetch-home from the declaration. The engine owns all buffer
arithmetic; layers only ever see typed :class:`ActivationSlot` objects.

Key resolutions
---------------
* ``tier: int``: save level L persists all fields with ``tier <= L``. Matches
  the paper's 4-level enum. Per-layer ``max_tier`` supports heterogeneous
  backbones (see :class:`SaveLevel`).
* ``offload=False``: device-only field (e.g. MoE router metadata); never in
  the home slot, never crosses PCIe.
* ``persist=False``: engine-owned scratch reused across chunks; no per-(chunk,
  layer) home slot at all.
* ``token_axis``: which axis of ``shape_fn``'s output is ``num_tokens``.
  Handles the ``softmax_lse`` (n_heads, num_tokens) transpose case without
  per-layer special-case code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import torch


ShapeFn = Callable[[int, Mapping[str, int]], tuple[int, ...]]
"""(num_tokens, model_dims) -> concrete tensor shape.

MUST be deterministic in ``num_tokens``. For the MoE ``x_up`` case, the outer
buffer shape is ``(num_tokens * top_k, 2 * expert_dim)`` -- deterministic;
per-expert slicing is a runtime derivation inside the MoE FFN's forward, not
part of the schema.
"""


# Per-slot byte alignment for the host activation buffer + GPU activation
# ring. Slots are packed back-to-back; if a slot's total byte size is not
# aligned to the LCM of all field dtypes' itemsizes, the next slot's
# views (e.g. ``buffer[off:off+nbytes].view(torch.float32)``) blow up
# with ``storage_offset() must be divisible by 4`` once a prior slot
# ends with a bf16/int16 field at an odd ``num_tokens``. fp32 is the
# largest field dtype in any current schema, so 4 bytes is sufficient.
# Per-slot alignment: end-of-slot is rounded to this so the next slot's
# first field starts cleanly. 256 covers (a) TMA on Hopper (128B), (b)
# 16-byte vectorized loads in Hopper-class kernels (quack, fbgemm),
# and (c) L1/L2 cache-line alignment. Cheap — wastes at most 255 bytes
# per slot.
_SLOT_ALIGN_BYTES = 256

# Per-field alignment: each field's storage_offset (within the slot's
# uint8 buffer) is rounded up to this. Same reasoning as _SLOT_ALIGN_BYTES;
# Hopper kernels (e.g. quack's gemm_gated) reject misaligned tensor data
# with `Misaligned Tensor data ... expected data alignment=16 bytes`.
# Wastes at most (FIELD_ALIGN - 1) bytes per field; a slot with ~10
# fields wastes ~2.5KB which is negligible vs typical slot size (MB).
_FIELD_ALIGN_BYTES = 256


def _padded_layout_bytes(
    fields,
    num_tokens: int,
    dims: Mapping[str, int],
) -> int:
    """Return the slot's actual byte size after per-field + end-of-slot
    padding. Mirrors the layout in :meth:`ActivationSlot.from_buffer`
    so size calculations match what's actually consumed."""
    offset = 0
    for f in fields:
        align = max(_FIELD_ALIGN_BYTES, f.dtype.itemsize)
        if offset % align != 0:
            offset = _round_up(offset, align)
        offset += f.byte_size(num_tokens, dims)
    return _round_up(offset, _SLOT_ALIGN_BYTES)


def _round_up(n: int, align: int) -> int:
    return (n + align - 1) // align * align


@dataclass(frozen=True)
class ActivationField:
    """One declared activation tensor.

    Parameters
    ----------
    name
        Attribute name on :class:`ActivationSlot` (e.g. ``"x_inp"``, ``"xk"``).
    shape_fn
        ``(num_tokens, dims) -> shape``. Deterministic in ``num_tokens``.
    dtype
        Tensor dtype (e.g. ``torch.bfloat16``, ``torch.float32`` for rstds).
    tier
        0..schema.max_tier. Save level ``L`` persists all fields with
        ``tier <= L``.
    offload
        If ``False``, device-only -- never in the home slot, never sent home or
        fetched back. Used for e.g. MoE router metadata trivially recomputable
        on device.
    persist
        If ``False``, the field has no per-(chunk, layer) home slot at all --
        engine owns it as scratch reused across chunks.
    token_axis
        Which axis of the shape is ``num_tokens``. 0 for most fields; 1 for
        ``softmax_lse`` which is ``(n_heads, num_tokens)``.
    """

    name: str
    shape_fn: ShapeFn
    dtype: torch.dtype
    tier: int
    offload: bool = True
    persist: bool = True
    token_axis: int = 0

    def shape(self, num_tokens: int, dims: Mapping[str, int]) -> tuple[int, ...]:
        return self.shape_fn(num_tokens, dims)

    def numel(self, num_tokens: int, dims: Mapping[str, int]) -> int:
        n = 1
        for s in self.shape(num_tokens, dims):
            n *= s
        return n

    def byte_size(self, num_tokens: int, dims: Mapping[str, int]) -> int:
        return self.numel(num_tokens, dims) * self.dtype.itemsize


@dataclass(frozen=True)
class ActivationSchema:
    """The declared activation tensors of one layer type.

    Parameters
    ----------
    fields
        Tuple of :class:`ActivationField`. Order determines layout in the
        host-pinned buffer.
    max_tier
        Highest save level this layer supports. DP solver sees levels
        0..max_tier.

    The engine derives ``home_size_bytes``, ``device_size_bytes``, buffer
    layout, and send/fetch iteration from this alone -- no per-layer
    overrides.
    """

    fields: tuple[ActivationField, ...]
    max_tier: int

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate field names: {dupes}")
        for f in self.fields:
            if not (0 <= f.tier <= self.max_tier):
                raise ValueError(
                    f"field {f.name!r} has tier={f.tier}, max_tier={self.max_tier}"
                )

    def fields_at_level(self, level: int) -> tuple[ActivationField, ...]:
        """Fields whose ``tier <= level``. Empty at level < 0."""
        if level < 0:
            return ()
        return tuple(f for f in self.fields if f.tier <= level)

    def has_field(self, name: str) -> bool:
        """True iff this schema declares a field named ``name``.

        Used by the engine to dispatch type-specific cross-chunk
        machinery (e.g. ``has_field("xk")`` for dense KV-window
        refresh, ``has_field("lin_final_state")`` for linear-attn
        state-window refresh) without hardcoding layer-class checks.
        """
        return any(f.name == name for f in self.fields)

    def _field_tiers_cache(self) -> dict[str, int]:
        """Map field name → tier. Used by :meth:`ActivationSlot.has`
        to check field validity at the slot's save level. Computed
        per-call (schema is frozen and this is called rarely —
        forward_recompute entry per layer per chunk).
        """
        return {f.name: f.tier for f in self.fields}

    def persistent_fields_at_level(self, level: int) -> tuple[ActivationField, ...]:
        """Persistent fields at a given level -- i.e. the ones that consume a
        slot in the host-pinned buffer."""
        return tuple(f for f in self.fields_at_level(level) if f.persist)

    def offloadable_fields_at_level(self, level: int) -> tuple[ActivationField, ...]:
        """Persistent + offloadable fields at a level -- the ones that actually
        cross PCIe on send_home/fetch_home."""
        return tuple(
            f for f in self.persistent_fields_at_level(level) if f.offload
        )

    def device_fields(self) -> tuple[ActivationField, ...]:
        """Fields that always live on device (tier <= max_tier; persist may be
        True or False). Used for the GPU activation-slot sizing."""
        return self.fields

    def home_size_bytes(
        self, num_tokens: int, dims: Mapping[str, int], level: int
    ) -> int:
        """Bytes this layer/chunk needs in the host-pinned buffer at the given
        save level.

        Rounded up to ``_SLOT_ALIGN_BYTES`` so the next slot's first field
        (often fp32 ``attn_norm_rstd``) lands at a dtype-aligned storage
        offset — without padding, a slot ending in a bf16 field with odd
        ``num_tokens`` (e.g. ``x_shared_gate (T, 1)`` with T odd) leaves
        the cursor at an offset that's 2 mod 4 and the subsequent
        ``buffer.view(torch.float32)`` blows up at line 334's
        ``view().reshape()``.
        """
        return _padded_layout_bytes(
            self.persistent_fields_at_level(level), num_tokens, dims,
        )

    def device_size_bytes(self, num_tokens: int, dims: Mapping[str, int]) -> int:
        """Bytes this layer/chunk needs in the GPU activation slot. ALL fields
        (every tier) live on device during forward; selection only matters for
        send-home.

        Rounded up to ``_SLOT_ALIGN_BYTES`` for the same reason as
        :meth:`home_size_bytes` — the GPU activation ring stores slots
        back-to-back at ``i * max_act_slot_bytes``, so if the per-slot
        size isn't dtype-aligned, slot 1 onward gets misaligned views.
        """
        return _padded_layout_bytes(self.fields, num_tokens, dims)

    def offloaded_bytes_at_level(
        self, num_tokens: int, dims: Mapping[str, int], level: int
    ) -> int:
        """Bytes that actually cross PCIe at this save level. Used by the DP
        solver's transfer-duration calculation."""
        return sum(
            f.byte_size(num_tokens, dims)
            for f in self.offloadable_fields_at_level(level)
        )


# ---------------------------------------------------------------------------
# ActivationSlot: the typed view a layer sees at runtime.
# ---------------------------------------------------------------------------


class ActivationSlot:
    """Typed view over a set of tensors for one (chunk, layer) pair.

    Layers access fields by attribute: ``slot.x_inp``, ``slot.attn_result``.
    They check presence with ``slot.has(name)`` instead of
    ``"key" in fwd_act_slot`` dict introspection.

    Instances are constructed by the engine -- layers never allocate their own
    slots. The engine fills ``_tensors`` by either:

    * slicing the host-pinned buffer (for home slots and the recirculating
      GPU activation ring -- see :meth:`from_buffer`), or
    * allocating fresh tensors (rare; only for unit tests and for freshly
      computed device-only tensors in ``forward``).

    The ``aux`` dict is a per-slot mutable scratch stash. Algorithmic
    blocks use it to pass tensors between their fwd / bwd helper methods
    (e.g. an attention block stashing the local dQ/dK/dV from flash-attn
    bwd so the RMSNorm-bwd call downstream can hand them off for the
    weight-grad matmuls). Engine-owned; layers write/read but do not
    assume ordering with other layers' slots.
    """

    __slots__ = ("schema", "level", "_tensors", "aux")

    def __init__(
        self,
        schema: ActivationSchema,
        level: int,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        self.schema = schema
        self.level = level
        self._tensors: dict[str, torch.Tensor] = dict(tensors)
        self.aux: dict[str, torch.Tensor] = {}

    # -- access --

    def __getattr__(self, name: str) -> torch.Tensor:
        # __getattr__ is only called when the normal attribute lookup fails,
        # so __slots__ members above resolve directly and don't recurse here.
        try:
            return self._tensors[name]
        except KeyError:
            raise AttributeError(
                f"activation field {name!r} not present at level={self.level} "
                f"(available: {sorted(self._tensors)})"
            ) from None

    def has(self, name: str) -> bool:
        """Is the field named ``name`` VALID at this slot's save level?

        Semantics: a field is considered present iff its tier is <=
        ``self.level``. The underlying ``_tensors`` dict may contain
        views for higher-tier fields too (the engine hands out
        max-tier GPU ring slots even when only lower-tier fields are
        actually populated from host), but those higher-tier views
        point to STALE memory left over from the prior use of the
        ring slot and MUST be treated as absent so that
        ``forward_recompute`` will overwrite them.

        The previous implementation (``return name in self._tensors``)
        incorrectly returned True for higher-tier fields on prefetched
        slots, causing LlamaBlock/Qwen3Block forward_recompute to
        silently skip the xq/xo/x1/x3 recompute — see [FINDING 17]
        in ``docs/internal/NOTES.md``. Symptom: bwd at an offloaded layer
        would read a previous layer's xq from the ring slot,
        producing gradients that explode through the residual chain.
        """
        field_tiers = self.schema._field_tiers_cache()
        tier = field_tiers.get(name)
        if tier is None:
            # Not a declared field -- default to dict membership (e.g.
            # for aux scratch fields installed via slot.set()).
            return name in self._tensors
        return tier <= self.level

    def set(self, name: str, tensor: torch.Tensor) -> None:
        """Install a freshly computed tensor into the slot. Used by
        ``forward`` when writing into engine-provided output buffers and by
        ``forward_recompute`` when recovering higher-tier fields."""
        self._tensors[name] = tensor

    def items(self):
        return self._tensors.items()

    def __repr__(self) -> str:
        keys = sorted(self._tensors)
        return f"ActivationSlot(level={self.level}, fields={keys})"

    # -- construction helpers (engine-side, but kept here to stay adjacent
    #    to the field/dtype math so they can't drift) --

    @classmethod
    def from_buffer(
        cls,
        schema: ActivationSchema,
        level: int,
        num_tokens: int,
        dims: Mapping[str, int],
        buffer: torch.Tensor,
        *,
        include_nonpersistent: bool = False,
    ) -> tuple["ActivationSlot", int]:
        """Slice ``buffer`` (1D uint8) into views for every persistent field at
        ``level``. Returns the slot and the number of bytes consumed.

        ``include_nonpersistent=True`` is used for the GPU activation-ring
        slots, where even ``persist=False`` fields need on-device storage; the
        host-pinned home slot uses the default ``False``.
        """
        if buffer.dtype is not torch.uint8 or buffer.dim() != 1:
            raise TypeError("buffer must be a 1-D uint8 tensor")

        fields = (
            schema.fields_at_level(level)
            if include_nonpersistent
            else schema.persistent_fields_at_level(level)
        )

        tensors: dict[str, torch.Tensor] = {}
        offset = 0
        for f in fields:
            shape = f.shape(num_tokens, dims)
            nbytes = f.byte_size(num_tokens, dims)
            # Each field's view requires storage_offset divisible by its
            # dtype itemsize for the typed view to work. We over-align to
            # _FIELD_ALIGN_BYTES (256) so Hopper-class kernels (quack
            # gemm_gated etc.) get the 16-byte alignment they require —
            # individual fields' first rows must sit on 16-byte boundaries
            # for vectorized loads.
            align = max(_FIELD_ALIGN_BYTES, f.dtype.itemsize)
            if offset % align != 0:
                offset = _round_up(offset, align)
            if offset + nbytes > buffer.numel():
                raise ValueError(
                    f"buffer too small: field {f.name!r} needs {nbytes}B at "
                    f"offset {offset}, buffer has {buffer.numel()}B"
                )
            view = buffer[offset : offset + nbytes].view(f.dtype).reshape(shape)
            tensors[f.name] = view
            offset += nbytes
        # Round the final cursor to slot alignment so the caller's
        # ``bytes_used`` matches what ``home_size_bytes`` reports —
        # otherwise ``BufferManager._host_act_cursor`` and the schema
        # disagree about slot boundaries by up to 3 bytes per slot.
        offset = _round_up(offset, _SLOT_ALIGN_BYTES)
        return cls(schema, level, tensors), offset

    def view_for(self, num_tokens: int, dims: Mapping[str, int]) -> "ActivationSlot":
        """Narrow every tensor in this slot to the first ``num_tokens`` along
        its declared ``token_axis``. Replaces the per-layer
        ``v[:num_tokens, :]`` / ``v[:, :num_tokens]`` branching in
        ``dense_layer.py:30-36``.

        Fields declared with ``token_axis=None`` don't scale with
        num_tokens and are passed through unchanged (e.g. MoE
        ``expert_counts`` with shape ``(num_experts,)``, or
        ``x_up`` which is already sized to ``(max_chunk * top_k,
        2*expert_dim)`` and consumed in sorted-by-expert order).
        """
        name_to_field = {f.name: f for f in self.schema.fields}
        narrowed: dict[str, torch.Tensor] = {}
        for name, tensor in self._tensors.items():
            f = name_to_field[name]
            axis = f.token_axis
            if axis is None:
                narrowed[name] = tensor
                continue
            if tensor.shape[axis] == num_tokens:
                narrowed[name] = tensor
            else:
                narrowed[name] = tensor.narrow(axis, 0, num_tokens)
        return ActivationSlot(self.schema, self.level, narrowed)


# ---------------------------------------------------------------------------
# Transfer helpers. The engine calls these; layers never do.
# ---------------------------------------------------------------------------


def _match_to(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Return a view of ``src`` whose shape matches ``dst``.

    The GPU activation ring is sized at ``max_chunk_size`` for predictable
    layout; the host slot is sized at the chunk's actual ``num_tokens``
    for memory efficiency. For most fields ``view_for`` already narrowed
    the GPU slot's ``token_axis`` dim to match. But fields with
    ``token_axis=None`` whose shape *does* depend on ``num_tokens`` (e.g.
    MoE ``scattered_router_weights`` of shape ``(T*top_k, 1)``,
    ``x_up`` of shape ``(T*top_k, 2*expert_dim)``) bypass that narrowing
    by design: the block does its own ``[:T*top_k, :]`` slicing at
    runtime. For ``send_home`` / ``fetch_home`` we slice the source down
    to the destination's first-dim extent — the ``[:T_actual*top_k]``
    prefix is the only valid content on device anyway.
    """
    if src.shape == dst.shape:
        return src
    # Slice each leading dim where the source is larger than the dest.
    out = src
    for axis, (s, d) in enumerate(zip(src.shape, dst.shape)):
        if s != d:
            if s < d:
                raise ValueError(
                    f"send_home/fetch_home: source shape {tuple(src.shape)} "
                    f"smaller than dest {tuple(dst.shape)} at axis {axis}"
                )
            out = out.narrow(axis, 0, d)
    return out


def send_home(
    home_slot: ActivationSlot,
    computed_slot: ActivationSlot,
    level: int,
    *,
    non_blocking: bool = True,
) -> None:
    """Copy every offloadable field at ``level`` from the device-resident
    ``computed_slot`` into the host-pinned ``home_slot``. Replaces the hand-
    maintained ``save_level_mapping`` dict in
    ``dense_layer.py:920-936`` (and its MoE twin).
    """
    if home_slot.schema is not computed_slot.schema:
        raise ValueError("home and computed slots must share a schema")
    for f in home_slot.schema.offloadable_fields_at_level(level):
        dst = home_slot._tensors[f.name]
        src = _match_to(computed_slot._tensors[f.name], dst)
        dst.copy_(src, non_blocking=non_blocking)


def fetch_home(
    dest_slot: ActivationSlot,
    home_slot: ActivationSlot,
    level: int,
    *,
    non_blocking: bool = True,
) -> None:
    """Copy every offloadable field at ``level`` from the host-pinned
    ``home_slot`` back into the device-resident ``dest_slot``."""
    if dest_slot.schema is not home_slot.schema:
        raise ValueError("dest and home slots must share a schema")
    for f in dest_slot.schema.offloadable_fields_at_level(level):
        dst = dest_slot._tensors[f.name]
        # Home tensor is at chunk-actual T*top_k; dest (GPU) is at
        # max_chunk*top_k. Narrow dest to home's extent before copy.
        dst_view = _match_to(dst, home_slot._tensors[f.name])
        dst_view.copy_(home_slot._tensors[f.name], non_blocking=non_blocking)


# ---------------------------------------------------------------------------
# Convenience for block composition (norm/attention/ffn declare their own
# fields; the layer builds a schema by concatenation).
# ---------------------------------------------------------------------------


def concat_fields(
    blocks: Sequence[Sequence[ActivationField]],
) -> tuple[ActivationField, ...]:
    """Concatenate per-block field tuples into the flat field sequence a
    :class:`ActivationSchema` expects. Errors on duplicate names."""
    out: list[ActivationField] = []
    for block in blocks:
        out.extend(block)
    names = [f.name for f in out]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate field names across blocks: {dupes}")
    return tuple(out)
