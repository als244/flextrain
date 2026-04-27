"""GPU and host buffer lifecycle.

Owns every training-time tensor that isn't a live compute-kernel
intermediate. Replaces the tangle of ``cpu_model_weights`` /
``cpu_grad_weights`` / ``cpu_opt_weights`` / ``model_weights_gpu`` /
``grad_weights_gpu`` / ``opt_weights_gpu`` / ``act_slot_gpu`` /
``cpu_act_buffer`` / ``transitions_gpu`` / ``fwd_context`` /
``bwd_context`` dicts in ``orig/active_model.py``.

Design constraints
------------------
* **Heterogeneous backbones supported**: the GPU param / grad / act
  rings are sized to the MAX across layer types. A layer with
  smaller tensors leaves the tail of its slot unused. See
  [DECISION 10] in docs/NOTES.md.
* **Host pinning uses cudaHostRegister** on a regular torch.zeros
  buffer, not ``pin_memory=True``. Matches orig (``active_model.py:
  266-273``) and sidesteps the "pin_memory only allocates in powers
  of 2" pathology that burns GB-scale host RAM on the activation
  buffer.
* **GPU activation buffer doubles as optimizer-state staging** during
  :meth:`ActiveModel.step` (paper §3.3). :meth:`swap_to_optimizer_state`
  carves opt-state slots out of the same underlying allocation and
  :meth:`restore_activation_ring` rebuilds the ring views.
* **Bodies, not stubs.** Phase 3 implementation lives here.

What this module does NOT do
----------------------------
* Forward / backward compute — :mod:`flextrain.nn.layers`.
* Stream synchronization — :mod:`flextrain.engine.streams`.
* Save-level DP — :mod:`flextrain.core.save_level`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from flextrain.core.activation_schema import (
    ActivationSchema,
    ActivationSlot,
)
from flextrain.core.layer import ParamSpec, TensorSpec
from flextrain.core.working_set import WorkingSetConfig
from flextrain.optim.base import OptimizerStateSpec

from .host_memory import HostMemoryBackend, default_host_backend


# ---------------------------------------------------------------------------
# KV context window.
# ---------------------------------------------------------------------------


@dataclass
class KVContextWindow:
    """One rolling K/V window + its backward counterparts.

    Sized to ``max_context_tokens * n_kv_heads * head_dim`` each in
    bf16; four tensors total (k, v, dk, dv). Orig allocates these once
    at :meth:`create_gpu_activations` (``active_model.py:432-446``) and
    reuses across all rounds and all layers — so do we.

    The ``max_context_tokens`` parameter is the larger of
    ``max_seq_len`` and ``max_chunk_size`` (orig:432).
    """

    k: torch.Tensor
    v: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        max_context_tokens: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "KVContextWindow":
        shape = (max_context_tokens, n_kv_heads, head_dim)
        return cls(
            k=torch.zeros(shape, dtype=dtype, device=device),
            v=torch.zeros(shape, dtype=dtype, device=device),
            dk=torch.zeros(shape, dtype=dtype, device=device),
            dv=torch.zeros(shape, dtype=dtype, device=device),
        )

    def zero_(self) -> "KVContextWindow":
        self.k.zero_()
        self.v.zero_()
        self.dk.zero_()
        self.dv.zero_()
        return self


# ---------------------------------------------------------------------------
# Scratch pool.
# ---------------------------------------------------------------------------


class ScratchPool:
    """Per-call allocator for ephemeral GPU workspace.

    Delegates to PyTorch's caching allocator — the reuse is already
    implicit. Centralizing here lets us swap in a ring-buffered arena
    later if a profiler flags scratch allocation cost.
    """

    def __init__(self, device: torch.device | str) -> None:
        self.device = device

    def __call__(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=self.device)


# ---------------------------------------------------------------------------
# BufferManager
# ---------------------------------------------------------------------------


def _byte_size_of_tensors(specs: Sequence[TensorSpec], dims: Mapping[str, int],
                          role: str) -> int:
    """Sum byte sizes of ``specs`` at the requested role.

    Frozen tensors (``TensorSpec.frozen=True``) are excluded from
    ``grad`` and ``opt_state`` roles since the engine doesn't allocate
    those buffers for them. They ARE included in ``compute`` and
    ``master`` (the forward still reads the frozen weight)."""
    total = 0
    for t in specs:
        if t.frozen and role in ("grad", "opt_state"):
            continue
        total += {
            "compute": t.compute_byte_size,
            "master": t.master_byte_size,
            "grad": t.grad_byte_size,
            "opt_state": t.opt_state_byte_size,
        }[role](dims)
    return total


def _grad_key(name: str) -> str:
    """Rename ``w_X -> g_X`` so the dict keys match orig's naming
    convention used by the kernels."""
    return "g_" + name[2:] if name.startswith("w_") else "g_" + name


def _alloc_dict_on_device(
    param_spec: ParamSpec,
    dims: Mapping[str, int],
    *,
    device: torch.device | str,
    role: str,
) -> dict[str, torch.Tensor]:
    """Allocate one tensor per ParamSpec entry on the *compute* device.

    Frozen tensors skip ``grad`` and ``opt_state`` allocation (mirrors
    :func:`_alloc_dict_on_host`)."""
    out: dict[str, torch.Tensor] = {}
    for t in param_spec.tensors:
        if t.frozen and role in ("grad", "opt_state"):
            continue
        dtype = {
            "compute": t.compute_dtype,
            "master": t.master_dtype,
            "grad": t.grad_dtype,
            "opt_state": t.opt_state_dtype,
        }[role]
        shape = t.shape(dims)
        key = _grad_key(t.name) if role == "grad" else t.name
        out[key] = torch.zeros(shape, dtype=dtype, device=device)
    return out


def _alloc_dict_on_host(
    param_spec: ParamSpec,
    dims: Mapping[str, int],
    *,
    role: str,
    backend: HostMemoryBackend,
) -> dict[str, torch.Tensor]:
    """Allocate one tensor per ParamSpec entry through the host
    backend. The backend decides where the memory comes from (local
    pinned, remote, etc.).

    Frozen tensors (``TensorSpec.frozen=True``) skip allocation for
    ``grad`` and ``opt_state`` roles — the engine doesn't accumulate
    grads or optimize them. The ``master`` (host source-of-truth) and
    ``compute`` (on-device) roles ARE allocated, since the forward
    still needs to read the frozen weight."""
    out: dict[str, torch.Tensor] = {}
    for t in param_spec.tensors:
        if t.frozen and role in ("grad", "opt_state"):
            continue
        dtype = {
            "compute": t.compute_dtype,
            "master": t.master_dtype,
            "grad": t.grad_dtype,
            "opt_state": t.opt_state_dtype,
        }[role]
        shape = t.shape(dims)
        key = _grad_key(t.name) if role == "grad" else t.name
        out[key] = backend.allocate_tensor(shape, dtype)
    return out


def _view_dict_from_spec_in_buffer(
    param_spec: ParamSpec,
    dims: Mapping[str, int],
    *,
    role: str,
    flat: torch.Tensor,  # uint8, on target device
    offset: int = 0,
) -> tuple[dict[str, torch.Tensor], int]:
    """Slice a uint8 ``flat`` buffer into a tensor-per-ParamSpec-entry
    dict. Returns ``(dict, bytes_used)``.

    Used to build the GPU param / grad / opt ring slots out of one
    contiguous allocation per ring.
    """
    out: dict[str, torch.Tensor] = {}
    cursor = offset
    for t in param_spec.tensors:
        if t.frozen and role in ("grad", "opt_state"):
            continue
        dtype = {
            "compute": t.compute_dtype,
            "master": t.master_dtype,
            "grad": t.grad_dtype,
            "opt_state": t.opt_state_dtype,
        }[role]
        shape = t.shape(dims)
        nelem = 1
        for s in shape:
            nelem *= s
        nbytes = nelem * dtype.itemsize
        if cursor + nbytes > flat.numel():
            raise ValueError(
                f"ring buffer too small: need {nbytes} at offset {cursor}, "
                f"buffer has {flat.numel()}"
            )
        key = (
            ("g_" + t.name[2:] if t.name.startswith("w_") else "g_" + t.name)
            if role == "grad"
            else t.name
        )
        out[key] = flat[cursor : cursor + nbytes].view(dtype).reshape(shape)
        cursor += nbytes
    return out, cursor - offset


@dataclass
class _OptStateBundle:
    """Host + device mirrors of one layer's optimizer state.

    AdamW has two tensors per param (``o_m``, ``o_v``); Muon has one
    (``o_momentum``). Driven by :class:`OptimizerStateSpec`.
    """

    host: dict[str, torch.Tensor] = field(default_factory=dict)
    # device mirror only exists while GPU opt-state ring is "swapped in"
    # (during ActiveModel.step()).
    device: dict[str, torch.Tensor] | None = None


def _alloc_opt_state_host(
    param_spec: ParamSpec,
    opt_spec: OptimizerStateSpec,
    dims: Mapping[str, int],
    *,
    backend: HostMemoryBackend,
) -> dict[str, torch.Tensor]:
    """Allocate host opt-state (``o_adam_m_<name>`` etc.) for all params in
    ``param_spec`` with the dtype from ``opt_spec``. Matches the name
    convention used by the kernels (``orig/dense_layer.py:344``).

    If ``opt_spec`` exposes ``per_param_state_tensors(p, dims)``
    (e.g. :class:`HybridStateSpec`), that is consulted per-parameter so
    only the applicable state tensors are allocated — no wasted bytes
    for e.g. Muon momentum on AdamW-classified params."""
    out: dict[str, torch.Tensor] = {}
    per_param_fn = getattr(opt_spec, "per_param_state_tensors", None)
    for p in param_spec.tensors:
        if p.frozen:
            continue
        shape = p.shape(dims)
        name_suffix = p.name[2:] if p.name.startswith("w_") else p.name
        tensors = (
            per_param_fn(p, dims) if per_param_fn is not None
            else opt_spec.tensors
        )
        for st in tensors:
            key = f"{st.name}_{name_suffix}"
            out[key] = backend.allocate_tensor(shape, st.dtype)
    return out


class BufferManager:
    """All host + device buffers the engine needs.

    Construction populates every host buffer and every GPU ring slot.
    After construction the engine only calls accessor methods
    (``gpu_param_slot``, ``host_param``, ``host_act_slot``, ...) and
    lifecycle methods (``prefetch_layer_params``,
    ``offload_layer_grads``, ``swap_to_optimizer_state``,
    ``restore_activation_ring``).

    Homogeneous vs heterogeneous
    ----------------------------
    Each ring is sized to the MAX across layer types of its per-slot
    byte budget. Smaller layers leave the tail of their slot unused.
    This lets us mix, e.g., a dense LlamaBlock with an MoE block in
    the same backbone without engine changes (see [DECISION 10]).
    """

    # --- the compute device ---
    device: torch.device
    # --- sizing + shapes ---
    working_set: WorkingSetConfig
    dims: Mapping[str, int]
    # --- per-layer specs (list indexed by layer position in backbone) ---
    layer_param_specs: list[ParamSpec]
    layer_schemas: list[ActivationSchema]
    # --- embed + head specs (optional) ---
    embed_param_spec: ParamSpec | None
    head_param_spec: ParamSpec | None
    # --- optimizer state spec (drives opt host/device allocations) ---
    opt_spec: OptimizerStateSpec | None

    # === HOST buffers (full-model mirror) ===
    host_params: list[dict[str, torch.Tensor]]  # per-layer
    host_grads: list[dict[str, torch.Tensor]]
    host_opt: list[_OptStateBundle]
    host_embed_params: dict[str, torch.Tensor]
    host_embed_grads: dict[str, torch.Tensor]
    host_embed_opt: _OptStateBundle
    host_head_params: dict[str, torch.Tensor]
    host_head_grads: dict[str, torch.Tensor]
    host_head_opt: _OptStateBundle
    host_act_buffer: torch.Tensor | None  # uint8, cudaHostRegistered

    # === GPU ring buffers ===
    gpu_param_ring: torch.Tensor  # uint8, (NP * max_param_bytes)
    gpu_grad_ring: torch.Tensor  # uint8
    gpu_act_ring: torch.Tensor  # uint8
    gpu_param_slot_views: list[tuple[int, int]]  # (offset, length) per slot
    gpu_grad_slot_views: list[tuple[int, int]]
    gpu_act_slot_views: list[tuple[int, int]]

    # === embed/head resident GPU buffers (no ring — they stay resident) ===
    gpu_embed_params: dict[str, torch.Tensor]
    gpu_embed_grads: dict[str, torch.Tensor]
    gpu_head_params: dict[str, torch.Tensor]
    gpu_head_grads: dict[str, torch.Tensor]

    # === KV context (shared across all layers, one window) ===
    kv_fwd: KVContextWindow
    kv_bwd_dk: torch.Tensor
    kv_bwd_dv: torch.Tensor

    # === transition table (residual-stream holder, one tensor per chunk) ===
    # Chunks are materialized on demand during fwd_bwd; we don't pre-
    # allocate a fixed (num_chunks, d_model) tensor because chunk count
    # varies per round. Instead, fwd_bwd allocates per-chunk via
    # ScratchPool-ish calls and stores them here.
    transitions: dict[int, torch.Tensor]  # chunk_id -> Tensor

    # === host-act-buffer cursor (advanced as we allocate per-(layer,
    #     chunk) home slots in a round) ===
    _host_act_cursor: int

    def __init__(
        self,
        *,
        working_set: WorkingSetConfig,
        dims: Mapping[str, int],
        layer_param_specs: Sequence[ParamSpec],
        layer_schemas: Sequence[ActivationSchema],
        embed_param_spec: ParamSpec | None = None,
        head_param_spec: ParamSpec | None = None,
        opt_spec: OptimizerStateSpec | None = None,
        device: torch.device | str = "cuda:0",
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        kv_dtype: torch.dtype = torch.bfloat16,
        host_backend: HostMemoryBackend | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.working_set = working_set
        self.dims = dict(dims)
        self.layer_param_specs = list(layer_param_specs)
        self.layer_schemas = list(layer_schemas)
        self.embed_param_spec = embed_param_spec
        self.head_param_spec = head_param_spec
        self.opt_spec = opt_spec
        self.host_backend = host_backend or default_host_backend()

        self.transitions = {}
        self._host_act_cursor = 0

        if n_kv_heads is None:
            n_kv_heads = int(self.dims["n_kv_heads"])
        if head_dim is None:
            head_dim = int(self.dims["head_dim"])

        # ---- Allocate host master params / grads / opt for every layer ----
        # All host-side allocation goes through self.host_backend so we
        # can swap in remote-memory backends later.
        self.host_params = []
        self.host_grads = []
        self.host_opt = []
        for ps in self.layer_param_specs:
            params = _alloc_dict_on_host(
                ps, self.dims, role="master", backend=self.host_backend
            )
            grads = _alloc_dict_on_host(
                ps, self.dims, role="grad", backend=self.host_backend
            )
            opt_b = _OptStateBundle()
            if opt_spec is not None:
                opt_b.host = _alloc_opt_state_host(
                    ps, opt_spec, self.dims, backend=self.host_backend
                )
            self.host_params.append(params)
            self.host_grads.append(grads)
            self.host_opt.append(opt_b)

        # embed
        self.host_embed_params = {}
        self.host_embed_grads = {}
        self.host_embed_opt = _OptStateBundle()
        if embed_param_spec is not None:
            self.host_embed_params = _alloc_dict_on_host(
                embed_param_spec, self.dims, role="master",
                backend=self.host_backend,
            )
            self.host_embed_grads = _alloc_dict_on_host(
                embed_param_spec, self.dims, role="grad",
                backend=self.host_backend,
            )
            if opt_spec is not None:
                self.host_embed_opt.host = _alloc_opt_state_host(
                    embed_param_spec, opt_spec, self.dims,
                    backend=self.host_backend,
                )

        # head
        self.host_head_params = {}
        self.host_head_grads = {}
        self.host_head_opt = _OptStateBundle()
        if head_param_spec is not None:
            self.host_head_params = _alloc_dict_on_host(
                head_param_spec, self.dims, role="master",
                backend=self.host_backend,
            )
            self.host_head_grads = _alloc_dict_on_host(
                head_param_spec, self.dims, role="grad",
                backend=self.host_backend,
            )
            if opt_spec is not None:
                self.host_head_opt.host = _alloc_opt_state_host(
                    head_param_spec, opt_spec, self.dims,
                    backend=self.host_backend,
                )

        # ---- Size the GPU param / grad rings (max across layer types) ----
        max_param_bytes = max(
            _byte_size_of_tensors(ps.tensors, self.dims, "compute")
            for ps in self.layer_param_specs
        )
        max_grad_bytes = max(
            _byte_size_of_tensors(ps.tensors, self.dims, "grad")
            for ps in self.layer_param_specs
        )

        NP = working_set.n_gpu_layers
        NG = working_set.n_gpu_grads

        self.gpu_param_ring = torch.zeros(
            NP * max_param_bytes, dtype=torch.uint8, device=self.device
        )
        self.gpu_grad_ring = torch.zeros(
            NG * max_grad_bytes, dtype=torch.uint8, device=self.device
        )
        self.gpu_param_slot_views = [
            (i * max_param_bytes, max_param_bytes) for i in range(NP)
        ]
        self.gpu_grad_slot_views = [
            (i * max_grad_bytes, max_grad_bytes) for i in range(NG)
        ]

        # ---- GPU activation ring ----
        self.gpu_act_ring = torch.zeros(
            working_set.gpu_act_buffer_size, dtype=torch.uint8, device=self.device
        )
        # Slot size = max across layer schemas at max_chunk_size.
        max_act_slot_bytes = max(
            schema.device_size_bytes(working_set.max_chunk_size, self.dims)
            for schema in self.layer_schemas
        )
        n_gpu_act_slots = (
            working_set.gpu_act_buffer_size // max_act_slot_bytes
            if max_act_slot_bytes > 0
            else 0
        )
        if n_gpu_act_slots == 0:
            raise RuntimeError(
                f"gpu_act_buffer_size ({working_set.gpu_act_buffer_size}) "
                f"too small for a single activation slot of "
                f"{max_act_slot_bytes} bytes"
            )
        self._max_act_slot_bytes = max_act_slot_bytes
        self.gpu_act_slot_views = [
            (i * max_act_slot_bytes, max_act_slot_bytes)
            for i in range(n_gpu_act_slots)
        ]
        self._n_gpu_act_slots = n_gpu_act_slots

        # ---- HARD INVARIANT: act buffer must hold ≥ 1 opt-state slot ----
        # The activation buffer is repurposed as the optimizer-state
        # ring during ``ActiveModel.step()`` (paper §3.3). If it
        # can't hold even a single layer's opt state, step() will
        # fail at swap_to_optimizer_state(). Catch this at init time
        # with a clear error message so callers can fix their sizing
        # before training starts.
        if opt_spec is not None and self.layer_param_specs:
            per_param_fn = getattr(opt_spec, "per_param_state_tensors", None)

            def _layer_opt_bytes(ps: ParamSpec) -> int:
                total = 0
                for p in ps.tensors:
                    if p.frozen:
                        continue
                    numel = p.numel(self.dims)
                    tensors = (
                        per_param_fn(p, self.dims)
                        if per_param_fn is not None
                        else opt_spec.tensors
                    )
                    for t in tensors:
                        total += numel * t.dtype.itemsize
                return total

            max_opt_bytes_per_layer = max(
                _layer_opt_bytes(ps) for ps in self.layer_param_specs
            )
            if working_set.gpu_act_buffer_size < max_opt_bytes_per_layer:
                raise RuntimeError(
                    f"gpu_act_buffer_size "
                    f"({working_set.gpu_act_buffer_size / (1 << 20):.1f} MiB) "
                    f"must be >= max per-layer optimizer state "
                    f"({max_opt_bytes_per_layer / (1 << 20):.1f} MiB) -- "
                    f"the activation buffer is repurposed as the opt-state "
                    f"ring during step(). Either increase "
                    f"gpu_act_buffer_size or use a smaller optimizer "
                    f"(e.g. bf16 opt-state dtype, or Muon instead of Adam)."
                )
            self._max_opt_bytes_per_layer = max_opt_bytes_per_layer
        else:
            self._max_opt_bytes_per_layer = 0

        # ---- Host activation buffer (routed through host backend) ----
        if working_set.host_act_buffer_size > 0:
            self.host_act_buffer = self.host_backend.allocate_tensor(
                (working_set.host_act_buffer_size,), torch.uint8
            )
        else:
            self.host_act_buffer = None

        # ---- Embed + head resident GPU buffers (small, always resident) ----
        self.gpu_embed_params = {}
        self.gpu_embed_grads = {}
        if embed_param_spec is not None:
            self.gpu_embed_params = _alloc_dict_on_device(
                embed_param_spec, self.dims, device=self.device, role="compute",
            )
            self.gpu_embed_grads = _alloc_dict_on_device(
                embed_param_spec, self.dims, device=self.device, role="grad",
            )
        self.gpu_head_params = {}
        self.gpu_head_grads = {}
        if head_param_spec is not None:
            self.gpu_head_params = _alloc_dict_on_device(
                head_param_spec, self.dims, device=self.device, role="compute",
            )
            self.gpu_head_grads = _alloc_dict_on_device(
                head_param_spec, self.dims, device=self.device, role="grad",
            )

        # ---- KV context ----
        context_window = max(working_set.max_seq_len, working_set.max_chunk_size)
        self.kv_fwd = KVContextWindow.create(
            max_context_tokens=context_window,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            device=self.device,
            dtype=kv_dtype,
        )
        # Back-compat alias for layers that read dk / dv as separate tensors.
        self.kv_bwd_dk = self.kv_fwd.dk
        self.kv_bwd_dv = self.kv_fwd.dv

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_gpu_act_slots(self) -> int:
        return self._n_gpu_act_slots

    def gpu_param_slot(
        self, slot_idx: int, layer_spec: ParamSpec
    ) -> dict[str, torch.Tensor]:
        """Return the per-name view dict for ring slot ``slot_idx`` at
        the layout implied by ``layer_spec`` (since layers have
        different layouts in a heterogeneous backbone)."""
        off, _length = self.gpu_param_slot_views[slot_idx]
        views, _ = _view_dict_from_spec_in_buffer(
            layer_spec, self.dims, role="compute",
            flat=self.gpu_param_ring, offset=off,
        )
        return views

    def gpu_grad_slot(
        self, slot_idx: int, layer_spec: ParamSpec
    ) -> dict[str, torch.Tensor]:
        off, _length = self.gpu_grad_slot_views[slot_idx]
        views, _ = _view_dict_from_spec_in_buffer(
            layer_spec, self.dims, role="grad",
            flat=self.gpu_grad_ring, offset=off,
        )
        return views

    def gpu_act_slot(
        self,
        slot_idx: int,
        schema: ActivationSchema,
        num_tokens: int,
    ) -> ActivationSlot:
        """Return the GPU activation slot at ring position ``slot_idx``,
        narrowed to ``num_tokens`` along each field's token_axis, with
        ALL fields (every tier) present — the GPU ring holds the
        computed state, not the saved state.
        """
        off, length = self.gpu_act_slot_views[slot_idx]
        # Build view at num_tokens=max_chunk_size first (full extents),
        # then narrow — the buffer was sized for max_chunk_size so the
        # fixed offsets stay valid.
        slot, _ = ActivationSlot.from_buffer(
            schema,
            level=schema.max_tier,
            num_tokens=self.working_set.max_chunk_size,
            dims=self.dims,
            buffer=self.gpu_act_ring[off : off + length],
            include_nonpersistent=True,
        )
        return slot.view_for(num_tokens, self.dims)

    def reset_host_act_cursor(self) -> None:
        """Call at the top of each gradient-accumulation round to reuse
        the host activation buffer from offset 0."""
        self._host_act_cursor = 0

    def host_act_slot(
        self,
        schema: ActivationSchema,
        num_tokens: int,
        level: int,
    ) -> tuple[ActivationSlot, int]:
        """Reserve the next contiguous slice of the host act buffer for
        one (layer, chunk) pair at save level ``level``.

        Returns ``(slot, bytes_used)``. Level < 0 returns
        ``(None, 0)``-equivalent handled by the engine's ``if level == -1``
        branch before calling here (we don't want to waste the offset
        math).
        """
        if self.host_act_buffer is None:
            raise RuntimeError(
                "No host act buffer (host_act_buffer_size=0). This is only "
                "valid when every (layer, chunk) is on-device — the engine "
                "should have taken the fast path without calling here."
            )
        need = schema.home_size_bytes(num_tokens, self.dims, level)
        if self._host_act_cursor + need > self.host_act_buffer.numel():
            raise RuntimeError(
                f"host act buffer exhausted: need {need} at cursor "
                f"{self._host_act_cursor}, buffer size "
                f"{self.host_act_buffer.numel()}"
            )
        flat = self.host_act_buffer[
            self._host_act_cursor : self._host_act_cursor + need
        ]
        slot, used = ActivationSlot.from_buffer(
            schema, level=level, num_tokens=num_tokens, dims=self.dims,
            buffer=flat, include_nonpersistent=False,
        )
        self._host_act_cursor += used
        return slot, used

    # ------------------------------------------------------------------
    # Prefetch / offload — these just do the tensor copies. Stream
    # management is the caller's job (see :class:`StreamBundle`).
    # ------------------------------------------------------------------

    def fetch_layer_params(
        self,
        layer_id: int,
        slot_idx: int,
        *,
        non_blocking: bool = True,
    ) -> None:
        """Copy host master params for ``layer_id`` into GPU ring slot
        ``slot_idx``. Casts master_dtype -> compute_dtype when they
        differ (handled implicitly by ``Tensor.copy_`` dtype promotion).
        """
        spec = self.layer_param_specs[layer_id]
        gpu = self.gpu_param_slot(slot_idx, spec)
        host = self.host_params[layer_id]
        for name, dev_t in gpu.items():
            dev_t.copy_(host[name], non_blocking=non_blocking)

    def fetch_layer_grads(
        self,
        layer_id: int,
        slot_idx: int,
        *,
        non_blocking: bool = True,
    ) -> None:
        spec = self.layer_param_specs[layer_id]
        gpu = self.gpu_grad_slot(slot_idx, spec)
        host = self.host_grads[layer_id]
        for name, dev_t in gpu.items():
            dev_t.copy_(host[name], non_blocking=non_blocking)

    def offload_layer_grads(
        self,
        layer_id: int,
        slot_idx: int,
        *,
        non_blocking: bool = True,
    ) -> None:
        spec = self.layer_param_specs[layer_id]
        gpu = self.gpu_grad_slot(slot_idx, spec)
        host = self.host_grads[layer_id]
        for name, dev_t in gpu.items():
            host[name].copy_(dev_t, non_blocking=non_blocking)

    def offload_layer_params(
        self,
        layer_id: int,
        slot_idx: int,
        *,
        non_blocking: bool = True,
    ) -> None:
        """Mirror of :meth:`fetch_layer_params` — used during
        :meth:`ActiveModel.step` to write updated master params back
        to the host copy."""
        spec = self.layer_param_specs[layer_id]
        gpu = self.gpu_param_slot(slot_idx, spec)
        host = self.host_params[layer_id]
        for name, dev_t in gpu.items():
            host[name].copy_(dev_t, non_blocking=non_blocking)

    # ------------------------------------------------------------------
    # Optimizer state ring — repurposes the GPU act ring during step().
    # ------------------------------------------------------------------

    def swap_to_optimizer_state(
        self, n_gpu_opt_layers: int
    ) -> list[dict[str, torch.Tensor]]:
        """Carve ``n_gpu_opt_layers`` opt-state slots out of the GPU
        activation ring. Returns a list of view dicts (one per slot).

        Paper §3.3: after fwd/bwd completes, the activation ring is
        free, so we reuse it to stage optimizer state for the step.
        The buffer is the SAME underlying storage as
        ``self.gpu_act_ring``; callers must not touch the ring views
        again until :meth:`restore_activation_ring` is called.
        """
        if self.opt_spec is None:
            raise RuntimeError(
                "swap_to_optimizer_state called but opt_spec is None"
            )
        # Byte size per opt-state slot = biggest layer's total opt-state.
        per_param_fn = getattr(self.opt_spec, "per_param_state_tensors", None)

        def _opt_bytes(ps: ParamSpec) -> int:
            s = 0
            for p in ps.tensors:
                if p.frozen:
                    continue
                numel = p.numel(self.dims)
                tensors = (
                    per_param_fn(p, self.dims)
                    if per_param_fn is not None
                    else self.opt_spec.tensors
                )
                for t in tensors:
                    s += numel * t.dtype.itemsize
            return s
        max_opt_bytes = max(_opt_bytes(ps) for ps in self.layer_param_specs)
        if n_gpu_opt_layers * max_opt_bytes > self.gpu_act_ring.numel():
            raise RuntimeError(
                f"opt-state ring ({n_gpu_opt_layers} * {max_opt_bytes}) "
                f"exceeds gpu_act_buffer ({self.gpu_act_ring.numel()})"
            )
        slots: list[dict[str, torch.Tensor]] = []
        for i in range(n_gpu_opt_layers):
            # Build view dict for each layer's opt state using layer_0's
            # spec as a template — since all layers share a schema for
            # the homogeneous case. Heterogeneous callers get the layer-
            # specific view through :meth:`gpu_opt_slot`.
            off = i * max_opt_bytes
            # Leave slots as plain memory; the caller (ActiveModel.step)
            # iterates layers and calls :meth:`gpu_opt_slot(i, layer_id)`
            # to get the properly-shaped view.
            slots.append({"_raw_offset": off, "_raw_length": max_opt_bytes})  # type: ignore[dict-item]
        self._gpu_opt_slot_views = [
            (s["_raw_offset"], s["_raw_length"]) for s in slots  # type: ignore[arg-type]
        ]
        self._max_opt_bytes = max_opt_bytes
        self._in_opt_mode = True
        return []  # caller uses gpu_opt_slot(slot_idx, layer_id) instead

    def gpu_opt_slot(
        self, slot_idx: int, layer_id: int
    ) -> dict[str, torch.Tensor]:
        """Get the per-(opt-state-name, param-name) view for a given
        layer at opt-ring slot ``slot_idx``. Only valid between
        :meth:`swap_to_optimizer_state` and :meth:`restore_activation_ring`.
        """
        if not getattr(self, "_in_opt_mode", False):
            raise RuntimeError(
                "gpu_opt_slot called outside swap_to_optimizer_state() window"
            )
        off, length = self._gpu_opt_slot_views[slot_idx]
        flat = self.gpu_act_ring[off : off + length]
        ps = self.layer_param_specs[layer_id]
        per_param_fn = getattr(self.opt_spec, "per_param_state_tensors", None)
        out: dict[str, torch.Tensor] = {}
        cursor = 0
        for p in ps.tensors:
            if p.frozen:
                continue
            nm_suffix = p.name[2:] if p.name.startswith("w_") else p.name
            tensors = (
                per_param_fn(p, self.dims)
                if per_param_fn is not None
                else self.opt_spec.tensors  # type: ignore[union-attr]
            )
            for t in tensors:
                nelem = p.numel(self.dims)
                nbytes = nelem * t.dtype.itemsize
                if cursor + nbytes > length:
                    raise RuntimeError(
                        f"opt slot {slot_idx} underflow for layer {layer_id}"
                    )
                key = f"{t.name}_{nm_suffix}"
                out[key] = (
                    flat[cursor : cursor + nbytes]
                    .view(t.dtype)
                    .reshape(p.shape(self.dims))
                )
                cursor += nbytes
        return out

    def fetch_layer_opt(
        self,
        layer_id: int,
        slot_idx: int,
        *,
        non_blocking: bool = True,
    ) -> None:
        gpu = self.gpu_opt_slot(slot_idx, layer_id)
        host = self.host_opt[layer_id].host
        for name, dev_t in gpu.items():
            dev_t.copy_(host[name], non_blocking=non_blocking)

    def offload_layer_opt(
        self,
        layer_id: int,
        slot_idx: int,
        *,
        non_blocking: bool = True,
    ) -> None:
        gpu = self.gpu_opt_slot(slot_idx, layer_id)
        host = self.host_opt[layer_id].host
        for name, dev_t in gpu.items():
            host[name].copy_(dev_t, non_blocking=non_blocking)

    def restore_activation_ring(self) -> None:
        """Leave opt-state mode; the GPU act ring is now addressable as
        activation slots again."""
        self._in_opt_mode = False
        self._gpu_opt_slot_views = []

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Release host buffers through the host backend. Safe to call
        once; subsequent calls are no-ops (backend handles dedup)."""
        for lst in (self.host_params, self.host_grads):
            for d in lst:
                for t in d.values():
                    self.host_backend.release(t)
        for bundle in self.host_opt:
            for t in bundle.host.values():
                self.host_backend.release(t)
        for d in (self.host_embed_params, self.host_embed_grads,
                  self.host_head_params, self.host_head_grads):
            for t in d.values():
                self.host_backend.release(t)
        for t in self.host_embed_opt.host.values():
            self.host_backend.release(t)
        for t in self.host_head_opt.host.values():
            self.host_backend.release(t)
        if self.host_act_buffer is not None:
            self.host_backend.release(self.host_act_buffer)
            self.host_act_buffer = None


__all__ = [
    "BufferManager",
    "KVContextWindow",
    "ScratchPool",
]
