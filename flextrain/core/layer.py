"""Layer contract and supporting types.

Three Protocols -- :class:`Layer`, :class:`InputLayer`, :class:`OutputLayer` --
define the only surface the engine consumes. Everything that was a layer
concern in ``orig/`` but is really engine bookkeeping (buffer slicing, send/
fetch, create/load/save/step, ephemeral workspace) moves engine-side.

Scope of the layer contract
---------------------------
A layer must provide:

* ``schema``             : what activations it declares (see
                           :class:`ActivationSchema`).
* ``param_spec``         : what parameters / grads / optimizer state it owns
                           (see :class:`ParamSpec`). The engine derives
                           create/load/save/step from this.
* ``forward``            : one chunk's compute, writing into a slot.
* ``forward_recompute``  : fill in the higher-tier fields that weren't saved,
                           in-place into the slot. Checks ``slot.has(name)``
                           to decide what to recompute.
* ``backward``           : one chunk's backward pass.
* ``compute_cost``       : total fwd FLOPs + per-tier avoided-recompute FLOPs
                           for the DP solver.

A layer does NOT allocate buffers, compute sizes, move bytes over PCIe,
manage optimizer state, open CUDA streams, or know about chunk-view
re-slicing.

``ParamSpec`` vs. ``ActivationSchema``
--------------------------------------
``ParamSpec`` is the STATIC weight set (loaded from an HF checkpoint,
persisted through training). ``ActivationSchema`` is the DYNAMIC per-chunk
state that lives in the AdaWS working set. Keeping them disjoint is what lets
the engine drive create/load/save/step without touching the compute path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import torch

from .activation_schema import ActivationSchema, ActivationSlot


# ---------------------------------------------------------------------------
# Parameter specs. The engine uses these to allocate parameter/grad/opt-state
# buffers on both host and device, and to drive HF-safetensors load/export.
# ---------------------------------------------------------------------------


ParamShapeFn = Callable[[Mapping[str, int]], tuple[int, ...]]
"""``(model_dims) -> shape``. Unlike activation shapes, parameter shapes do
not depend on ``num_tokens``."""


@dataclass(frozen=True)
class TensorSpec:
    """One parameter tensor's shape + per-role dtypes.

    A tensor can appear in training with up to four different dtypes:

    * ``master_dtype``      -- the authoritative copy held in host memory
                               across training; usually fp32 for numeric
                               stability during the optimizer step, bf16
                               otherwise.
    * ``compute_dtype``     -- the dtype the device buffer holds during the
                               forward / backward pass; typically bf16.
    * ``grad_dtype``        -- accumulated gradient dtype; bf16 or fp32.
    * ``opt_state_dtype``   -- moments / Muon orthogonalization workspace
                               dtype; fp32 by default.

    These default to ``torch.bfloat16`` / ``torch.float32`` so most layers can
    pass a single ``dtype=`` for compatibility with ``orig``. Individual
    architectures override per-tensor (e.g. embedding master in bf16 but
    optimizer state in fp32).

    The engine decides when to cast between master and compute: on parameter
    prefetch (host->device) it casts master_dtype -> compute_dtype if they
    differ, and on outbound (device->host for optimizer step) it casts
    grad_dtype -> master_dtype for the update.
    """

    name: str
    shape_fn: ParamShapeFn
    compute_dtype: torch.dtype
    master_dtype: torch.dtype | None = None  # defaults to compute_dtype
    grad_dtype: torch.dtype | None = None  # defaults to compute_dtype
    opt_state_dtype: torch.dtype | None = None  # defaults to torch.bfloat16
    # Optimizer hint for hybrid optimizers (e.g. HybridMuonAdamW). Plain
    # single-algorithm optimizers ignore this field and update every
    # parameter with their rule. Valid values: "adamw", "muon". If None,
    # the hybrid optimizer auto-infers: 2-D weights that look like
    # dense projections ("w_q", "w_1", "w_up", ...) → "muon"; everything
    # else (norms, embeddings, head, router, 1-D biases, per-expert
    # routers) → "adamw". Explicit setting always wins.
    optimizer: str | None = None
    # If True, this parameter is frozen — the engine still allocates
    # the master copy (needed for forward), but skips grad and optimizer
    # state allocation. Used by LoRA, where the base weights are frozen
    # and only the (much smaller) A/B delta tensors are trained. Frozen
    # tensors are silently skipped by the optimizer step too.
    frozen: bool = False

    # Backwards-compat alias so ``TensorSpec(name, shape, dtype)`` keeps working
    # for block code that doesn't care about mixed precision.
    @classmethod
    def simple(
        cls, name: str, shape_fn: ParamShapeFn, dtype: torch.dtype
    ) -> "TensorSpec":
        return cls(name=name, shape_fn=shape_fn, compute_dtype=dtype)

    def __post_init__(self) -> None:
        # Fill role-defaulted dtypes. Use object.__setattr__ because the
        # dataclass is frozen.
        object.__setattr__(
            self, "master_dtype", self.master_dtype or self.compute_dtype
        )
        object.__setattr__(
            self, "grad_dtype", self.grad_dtype or self.compute_dtype
        )
        object.__setattr__(
            self, "opt_state_dtype", self.opt_state_dtype or torch.bfloat16
        )

    # --- read-only accessors ---

    @property
    def dtype(self) -> torch.dtype:
        """Back-compat: the compute-time dtype."""
        return self.compute_dtype

    def shape(self, dims: Mapping[str, int]) -> tuple[int, ...]:
        return self.shape_fn(dims)

    def numel(self, dims: Mapping[str, int]) -> int:
        n = 1
        for s in self.shape(dims):
            n *= s
        return n

    # Byte-size queries, one per role:
    def compute_byte_size(self, dims: Mapping[str, int]) -> int:
        return self.numel(dims) * self.compute_dtype.itemsize

    def master_byte_size(self, dims: Mapping[str, int]) -> int:
        return self.numel(dims) * self.master_dtype.itemsize

    def grad_byte_size(self, dims: Mapping[str, int]) -> int:
        return self.numel(dims) * self.grad_dtype.itemsize

    def opt_state_byte_size(self, dims: Mapping[str, int]) -> int:
        return self.numel(dims) * self.opt_state_dtype.itemsize

    # Default ``byte_size`` = compute dtype (most asked-for). Callers who
    # specifically need master / grad / opt sizes should use the role helpers
    # above.
    def byte_size(self, dims: Mapping[str, int]) -> int:
        return self.compute_byte_size(dims)


@dataclass(frozen=True)
class ParamSpec:
    """All parameters one layer owns.

    The engine derives from this:

    * host + device parameter buffers (bf16, see ``tensor.dtype``)
    * gradient buffers (same shapes, grad-dtype per :meth:`grad_dtype_of`)
    * AdamW optimizer state (two tensors per param, fp32 by default)
    * Muon optimizer state (one tensor per param + transient ortho workspace)
    * HF-safetensors load/export via a per-architecture name map (see
      ``io/arch/<family>.py``).
    """

    tensors: tuple[TensorSpec, ...]

    def __post_init__(self) -> None:
        names = [t.name for t in self.tensors]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate param names: {dupes}")

    def byte_size(
        self,
        dims: Mapping[str, int],
        role: str = "compute",
    ) -> int:
        """Total bytes across all tensors for a given role.

        ``role`` is one of ``"compute"`` (default), ``"master"``, ``"grad"``,
        ``"opt_state"`` -- corresponding to the ``TensorSpec`` per-role dtype
        field. The engine queries this to size its GPU / host buffers.

        The AdamW optimizer stores TWO ``opt_state``-dtype tensors per
        parameter (m + v); callers that care should multiply by 2 for Adam
        or 1 for Muon. Keeping that multiplier out of here lets us stay
        optimizer-agnostic.
        """
        if role == "compute":
            return sum(t.compute_byte_size(dims) for t in self.tensors)
        if role == "master":
            return sum(t.master_byte_size(dims) for t in self.tensors)
        if role == "grad":
            return sum(t.grad_byte_size(dims) for t in self.tensors)
        if role == "opt_state":
            return sum(t.opt_state_byte_size(dims) for t in self.tensors)
        raise ValueError(
            f"unknown role {role!r}; expected compute|master|grad|opt_state"
        )

    def names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tensors)

    def get(self, name: str) -> TensorSpec:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(name)

    @classmethod
    def merge(cls, specs: Sequence["ParamSpec"]) -> "ParamSpec":
        """Concatenate block-level specs into a layer-level one. Errors on
        name collisions."""
        tensors: list[TensorSpec] = []
        for s in specs:
            tensors.extend(s.tensors)
        return cls(tensors=tuple(tensors))


# ---------------------------------------------------------------------------
# Compute cost: total fwd FLOPs + per-tier avoided-recompute FLOPs. Consumed
# by the DP solver (see core/save_level.py).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeCost:
    """Per-(chunk, layer) FLOP accounting for the save-level DP.

    Parameters
    ----------
    total_fwd_flops
        FLOPs for the full forward pass of this chunk-on-this-layer.
    avoided_recompute_flops
        Tuple of length ``schema.max_tier + 1``. Element ``L`` is the FLOPs we
        DON'T have to redo in backward if we saved at tier ``L``. MUST be
        monotone non-decreasing (saving more saves more).

    Derivation
    ----------
    Each block type knows its own FLOPs per-field. The layer's
    :meth:`Layer.compute_cost` sums block contributions via :meth:`ComputeCost.sum`
    so the per-level dict is derived, not hand-written.
    """

    total_fwd_flops: int
    avoided_recompute_flops: tuple[int, ...]

    def __post_init__(self) -> None:
        for i in range(1, len(self.avoided_recompute_flops)):
            if (
                self.avoided_recompute_flops[i]
                < self.avoided_recompute_flops[i - 1]
            ):
                raise ValueError(
                    f"avoided_recompute_flops must be monotone non-decreasing, "
                    f"got {self.avoided_recompute_flops}"
                )
        if (
            self.avoided_recompute_flops
            and self.avoided_recompute_flops[-1] > self.total_fwd_flops
        ):
            raise ValueError(
                "avoided_recompute_flops cannot exceed total_fwd_flops"
            )

    @classmethod
    def sum(cls, parts: Sequence["ComputeCost"], max_tier: int) -> "ComputeCost":
        """Sum block-level costs into a layer-level cost. All parts must have
        the same ``max_tier + 1`` length."""
        k = max_tier + 1
        total = 0
        avoided = [0] * k
        for p in parts:
            if len(p.avoided_recompute_flops) != k:
                raise ValueError(
                    f"part has {len(p.avoided_recompute_flops)} tiers, expected {k}"
                )
            total += p.total_fwd_flops
            for i in range(k):
                avoided[i] += p.avoided_recompute_flops[i]
        return cls(total_fwd_flops=total, avoided_recompute_flops=tuple(avoided))


# ---------------------------------------------------------------------------
# ChunkMeta + LayerContext: what forward/backward see per-call.
# ---------------------------------------------------------------------------


@dataclass
class ChunkMeta:
    """Per-chunk scheduling metadata supplied by the engine.

    Mirrors the ``chunk_metadata`` dict that ``orig/active_model.py`` built
    in ``make_chunk_metadata`` (``dense_layer.py:707``). Keeping the same
    field names to minimize porting surface.
    """

    # number of "new" tokens in this chunk
    total_q: int
    # number of K tokens visible during attention (new + prior from kv cache)
    total_k: int
    # per-sequence new lengths (host list; no device tensor in orig)
    seq_lens_host: Sequence[int]
    # per-sequence positions within each sequence, shape (total_q, 1) int32
    seq_positions: torch.Tensor
    # cu_seqlens-style offsets for flash-attn varlen (device int32 tensors)
    q_seq_offsets: torch.Tensor
    k_seq_offsets: torch.Tensor
    q_seq_lens: torch.Tensor
    k_seq_lens: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    # per-sequence prior context length (host copies used for KV placement)
    prior_seq_lens_host: Sequence[int]
    prior_seq_offsets_host: Sequence[int]
    # extensible -- some ops (MoE router, sliding window) need more
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        seq_lens: Sequence[int],
        seq_positions: Sequence[int],
        prior_seq_lens: Sequence[int],
        prior_seq_offsets: Sequence[int],
        *,
        device: torch.device | str,
    ) -> "ChunkMeta":
        """Mirrors ``dense_layer.py:707`` ``make_chunk_metadata``."""
        import numpy as np

        num_seqs = len(seq_lens)
        assert (
            len(prior_seq_lens) == num_seqs
        ), "num_prior_seqs must match num_seqs"

        total_q = int(sum(seq_lens))
        total_k = int(sum(prior_seq_lens)) + total_q

        q_seq_offsets = torch.tensor(
            [0] + list(np.cumsum(seq_lens)), dtype=torch.int32, device=device
        )
        k_seq_offsets = torch.tensor(
            [0]
            + list(np.cumsum(np.array(seq_lens) + np.array(prior_seq_lens))),
            dtype=torch.int32,
            device=device,
        )
        q_seq_lens = torch.tensor(
            list(seq_lens), dtype=torch.int32, device=device
        )
        k_seq_lens = torch.tensor(
            [prior_seq_lens[i] + seq_lens[i] for i in range(num_seqs)],
            dtype=torch.int32,
            device=device,
        )
        max_seqlen_q = int(max(seq_lens))
        max_seqlen_k = int(
            max(prior_seq_lens[i] + seq_lens[i] for i in range(num_seqs))
        )
        seq_positions_t = torch.tensor(
            list(seq_positions), dtype=torch.int32, device=device
        ).reshape(-1, 1)

        return cls(
            total_q=total_q,
            total_k=total_k,
            seq_lens_host=list(seq_lens),
            seq_positions=seq_positions_t,
            q_seq_offsets=q_seq_offsets,
            k_seq_offsets=k_seq_offsets,
            q_seq_lens=q_seq_lens,
            k_seq_lens=k_seq_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            prior_seq_lens_host=list(prior_seq_lens),
            prior_seq_offsets_host=list(prior_seq_offsets),
        )

    def as_orig_dict(self) -> dict:
        """Adapter: emit the dict shape orig layers expect. Used during the
        port so ``orig.TransformerLayer.forward(..., chunk_metadata=...)``
        can be called with one of our ChunkMetas for parity testing."""
        return {
            "seq_lens_host": list(self.seq_lens_host),
            "prior_seq_lens_host": list(self.prior_seq_lens_host),
            "prior_seq_offsets_host": list(self.prior_seq_offsets_host),
            "total_q": self.total_q,
            "total_k": self.total_k,
            "seq_positions": self.seq_positions,
            "q_seq_offsets": self.q_seq_offsets,
            "k_seq_offsets": self.k_seq_offsets,
            "q_seq_lens": self.q_seq_lens,
            "k_seq_lens": self.k_seq_lens,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_k": self.max_seqlen_k,
        }


ScratchFn = Callable[[tuple[int, ...], torch.dtype], torch.Tensor]


@dataclass(frozen=True)
class MoEChunkConfig:
    """Declaration that a layer needs per-chunk MoE scratch tensors
    (``token_index_mapping`` and ``expert_counts_host``).

    The engine inspects each backbone layer at init time; if any layer
    has ``.moe_chunk_config is not None``, the engine allocates the
    matching per-(layer_id, chunk_id) tensors and stashes them in
    ``ChunkMeta.extra`` before fwd/bwd runs. MoE layers read them via
    ``chunk.extra["moe_token_index_mapping"][layer_id]`` and
    ``chunk.extra["moe_expert_counts_host"][layer_id]``.

    Centralized here (not in the layer implementations) so the engine
    can allocate without a circular import on ``MoESwiGLUFFN``.
    """

    num_experts: int
    top_k: int


@dataclass
class BackwardIntermediates:
    """Per-projection ``(X, dY)`` pairs and any layer-internal cache the
    weight-grad pass needs to consume. Produced by
    :meth:`Layer.backward_dgrad`, consumed by :meth:`Layer.backward_wgrad`
    (or by a LoRA wrapper's fast accumulate path).

    Why a typed payload
    -------------------
    Splitting ``backward()`` into ``backward_dgrad()`` (returns ``dx``)
    and ``backward_wgrad()`` (accumulates ``grads[g_*]``) lets layers
    that wrap a frozen base — LoRA in particular — call dgrad alone and
    skip the expensive ``X^T @ dY`` Wgrad matmul on every projection.
    LoRA accumulates its own ``dA, dB`` directly from ``(X, dY)`` via
    rank-r matmuls, never materializing ``dW`` for the frozen base.

    Contract
    --------
    * ``proj_inputs_and_grads[name]`` = ``(X, dY)`` for projection
      ``name`` (a key in the layer's ``ParamSpec`` / ``weights`` /
      ``grads`` dicts -- e.g. ``"w_q"``, ``"w_o"``, ``"w_1"``,
      ``"w_up"``). The Wgrad matmul a layer would have run is
      ``grads[f"g_{name[2:]}"].addmm_(X.T, dY)``; LoRA replaces this
      with rank-r matmuls on the same ``(X, dY)``.
    * For MoE projections the ``(X, dY)`` tensors are 3-D
      ``(num_experts, T_e, dim)`` -- the same shape contract LoRA
      already handles via ``bmm`` (see ``lora_wrapper.py``).
    * ``aux`` carries layer-internal state -- e.g. a recomputed
      RMSNorm output that ``backward_wgrad`` needs as the left operand
      of a Wgrad matmul. Opaque to LoRA.

    Lifetime
    --------
    Short-lived: the engine calls dgrad immediately followed by wgrad
    on the compute stream, so an intermediates instance lives only for
    one (layer, chunk) backward iteration. No cross-layer accumulation,
    no extra long-lived GPU residency vs. today's monolithic backward.
    """

    proj_inputs_and_grads: dict[str, tuple[torch.Tensor, torch.Tensor]] = (
        field(default_factory=dict)
    )
    aux: dict[str, Any] = field(default_factory=dict)

    def __getitem__(
        self, name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.proj_inputs_and_grads[name]

    def __contains__(self, name: str) -> bool:
        return name in self.proj_inputs_and_grads

    def names(self) -> tuple[str, ...]:
        return tuple(self.proj_inputs_and_grads.keys())


@dataclass
class LayerContext:
    """Engine-provided per-call resources. Replaces the ad-hoc ``fwd_context``
    / ``bwd_context`` dicts that ``orig/`` passed around.

    Parameters
    ----------
    scratch
        ``(shape, dtype) -> Tensor`` allocator for ephemeral workspace
        (``x_temp``, per-expert slices, dQ/dK/dV accumulators, etc.). The
        engine owns the backing pool and frees on context exit -- layers do
        NOT allocate their own scratch with ``torch.empty``.
    kv_cache
        The attention K/V ring window. Opaque to this module; attention
        blocks cast to the concrete type defined in
        ``flextrain/engine/buffers.py``.
    stream
        The primary compute stream for this call.
    secondary_stream
        Optional second compute stream (used by MoE to overlap shared-expert
        with routed-expert work).
    """

    scratch: ScratchFn
    kv_cache: Any  # KVContextWindow | None -- resolved in engine/buffers.py
    stream: torch.cuda.Stream
    secondary_stream: torch.cuda.Stream | None = None
    # Total active tokens in the gradient-accumulation step (across all
    # rounds/chunks). MoE layers use this for the load-balance auxiliary
    # loss coefficient. None for non-MoE workloads.
    total_tokens_per_step: int | None = None


# ---------------------------------------------------------------------------
# Layer Protocols. Three variants:
#
#   Layer        -- the backbone block (dense, MoE, and everything in between)
#   InputLayer   -- the token embedding (runs once per chunk, no slot)
#   OutputLayer  -- the LM head (fuses fwd + loss + bwd to avoid materializing
#                   full (tokens, vocab) logits)
# ---------------------------------------------------------------------------


@runtime_checkable
class Layer(Protocol):
    """A transformer backbone layer.

    Attributes ``schema`` and ``param_spec`` must be set at ``__init__`` time
    and are read by the engine before the first forward.

    Layers MAY optionally declare a ``moe_chunk_config: MoEChunkConfig``
    attribute to request per-chunk MoE scratch tensors from the engine.
    The engine probes each layer via
    ``getattr(layer, "moe_chunk_config", None)`` at init time; layers
    that don't need MoE scratch simply omit the attribute.
    """

    layer_id: int
    schema: ActivationSchema
    param_spec: ParamSpec

    def forward(
        self,
        x: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        slot: ActivationSlot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        """One chunk forward. Writes declared activations into ``slot`` and
        returns the residual-added output tensor (shape ``(num_tokens,
        d_model)``)."""
        ...

    def forward_recompute(
        self,
        slot: ActivationSlot,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> None:
        """Fill in the activation fields that weren't saved.

        The engine has already populated ``slot`` with all fields at
        ``slot.level`` (the save level this chunk/layer was configured with).
        Higher-tier fields appear as unset -- layers check with
        ``slot.has(name)`` and ``slot.set(name, ...)`` rather than dict-key
        introspection.
        """
        ...

    def backward(
        self,
        dx: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot: ActivationSlot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        """One chunk backward. Accumulates param gradients into ``grads``
        in-place. Returns ``dx`` for the preceding layer (same shape as
        ``x``).

        Layers MAY also implement the optional split pair below
        (:meth:`backward_dgrad` + :meth:`backward_wgrad`). When both are
        implemented, the canonical pattern is for ``backward()`` to be
        a delegating shim: ``upstream_dx, inter = backward_dgrad(...);
        backward_wgrad(inter, ...); return upstream_dx``. This keeps
        zero-behavior-change for all current callers (engine, parity
        benches, tests) while enabling the LoRA fast-path consumer in
        :mod:`flextrain.engine.active_model` to call dgrad alone and
        skip the per-projection Wgrad matmuls for frozen base weights.
        See ``docs/lora_fast_backward.md`` for the contract.
        """
        ...

    # ------------------------------------------------------------------
    # Optional: split backward into dgrad / wgrad. Phase 1 of the LoRA
    # fast-backward refactor (docs/lora_fast_backward.md). Layers that
    # implement these two methods AND have ``backward()`` delegate to
    # them gain the ability to skip Wgrad on a per-projection basis
    # (LoRA does this for every wrapped projection). Layers that don't
    # implement them keep using the monolithic ``backward()`` -- the
    # engine falls back automatically in that case.
    # ------------------------------------------------------------------

    def backward_dgrad(
        self,
        dx: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot: ActivationSlot,
        ctx: LayerContext,
        *,
        skip_target_names: frozenset[str] = frozenset(),
    ) -> tuple[torch.Tensor, BackwardIntermediates]:
        """Compute dL/dx (the upstream gradient) and return everything
        :meth:`backward_wgrad` will need to accumulate dL/dW.

        Returns ``(upstream_dx, intermediates)``.

        ``intermediates`` is a :class:`BackwardIntermediates` carrying
        per-projection ``(X, dY)`` pairs (``proj_inputs_and_grads``) so
        a LoRA wrapper can compute ``dA, dB`` directly via rank-r
        matmuls without materializing the full ``dW = X^T @ dY`` for
        the frozen base, plus opaque layer-internal state in ``aux``
        (e.g. recomputed RMSNorm outputs).

        Side effects on ``grads``
        -------------------------
        Some Wgrads are accumulated INLINE during dgrad in today's
        block implementations -- e.g. ``g_o`` (attention output proj)
        and ``g_2`` (FFN down-proj) in Llama, plus 1-D parameter grads
        like RMSNorm gain (``g_attn_norm`` etc.) and attention biases
        (``g_b_q`` etc.). These are still accumulated here.

        For a 2-D matmul projection whose name appears in
        ``skip_target_names``, the inline ``addmm`` is skipped and the
        ``(X, dY)`` pair is stashed into
        ``intermediates.proj_inputs_and_grads[name]`` so the LoRA
        wrapper can pick it up. 1-D parameter grads (norms, biases)
        are never LoRA targets and always accumulate normally.

        ``skip_target_names`` defaults to ``frozenset()`` -- full FT
        callers don't pass anything and behavior is identical to
        today's monolithic backward.
        """
        ...

    def backward_wgrad(
        self,
        intermediates: BackwardIntermediates,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot: ActivationSlot,
        ctx: LayerContext,
        *,
        skip_target_names: frozenset[str] = frozenset(),
    ) -> None:
        """Accumulate the deferred dL/dW into ``grads`` -- the Wgrads
        that need a recomputed RMSNorm output as their left operand
        and so couldn't run inline during ``backward_dgrad``.

        ``skip_target_names`` works the same way as in
        :meth:`backward_dgrad`: a 2-D projection whose name is in the
        set has its ``addmm`` skipped and its ``(X, dY)`` stashed in
        ``intermediates.proj_inputs_and_grads[name]`` so a LoRA
        wrapper can compute ``dA, dB`` from the same ``(X, dY)``.

        ``slot`` is passed through so the layer can reach activations
        the block-level callees still read from ``slot.aux`` (e.g.
        Llama's ``slot.aux["bwd_dq"]``).
        """
        ...

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        """Return forward FLOPs + per-tier avoided-recompute FLOPs for this
        chunk. Consumed by :func:`flextrain.core.save_level.build_dp_tables`."""
        ...


@runtime_checkable
class InputLayer(Protocol):
    """The token-embedding layer.

    Runs once per chunk at the start of forward. It has no activation slot
    because its "activation" is the ``token_ids`` tensor, already owned by
    :class:`ChunkMeta`. Its schema is still present (for symmetry with
    :class:`Layer`) but typically has ``max_tier=0`` and no fields.
    """

    schema: ActivationSchema
    param_spec: ParamSpec

    def forward(
        self,
        token_ids: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> torch.Tensor:
        """Return the embedded token tensor, shape ``(num_tokens, d_model)``."""
        ...

    def backward(
        self,
        dx: torch.Tensor,
        token_ids: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> None:
        """Accumulate embedding-table gradients from ``dx``. Returns no
        upstream gradient -- embedding is the first layer."""
        ...

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        ...


@dataclass
class LossStats:
    """Per-chunk loss bookkeeping returned by :class:`OutputLayer`.

    Fields mirror the side-effects orig's head wrote into
    ``chunk_metadata`` (see ``orig/awsm_transformer/head.py:149-151``)
    so the engine can reproduce orig's per-Sequence writeback at
    ``orig/active_model.py:1388-1390`` byte-for-byte.

    Attributes
    ----------
    per_token_loss
        ``(num_tokens,)`` fp32 tensor of per-token CE loss. The engine
        copies slices of this into each :class:`Sequence` object's
        ``per_token_loss`` buffer.
    next_prediction
        ``(num_tokens,)`` int64 tensor with argmax of the softmax. Orig
        also records this in chunk_metadata; we preserve for parity and
        for diagnostics (accuracy metrics).
    next_prediction_prob
        ``(num_tokens,)`` fp32 tensor of the argmax probability.
    token_count
        Convenience count (equals ``per_token_loss.numel()``) for
        per-round averaging.
    """

    per_token_loss: torch.Tensor
    next_prediction: torch.Tensor
    next_prediction_prob: torch.Tensor
    token_count: int


@runtime_checkable
class OutputLayer(Protocol):
    """The LM head. Fuses forward + loss + backward in one call to avoid
    materializing full ``(tokens, vocab)`` logits in memory.

    The loss objective is pluggable: the caller passes a ``loss_fn``
    (see :mod:`flextrain.nn.loss`) that turns a ``(T', V)`` logits slice
    into ``dZ`` inside the head's inner micro-chunk loop. This keeps
    the head arch-specific but loss-agnostic, so the same head drives
    SFT cross-entropy, RL (GRPO/PPO/DPO), distillation (MSE), etc.

    Memory invariant
    ----------------
    Implementations MUST micro-chunk along the token axis so that no
    full ``(num_tokens, vocab)`` intermediate is ever materialized.
    Peak logits VRAM is bounded by ``head_chunk_size * vocab_size``
    per call. Defeating this invariant defeats the whole point of
    AdaWS's tight memory budgets.

    Mirrors the ``process`` loop in
    ``orig/awsm_transformer/head.py:110``, generalized.
    """

    schema: ActivationSchema
    param_spec: ParamSpec

    def forward_backward(
        self,
        x: torch.Tensor,
        token_ctx: "TokenContext",  # type: ignore[name-defined]
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        ctx: LayerContext,
        *,
        loss_scale: float = 1.0,
        loss_fn: "LossFn | None" = None,  # type: ignore[name-defined]
    ) -> tuple[torch.Tensor, LossStats]:
        """Compute loss, accumulate head grads, and return ``(dx, stats)``.

        ``dx`` has shape ``(num_tokens, d_model)`` and is the gradient to
        hand to the last backbone layer's ``backward`` (orig semantics:
        aliased with ``x`` in place).
        """
        ...

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        ...
