"""MoE expert-compute backend protocol + the flextrain implementation.

The MoE block (:class:`flextrain.nn.blocks.ffn_moe.MoESwiGLUFFN`) routes
its scatter/per-expert-MLP/gather pipeline through one of these
backends. Today only :class:`FlextrainMoEExpertCompute` exists;
``ScatterMoEExpertCompute`` and ``SonicMoEExpertCompute`` will be added
later. The boundary is set so that:

* The router GEMM, top-k softmax, and router-gate-bwd stay outside the
  backend (they're flextrain-owned in every config).
* The backend owns scatter + per-expert MLP + gather, including saving
  whatever its bwd needs into caller-provided slot fields.
* No autograd: each backend exposes explicit ``fwd``/``bwd``/
  ``fwd_recompute`` methods. Caller pre-allocates the output tensors
  and grad accumulators (or backend allocates if ``None`` is passed).

The shared block-level slot schema is small (``x_router``,
``router_weights``, ``chosen_experts``, ``x_up``); each backend declares
its own private slot fields via :meth:`activation_fields`.

Pre-activation save model: every backend saves the (TK, 2F) pre-SwiGLU
activations into the shared ``slot.x_up``. ScatterMoE / sonic backends
deviate from their upstream defaults (which save autograd intermediates
or post-act tensors) so the save format is unified — recompute the
activation grad against the pre-act inside the backend's bwd.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping, Protocol, runtime_checkable

import torch

from flextrain.core.activation_schema import ActivationField


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MoEExpertCompute(Protocol):
    """Owns scatter + per-expert MLP + gather for one MoE block.

    Plugged into :class:`MoESwiGLUFFN` at construction. Defaults to
    :class:`FlextrainMoEExpertCompute` when no backend is passed.
    """

    name: str
    """Short identifier used in logs / errors / chunk_extra namespacing."""

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_residual_in_gather(self) -> bool:
        """If True, the backend's ``fwd`` accepts a non-None ``residual``
        and adds it inline during the gather. If False, callers must
        add the residual themselves after the backend returns; passing
        a non-None ``residual`` to such a backend raises ValueError."""
        ...

    @property
    def supported_tiers(self) -> frozenset[int]:
        """Save tiers this backend supports. The block's ``schema`` is
        capped at ``max(supported_tiers)``; the DP planner only sees
        tiers in this set."""
        ...

    # ------------------------------------------------------------------
    # Schema declaration
    # ------------------------------------------------------------------

    def activation_fields(
        self, num_experts: int, top_k: int, expert_dim: int, d_model: int,
        compute_dtype: torch.dtype,
    ) -> tuple[ActivationField, ...]:
        """Backend-private slot fields. Returned to the engine for
        tier planning and buffer sizing. The shared block fields
        (x_router, router_weights, chosen_experts, x_up) are NOT in
        here — they're declared by ``MoESwiGLUFFN`` itself."""
        ...

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def fwd(
        self,
        x: torch.Tensor,                       # (T, d_model) — input activations
        router_weights: torch.Tensor,          # (T, K) — softmax(topk(logits))
        chosen_experts: torch.Tensor,          # (T, K) int32 — topk indices
        weights: Mapping[str, torch.Tensor],   # {w_up, w_down}
        *,
        out: torch.Tensor,                     # (T, d_model) — written
        residual: torch.Tensor | None,         # (T, d_model) — if backend supports inline add
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> None:
        """Run scatter + per-expert MLP + gather. Writes ``out``;
        backend saves whatever its bwd needs into ``slot``'s fields
        (both shared and backend-private). ``chunk_extra`` is a per-chunk
        mutable dict the backend may use for its own host-side scratch
        (e.g., pinned-host counts) under backend-namespaced keys."""
        ...

    def fwd_recompute(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        *,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> Any:
        """Re-run the part of fwd that was dropped at save tier <
        max. Must repopulate every slot field at tier > save_level
        (typically ``slot.x_up``). May return an opaque object the
        backend later passes to ``bwd`` (e.g., the rescatter buffer)
        — caller stashes it in ``slot.aux``."""
        ...

    def expert_counts_gpu(self, slot: Any) -> torch.Tensor:
        """Return the per-expert token-count tensor on GPU for this
        layer's chunk. Backends store the same information in different
        forms (flextrain: counts directly; scattermoe: cumulative
        offsets); this is the unified accessor for the load-balance
        bwd kernel that needs (E,) int32 counts."""
        ...

    def bwd(
        self,
        dy: torch.Tensor,                      # (T, d_model)
        x: torch.Tensor,                       # (T, d_model) — saved fwd input
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        *,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
        dx: torch.Tensor | None = None,
        recompute_handoff: Any = None,
        lora_capture: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Run gather-bwd + per-expert MLP-bwd + scatter-bwd. Accumulates
        wgrads into ``grads`` (caller pre-zeroes if needed). Returns the
        per-token ``dx`` (fresh alloc if ``dx`` kwarg was None).

        Backend writes any router-grad scratch the caller needs into
        ``slot.aux`` under backend-specific keys; today the caller
        (``routed_swiglu_moe_bwd``) reads ``slot.aux["moe_dprobs"]`` (a
        ``(TK, 1)`` scattered-layout tensor for flextrain) to feed the
        router-gate-bwd. As more backends are added, this contract may
        grow into a typed return — for now slot.aux is the carrier.

        ``lora_capture`` (when not None) is a caller-owned dict the
        backend populates with per-expert grouped intermediates that a
        downstream LoRA wgrad finalize will consume. Required keys:

          * ``"scattered_x_grouped"`` (TK, d_model) bf16 — fwd input
            grouped by expert (sorted/scattered layout the backend
            already uses internally). Source for ``X`` in the LoRA
            wgrad of ``w_up``.
          * ``"dx_up_up_grouped"`` (TK, 2*F) bf16 — gradient at the
            up-projection output, post-activation-bwd, grouped by
            expert. Source for ``dY`` in w_up's LoRA wgrad.
          * ``"scattered_upstream_grouped"`` (TK, d_model) bf16 —
            upstream gradient at the down-projection output, grouped
            by expert. Source for ``dY`` in w_down's LoRA wgrad.
            (Backend MUST NOT in-place-overwrite this with dx_pre
            when capturing.)
          * ``"x_up_grouped"`` (TK, 2*F) bf16 — saved per-slot pre-
            SwiGLU activation, sliced to actual TK. The finalize
            recomputes ``fwd_act = silu(gate) * value`` from this as
            the ``X`` for w_down's LoRA wgrad. Reference to
            ``slot.x_up[:TK]`` — no new alloc.
          * ``"expert_offsets"`` (E,) int32 — cumulative ending
            offsets, ``offs[-1] == TK``. Used as ``offs`` for
            ``torch.nn.functional.grouped_mm``.
          * ``"TK"`` int — actual token-slot count this chunk
            (≤ TK_max). All staged tensors are sliced to this count.

        When ``lora_capture is None`` (default), backend behaves
        exactly as before — no staging, no extra allocations beyond
        what pre-LoRA code already used."""
        ...


# ---------------------------------------------------------------------------
# Flextrain implementation
# ---------------------------------------------------------------------------


class FlextrainMoEExpertCompute:
    """The current flextrain MoE pipeline, packaged behind the backend
    protocol. No behavior change vs. pre-refactor code — just moved.

    Activation slot contract (backend-private; shared fields belong to
    ``MoESwiGLUFFN``):

    * ``index_mapping: (T, K) int32`` (tier 0) — per-(layer, chunk)
      sort indices used by :func:`flextrain_moe_scatter` and the
      router-gate-bwd.
    * ``expert_counts: (E,) int32`` (tier 0) — GPU-side per-expert
      token counts; written by the sort kernel, mirrored to the
      pinned-host buffer in chunk_extra (see ``_host_counts``).
    * ``scattered_router_weights: (TK, 1) bf16`` (tier 0) — router
      weights in scatter order, used by the SwiGLU-bwd's per-slot
      rescale.

    Pinned-host scratch (``(E,) int32``) is allocated lazily into
    ``chunk_extra["flextrain.moe.expert_counts_host"][layer_id]`` so
    its lifetime matches the chunk (survives fwd→bwd including for
    offloaded layers where slot.aux is rebuilt). Pinned-host
    allocations of this size are ~µs each and amortize across the
    chunk's full fwd+bwd workload.
    """

    name = "flextrain"
    _CHUNK_EXTRA_HOST_COUNTS = "flextrain.moe.expert_counts_host"

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_residual_in_gather(self) -> bool:
        return True

    @property
    def supported_tiers(self) -> frozenset[int]:
        return frozenset({0, 1, 2, 3})

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def activation_fields(
        self, num_experts: int, top_k: int, expert_dim: int, d_model: int,
        compute_dtype: torch.dtype,
    ) -> tuple[ActivationField, ...]:
        return (
            ActivationField(
                "expert_counts",
                lambda n, d: (num_experts,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "index_mapping",
                lambda n, d: (n, top_k),
                torch.int32,
                tier=0,
            ),
            ActivationField(
                "scattered_router_weights",
                lambda n, d: (n * top_k, 1),
                compute_dtype,
                tier=0,
                token_axis=None,
            ),
        )

    def expert_counts_gpu(self, slot: Any) -> torch.Tensor:
        """Flextrain stores per-expert token counts directly on the slot."""
        return slot.expert_counts

    # ------------------------------------------------------------------
    # Pinned-host expert-counts helper. Stashed in ``chunk_extra`` keyed
    # by layer_id so its lifetime is per-(layer, chunk) — survives
    # fwd→bwd even for offloaded slots (whose ``aux`` is rebuilt at
    # bwd-time fetch). When host_pinned slot fields are added to the
    # schema this helper goes away and the buffer becomes a regular
    # tier-0 slot field.
    # ------------------------------------------------------------------

    @classmethod
    def _host_counts(
        cls,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        num_experts: int,
    ) -> torch.Tensor:
        per_layer = chunk_extra.setdefault(cls._CHUNK_EXTRA_HOST_COUNTS, {})
        buf = per_layer.get(layer_id)
        if buf is None:
            buf = torch.zeros(num_experts, dtype=torch.int32, pin_memory=True)
            per_layer[layer_id] = buf
        return buf

    # ------------------------------------------------------------------
    # Compute (delegates to existing primitives in full_moe.py)
    # ------------------------------------------------------------------

    def fwd(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        chosen_experts: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        *,
        out: torch.Tensor,
        residual: torch.Tensor | None,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> None:
        from flextrain.ops.full_moe import (
            dispatch_scatter,
            swiglu_expert_loop_fwd,
            combine_gather,
        )

        # w_up: (E, 2F, d), w_down: (E, d, F). Pull dims from weights.
        num_experts = weights["w_up"].shape[0]
        expert_dim = weights["w_down"].shape[2]
        num_tokens, d_model = x.shape
        TK = num_tokens * router_weights.shape[1]

        # 1. Scatter inputs by expert.
        expert_counts_cpu = self._host_counts(chunk_extra, layer_id, num_experts)
        scattered_x = scratch_fn((TK, d_model), x.dtype)
        srw = slot.scattered_router_weights[:TK, :]
        indices, expert_counts_cpu = dispatch_scatter(
            x, router_weights, chosen_experts,
            num_experts=num_experts,
            index_mapping=slot.index_mapping,
            expert_counts_gpu=slot.expert_counts,
            expert_counts_cpu=expert_counts_cpu,
            scattered_x_out=scattered_x,
            scattered_router_weights_out=srw,
        )

        # 2. Per-expert SwiGLU MLP.
        max_exp_tokens = int(expert_counts_cpu.max())
        use_secondary = secondary_stream is not None
        x_act_even = scratch_fn((max_exp_tokens, expert_dim), x.dtype)
        x_act_odd = (
            scratch_fn((max_exp_tokens, expert_dim), x.dtype)
            if use_secondary else x_act_even
        )
        swiglu_expert_loop_fwd(
            scattered_x=scattered_x,
            x_preact_buf=slot.x_up,
            w_up=weights["w_up"],
            w_down=weights["w_down"],
            expert_counts_cpu=expert_counts_cpu,
            primary_stream=primary_stream,
            secondary_stream=secondary_stream,
            x_act_even=x_act_even,
            x_act_odd=x_act_odd,
        )

        # 3. Combine + residual + gather.
        combine_gather(
            scattered_x, indices,
            router_weights=router_weights,
            residual=residual.view(-1, d_model) if residual is not None else None,
            out_tensor=out,
        )

    def fwd_recompute(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        *,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> torch.Tensor:
        """Re-run the up-projection part of fwd to repopulate ``slot.x_up``
        when the layer's save tier was < 3. Returns the rescatter buffer
        for bwd to reuse (so we don't scatter ``x`` twice)."""
        from flextrain.ops.full_moe import swiglu_expert_loop_recompute_x_up
        from flextrain.ops import flextrain_moe_scatter

        # x: (T, d_model)
        num_tokens, d_model = x.shape
        top_k = slot.index_mapping.shape[1]
        TK = num_tokens * top_k

        expert_counts_cpu = self._host_counts(
            chunk_extra, layer_id, weights["w_up"].shape[0],
        )
        scattered_x = scratch_fn((TK, d_model), x.dtype)
        flextrain_moe_scatter(x, slot.index_mapping, out=scattered_x)
        swiglu_expert_loop_recompute_x_up(
            scattered_x=scattered_x,
            x_preact_buf=slot.x_up,
            w_up=weights["w_up"],
            expert_counts_cpu=expert_counts_cpu,
            primary_stream=primary_stream,
            secondary_stream=secondary_stream,
        )
        return scattered_x

    def bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        *,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
        dx: torch.Tensor | None = None,
        recompute_handoff: Any = None,
        # Generic LoRA capture (when not None, backend stages per-expert
        # grouped intermediates that the wrapper's backward_wgrad
        # consumes via grouped_mm finalize). See protocol docstring.
        lora_capture: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        from flextrain.ops.full_moe import (
            swiglu_expert_loop_bwd,
            flextrain_moe_scatter,
            flextrain_moe_gather,
        )

        num_tokens, d_model = dy.shape
        top_k = slot.index_mapping.shape[1]
        TK = num_tokens * top_k
        num_experts = weights["w_up"].shape[0]
        # w_down: (E, d, F). expert_dim (F) is now the last axis.
        expert_dim = weights["w_down"].shape[2]
        expert_counts_cpu = self._host_counts(chunk_extra, layer_id, num_experts)

        # 1. Scatter dy. READ-ONLY across the expert loop now (under
        #    the deferred-LoRA-friendly path); the dx_pre output goes
        #    to its own buffer, not in-place over scattered_upstream.
        scattered_upstream = scratch_fn((TK, d_model), dy.dtype)
        flextrain_moe_scatter(dy, slot.index_mapping, out=scattered_upstream)

        # 2. Get scattered_x (either from recompute handoff or scatter
        #    fresh from saved x).
        if recompute_handoff is not None:
            scattered_x = recompute_handoff
        else:
            scattered_x = scratch_fn((TK, d_model), dy.dtype)
            flextrain_moe_scatter(x, slot.index_mapping, out=scattered_x)

        # 3. Per-expert bwd loop. Writes per-slot dx_up_up into
        #    ``dx_up_up_grouped`` and per-slot dx_pre into
        #    ``dx_pre_grouped`` (caller-allocated TK-sized buffers).
        #    Accumulates g_up/g_down per-expert when grads.get(...) is
        #    not None; otherwise the inline addmm is skipped (frozen
        #    base / LoRA-only training pattern). Writes dprobs.
        srw = slot.scattered_router_weights[:TK, :]
        dprobs = torch.zeros_like(srw)
        # TK-sized output buffers — survive the loop so LoRA finalize
        # can read dx_up_up_grouped, and gather can source dx_pre_grouped.
        dx_up_up_grouped = scratch_fn((TK, 2 * expert_dim), dy.dtype)
        dx_pre_grouped = scratch_fn((TK, d_model), dy.dtype)
        swiglu_expert_loop_bwd(
            scattered_upstream=scattered_upstream,
            scattered_x=scattered_x,
            x_preact_buf=slot.x_up,
            srw=srw,
            dprobs=dprobs,
            w_up=weights["w_up"],
            w_down=weights["w_down"],
            grads=grads,
            expert_counts_cpu=expert_counts_cpu,
            dx_up_up_grouped=dx_up_up_grouped,
            dx_pre_grouped=dx_pre_grouped,
            expert_dim=expert_dim,
            primary_stream=primary_stream,
            secondary_stream=secondary_stream,
        )

        # 4. Stage per-expert grouped intermediates for downstream LoRA
        # finalize. All four are existing buffers (no new alloc) — just
        # references the wrapper's grouped_mm driver will consume.
        if lora_capture is not None:
            # Cumulative ending offsets, GPU-side, ending in TK. The
            # expert_counts_cpu tensor is host-pinned for the python
            # loop; we need a GPU int32 cumsum for grouped_mm's `offs`.
            # (Future opt: cache in slot.aux to avoid recomputing across
            # layers — for now the cumsum is ~µs.)
            expert_offsets = (
                self.expert_counts_gpu(slot).cumsum(0).to(torch.int32)
            )
            lora_capture["scattered_x_grouped"] = scattered_x
            lora_capture["dx_up_up_grouped"] = dx_up_up_grouped
            lora_capture["scattered_upstream_grouped"] = scattered_upstream
            lora_capture["x_up_grouped"] = slot.x_up[:TK]
            lora_capture["expert_offsets"] = expert_offsets
            lora_capture["TK"] = TK

        # 5. Gather per-slot dx back to per-token dx.
        if dx is None:
            dx = torch.empty_like(dy)
        flextrain_moe_gather(dx_pre_grouped, slot.index_mapping, out=dx)

        # 6. Hand dprobs (scattered layout) to the caller for the
        #    router-gate-bwd via slot.aux.
        slot.aux["moe_dprobs"] = dprobs

        return dx


# ---------------------------------------------------------------------------
# ScatterMoE implementation
# ---------------------------------------------------------------------------


class ScatterMoEExpertCompute:
    """Drives the routed-MLP pipeline via scattermoe's Triton kernels.

    Lower-level kernel calls (no autograd, no torch.autograd.Function):
    we directly drive ``scatter2scatter``, ``group``, and ``group_bwd_W``
    with our own intermediate tensors so we can save only the pre-act
    ``(TK, 2F)`` into ``slot.x_up`` (matching the flextrain convention)
    and recompute the SwiGLU activation in bwd.

    Backend-private slot fields (all tier 0):
      * ``scattermoe_sorted_expert_idxs: (TK,) int32`` — scattermoe's
        sort-by-expert tensor (output of flatten_sort_count).
      * ``scattermoe_sorted_scattered_idxs: (TK,) int32`` — inverse
        permutation; needed by every scatter2scatter call.
      * ``scattermoe_expert_offsets: (E,) int32`` — bincount cumsum;
        equivalent of flextrain's expert_counts cumulated.
      * ``index_mapping: (T, K) int32`` — flextrain-shaped permutation,
        derived from sorted_scattered_idxs once at fwd. Used by the
        unified ``flextrain_moe_router_gate_bwd`` so both backends
        produce dprobs in scattered layout for the router-gate-bwd.

    Limitations:
      * No inline residual add in gather. Caller adds residual after.
    """

    name = "scattermoe"

    @property
    def supports_residual_in_gather(self) -> bool:
        return False

    @property
    def supported_tiers(self) -> frozenset[int]:
        return frozenset({0, 1, 2, 3})

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def activation_fields(
        self, num_experts: int, top_k: int, expert_dim: int, d_model: int,
        compute_dtype: torch.dtype,
    ) -> tuple[ActivationField, ...]:
        return (
            ActivationField(
                "scattermoe_sorted_expert_idxs",
                lambda n, d: (n * top_k,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "scattermoe_sorted_scattered_idxs",
                lambda n, d: (n * top_k,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "scattermoe_expert_offsets",
                lambda n, d: (num_experts,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            # Flextrain-shaped index_mapping — populated at fwd from
            # sorted_scattered_idxs so the unified router-gate-bwd can
            # consume scattered dprobs the same way it does for the
            # flextrain backend.
            ActivationField(
                "index_mapping",
                lambda n, d: (n, top_k),
                torch.int32,
                tier=0,
            ),
        )

    def expert_counts_gpu(self, slot: Any) -> torch.Tensor:
        """ScatterMoE stores cumulative offsets; per-expert counts are
        the first-difference. Returns (E,) int32."""
        offsets = slot.scattermoe_expert_offsets  # (E,) cumsum int32
        # counts[e] = offsets[e] - offsets[e-1]; offsets[-1] = 0.
        prev = torch.zeros(1, dtype=offsets.dtype, device=offsets.device)
        return torch.diff(offsets, prepend=prev)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def fwd(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        chosen_experts: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        *,
        out: torch.Tensor,
        residual: torch.Tensor | None,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> None:
        if residual is not None:
            raise ValueError(
                "ScatterMoEExpertCompute does not support inline residual "
                "in the gather; caller must add it after fwd returns."
            )
        from scattermoe import kernels as scm_kernels
        from scattermoe.parallel_experts import flatten_sort_count

        # Dimensions. Option-B layout: w_up (E, 2F, d), w_down (E, d, F).
        # Note: scattermoe's own ``ParallelExperts.weight`` stores
        # (num_experts, output_size, input_size) = (E, out, in), which is
        # exactly our new layout — see external_moe_impl/scattermoe/
        # scattermoe/parallel_experts.py:151. We pass W.permute(0, 2, 1)
        # to the scatter2scatter kernel below, mirroring scattermoe's own
        # ``ParallelExperts.forward`` (line 177).
        num_experts = weights["w_up"].shape[0]
        d_model = weights["w_up"].shape[2]
        F = weights["w_down"].shape[2]
        T = x.shape[0]
        K = router_weights.shape[1]
        TK = T * K

        # 1. Sort + count. flatten_sort_count is a torch.compile'd
        # routine that returns three tensors sized to the actual
        # chunk's TK. Slot fields are allocated for max-TK (max
        # num_tokens * top_k), so we copy into the [:TK] prefix and
        # downstream code slices accordingly. Same pattern as sonic.
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = (
            flatten_sort_count(chosen_experts, num_experts=num_experts)
        )
        slot.scattermoe_sorted_expert_idxs[:TK].copy_(sorted_expert_idxs)
        slot.scattermoe_sorted_scattered_idxs[:TK].copy_(sorted_scattered_idxs)
        slot.scattermoe_expert_offsets.copy_(expert_offsets)

        # 2. Build flextrain-shaped index_mapping for the
        # router-gate-bwd. index_mapping[t, k] = scatter position of
        # token t's k-th routed expert. sorted_scattered_idxs[i] gives
        # the original (t*K + k) for sorted position i; the inverse
        # permutation is what we need.
        # We compute it on GPU via scatter: for each i in [0..TK),
        # index_mapping.flatten()[sorted_scattered_idxs[i]] = i.
        # PyTorch supports this via index_copy_ on a flattened view.
        # Slice both the index_mapping target view and the source idxs
        # to the actual TK (max-TK allocation, actual chunk smaller).
        index_mapping_flat = slot.index_mapping.view(-1)[:TK]
        # Note: sorted_scattered_idxs is int32; index_copy_ wants int64.
        # The conversion here is ~µs.
        index_mapping_flat.index_copy_(
            0,
            slot.scattermoe_sorted_scattered_idxs[:TK].long(),
            torch.arange(TK, device=x.device, dtype=torch.int32),
        )

        # 3. Up-projection: scatter2scatter into pre-act.
        # x: (T, d), w_up: (E, 2F, d) → permute to (E, d, 2F) view.
        # Output: (TK, 2F) into slot.x_up. Slot fields and slot.x_up
        # are allocated for max-TK; slice all to actual chunk TK.
        scm_kernels.ops.scatter2scatter(
            X=x, W=weights["w_up"].permute(0, 2, 1),
            sorted_expert_idxs=slot.scattermoe_sorted_expert_idxs[:TK],
            sorted_scattered_idxs=slot.scattermoe_sorted_scattered_idxs[:TK],
            k=K, x_grouped=False, y_grouped=True,
            out=slot.x_up[:TK],
        )

        # 4. Activation: silu(gate) * value. slot.x_up packs as
        # [value || gate] (flextrain convention; first half value,
        # second half gate). Slice to actual TK.
        post_act = scratch_fn((TK, F), x.dtype)
        value, gate = slot.x_up[:TK].chunk(2, dim=-1)
        post_act.copy_(torch.nn.functional.silu(gate) * value)

        # 5. Down-projection with gate-weighting.
        # scatter2scatter w/ k=1, x_grouped=True, y_grouped=False:
        # input is grouped (sorted by expert), output is token-major
        # (TK,d) viewable as (T, K, d) for the per-token-per-k weighted
        # sum below. Matches scattermoe's GLUMLP output_experts call
        # which uses (grouped_in=True, grouped_out=False).
        out_expanded = scratch_fn((TK, d_model), x.dtype)
        # w_down: (E, d, F) → permute to (E, F, d) view for the kernel
        # (K=F = post_act.size(-1), N=d = output's last dim).
        scm_kernels.ops.scatter2scatter(
            X=post_act, W=weights["w_down"].permute(0, 2, 1),
            sorted_expert_idxs=slot.scattermoe_sorted_expert_idxs[:TK],
            sorted_scattered_idxs=slot.scattermoe_sorted_scattered_idxs[:TK],
            k=1, x_grouped=True, y_grouped=False,
            out=out_expanded,
        )
        # Reshape (TK, d) -> (T, K, d) -> weighted-sum to (T, d).
        # Match scattermoe's MLP wrapper:
        #   output_expanded = output.view(T, K, d)
        #   output = (gates.unsqueeze(1) @ output_expanded).squeeze(1)
        out_view = out_expanded.view(T, K, d_model)
        # router_weights (T, K) → (T, 1, K). bmm with (T, K, d) → (T, 1, d) → (T, d).
        torch.bmm(
            router_weights.unsqueeze(1),
            out_view,
            out=out.view(T, 1, d_model),
        )

        # NOTE: ``out_expanded`` (post-down, pre-gate) and ``post_act``
        # (post-SwiGLU activation) are NOT saved here. Both are
        # recomputed in ``bwd`` from ``slot.x_up`` (saved at tier ≥3 or
        # repopulated by ``fwd_recompute`` at lower tiers) plus a fresh
        # down-projection scatter2scatter call. This keeps memory
        # pressure bounded — chunk_extra only carried these as a
        # stop-gap pre-stage-1 and bloated GPU memory by ~10GB / 40
        # layers for 35B-A3B.

    def fwd_recompute(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        *,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> Any:
        """Re-run the up-projection scatter2scatter to repopulate
        ``slot.x_up`` when the layer's save tier was < 3. Skips the
        down-projection / activation / combine — bwd recomputes those
        from slot.x_up.

        All routing metadata (``scattermoe_*``, ``index_mapping``) is
        tier 0 and already populated from fwd; we only re-run the
        up-projection. Returns ``None`` (no handoff needed; bwd reads
        slot fields the same way it does after a tier-3 fwd).
        """
        from scattermoe import kernels as scm_kernels

        # Option-B layout: w_up (E, 2F, d), w_down (E, d, F).
        T = x.shape[0]
        K = slot.index_mapping.shape[1]
        TK = T * K

        scm_kernels.ops.scatter2scatter(
            X=x, W=weights["w_up"].permute(0, 2, 1),
            sorted_expert_idxs=slot.scattermoe_sorted_expert_idxs[:TK],
            sorted_scattered_idxs=slot.scattermoe_sorted_scattered_idxs[:TK],
            k=K, x_grouped=False, y_grouped=True,
            out=slot.x_up[:TK],
        )
        return None

    def bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        *,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
        dx: torch.Tensor | None = None,
        recompute_handoff: Any = None,
        # Generic LoRA capture (populated below when not None).
        lora_capture: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if recompute_handoff is not None:
            raise NotImplementedError(
                "ScatterMoEExpertCompute does not support tier <3 recompute."
            )
        from scattermoe import kernels as scm_kernels

        # Dimensions. Option-B layout: w_up (E, 2F, d), w_down (E, d, F).
        num_experts = weights["w_up"].shape[0]
        d_model = weights["w_up"].shape[2]
        F = weights["w_down"].shape[2]
        T = dy.shape[0]
        K = slot.index_mapping.shape[1]
        TK = T * K

        # Pull saved fwd state. Slot fields are allocated for max-TK;
        # slice to actual chunk TK to match what the kernels iterate over.
        sorted_expert_idxs = slot.scattermoe_sorted_expert_idxs[:TK]
        sorted_scattered_idxs = slot.scattermoe_sorted_scattered_idxs[:TK]
        expert_offsets = slot.scattermoe_expert_offsets
        # Recover router_weights from the shared slot field.
        router_weights = slot.router_weights[:T]  # (T, K) bf16; slice off max-T padding

        # Recompute ``post_act`` from saved ``slot.x_up`` (free — just a
        # SwiGLU on the saved pre-act). Avoids stashing it in chunk_extra
        # at fwd time (the v1 approach bloated GPU mem by ~|TK*F| bytes
        # per layer for the entire fwd-bwd window).
        value, gate = slot.x_up[:TK].chunk(2, dim=-1)
        post_act = scratch_fn((TK, F), dy.dtype)
        post_act.copy_(torch.nn.functional.silu(gate) * value)

        # Recompute ``out_expanded`` (down-projection result) by re-
        # running the down-projection scatter2scatter. One extra GEMM
        # in bwd (vs none-recompute), but saves |TK*d| bytes per layer
        # of held GPU memory across the fwd-bwd window. At 35B-A3B,
        # 40 layers, TK=512K, d=2048, that's ~20GB saved.
        out_expanded = scratch_fn((TK, d_model), dy.dtype)
        scm_kernels.ops.scatter2scatter(
            X=post_act, W=weights["w_down"].permute(0, 2, 1),
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=1, x_grouped=True, y_grouped=False,
            out=out_expanded,
        )

        # 1. d_gates: dy ⋅ out_expanded (token-major). Mirrors scattermoe
        # ParallelLinear.backward line 79.
        d_gates = (out_expanded.view(T, K, d_model) @ dy.unsqueeze(-1)).squeeze(-1)
        # (T, K) bf16. Will be scattered to (TK, 1) below for the unified
        # router-gate-bwd.

        # 2. Build grouped_grad_out = group(dy, sorted_scattered_idxs,
        # fan_out=K, coeff=router_weights.flatten()) — the gather-bwd.
        grouped_grad_out = scratch_fn((TK, d_model), dy.dtype)
        scm_kernels.ops.group(
            dy, sorted_scattered_idxs,
            fan_out=K,
            coeff=router_weights.flatten(),
            out=grouped_grad_out,
        )

        # 3. d_w_down = group_bwd_W(DY=grouped_grad_out, X=post_act,
        # offsets=expert_offsets). Gated on grads.get("g_down") so the
        # full GEMM is skipped under frozen-base / LoRA-only training
        # (where g_down isn't in the grads dict).
        if grads.get("g_down") is not None:
            d_w_down, _ = scm_kernels.ops.group_bwd_W(
                DY=grouped_grad_out, X=post_act,
                expert_offsets=expert_offsets,
                E=num_experts, has_bias=False,
            )
            # group_bwd_W returns DW as a (E, X_dim=F, DY_dim=d) view
            # over storage that is physically (E, d, F). Under option-B
            # layout, g_down is (E, d, F) — matches the underlying
            # storage. Permute the returned view to add into g_down.
            grads["g_down"].add_(d_w_down.permute(0, 2, 1))

        # 4. d_post_act = scatter2scatter(grouped_grad_out, W=w_down, ...)
        # produces (TK, F). Under option-B, w_down: (E, d, F) — already
        # the right orientation for the dgrad's K=d, N=F (drop the permute
        # that the old layout needed). Non-contiguous strided W is fine
        # — scattermoe's kernel uses raw strides.
        d_post_act = scratch_fn((TK, F), dy.dtype)
        scm_kernels.ops.scatter2scatter(
            X=grouped_grad_out, W=weights["w_down"],
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=1, x_grouped=True, y_grouped=True,
            out=d_post_act,
        )

        # 5. Activation bwd: post_act = silu(gate) * value. Need
        # d_pre_act = (d_post_act * d/d(silu*value)/d(pre_act)).
        # slot.x_up is allocated for max-TK; slice to actual TK.
        value, gate = slot.x_up[:TK].chunk(2, dim=-1)
        # silu_grad: derivative of silu. silu(x) = x * sigmoid(x).
        # d_silu/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sig_gate = torch.sigmoid(gate.float())
        silu_gate = gate.float() * sig_gate
        d_silu_d_gate = (sig_gate + silu_gate * (1.0 - sig_gate))
        # post_act = silu(gate) * value
        # d/d(value) = d_post_act * silu(gate)
        # d/d(gate) = d_post_act * value * d_silu_d_gate
        d_pre_act = scratch_fn((TK, 2 * F), dy.dtype)
        d_value, d_gate = d_pre_act.chunk(2, dim=-1)
        d_value.copy_((d_post_act.float() * silu_gate).to(d_post_act.dtype))
        d_gate.copy_(
            (d_post_act.float() * value.float() * d_silu_d_gate).to(d_post_act.dtype)
        )

        # 6. d_w_up = group_bwd_W(DY=d_pre_act, X=grouped_x, ...).
        # ``grouped_x`` is needed if EITHER the base wgrad runs OR the
        # LoRA capture path is active (it stages grouped_x as
        # scattered_x_grouped). Skip both the group() and group_bwd_W()
        # calls when neither path needs them.
        need_grouped_x = (
            grads.get("g_up") is not None or lora_capture is not None
        )
        grouped_x = None
        if need_grouped_x:
            grouped_x = scratch_fn((TK, d_model), x.dtype)
            scm_kernels.ops.group(
                x, sorted_scattered_idxs, fan_out=K, out=grouped_x,
            )
        if grads.get("g_up") is not None:
            d_w_up, _ = scm_kernels.ops.group_bwd_W(
                DY=d_pre_act, X=grouped_x,
                expert_offsets=expert_offsets,
                E=num_experts, has_bias=False,
            )
            # Same permute pattern as g_down above: returned DW view is
            # (E, d, 2F) over (E, 2F, d) physical storage. g_up is
            # (E, 2F, d), so permute the view before .add_().
            grads["g_up"].add_(d_w_up.permute(0, 2, 1))

        # Stage per-expert grouped intermediates for downstream LoRA
        # finalize. Scattermoe naturally produces all three as standalone
        # buffers — just pass references; lifetime extends to end of bwd
        # via the dict the wrapper holds. See MoEExpertCompute.bwd
        # docstring for the contract.
        if lora_capture is not None:
            lora_capture["scattered_x_grouped"] = grouped_x
            lora_capture["dx_up_up_grouped"] = d_pre_act
            lora_capture["scattered_upstream_grouped"] = grouped_grad_out
            lora_capture["x_up_grouped"] = slot.x_up[:TK]
            lora_capture["expert_offsets"] = expert_offsets  # (E,) int32 cumsum
            lora_capture["TK"] = TK

        # 7. d_x = scatter2scatter(d_pre_act, W=w_up, k=1, ...)
        # then sum over K. Under option-B, w_up: (E, 2F, d) — already the
        # right orientation for the dgrad (K=2F, N=d). Drop the permute
        # the old layout needed.
        d_expanded_input = scratch_fn((TK, d_model), x.dtype)
        scm_kernels.ops.scatter2scatter(
            X=d_pre_act, W=weights["w_up"],
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=1, x_grouped=True, y_grouped=False,
            out=d_expanded_input,
        )
        if dx is None:
            dx = torch.empty_like(dy)
        # d_expanded_input is (TK, d) but in token-major layout
        # (y_grouped=False). View as (T, K, d) and sum over K.
        dx.copy_(d_expanded_input.view(T, K, d_model).sum(dim=1))

        # 8. dprobs: scatter d_gates (T, K) into (TK, 1) via slot.index_mapping.
        # The unified router-gate-bwd consumes scattered layout.
        # Slice index_mapping to actual T (max-T allocation).
        from flextrain.ops import flextrain_moe_scatter_routing_weights
        dprobs_flat = scratch_fn((TK,), dy.dtype)
        flextrain_moe_scatter_routing_weights(
            d_gates.contiguous(), slot.index_mapping[:T], out=dprobs_flat,
        )
        # Router-gate-bwd expects (TK, 1).
        slot.aux["moe_dprobs"] = dprobs_flat.unsqueeze(-1)

        return dx


# ---------------------------------------------------------------------------
# SonicMoE implementation
# ---------------------------------------------------------------------------


class SonicMoEExpertCompute:
    """Drives the routed-MLP pipeline via sonic-moe's CUTLASS DSL kernels.

    Lower-level kernel calls (no autograd, no Function.apply): we drive
    ``general_routing_router_metadata_triton`` (routing bookkeeping),
    ``gemm_gated`` (fused up-proj + activation), ``gemm`` (down-proj),
    ``_router_forward`` (gather + weighted-sum) for fwd, and
    ``_down_projection_backward_act`` (fused down-bwd + activation-bwd via
    ``gemm_dgated``), ``_up_projection_backward_act`` (up-bwd dgrad),
    ``gemm`` (×2 for wgrads), ``_token_broadcast_backward`` (K-reduction)
    for bwd.

    Backend-private slot fields (all tier 0):
      * ``sonic_s_scatter_idx: (TK,) int32`` — scattered → token-major idx.
      * ``sonic_s_reverse_scatter_idx: (TK,) int32`` — token-major → scattered idx.
      * ``sonic_x_gather_idx: (TK,) int32`` — scattered → x token idx.
      * ``sonic_expert_frequency: (E,) int32`` — per-expert counts.
      * ``sonic_expert_frequency_offset: (E+1,) int32`` — cumsum w/ leading 0.
      * ``index_mapping: (T, K) int32`` — flextrain-shape, derived from
        s_scatter_idx at fwd. Lets the unified
        ``flextrain_moe_router_gate_bwd`` consume scattered dprobs the
        same way both flextrain and sonic backends do.

    Pre-act save: sonic's natural format. ``slot.x_up: (TK, 2F)`` is
    populated by ``gemm_gated``'s ``preact_out=`` arg in fwd; bwd's
    ``_down_projection_backward_act`` consumes it as ``PreAct=h``. No
    deviation from sonic's convention required.

    Limitations:
      * No LoRA — no per-expert wgrad hook in sonic.
      * No inline residual add in gather (caller adds it).
      * Hopper (sm_90+) only — sonic's CUTLASS DSL targets sm_90.
        Earlier archs hit "Gemm Sm80 is not implemented yet" at the
        first kernel call. ``__init__`` checks compute capability and
        raises with a clear message; set ``SONICMOE_FORCE=1`` to bypass.
    """

    name = "sonicmoe"

    def __init__(self) -> None:
        import os
        cap = torch.cuda.get_device_capability()
        force = os.environ.get("SONICMOE_FORCE", "0") == "1"
        if cap < (9, 0) and not force:
            raise RuntimeError(
                f"SonicMoEExpertCompute requires sm_90+ (Hopper). "
                f"Got sm_{cap[0]}{cap[1]}. Set SONICMOE_FORCE=1 to bypass."
            )
        # Try-import at construction so we fail fast if sonic / quack
        # aren't installed. The CUTLASS DSL JIT compiles on first call,
        # so the imports themselves don't run kernels — safe.
        try:
            from quack.gemm_interface import gemm, gemm_gated, gemm_dgated  # noqa: F401
            from sonicmoe.functional.triton_kernels import (  # noqa: F401
                general_routing_router_metadata_triton,
            )
            from sonicmoe.functional.forward import _router_forward  # noqa: F401
            from sonicmoe.functional.backward import (  # noqa: F401
                _down_projection_backward_act,
                _up_projection_backward_act,
                _token_broadcast_backward,
            )
        except ImportError as e:
            raise RuntimeError(
                "sonic-moe / quack-kernels not installed. Install with:\n"
                "  pip install -e <flextrain>/external_moe_impl/sonic-moe[cu13]\n"
                f"Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_residual_in_gather(self) -> bool:
        return False

    @property
    def supported_tiers(self) -> frozenset[int]:
        # Tier 3 saves slot.x_up locally; tiers 0/1/2 drop x_up and
        # rely on ``fwd_recompute`` to repopulate it via the up-proj
        # gemm_gated. Routing metadata (sonic_*, index_mapping) is
        # tier 0 in the schema and survives all tier choices.
        return frozenset({0, 1, 2, 3})

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def activation_fields(
        self, num_experts: int, top_k: int, expert_dim: int, d_model: int,
        compute_dtype: torch.dtype,
    ) -> tuple[ActivationField, ...]:
        return (
            ActivationField(
                "sonic_s_scatter_idx",
                lambda n, d: (n * top_k,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "sonic_s_reverse_scatter_idx",
                lambda n, d: (n * top_k,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "sonic_x_gather_idx",
                lambda n, d: (n * top_k,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "sonic_expert_frequency",
                lambda n, d: (num_experts,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "sonic_expert_frequency_offset",
                lambda n, d: (num_experts + 1,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            # Flextrain-shape index_mapping for unified router-gate-bwd.
            ActivationField(
                "index_mapping",
                lambda n, d: (n, top_k),
                torch.int32,
                tier=0,
            ),
        )

    def expert_counts_gpu(self, slot: Any) -> torch.Tensor:
        """Sonic stores per-expert counts directly in expert_frequency."""
        return slot.sonic_expert_frequency

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def fwd(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        chosen_experts: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        *,
        out: torch.Tensor,
        residual: torch.Tensor | None,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> None:
        if residual is not None:
            raise ValueError(
                "SonicMoEExpertCompute does not support inline residual "
                "in the gather; caller must add it after fwd returns."
            )
        from quack.gemm_interface import gemm, gemm_gated
        from sonicmoe.functional.triton_kernels import (
            TC_topk_router_metadata_triton,
        )
        from sonicmoe.functional.forward import _router_forward

        # Dimensions. Option-B layout: w_up (E, 2F, d), w_down (E, d, F).
        num_experts = weights["w_up"].shape[0]
        d_model = weights["w_up"].shape[2]
        F = weights["w_down"].shape[2]
        T = x.shape[0]
        K = router_weights.shape[1]
        TK = T * K

        # router_scores_flat: router_weights viewed as (TK,) float32
        # (sonic's gemm_dgated needs fp32 for the scaling; we cast here).
        router_scores_flat = router_weights.view(-1).float().contiguous()

        # 1. Build routing bookkeeping using sonicmoe's TC (token-choice)
        # path — fixed top-K, no num_activated_expert_per_token_offset.
        # Mirrors sonicmoe/functional/__init__.py:364 (moe_TC_softmax_topk_layer).
        TC_topk_router_metadata_triton(
            chosen_experts.contiguous(),  # (T, K) int32
            num_experts,
            slot.sonic_expert_frequency,
            slot.sonic_expert_frequency_offset,
            slot.sonic_x_gather_idx,
            slot.sonic_s_scatter_idx,
            slot.sonic_s_reverse_scatter_idx,
        )

        # 3. Build flextrain-shape index_mapping (T, K) for unified
        # router-gate-bwd. s_scatter_idx[i] = j means scattered position
        # i was originally at token-major flat position j. So:
        #   index_mapping.flatten()[s_scatter_idx[i]] = i
        # produces the inverse permutation.
        #
        # NOTE — slot.sonic_s_scatter_idx and slot.index_mapping are
        # both allocated for the MAX (T*K); the metadata kernel only
        # writes the first actual-TK entries. Slice both to the
        # current-chunk TK before scattering.
        slot.index_mapping.view(-1)[:TK].index_copy_(
            0,
            slot.sonic_s_scatter_idx[:TK].long(),
            torch.arange(TK, device=x.device, dtype=torch.int32),
        )

        # Layout bridge to sonicmoe's (E, out, in) storage:
        #
        # GATING CONVENTION for w_up (unchanged from old layout):
        #   flextrain (and the parity-test reference) chunk(2) the
        #   pre-activation as `value, gate = pre.chunk(2, dim=-1)`,
        #   then compute `silu(gate) * value`. So flextrain's chunked
        #   layout along the 2F axis is [up_F, gate_F] — UP FIRST,
        #   GATE SECOND.
        #   sonicmoe's _swiglu does `g = x[..., 0::2]; u = x[..., 1::2]`
        #   then `u * silu(g)` — EVEN=gate, ODD=up, concat_layout=False.
        # Conversion: split flextrain's chunked (up, gate) along the 2F
        # axis (now dim=1 under option-B), interleave gate at even
        # positions, up at odd.
        #
        # Option-B layout makes w_down a free pass-through: flextrain
        # stores (E, d, F) which matches sonicmoe's expected w_down_native
        # exactly — no transpose, no copy.
        up, gate = weights["w_up"].chunk(2, dim=1)        # each (E, F, d); flextrain [up; gate] along dim=1
        w_up_native = torch.stack([gate, up], dim=2).reshape(
            num_experts, 2 * F, d_model,
        )                                                  # (E, 2F_interleaved, d) — even=gate, odd=up; contiguous (stack+reshape allocs fresh storage)
        w_down_native = weights["w_down"]                  # (E, d, F) — direct, sonicmoe's native shape under option-B

        # 4. Up-projection via gemm_gated. Mirror sonicmoe verbatim
        # (functional/__init__.py:114). MoE.forward passes
        # `c_fc.weight.permute(1, 2, 0)` as w1, and _UpProjection.forward
        # calls `gemm_gated(.., w1.permute(2, 1, 0), ..)`. We replicate.
        # NOTE — slot.x_up is allocated for max TK; quack's gemm_gated
        # asserts preact_out.shape[0] == postact_out.shape[0], so slice
        # to the current chunk's TK.
        a_post = scratch_fn((TK, F), x.dtype)
        w1_view = w_up_native.permute(1, 2, 0)            # (2F, d, E) — sonicmoe's saved w1
        gemm_gated(
            x,
            w1_view.permute(2, 1, 0),                     # (E, d, 2F)
            activation="swiglu",
            cu_seqlens_m=slot.sonic_expert_frequency_offset,
            A_idx=slot.sonic_x_gather_idx[:TK],
            preact_out=slot.x_up[:TK],
            postact_out=a_post,
            store_preact=True,
            bias=None,
        )

        # 5. Down-projection. Mirror sonicmoe (_DownProjection.forward
        # line 237): `gemm(a, w2.permute(2, 1, 0), ...)` where
        # `w2 = c_proj.weight.permute(1, 2, 0)`.
        y = scratch_fn((TK, d_model), x.dtype)
        w2_view = w_down_native.permute(1, 2, 0)          # (d, F, E) — sonicmoe's saved w2
        gemm(
            a_post,
            w2_view.permute(2, 1, 0),                     # (E, F, d)
            out=y,
            cu_seqlens_m=slot.sonic_expert_frequency_offset,
            bias=None,
        )

        # 6. Gather + weighted-sum: out (T, d) = sum over K of
        # y[scatter_pos(t, k)] * router_scores[t, k]. TC fixed-K
        # routing — pass None for num_activated_expert_per_token_offset
        # and is_varlen_K=False. Mirrors sonicmoe's TC path
        # (functional/__init__.py:391).
        _router_forward(
            y=y,
            o=out,
            topk_scores=router_scores_flat,
            s_reverse_scatter_idx=slot.sonic_s_reverse_scatter_idx[:TK],
            num_activated_expert_per_token_offset=None,
            varlen_K_max=K,
            H=d_model,
            is_varlen_K=False,
        )

    def fwd_recompute(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        *,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
    ) -> Any:
        """Re-run the up-projection part of fwd to repopulate
        ``slot.x_up`` when the layer's save tier was < 3. Skips the
        down-projection and combine — bwd doesn't need ``y`` or ``out``.

        All routing metadata fields (sonic_*, index_mapping) are tier 0
        and already populated from fwd; we only re-run the up-proj
        ``gemm_gated``. Returns ``None`` (no handoff needed; bwd reads
        slot fields the same way it does after a tier-3 fwd).
        """
        from quack.gemm_interface import gemm_gated

        # Option-B layout: w_up (E, 2F, d), w_down (E, d, F).
        num_experts = weights["w_up"].shape[0]
        d_model = weights["w_up"].shape[2]
        F = weights["w_down"].shape[2]
        T = x.shape[0]
        K = slot.chosen_experts.shape[1]
        TK = T * K

        # Materialize sonicmoe-layout w_up (interleaved gate/up). See fwd
        # for the full gating convention discussion. Under option-B the
        # chunk happens along dim=1 (the 2F axis), no extra transpose.
        up, gate = weights["w_up"].chunk(2, dim=1)        # (E, F, d) each
        w_up_native = torch.stack([gate, up], dim=2).reshape(
            num_experts, 2 * F, d_model,
        )                                                  # (E, 2F_interleaved, d)

        # Up-projection only — repopulates slot.x_up. Throwaway
        # postact_out scratch since we don't need a_post here.
        a_post_scratch = scratch_fn((TK, F), x.dtype)
        w1_view = w_up_native.permute(1, 2, 0)
        gemm_gated(
            x,
            w1_view.permute(2, 1, 0),
            activation="swiglu",
            cu_seqlens_m=slot.sonic_expert_frequency_offset,
            A_idx=slot.sonic_x_gather_idx[:TK],
            preact_out=slot.x_up[:TK],
            postact_out=a_post_scratch,
            store_preact=True,
            bias=None,
        )
        return None

    def bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        *,
        slot: Any,
        chunk_extra: MutableMapping[str, Any],
        layer_id: int,
        primary_stream: torch.cuda.Stream,
        secondary_stream: torch.cuda.Stream | None,
        scratch_fn: Callable[[tuple[int, ...], torch.dtype], torch.Tensor],
        dx: torch.Tensor | None = None,
        recompute_handoff: Any = None,
        # Generic LoRA capture (populated below when not None).
        lora_capture: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if recompute_handoff is not None:
            raise NotImplementedError(
                "SonicMoEExpertCompute does not support tier <3 recompute."
            )
        from quack.gemm_interface import gemm
        from sonicmoe.functional.backward import (
            _down_projection_backward_act,
            _up_projection_backward_act,
            _token_broadcast_backward,
        )
        from flextrain.ops import flextrain_moe_scatter_routing_weights

        # Dimensions. Option-B layout: w_up (E, 2F, d), w_down (E, d, F).
        num_experts = weights["w_up"].shape[0]
        d_model = weights["w_up"].shape[2]
        F = weights["w_down"].shape[2]
        T = dy.shape[0]
        K = slot.index_mapping.shape[1]
        TK = T * K

        # router_scores in fp32 (sonic's down-bwd kernel uses fp32 for
        # the gradient scaling chain). slot.router_weights is allocated
        # for max-T; slice to actual T.
        router_scores_flat = (
            slot.router_weights[:T].reshape(-1).float().contiguous()
        )

        # Layout bridge — see fwd for full discussion. Same construction:
        # gating-aware interleave for w_up; w_down passes through
        # unchanged (option-B's (E, d, F) matches sonicmoe natively).
        up, gate = weights["w_up"].chunk(2, dim=1)        # (E, F, d) each; flextrain [up; gate] along dim 1
        w_up_native = torch.stack([gate, up], dim=2).reshape(
            num_experts, 2 * F, d_model,
        )                                                  # (E, 2F_interleaved, d) — even=gate, odd=up
        w_down_native = weights["w_down"]                  # (E, d, F) — direct, no copy

        # sonicmoe's bwd helpers expect `w1`/`w2` in the form that
        # MoE.forward constructs: `Experts.weight.permute(1, 2, 0)`.
        w1_sonic = w_up_native.permute(1, 2, 0)           # (2F, d, E)
        w2_sonic = w_down_native.permute(1, 2, 0)         # (d, F, E)

        # 1. Down-projection bwd + activation bwd, fused via gemm_dgated.
        #    - dh: (TK, 2F) — gradient at the pre-SwiGLU activation.
        #    - a_prime: (TK, F) — recomputed post-act, used as X for dw_down.
        #    - ds: (TK,) token-major — d/d-router-score per slot.
        dh = scratch_fn((TK, 2 * F), dy.dtype)
        a_prime = scratch_fn((TK, F), dy.dtype)
        ds = torch.empty(TK, dtype=router_scores_flat.dtype, device=dy.device)

        _down_projection_backward_act(
            dout=dy,
            h=slot.x_up[:TK],
            w2=w2_sonic,
            dh=dh,
            ds=ds,
            b2=None,
            db2=None,
            a_prime=a_prime,
            topk_scores=router_scores_flat,
            expert_frequency_offset=slot.sonic_expert_frequency_offset,
            x_gather_idx=slot.sonic_x_gather_idx[:TK],
            s_scatter_idx=slot.sonic_s_scatter_idx[:TK],
            activation_type="swiglu",
        )

        # 2. dw_down = gemm(dy.T, a_prime). Mirror sonicmoe's exact
        # allocation pattern (functional/__init__.py:313): in sonicmoe
        # `dw2 = torch.empty_like(w2)` where w2 is the (d, F, E) view of
        # the underlying (E, d, F) storage. `empty_like` of a non-
        # contiguous view PRESERVES strides, so dw2 has shape (d, F, E)
        # with strides (F, 1, d*F) — NOT a fresh contiguous allocation.
        # Then `dw2.permute(2, 0, 1)` → (E, d, F) view with strides
        # (d*F, F, 1). Replicate by allocating (E, d, F) contiguous
        # directly and taking that as the out= view.
        if grads.get("g_down") is not None:
            dw_down_storage = torch.empty(
                num_experts, d_model, F,    # (E, d, F) sonicmoe storage
                dtype=weights["w_down"].dtype, device=dy.device,
            )
            dw2 = dw_down_storage.permute(1, 2, 0)            # (d, F, E) view
            gemm(
                dy.T,
                a_prime,
                out=dw2.permute(2, 0, 1),                     # (E, d, F)
                cu_seqlens_k=slot.sonic_expert_frequency_offset,
                A_idx=slot.sonic_x_gather_idx[:TK],
                batch_idx_permute=None,
                dynamic_scheduler=False,
            )
            # Under option-B layout, g_down is (E, d, F) — identical to
            # dw_down_storage. Direct add, no transpose needed.
            grads["g_down"].add_(dw_down_storage)

        # 3. Up-projection dgrad: dx_expanded (TK, d) = dh @ w_up^T.
        # _up_projection_backward_act handles the gemm internally — it
        # expects w1 in sonicmoe's (2F, d, E) storage and does the
        # `.permute(2, 0, 1)` itself.
        dx_expanded = scratch_fn((TK, d_model), dy.dtype)
        _up_projection_backward_act(
            w1=w1_sonic,
            dx_expanded=dx_expanded,
            dh=dh,
            db1=None,
            expert_frequency_offset=slot.sonic_expert_frequency_offset,
            is_glu_activation=True,
            concat_layout=False,
        )

        # Stage per-expert grouped intermediates for downstream LoRA
        # finalize. Sonic differs from flextrain/scatter in three ways:
        #
        # 1. Sonic uses gather-indexing internally — no naturally-
        #    materialized gathered x / dy. Materialize via index_select
        #    (~256 MiB each at TK=65536, d=2048, bf16).
        #
        # 2. Sonic stores the pre-SwiGLU activation and its gradient
        #    in INTERLEAVED gate/up layout (even=gate, odd=up); the
        #    LoRA finalize and the LoRA factor B both use the
        #    flextrain CHUNKED convention ([up_F, gate_F] along the
        #    2F axis). Deinterleave both ``dh`` and ``slot.x_up`` to
        #    chunked before staging.
        #
        # 3. expert_offsets — sonic_expert_frequency_offset has leading
        #    0, length E+1. grouped_mm wants (E,) cumulative ending
        #    in TK; slice off the 0.
        if lora_capture is not None:
            x_gather_idx_long = slot.sonic_x_gather_idx[:TK].long()
            scattered_x_grouped = scratch_fn((TK, d_model), x.dtype)
            scattered_x_grouped.copy_(x.index_select(0, x_gather_idx_long))
            scattered_upstream_grouped = scratch_fn((TK, d_model), dy.dtype)
            scattered_upstream_grouped.copy_(
                dy.index_select(0, x_gather_idx_long)
            )
            # Deinterleave dh: even=gate, odd=up → chunked [up, gate].
            # Two slices are non-contiguous; concat materializes a fresh
            # contiguous (TK, 2F) tensor.
            dh_up = dh[:, 1::2]                          # (TK, F) — up
            dh_gate = dh[:, 0::2]                        # (TK, F) — gate
            dx_up_up_chunked = torch.cat([dh_up, dh_gate], dim=-1)
            # Deinterleave slot.x_up the same way (sonic's preact is
            # also interleaved — gemm_gated wrote it in sonic's native
            # format).
            x_up_inter = slot.x_up[:TK]
            x_up_up_part = x_up_inter[:, 1::2]            # (TK, F) — up
            x_up_gate_part = x_up_inter[:, 0::2]          # (TK, F) — gate
            x_up_chunked = torch.cat([x_up_up_part, x_up_gate_part], dim=-1)

            lora_capture["scattered_x_grouped"] = scattered_x_grouped
            lora_capture["dx_up_up_grouped"] = dx_up_up_chunked
            lora_capture["scattered_upstream_grouped"] = scattered_upstream_grouped
            lora_capture["x_up_grouped"] = x_up_chunked
            lora_capture["expert_offsets"] = (
                slot.sonic_expert_frequency_offset[1:]
            )
            lora_capture["TK"] = TK

        # 4. dw_up = gemm(x.T, dh). Same allocation pattern as step 2.
        # sonicmoe (functional/__init__.py:187): `dw1 = torch.empty_like(w1)`
        # where w1 is the (2F, d, E) view of (E, 2F, d) storage, then
        # writes via dw1.permute(2, 1, 0) → (E, d, 2F).
        if grads.get("g_up") is not None:
            dw_up_storage = torch.empty(
                num_experts, 2 * F, d_model,    # (E, 2F, d) sonicmoe storage
                dtype=weights["w_up"].dtype, device=dy.device,
            )
            dw1 = dw_up_storage.permute(1, 2, 0)              # (2F, d, E) view
            gemm(
                x.T,
                dh,
                out=dw1.permute(2, 1, 0),                     # (E, d, 2F)
                cu_seqlens_k=slot.sonic_expert_frequency_offset,
                A_idx=slot.sonic_x_gather_idx[:TK],
                batch_idx_permute=None,
                dynamic_scheduler=False,
            )
            # dw_up_storage is (E, 2F_interleaved, d) — sonicmoe layout
            # with interleaved gate(even)/up(odd) along dim 1. Under
            # option-B, g_up is (E, 2F, d) chunked [up; gate] along dim 1.
            # The 2F axis is dim 1 in both — de-interleave directly along
            # dim 1 and concat as [up; gate].
            dw_gate = dw_up_storage[:, 0::2, :]                # (E, F, d) — gate
            dw_up_part = dw_up_storage[:, 1::2, :]             # (E, F, d) — up
            grads["g_up"].add_(torch.cat([dw_up_part, dw_gate], dim=1))

        # 5. dx_reduced = sum_k dx_expanded[scatter_pos(t, k)]: K-dim
        # reduction back to per-token dx.
        if dx is None:
            dx = torch.empty_like(dy)
        _token_broadcast_backward(
            dx_reduced=dx,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=slot.sonic_s_reverse_scatter_idx[:TK],
            num_activated_expert_per_token_offset=None,
            varlen_K_max=K,
            H=d_model,
            is_varlen_K=False,
        )

        # 6. dprobs for the unified router-gate-bwd: ds is (TK,)
        # token-major fp32. Cast to compute dtype, view as (T, K),
        # scatter to (TK,) scattered, expose as (TK, 1).
        dprobs_flat = scratch_fn((TK,), dy.dtype)
        flextrain_moe_scatter_routing_weights(
            ds.view(T, K).to(dy.dtype).contiguous(),
            slot.index_mapping[:T],
            out=dprobs_flat,
        )
        slot.aux["moe_dprobs"] = dprobs_flat.unsqueeze(-1)

        return dx

