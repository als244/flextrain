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
    ) -> torch.Tensor:
        """Run gather-bwd + per-expert MLP-bwd + scatter-bwd. Accumulates
        wgrads into ``grads`` (caller pre-zeroes if needed). Returns the
        per-token ``dx`` (fresh alloc if ``dx`` kwarg was None).

        Backend writes any router-grad scratch the caller needs into
        ``slot.aux`` under backend-specific keys; today the caller
        (``routed_swiglu_moe_bwd``) reads ``slot.aux["moe_dprobs"]`` (a
        ``(TK, 1)`` scattered-layout tensor for flextrain) to feed the
        router-gate-bwd. As more backends are added, this contract may
        grow into a typed return — for now slot.aux is the carrier."""
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

        # w_up: (E, d, 2F), w_down: (E, F, d). Pull dims from weights.
        num_experts = weights["w_up"].shape[0]
        expert_dim = weights["w_down"].shape[1]
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
        # Flextrain-only optional kwargs (LoRA hook + skip_grads).
        # Not part of the protocol; this backend's caller in
        # routed_swiglu_moe_bwd forwards them through. Other backends
        # do not accept these.
        skip_grads: frozenset[str] = frozenset(),
        lora_per_expert_callback: object | None = None,
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
        expert_dim = weights["w_down"].shape[1]
        expert_counts_cpu = self._host_counts(chunk_extra, layer_id, num_experts)

        # 1. Scatter dy.
        scattered_upstream = scratch_fn((TK, d_model), dy.dtype)
        flextrain_moe_scatter(dy, slot.index_mapping, out=scattered_upstream)

        # 2. Get scattered_x (either from recompute handoff or scatter
        #    fresh from saved x).
        if recompute_handoff is not None:
            scattered_x = recompute_handoff
        else:
            scattered_x = scratch_fn((TK, d_model), dy.dtype)
            flextrain_moe_scatter(x, slot.index_mapping, out=scattered_x)

        # 3. Per-expert bwd loop. Overwrites scattered_upstream with
        #    per-slot dx; accumulates g_up/g_down (or fires LoRA cb);
        #    writes dprobs.
        srw = slot.scattered_router_weights[:TK, :]
        dprobs = torch.zeros_like(srw)
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
            expert_dim=expert_dim,
            primary_stream=primary_stream,
            secondary_stream=secondary_stream,
            skip_grads=skip_grads,
            lora_per_expert_callback=lora_per_expert_callback,
        )

        # 4. Gather per-slot dx back to per-token dx.
        if dx is None:
            dx = torch.empty_like(dy)
        flextrain_moe_gather(scattered_upstream, slot.index_mapping, out=dx)

        # 5. Hand dprobs (scattered layout) to the caller for the
        #    router-gate-bwd via slot.aux.
        slot.aux["moe_dprobs"] = dprobs

        return dx
