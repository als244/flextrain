"""LoRA wrapper layer — adds LoRA support to ANY block-composed Layer.

Design: the wrapper holds a base :class:`Layer` and a set of LoRA
configs. At forward time it computes effective weights ``W' = W + A @ B * scale``
for each LoRA target and passes them to the base layer's forward (the
base layer sees a normal ``weights`` dict and does its plain matmul).

At backward time the wrapper receives the engine's gradient routing,
calls the base layer's backward with a temporary ``g_<target>``
buffer (which the base layer expects to write the dL/dW_effective
into), then decomposes that into ``g_<target>_lora_a`` / ``g_<target>_lora_b``::

    dL/dA = dL/dW_effective @ B^T * scale
    dL/dB = A^T @ dL/dW_effective * scale

The base block's frozen ``W`` is never updated. The two small LoRA
matrices A and B are trained as standard AdamW parameters.

This keeps :class:`LlamaBlock`, :class:`OLMoEBlock`, etc., **unchanged**.
LoRA is purely a layer-level wrapper.

Caveats / current limitations
-----------------------------
* W' = W + B@A*s is materialized fresh each forward call. For very
  large W this adds memory + compute (one extra GEMM per LoRA target
  per fwd). With the default rank=16 the cost is ~r/d_out fraction
  of the base matmul — negligible.
* LoRA targets must be 2-D linear projections. 3-D MoE expert stacks
  are not supported by this wrapper (use a per-expert LoRA in the MoE
  block as a future extension).
* The wrapper currently supports any base layer that exposes its
  ``param_spec``, ``schema``, and the standard
  ``forward / forward_recompute / backward / compute_cost`` methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping

import torch

from flextrain.core.layer import (
    BackwardIntermediates,
    ChunkMeta, ComputeCost, LayerContext, ParamSpec, TensorSpec,
)


@dataclass(frozen=True)
class LoRATargetConfig:
    """LoRA on one base parameter.

    ``target_name`` is the base layer's ParamSpec entry name (e.g. ``"w_q"``).
    The wrapper auto-discovers the base shape by reading the entry's
    ``shape_fn`` at attach time, so users only need to specify the name +
    rank + alpha.

    Optional dtype overrides apply only to the A/B adapters (not the
    base ``W``). When ``None``, the adapters inherit the base spec's
    dtypes. Use these to keep base bf16 while training A/B in fp32 —
    matches HF PEFT's default (base bf16 + LoRA fp32).
    """

    target_name: str
    rank: int = 16
    alpha: float = 16.0
    adapter_compute_dtype: torch.dtype | None = None
    adapter_master_dtype: torch.dtype | None = None
    adapter_grad_dtype: torch.dtype | None = None
    adapter_opt_state_dtype: torch.dtype | None = None

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def a_name(self) -> str:
        return f"{self.target_name}_lora_a"

    @property
    def b_name(self) -> str:
        return f"{self.target_name}_lora_b"


def expand_targets(
    targets: str | Iterable[str] | None,
    available_2d_names: Iterable[str],
) -> tuple[str, ...]:
    """Resolve ``"all"`` / explicit names / ``None`` / empty into a tuple."""
    if targets is None or targets == ():
        return ()
    if targets == "all":
        return tuple(available_2d_names)
    return tuple(targets)


def _discover_lora_eligible_names(
    param_spec: ParamSpec, dims: Mapping[str, int],
) -> tuple[str, ...]:
    """Find every base param that's a 2-D matrix or a 3-D MoE expert
    stack — both are LoRA-eligible.

    Includes 3-D shared-expert stacks (e.g. ``w_shared_up`` /
    ``w_shared_down`` of shape ``(S, d_in, d_out)``) — each of the S
    shared experts gets its own per-expert A/B matrices, mirroring
    routed expert handling.

    Excludes:
    * 1-D tensors (norm γ, biases): not linear projections.
    * Routers (name contains ``router``): typically not adapted in
      LoRA fine-tuning. Users can still target them explicitly.
    * Shared-expert gates (name contains ``shared_expert_gate``): the
      per-token sigmoid scalar gate is router-like (single-output
      projection) and not normally adapted via LoRA.
    """
    out = []
    for t in param_spec.tensors:
        shape = t.shape(dims)
        if len(shape) not in (2, 3):
            continue
        nm = t.name.lower()
        if "router" in nm:
            continue
        if "shared_expert_gate" in nm:
            continue
        out.append(t.name)
    return tuple(out)


# Back-compat alias.
_discover_2d_param_names = _discover_lora_eligible_names


def _make_lora_specs(
    base_spec: TensorSpec, cfg: LoRATargetConfig, dims: Mapping[str, int],
) -> tuple[TensorSpec, TensorSpec]:
    """Build the (A, B) TensorSpecs for one LoRA target.

    Two layouts:

    * Base 2-D ``(d_in, d_out)`` → ``A: (d_in, r)``, ``B: (r, d_out)``.
    * Base 3-D MoE expert stack ``(E, d_in, d_out)`` → per-expert
      adapters ``A: (E, d_in, r)``, ``B: (E, r, d_out)``. Each expert
      gets its own independent low-rank delta.
    """
    base_shape = base_spec.shape(dims)
    r = cfg.rank
    a_compute = cfg.adapter_compute_dtype or base_spec.compute_dtype
    a_master = cfg.adapter_master_dtype or base_spec.master_dtype
    a_grad = cfg.adapter_grad_dtype or base_spec.grad_dtype
    a_opt = cfg.adapter_opt_state_dtype or base_spec.opt_state_dtype
    if len(base_shape) == 2:
        d_in, d_out = base_shape
        a = TensorSpec(
            name=cfg.a_name,
            shape_fn=lambda d, di=d_in, R=r: (di, R),
            compute_dtype=a_compute,
            master_dtype=a_master,
            grad_dtype=a_grad,
            opt_state_dtype=a_opt,
            optimizer="adamw",
        )
        b = TensorSpec(
            name=cfg.b_name,
            shape_fn=lambda d, do=d_out, R=r: (R, do),
            compute_dtype=a_compute,
            master_dtype=a_master,
            grad_dtype=a_grad,
            opt_state_dtype=a_opt,
            optimizer="adamw",
        )
        return a, b
    if len(base_shape) == 3:
        E, d_in, d_out = base_shape
        a = TensorSpec(
            name=cfg.a_name,
            shape_fn=lambda d, EE=E, di=d_in, R=r: (EE, di, R),
            compute_dtype=a_compute,
            master_dtype=a_master,
            grad_dtype=a_grad,
            opt_state_dtype=a_opt,
            optimizer="adamw",
        )
        b = TensorSpec(
            name=cfg.b_name,
            shape_fn=lambda d, EE=E, do=d_out, R=r: (EE, R, do),
            compute_dtype=a_compute,
            master_dtype=a_master,
            grad_dtype=a_grad,
            opt_state_dtype=a_opt,
            optimizer="adamw",
        )
        return a, b
    raise ValueError(
        f"LoRA target {base_spec.name!r} must be 2-D or 3-D, "
        f"got shape {base_shape}"
    )


class LoRAWrapperLayer:
    """Wrap any base :class:`Layer` to add LoRA on selected targets.

    Forward path: build effective weights ``W' = W + A @ B * scale`` for
    each LoRA target, call ``base.forward(...)`` with the modified
    weights dict.

    Backward path: allocate temporary ``g_<target>`` buffers for the
    base layer to accumulate dL/dW_effective into; after backward,
    decompose into LoRA A/B grads and discard the base grad (frozen).
    """

    def __init__(
        self,
        base: object,                                 # any Layer instance
        lora_targets: str | Iterable[str] | None,
        *,
        rank: int = 16,
        alpha: float = 16.0,
        dims: Mapping[str, int],
        adapter_compute_dtype: torch.dtype | None = None,
        adapter_master_dtype: torch.dtype | None = None,
        adapter_grad_dtype: torch.dtype | None = None,
        adapter_opt_state_dtype: torch.dtype | None = None,
    ) -> None:
        self.base = base
        self.layer_id = base.layer_id
        # MoE marker passes through transparently.
        self.moe_chunk_config = getattr(base, "moe_chunk_config", None)
        if self.moe_chunk_config is None:
            # `Layer` Protocol probes this; absent on non-MoE base layers.
            pass

        all_2d = _discover_2d_param_names(base.param_spec, dims)
        target_names = expand_targets(lora_targets, all_2d)
        unknown = set(target_names) - set(all_2d)
        if unknown:
            raise ValueError(
                f"LoRA targets {unknown!r} are not 2-D params of the base "
                f"layer. Available: {all_2d}"
            )

        # Build target configs.
        self.targets: tuple[LoRATargetConfig, ...] = tuple(
            LoRATargetConfig(
                target_name=n, rank=rank, alpha=alpha,
                adapter_compute_dtype=adapter_compute_dtype,
                adapter_master_dtype=adapter_master_dtype,
                adapter_grad_dtype=adapter_grad_dtype,
                adapter_opt_state_dtype=adapter_opt_state_dtype,
            )
            for n in target_names
        )
        self._target_set: frozenset[str] = frozenset(target_names)

        # Compose the param spec: mark targets frozen, add A/B.
        new_base_specs = []
        target_specs: dict[str, TensorSpec] = {}
        for t in base.param_spec.tensors:
            if t.name in self._target_set:
                target_specs[t.name] = t
                new_base_specs.append(
                    TensorSpec(
                        name=t.name, shape_fn=t.shape_fn,
                        compute_dtype=t.compute_dtype,
                        master_dtype=t.master_dtype,
                        grad_dtype=t.grad_dtype,
                        opt_state_dtype=t.opt_state_dtype,
                        optimizer=t.optimizer,
                        frozen=True,
                    )
                )
            else:
                new_base_specs.append(t)
        ab_specs: list[TensorSpec] = []
        for cfg in self.targets:
            a_spec, b_spec = _make_lora_specs(
                target_specs[cfg.target_name], cfg, dims,
            )
            ab_specs.extend([a_spec, b_spec])
        self.param_spec = ParamSpec(tensors=tuple(new_base_specs + ab_specs))
        self.schema = base.schema  # unchanged

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    def _build_effective_weights(
        self, weights: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return a new weights dict with ``W' = W + A @ B * scale`` for
        every LoRA target. Non-target weights pass through unchanged.

        For 3-D MoE expert stacks ``W: (E, d_in, d_out)``,
        ``A: (E, d_in, r)``, ``B: (E, r, d_out)``: per-expert
        ``W'[e] = W[e] + A[e] @ B[e] * scale`` via batched matmul.
        """
        eff = dict(weights)
        for cfg in self.targets:
            W = weights[cfg.target_name]
            A = weights[cfg.a_name]
            B = weights[cfg.b_name]
            # If A/B are higher precision than W, the LoRA delta is
            # computed at the higher precision and cast back to W's
            # dtype so the base block's matmul receives the expected
            # dtype. (HF PEFT does the analogous thing in the unmerged
            # path: the LoRA branch runs in fp32 and the result is
            # cast to bf16 before adding to the base output.)
            if W.dim() == 2:
                delta = (A @ B) * cfg.scale
            elif W.dim() == 3:
                delta = torch.bmm(A, B) * cfg.scale
            else:
                raise ValueError(
                    f"LoRA: unexpected W rank {W.dim()} for {cfg.target_name!r}"
                )
            eff[cfg.target_name] = (W + delta.to(W.dtype)).contiguous()
        return eff

    # ------------------------------------------------------------------
    # Layer protocol.
    # ------------------------------------------------------------------

    def forward(self, x, chunk: ChunkMeta, weights, slot, ctx: LayerContext):
        # No slot.aux state — bwd reads A/B from `weights` directly and
        # rebuilds ``eff`` lazily. This avoids the issue where the
        # engine's activation prefetch creates a fresh ``ActivationSlot``
        # (with empty ``aux``) when restoring offloaded activations,
        # which would lose any state stashed in fwd.
        eff = self._build_effective_weights(weights)
        return self.base.forward(x, chunk, eff, slot, ctx)

    def forward_recompute(self, slot, chunk, weights, ctx) -> None:
        # Recompute uses the effective weights — rebuild fresh each call
        # to avoid the cross-layer OOM that storing them would cause.
        eff = self._build_effective_weights(weights)
        self.base.forward_recompute(slot, chunk, eff, ctx)

    # ------------------------------------------------------------------
    # Fast-path infra: which projections support the dgrad/wgrad split.
    # ------------------------------------------------------------------
    #
    # All known projection names that have a fast-path:
    #   w_q, w_k, w_v        (attention deferred Wgrads -> dense fast path)
    #   w_o                  (attention inline Wgrad -> dense fast path)
    #   w_1, w_3             (FFN deferred Wgrads -> dense fast path)
    #   w_2                  (FFN inline Wgrad -> dense fast path)
    #   w_up, w_down         (MoE per-expert -> rank-r in callback)
    #   w_router             (MoE 2-D -> rank-r in callback) -- normally
    #                        excluded by default lora_targets="all" via
    #                        _discover_lora_eligible_names; only fires
    #                        when a user explicitly targets it.
    _FAST_PATH_TARGETS = frozenset(
        ("w_q", "w_k", "w_v", "w_o",
         "w_1", "w_2", "w_3",
         "w_up", "w_down", "w_router",
         "w_shared_up", "w_shared_down", "w_shared_expert_gate",
         "w_lin_out", "w_lin_qkvz", "w_lin_ba")
    )

    def lora_target_names(self) -> frozenset[str]:
        """Names of base-layer projections this wrapper has LoRA on
        (e.g. ``frozenset({"w_q", "w_k", "w_o", ...})``). The engine
        passes this to ``backward_wgrad`` as ``skip_target_names``."""
        return self._target_set

    def _fast_path_target_names(self) -> frozenset[str]:
        """Subset of LoRA targets that support the dgrad/wgrad
        skip-and-capture protocol. Currently the dense Llama
        deferred-Wgrad set; non-overlap (w_o, w_2, MoE) goes through
        the slow path."""
        return self._target_set & self._FAST_PATH_TARGETS

    def _slow_path_target_names(self) -> frozenset[str]:
        return self._target_set - self._fast_path_target_names()

    # ------------------------------------------------------------------
    # backward(): legacy monolithic shim (engine still calls this for
    # non-LoRA-aware paths and for any code that wraps a non-Llama base).
    # ------------------------------------------------------------------

    def backward(self, dx, chunk, weights, grads, slot, ctx) -> torch.Tensor:
        """Per-chunk backward. Delegates to ``backward_dgrad`` +
        ``backward_wgrad`` + ``accumulate_lora_grads``. Bit-equivalent
        (modulo bf16 reorder) to the pre-Phase-2 monolithic
        materialize-then-decompose path for slow-path projections,
        and uses rank-r matmuls for fast-path projections."""
        upstream_dx, intermediates = self.backward_dgrad(
            dx, chunk, weights, grads, slot, ctx,
            skip_target_names=self._target_set,
        )
        self.backward_wgrad(
            intermediates, weights, grads, slot, ctx,
            skip_target_names=self._target_set,
        )
        self.accumulate_lora_grads(intermediates, weights, grads, ctx)
        return upstream_dx

    # ------------------------------------------------------------------
    # backward_dgrad / backward_wgrad: split form. The wrapper builds
    # ``eff`` once per call and forwards to the base layer's split.
    # ------------------------------------------------------------------

    def backward_dgrad(
        self, dx, chunk, weights, grads, slot, ctx,
        *, skip_target_names: frozenset[str] = frozenset(),
    ) -> tuple[torch.Tensor, BackwardIntermediates]:
        """Wrap the base's ``backward_dgrad``. ``skip_target_names`` is
        widened to include both this wrapper's LoRA targets AND
        whatever the engine already wanted to skip (the LoRA fast-path
        set inside this wrapper specifically)."""
        eff = self._build_effective_weights(weights)
        # The base also needs slow-path scratch grad buffers for any
        # LoRA target that can't take the fast path. With the current
        # set _FAST_PATH_TARGETS this is empty for all supported
        # archs, but kept for safety / future projections.
        slow_path = self._slow_path_target_names()
        if slow_path:
            scratch_grads: dict[str, torch.Tensor] = {}
            for cfg in self.targets:
                if cfg.target_name in slow_path:
                    W = eff[cfg.target_name]
                    scratch_grads[
                        f"g_{cfg.target_name[2:]}"
                    ] = torch.zeros_like(W, dtype=W.dtype)
            combined_grads = {**grads, **scratch_grads}
        else:
            scratch_grads = {}
            combined_grads = grads

        skip_for_base = skip_target_names | self._fast_path_target_names()

        # Install MoE per-expert LoRA callback before calling base, IF
        # any LoRA target is a 3-D MoE expert weight (w_up/w_down +
        # w_shared_up/w_shared_down) or one of the 2-D router-like
        # projections (w_router, w_shared_expert_gate). The callback
        # fires inside ffn_moe.bwd / ffn_moe_shared.bwd much more
        # cheaply than the dense fast path (no per-expert clones).
        moe_targets = self._target_set & self._MOE_CALLBACK_TARGETS
        if moe_targets:
            slot.aux["__lora_moe_callback__"] = self._make_moe_callback(
                moe_targets, weights, grads,
            )

        upstream_dx, inter = self.base.backward_dgrad(
            dx, chunk, eff, combined_grads, slot, ctx,
            skip_target_names=skip_for_base,
        )
        if scratch_grads:
            inter.aux["lora_slow_scratch_grads"] = scratch_grads
        return upstream_dx, inter

    def _make_moe_callback(
        self,
        moe_targets: frozenset[str],
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
    ):
        """Construct a closure ``(g_name, eid, X, dY) -> None`` that
        ``ffn_moe.bwd`` calls per-expert (or once for router) for
        each LoRA-targeted projection. The closure does the rank-r
        matmul into the LoRA A/B grad accumulators directly, so we
        never materialize the full per-expert ``dW``.

        Math (per expert e, projection name with stored ``W: (E, in,
        out)``, ``A: (E, in, r)``, ``B: (E, r, out)``):

            dA[e] = X^T @ (dY @ B[e]^T) * scale       shape (in, r)
            dB[e] = (X @ A[e])^T @ dY * scale         shape (r, out)

        For the 2-D router (eid=-1, ``W: (in, out)``, ``A: (in, r)``,
        ``B: (r, out)``):

            dA = X^T @ (dY @ B^T) * scale
            dB = (X @ A)^T @ dY * scale
        """
        # Index targets by g_* name for fast lookup inside the loop.
        cfg_by_g = {
            f"g_{cfg.target_name[2:]}": cfg
            for cfg in self.targets
            if cfg.target_name in moe_targets
        }

        def _cb(g_name: str, eid: int,
                X: torch.Tensor, dY: torch.Tensor) -> None:
            cfg = cfg_by_g.get(g_name)
            if cfg is None:
                return
            A_full = weights[cfg.a_name]
            B_full = weights[cfg.b_name]
            ga_full = grads["g_" + cfg.a_name[2:]]
            gb_full = grads["g_" + cfg.b_name[2:]]
            scale = cfg.scale
            if eid >= 0:
                # 3-D MoE per-expert: pick the e-th slice.
                A = A_full[eid]                             # (in, r)
                B = B_full[eid]                             # (r, out)
                ga = ga_full[eid]                           # (in, r)
                gb = gb_full[eid]                           # (r, out)
            else:
                # 2-D router: full tensors.
                A, B, ga, gb = A_full, B_full, ga_full, gb_full
            # Rank-r matmuls -- never materialize dW = X^T @ dY.
            dY_B = dY @ B.transpose(-1, -2)                 # (T_e, r)
            dA = (X.transpose(-1, -2) @ dY_B) * scale       # (in, r)
            X_A = X @ A                                     # (T_e, r)
            dB = (X_A.transpose(-1, -2) @ dY) * scale       # (r, out)
            ga.add_(dA.to(ga.dtype))
            gb.add_(dB.to(gb.dtype))

        return _cb

    def backward_wgrad(
        self, intermediates: BackwardIntermediates,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot, ctx,
        *, skip_target_names: frozenset[str] = frozenset(),
    ) -> None:
        """Wrap the base's ``backward_wgrad``. The base sees
        ``eff`` (effective weights) and ``combined_grads`` (with
        scratch slots for slow-path projections), and skips the
        addmm for any name in our fast-path LoRA target set."""
        eff = self._build_effective_weights(weights)
        scratch_grads = intermediates.aux.get(
            "lora_slow_scratch_grads", {}
        )
        combined_grads = (
            {**grads, **scratch_grads} if scratch_grads else grads
        )
        skip_for_base = skip_target_names | self._fast_path_target_names()
        self.base.backward_wgrad(
            intermediates, eff, combined_grads, slot, ctx,
            skip_target_names=skip_for_base,
        )

    # ------------------------------------------------------------------
    # accumulate_lora_grads: the actual LoRA Wgrad math.
    # ------------------------------------------------------------------

    # Names that go through the MoE per-expert callback path (accumulated
    # inline inside ffn_moe.bwd / ffn_moe_shared.bwd, not via
    # accumulate_lora_grads).
    _MOE_CALLBACK_TARGETS = frozenset((
        "w_up", "w_down", "w_router",
        "w_shared_up", "w_shared_down", "w_shared_expert_gate",
    ))

    def accumulate_lora_grads(
        self, intermediates: BackwardIntermediates,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> None:
        """Compute and accumulate ``dA, dB`` for LoRA targets that
        weren't handled inline.

        Three routing paths:

        1. **Inline MoE callback** (``w_up, w_down, w_router``): handled
           per-expert inside ``ffn_moe.bwd`` via the callback we
           installed in :meth:`backward_dgrad`. ``dA, dB`` are already
           accumulated. Skipped here.

        2. **Dense fast path** (entry in
           ``intermediates.proj_inputs_and_grads``): rank-r matmuls
           on the captured ``(X, dY)``. Never materializes ``dW``.

        3. **Slow path** (entry in
           ``intermediates.aux["lora_slow_scratch_grads"]``): the base
           materialized the full ``dW`` into a scratch tensor and we
           decompose. With the current ``_FAST_PATH_TARGETS`` the
           slow path is unused for all supported archs; kept for
           safety.
        """
        del ctx
        scratch_grads = intermediates.aux.get(
            "lora_slow_scratch_grads", {}
        )
        for cfg in self.targets:
            target = cfg.target_name
            if target in self._MOE_CALLBACK_TARGETS:
                # Already accumulated inside ffn_moe.bwd via the
                # per-expert callback. Skip here.
                continue
            ga = grads["g_" + cfg.a_name[2:]]
            gb = grads["g_" + cfg.b_name[2:]]
            A = weights[cfg.a_name]
            B = weights[cfg.b_name]

            if target in intermediates.proj_inputs_and_grads:
                # Dense fast path. (X, dY) shapes:
                #   X: (T, in), dY: (T, out)
                #   A: (in, r), B: (r, out)
                X, dY = intermediates.proj_inputs_and_grads[target]
                dY_B = (dY @ B.transpose(-1, -2))  # (T, r)
                dA = (X.transpose(-1, -2) @ dY_B) * cfg.scale  # (in, r)
                X_A = (X @ A)  # (T, r)
                dB = (X_A.transpose(-1, -2) @ dY) * cfg.scale  # (r, out)
                ga.add_(dA.to(ga.dtype))
                gb.add_(dB.to(gb.dtype))
            elif f"g_{target[2:]}" in scratch_grads:
                # Slow path: dW was materialized in scratch and now we
                # decompose. Every supported arch should hit a fast
                # path (dense or MoE-callback) instead -- if we land
                # here it's a bug. Raise loudly so we don't silently
                # waste the full Wgrad matmul on frozen base weights.
                raise RuntimeError(
                    f"LoRA target {target!r}: hit slow scratch-dW path. "
                    "This means the wrapper allocated a scratch grad "
                    "buffer for this projection, the base layer's "
                    "addmm wrote into it, and we'd now decompose -- "
                    "exactly the wasted-compute path the fast-backward "
                    "refactor is meant to eliminate. Check that the "
                    "base layer's backward_dgrad/backward_wgrad honors "
                    "skip_target_names for this projection."
                )
            else:
                raise RuntimeError(
                    f"LoRA target {target!r}: neither fast-path "
                    "(X, dY) nor slow-path scratch dW was supplied. "
                    "Either backward_dgrad/backward_wgrad weren't "
                    "called with skip_target_names containing this "
                    "name, or the base layer doesn't honor it."
                )

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        # Approximate: same as base. The LoRA weight-merge adds a small
        # constant cost per fwd that's negligible at typical ranks.
        return self.base.compute_cost(chunk)


__all__ = [
    "LoRATargetConfig",
    "LoRAWrapperLayer",
    "expand_targets",
]
