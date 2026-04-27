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

    def backward(self, dx, chunk, weights, grads, slot, ctx) -> torch.Tensor:
        """Per-chunk backward.

        The base layer's backward accumulates ``dL/dW_effective`` into a
        scratch buffer we provide per-chunk. We decompose into ``dL/dA``
        and ``dL/dB`` per-chunk and add to the engine's grad accumulators.
        Per-chunk decomposition is correct because the chain rule is
        linear in dL/dW_effective::

            dL/dA = (sum_chunks dL/dW_eff_chunk) @ B^T * scale
                  = sum_chunks (dL/dW_eff_chunk @ B^T * scale)

        so summing per-chunk decompositions equals decomposing the sum.
        """
        # Rebuild eff lazily — see fwd comment about why we don't
        # cache it across layers.
        eff = self._build_effective_weights(weights)

        # Per-chunk scratch buffer (zeroed each chunk).
        scratch_grads: dict[str, torch.Tensor] = {}
        for cfg in self.targets:
            W = eff[cfg.target_name]
            scratch_grads[f"g_{cfg.target_name[2:]}"] = torch.zeros_like(
                W, dtype=W.dtype,
            )
        combined = dict(grads)
        combined.update(scratch_grads)

        dx_out = self.base.backward(dx, chunk, eff, combined, slot, ctx)

        # Decompose this chunk's dL/dW_effective into LoRA A/B grads.
        # Read A,B directly from ``weights`` (the engine guarantees the
        # same weights dict for fwd and bwd of a given layer/chunk; LoRA
        # A/B values are unchanged between fwd and bwd within a step).
        for cfg in self.targets:
            dW = scratch_grads[f"g_{cfg.target_name[2:]}"]
            A = weights[cfg.a_name]
            B = weights[cfg.b_name]
            ga = grads["g_" + cfg.a_name[2:]]
            gb = grads["g_" + cfg.b_name[2:]]
            if dW.dim() == 2:
                # 2-D: dA = dW @ B^T * s, dB = A^T @ dW * s.
                ga.add_(((dW @ B.transpose(-1, -2)).to(ga.dtype)) * cfg.scale)
                gb.add_(((A.transpose(-1, -2) @ dW).to(gb.dtype)) * cfg.scale)
            elif dW.dim() == 3:
                # 3-D MoE: per-expert matmuls via bmm.
                ga.add_(
                    (torch.bmm(dW, B.transpose(-1, -2)).to(ga.dtype)) * cfg.scale
                )
                gb.add_(
                    (torch.bmm(A.transpose(-1, -2), dW).to(gb.dtype)) * cfg.scale
                )
            else:
                raise ValueError(f"unexpected dW rank {dW.dim()}")

        return dx_out

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        # Approximate: same as base. The LoRA weight-merge adds a small
        # constant cost per fwd that's negligible at typical ranks.
        return self.base.compute_cost(chunk)


__all__ = [
    "LoRATargetConfig",
    "LoRAWrapperLayer",
    "expand_targets",
]
