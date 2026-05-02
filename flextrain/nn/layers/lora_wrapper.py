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


# ---------------------------------------------------------------------------
# Rank-r dA/dB grad accumulation.
#
# Per LoRA target on the dense fast path we accumulate:
#     dA = (X.T @ (dY @ B.T)) * scale   -> (in, r)
#     dB = ((X @ A).T @ dY)   * scale   -> (r, out)
#     ga += dA.cast;  gb += dB.cast
#
# Two paths:
#
# 1. Same-dtype "fused" path: when X / dY / A / B / ga / gb all share
#    a dtype (the production all-bf16 LoRA config), route the two big
#    GEMMs through cuBLASLt via the matmul_dispatcher with
#    ``alpha=scale, beta=1, C=ga, D=ga``. cuBLASLt folds the scale +
#    accumulate into the GEMM epilogue so we skip the un-fused
#    ``ga.add_(dA.to(...))`` pass. Bit-identical to eager (rel=0.0
#    in our tests; same cuBLAS kernels, same fp32 internal accum,
#    different epilogue). ~1.05-1.12x at small T (1024-4096), wash
#    at large T (>=16384) but never a regression. See
#    docs/lora_perf_notes.md for measured numbers.
#
# 2. Mixed-dtype "eager" path: when accumulators ga/gb don't match
#    the input dtype (e.g. bf16 inputs + fp32 grads when
#    lora_adapter_grad_dtype=fp32), the dispatcher's same-dtype
#    Python wrapper can't express the mixed cast in one call, so we
#    fall back to the 4-GEMM eager pipeline plus a separate
#    ga.add_(dA.to(ga.dtype)) cast/add. Correct, just doesn't get the
#    cuBLASLt epilogue fusion.
#
# We tried other approaches (torch.compile, hand-rolled Triton chain
# fusion, hand-rolled Triton epilogue fusion) -- all measured worse
# than the cuBLASLt route. See docs/lora_perf_notes.md.
# ---------------------------------------------------------------------------


# Lazy-import the dispatcher: ops/__init__.py has GPU-only side effects
# (matmul_dispatcher workspace allocation), and we want lora_wrapper
# itself to remain importable on head nodes (download.py path).
_dispatcher = None


def _get_dispatcher():
    global _dispatcher
    if _dispatcher is None:
        from flextrain.ops import dispatcher as _d
        _dispatcher = _d
    return _dispatcher


def _lora_dadb(
    X: torch.Tensor, dY: torch.Tensor,
    A: torch.Tensor, B: torch.Tensor,
    ga: torch.Tensor, gb: torch.Tensor,
    scale: float,
) -> None:
    """Compute and accumulate dA/dB into (ga, gb) for one LoRA target.

    Routes to the cuBLASLt fused-epilogue path when all six tensors
    share dtype; falls back to the 4-GEMM eager pipeline otherwise.
    """
    same_dtype = (
        ga.dtype == X.dtype and gb.dtype == X.dtype
        and dY.dtype == X.dtype and A.dtype == X.dtype
        and B.dtype == X.dtype
    )
    if same_dtype and X.is_cuda:
        dispatcher = _get_dispatcher()
        stream_ptr = torch.cuda.current_stream().cuda_stream
        # Cheap (T, r) intermediates -- still cuBLAS, just via PyTorch.
        dY_B = dY @ B.transpose(-1, -2)
        X_A = X @ A
        # ga += scale * X.T @ dY_B  (fused alpha*AB + beta*C in one call)
        dispatcher.matmul(
            stream_ptr, A=X.transpose(0, 1), B=dY_B,
            C=ga, D=ga, alpha=float(scale), beta=1.0,
        )
        # gb += scale * X_A.T @ dY
        dispatcher.matmul(
            stream_ptr, A=X_A.transpose(0, 1), B=dY,
            C=gb, D=gb, alpha=float(scale), beta=1.0,
        )
        return

    # Fallback: 4 cuBLAS GEMMs + 2 separate cast/add.
    dY_B = dY @ B.transpose(-1, -2)              # (T, r)
    dA = (X.transpose(-1, -2) @ dY_B) * scale    # (in, r)
    X_A = X @ A                                   # (T, r)
    dB = (X_A.transpose(-1, -2) @ dY) * scale    # (r, out)
    ga.add_(dA.to(ga.dtype))
    gb.add_(dB.to(gb.dtype))


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
    * Depthwise convolution kernels (name contains ``conv``): 3-D shape
      ``(channels, 1, K)`` is not a per-expert linear stack and the
      wrapper's MoE-style adapter math doesn't apply.
    * Bundled gated-DeltaNet linear-attn projections
      (``w_lin_qkvz``, ``w_lin_ba``): FT bundles HF's split
      ``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a`` into one
      matrix for compute efficiency. A single rank-r adapter on the
      bundled matrix is mathematically NOT equivalent to four
      independent rank-r adapters on HF's split projections, so we
      skip these to preserve HF PEFT parity. Users targeting LoRA on
      linear-attn layers should adapt ``w_lin_out`` plus the layer's
      MLP only (the typical practical recipe). To LoRA the bundled
      projections explicitly, name them in ``lora_targets``.
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
        if "conv" in nm:
            continue
        if nm in ("w_lin_qkvz", "w_lin_ba"):
            continue
        # Shared-expert weights (w_shared_up, w_shared_down): not yet
        # migrated to the deferred-LoRA-wgrad capture path. Their
        # ParamSpec is independent of the routed-expert option-B
        # layout migration. Excluded from lora_targets="all" until we
        # wire shared experts through. Users who explicitly name them
        # will hit a clear "slow scratch-dW path" error in
        # accumulate_lora_grads (better than silently leaving the
        # adapter grads at zero).
        if nm.startswith("w_shared_"):
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
        # Option-B layout: 3-D MoE expert weights are stored as
        # ``(E, out, in)`` (matching HF gate_up_proj / down_proj). LoRA
        # convention (PEFT-compatible): ``A: (E, in, R)``, ``B: (E, R, out)``
        # so that ``delta_W = A @ B`` is ``(E, in, out)``, which is the
        # transpose of the underlying base weight orientation. The
        # effective-weights builder accumulates this delta into ``W``
        # by computing ``W + (B.T @ A.T) * scale`` per expert (= a
        # transposed-delta view written into the (out, in) buffer).
        E, d_out, d_in = base_shape
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
        # Secondary-compute-stream marker passes through too. Without
        # this, the engine sees ``getattr(wrapper, 'uses_secondary_stream',
        # False) → False`` (the wrapper itself doesn't define the attr,
        # only the inner base layer does), declines to allocate a
        # secondary stream, and the MoE expert-loop alternation
        # silently degrades to single-stream — losing the per-expert
        # primary/secondary overlap entirely under --mode lora.
        if getattr(base, "uses_secondary_stream", False):
            self.uses_secondary_stream = True

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

        # Compose the param spec: every base tensor is frozen (matches
        # HF PEFT default: ``requires_grad_(False)`` is applied
        # everywhere, then only the LoRA A/B adapters get added back as
        # trainable). For LoRA targets we keep a handle on the original
        # spec so we can build matching A/B shapes; the spec we install
        # for the target is also frozen.
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

        2-D path: standard ``W' = W + (A @ B) * scale`` via PyTorch
        matmul. ``W: (in, out)``, ``A: (in, r)``, ``B: (r, out)``.
        The intermediate delta tensor is small enough not to matter.

        3-D MoE expert stack path under option-B layout
        (``W: (E, out, in)``, ``A: (E, in, r)``, ``B: (E, r, out)``):
        the LoRA delta is ``A @ B`` shaped ``(E, in, out)``, which
        is the *transpose* of ``W``'s ``(E, out, in)`` orientation.
        Per expert, we compute ``eff_W[e] = W[e] + (B[e].T @ A[e].T) * scale``
        — equivalently ``W[e] + (A[e] @ B[e]).T * scale`` but written
        as a single dispatcher.matmul without materializing the
        intermediate transpose. ``B[e].T: (out, r)`` and
        ``A[e].T: (r, in)`` are non-contiguous views; the dispatcher
        handles row-major OR col-major inputs natively, so no
        ``.contiguous()`` is needed.

        Pre-allocate one full effective buffer in W's dtype; the
        caching allocator reuses it between layer forwards. Total
        steady-state cost is one ``(E, out, in)`` buffer per LoRA
        target at bf16 (~256 MiB for Qwen3.5-MoE-35B's w_up).

        The dispatcher requires A/B/C/D share dtype, so we cast A and
        B to W's dtype. For typical attn LoRA the master/grad dtypes
        default to bf16 so this is a no-op; for fp32 LoRA masters
        (HF PEFT parity) the cast happens at compute time and the
        higher-precision update flows through the optimizer step on
        host masters as usual.
        """
        from flextrain.ops import dispatcher
        eff = dict(weights)
        stream_ptr = torch.cuda.current_stream().cuda_stream
        for cfg in self.targets:
            W = weights[cfg.target_name]
            A = weights[cfg.a_name]
            B = weights[cfg.b_name]
            if W.dim() == 2:
                # Small-ish delta; PyTorch matmul is fine and avoids the
                # extra dispatcher-allocator path. Keep as-is.
                delta = (A @ B) * cfg.scale
                eff[cfg.target_name] = (W + delta.to(W.dtype)).contiguous()
            elif W.dim() == 3:
                E, d_out, d_in = W.shape
                eff_W = torch.empty(
                    (E, d_out, d_in), dtype=W.dtype, device=W.device,
                )
                A_w = A if A.dtype == W.dtype else A.to(W.dtype)
                B_w = B if B.dtype == W.dtype else B.to(W.dtype)
                for e in range(E):
                    # eff_W[e] = scale * (B[e].T @ A[e].T) + W[e]
                    # Shapes: (out, r) @ (r, in) = (out, in) ✓
                    dispatcher.matmul(
                        stream_ptr,
                        A=B_w[e].T, B=A_w[e].T,
                        C=W[e], D=eff_W[e],
                        alpha=cfg.scale, beta=1.0,
                    )
                eff[cfg.target_name] = eff_W
            else:
                raise ValueError(
                    f"LoRA: unexpected W rank {W.dim()} for {cfg.target_name!r}"
                )
        return eff

    # ------------------------------------------------------------------
    # Layer protocol.
    # ------------------------------------------------------------------

    def forward(self, x, chunk: ChunkMeta, weights, slot, ctx: LayerContext):
        # Forward builds ``eff`` (W + A@B*scale) once and the saved
        # activations get to see it. We do NOT cache eff across fwd+bwd:
        # the engine's activation prefetch creates a fresh
        # ``ActivationSlot`` (with empty ``aux``) when restoring
        # offloaded activations, so any stash in ``slot.aux`` here is
        # lost by the time bwd runs.
        eff = self._build_effective_weights(weights)
        return self.base.forward(x, chunk, eff, slot, ctx)

    def forward_recompute(self, slot, chunk, weights, ctx) -> None:
        # Recompute happens during the bwd phase. We stash ``eff`` on
        # ``slot.aux`` so the immediately-following ``backward_dgrad``
        # / ``backward_wgrad`` pair can reuse it without rebuilding —
        # rebuilding 3-D MoE eff is expensive (256 sequential
        # dispatcher calls per LoRA target) and doubles peak memory.
        # The slot used here is the SAME ``dev_slot`` the engine then
        # passes to ``backward`` (active_model.py:1292/1300), so
        # stashing on slot.aux is safe.
        eff = self._build_effective_weights(weights)
        slot.aux["__lora_eff__"] = eff
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
    #
    # MoE-expert targets (w_up, w_down, w_shared_*) and MoE router
    # targets (w_router, w_shared_expert_gate) are NOT in this set —
    # they take the deferred-LoRA-wgrad path via the
    # ``__lora_moe_capture__`` slot.aux dict + grouped_mm finalize in
    # ``_accumulate_moe_lora_grads_from_capture``. The base accumulates
    # base ``g_up`` / ``g_down`` as usual; LoRA finalize adds dA/dB on
    # top. To freeze the base, simply don't allocate g_up/g_down in
    # the optimizer state (backend skips inline addmm when grads.get
    # is None).
    _FAST_PATH_TARGETS = frozenset(
        ("w_q", "w_k", "w_v", "w_o",
         "w_1", "w_2", "w_3",
         "w_lin_out", "w_lin_qkvz", "w_lin_ba")
    )

    def lora_target_names(self) -> frozenset[str]:
        """Names of base-layer projections this wrapper has LoRA on
        (e.g. ``frozenset({"w_q", "w_k", "w_o", ...})``). The engine
        passes this to ``backward_wgrad`` as ``skip_target_names``."""
        return self._target_set

    def _fast_path_target_names(self) -> frozenset[str]:
        """Subset of LoRA targets that support the dgrad/wgrad
        skip-and-capture protocol (dense projections only). MoE
        targets take the separate capture-dict path."""
        return self._target_set & self._FAST_PATH_TARGETS

    def _moe_lora_target_names(self) -> frozenset[str]:
        """Subset of LoRA targets that take the deferred-LoRA capture
        path (MoE expert weights + MoE routers). Disjoint from
        fast-path and slow-path."""
        return self._target_set & self._MOE_CALLBACK_TARGETS

    def _slow_path_target_names(self) -> frozenset[str]:
        return (
            self._target_set
            - self._fast_path_target_names()
            - self._moe_lora_target_names()
        )

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
        set inside this wrapper specifically).

        Reuses ``slot.aux["__lora_eff__"]`` if ``forward_recompute``
        stashed one there earlier in this same bwd phase.
        """
        eff = slot.aux.get("__lora_eff__")
        if eff is None:
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

        # Install MoE LoRA capture dict — backend will populate it with
        # per-expert grouped intermediates that ``backward_wgrad`` then
        # consumes via grouped_mm-batched finalize. MoE targets are NOT
        # added to skip_for_base: the base's backend accumulates base
        # ``g_up`` / ``g_down`` as usual (or skips when grads.get(...)
        # is None for frozen-base training). LoRA finalize adds dA/dB
        # on top. See _accumulate_moe_lora_grads_from_capture.
        moe_targets = self._moe_lora_target_names()
        if moe_targets:
            slot.aux["__lora_moe_capture__"] = {}

        upstream_dx, inter = self.base.backward_dgrad(
            dx, chunk, eff, combined_grads, slot, ctx,
            skip_target_names=skip_for_base,
        )
        if scratch_grads:
            inter.aux["lora_slow_scratch_grads"] = scratch_grads
        return upstream_dx, inter

    def _accumulate_moe_lora_grads_from_capture(
        self,
        capture: dict[str, torch.Tensor],
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
    ) -> None:
        """Generic deferred LoRA wgrad finalize for MoE expert weights.

        Consumes the per-expert grouped intermediates that the backend
        staged in ``capture`` and accumulates ``dA``, ``dB`` into the
        LoRA factor grad buffers. Backend-agnostic: same code path runs
        regardless of which backend produced ``capture``.

        Math (per LoRA target — w_up or w_down — under option-B
        layout; see design memory ``moe_lora_integration_design_2026_05_02``):

            X         (TK, in)   X_grouped — fwd input to the projection
            dY        (TK, out)  upstream grad at the projection output
            A         (E, in, r) LoRA down-projection factor
            B         (E, r, out) LoRA up-projection factor

        Per-expert (with offs partitioning the TK axis):
            dY_B  = dY  @ B^T          (TK, r)
            dA   += X^T @ dY_B  * scale shape (E, in, r)
            X_A   = X   @ A             (TK, r)
            dB   += X_A^T @ dY  * scale shape (E, r, out)

        Realized via four ``torch.nn.functional.grouped_mm`` calls per
        target — no python expert loop. The grouped GEMM kernels are
        purpose-built for MoE wgrad on Hopper (sm_90+; bf16-only at
        time of writing). Caller responsibility to ensure inputs are
        bf16 and CUDA.

        For w_down's ``X = fwd_act``, recompute via SwiGLU on the
        saved ``slot.x_up`` (free; just a fused activation). The
        capture dict supplies ``dx_up_up_grouped`` directly as w_up's
        dY, ``scattered_upstream_grouped`` as w_down's dY.

        Args:
            capture: dict populated by ``MoEExpertCompute.bwd``.
                Required keys (see protocol docstring):
                ``scattered_x_grouped``, ``dx_up_up_grouped``,
                ``scattered_upstream_grouped``, ``expert_offsets``,
                ``TK``.
            weights: full weights dict — used to read A/B/scale per
                LoRA target. (Effective W is irrelevant here; the
                base's wgrad math doesn't run on these targets.)
            grads: full grads dict — accumulators for dA/dB.

        No-op when no LoRA target is in ``_MOE_CALLBACK_TARGETS``.
        """
        from torch.nn.functional import grouped_mm

        moe_targets = self._target_set & self._MOE_CALLBACK_TARGETS
        # Filter to MoE-expert targets only (w_up, w_down,
        # w_shared_up, w_shared_down). Router targets (w_router,
        # w_shared_expert_gate) go through the 2-D path.
        moe_expert_targets = moe_targets & frozenset(
            ("w_up", "w_down", "w_shared_up", "w_shared_down"),
        )
        if not moe_expert_targets:
            return

        TK = capture["TK"]
        offs = capture["expert_offsets"]  # (E,) int32 cumulative ending in TK

        # Per-target X / dY mapping. Both X and dY must be bf16 for
        # grouped_mm; capture stages full TK so explicit [:TK] is a
        # no-op when backend already sliced.
        scattered_x_grouped = capture["scattered_x_grouped"][:TK]
        dx_up_up_grouped = capture["dx_up_up_grouped"][:TK]
        scattered_upstream_grouped = capture["scattered_upstream_grouped"][:TK]
        x_up_grouped = capture["x_up_grouped"][:TK]   # (TK, 2F) saved pre-SwiGLU

        # Recompute fwd_act = silu(gate) * value (option-B chunked
        # convention: x_up_grouped packs [up_F, gate_F] along last dim).
        # Done lazily below only if w_down is targeted, since the
        # SwiGLU forward is non-trivial bandwidth even at TK*F.
        fwd_act_grouped: torch.Tensor | None = None

        def _ensure_fwd_act() -> torch.Tensor:
            nonlocal fwd_act_grouped
            if fwd_act_grouped is None:
                value, gate = x_up_grouped.chunk(2, dim=-1)
                fwd_act_grouped = (
                    torch.nn.functional.silu(gate) * value
                ).contiguous()
            return fwd_act_grouped

        for cfg in self.targets:
            if cfg.target_name not in moe_expert_targets:
                continue
            A = weights[cfg.a_name]   # (E, in, r) — PEFT convention
            B = weights[cfg.b_name]   # (E, r, out)
            ga = grads.get(f"g_{cfg.a_name[2:]}")
            gb = grads.get(f"g_{cfg.b_name[2:]}")
            if ga is None or gb is None:
                # User froze the LoRA factors entirely (degenerate).
                continue

            # X / dY routing per target. Names follow the same
            # X/dY-contract semantics across all backends.
            if cfg.target_name == "w_up":
                X = scattered_x_grouped              # (TK, d=in)
                dY = dx_up_up_grouped                # (TK, 2F=out)
            elif cfg.target_name == "w_down":
                X = _ensure_fwd_act()                # (TK, F=in)
                dY = scattered_upstream_grouped      # (TK, d=out)
            elif cfg.target_name in ("w_shared_up", "w_shared_down"):
                # Shared experts not yet wired — they have a separate
                # capture path through MoESharedFFN. Will be added if
                # we ever migrate shared experts to option-B.
                continue
            else:
                continue

            r = cfg.rank
            scale = float(cfg.scale)

            # 1. dY_B = dY @ B^T per-expert. 2D × 3D forward-style.
            #    dY: (TK, out), B^T: (E, out, r) → (TK, r)
            dY_B = grouped_mm(dY, B.transpose(-1, -2), offs=offs)

            # 2. dA = X^T @ dY_B per-expert. 2D × 2D wgrad-style.
            #    X^T: (in, TK), dY_B: (TK, r) → (E, in, r)
            dA = grouped_mm(X.transpose(-1, -2), dY_B, offs=offs)

            # 3. X_A = X @ A per-expert. 2D × 3D forward-style.
            #    X: (TK, in), A: (E, in, r) → (TK, r)
            X_A = grouped_mm(X, A, offs=offs)

            # 4. dB = X_A^T @ dY per-expert. 2D × 2D wgrad-style.
            #    X_A^T: (r, TK), dY: (TK, out) → (E, r, out)
            dB = grouped_mm(X_A.transpose(-1, -2), dY, offs=offs)

            # 5. Accumulate.
            ga.add_((dA * scale).to(ga.dtype))
            gb.add_((dB * scale).to(gb.dtype))

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
        addmm for any name in our fast-path LoRA target set.

        Reuses ``slot.aux["__lora_eff__"]`` if ``forward_recompute``
        / ``backward_dgrad`` already populated one this bwd phase.
        """
        eff = slot.aux.get("__lora_eff__")
        if eff is None:
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

        # Deferred LoRA wgrad finalize for MoE-LoRA targets. The base
        # layer's backward_dgrad populated slot.aux["__lora_moe_capture__"]
        # with per-expert grouped intermediates (via the backend's
        # ``lora_capture`` kwarg). Run grouped_mm-batched dA/dB
        # accumulation for w_up / w_down / w_shared_* now.
        moe_capture = slot.aux.pop("__lora_moe_capture__", None)
        if moe_capture is not None and moe_capture:
            # Empty dict means the base layer didn't actually populate
            # anything (e.g., dense-only base; capture was installed
            # speculatively). Skip the finalize when the dict is empty.
            self._accumulate_moe_lora_grads_from_capture(
                moe_capture, weights, grads,
            )

        # Release the cached eff at the end of the bwd pair so the
        # caching allocator can reuse the (E, d_in, d_out) buffer for
        # the next layer. Without this the cache would persist across
        # all 40 layers' bwd calls and defeat the memory savings.
        slot.aux.pop("__lora_eff__", None)

    # ------------------------------------------------------------------
    # accumulate_lora_grads: the actual LoRA Wgrad math.
    # ------------------------------------------------------------------

    # Names that go through the deferred-LoRA-wgrad capture path
    # (accumulated by ``_accumulate_moe_lora_grads_from_capture`` from
    # the ``slot.aux["__lora_moe_capture__"]`` dict the backend
    # populated). MoE expert weights only — 3-D ``(E, out, in)``
    # tensors that the backend processes per-expert.
    #
    # NOT in this set:
    # * ``w_router`` (2-D, MoE router): historically went through the
    #   per-expert callback at routed_swiglu_moe_bwd's router step,
    #   firing with ``eid=-1``. That legacy path was removed in Phase 7.
    #   Router LoRA is currently unsupported. If needed, add a small
    #   dedicated path that captures ``(ffn_norm_output, dlogits)``
    #   from routed_swiglu_moe_bwd into the capture dict.
    # * ``w_shared_expert_gate`` (2-D, shared expert gate): same story.
    # * ``w_shared_up``, ``w_shared_down`` (3-D shared experts): not
    #   yet migrated to option-B. Stay on the legacy callback path
    #   (which goes through ffn_moe_shared.bwd) until shared experts
    #   are migrated.
    _MOE_CALLBACK_TARGETS = frozenset((
        "w_up", "w_down",
    ))

    # torch.compile of the rank-r dA/dB block. Set FLEXTRAIN_LORA_COMPILE=0
    # to disable (e.g. for diagnostic runs or environments where Inductor
    # can't lower the kernel). Compiled per-shape; we cache one entry
    # per (in_dim, out_dim, T, dtype) tuple via dynamic=True so a single
    # compile covers all chunks once T is recompiled-as-dynamic.
    _COMPILED_DADB = None

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
                # Delegated to a torch.compile-ed kernel (cast/add fuse
                # into the matmul epilogue + CUDA Graphs remove launch
                # overhead). Eager fallback is bit-identical; see the
                # ``_lora_dadb`` definition at module top.
                X, dY = intermediates.proj_inputs_and_grads[target]
                _lora_dadb(X, dY, A, B, ga, gb, cfg.scale)
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
