"""RMSNorm block.

One algorithmic unit that covers BOTH variants used in modern LLMs:

* **Residual-stream RMSNorm** (Llama/Qwen/Mistral attn_norm, ffn_norm,
  final_norm). Input shape ``(num_tokens, d_model)``; a single
  ``(d_model,)`` weight tensor; rstd shape ``(num_tokens, 1)``.

* **Per-head RMSNorm** (aka "QK-norm" — Gemma2/3, Qwen3 applying RMSNorm
  to Q and K before attention). Input shape
  ``(num_tokens, n_heads, head_dim)``; a single ``(head_dim,)`` weight
  tensor broadcast across heads; rstd shape ``(num_tokens, n_heads)``.

Both are exactly the same kernel -- ``flextrain_rmsnorm_fwd`` just needs
``head_dim`` set to select the per-head code path
(``ops/rmsnorm.py:_get_norm_configs`` ``IS_BY_HEAD``). Same bwd kernel,
same weight tensor layout. Naming "QKNorm" would have introduced a
spurious second class -- per-head RMSNorm IS RMSNorm.

Usage
-----
Residual-stream (Llama's attn/ffn norms):
    RMSNormBlock(prefix="attn_norm")

Per-head for Q (Qwen3 q_norm):
    RMSNormBlock(prefix="q_norm", normalized_shape=("head_dim",),
                 per_head=True, heads_dim_name="n_heads")

The ``heads_dim_name`` entry drives rstd's second axis: ``"n_heads"`` for
Q, ``"n_kv_heads"`` for K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import ActivationField
from flextrain.core.layer import ComputeCost, ParamSpec, TensorSpec
from flextrain.ops import (
    flextrain_rmsnorm_bwd,
    flextrain_rmsnorm_fwd,
    flextrain_rmsnorm_fwd_recompute,
)


@dataclass(frozen=True)
class RMSNormBlock:
    """RMSNorm over a configurable normalization axis.

    Parameters
    ----------
    prefix
        Uniquifier for field / weight names (e.g. ``"attn_norm"``,
        ``"ffn_norm"``, ``"q_norm"``, ``"k_norm"``).
    eps
        ``rms_norm_eps``. Default 1e-5 (Llama / Qwen). Gemma uses 1e-6.
    per_head
        If True, apply per-head normalization: weight broadcast across
        ``head_dim``, rstd is ``(num_tokens, heads)`` instead of
        ``(num_tokens, 1)``. Used for QK-norm.
    heads_dim_name
        When ``per_head=True``, which model dim provides the head count
        ("n_heads" for Q, "n_kv_heads" for K). Irrelevant otherwise.
    weight_dim_name
        Which model dim provides the weight-vector size. Default
        ``"d_model"`` (full residual stream); ``"head_dim"`` for QK-norm.
    param_compute_dtype / param_master_dtype / param_grad_dtype
        Per-role dtypes (see :class:`TensorSpec`). ``grad_dtype`` defaults
        to fp32 because ``flextrain_rmsnorm_bwd`` accumulates via atomics.

    Declared activation field
    -------------------------
    ``{prefix}_rstd``    shape depends on ``per_head``:
                         * False -> ``(num_tokens, 1)``, fp32
                         * True  -> ``(num_tokens, heads)``, fp32
                         Always tier 0 (always saved -- tiny memory cost
                         and required to make fwd_from_rstd a true
                         re-projection rather than a re-computation).
    """

    prefix: str
    eps: float = 1e-5
    per_head: bool = False
    heads_dim_name: str = "n_heads"
    weight_dim_name: str = "d_model"
    param_compute_dtype: torch.dtype = torch.bfloat16
    param_master_dtype: torch.dtype | None = None
    param_grad_dtype: torch.dtype = torch.float32

    @property
    def rstd_name(self) -> str:
        return f"{self.prefix}_rstd"

    @property
    def weight_name(self) -> str:
        return f"w_{self.prefix}"

    @property
    def grad_name(self) -> str:
        return f"g_{self.prefix}"

    # ------------------------------------------------------------------
    # Declarations.
    # ------------------------------------------------------------------

    def fields(self) -> tuple[ActivationField, ...]:
        if self.per_head:
            heads_name = self.heads_dim_name
            shape_fn = lambda n, d: (n, d[heads_name])
        else:
            shape_fn = lambda n, d: (n, 1)
        return (
            ActivationField(
                name=self.rstd_name,
                shape_fn=shape_fn,
                dtype=torch.float32,
                tier=0,
            ),
        )

    def param_spec(self) -> ParamSpec:
        weight_name = self.weight_dim_name
        return ParamSpec(
            tensors=(
                TensorSpec(
                    name=self.weight_name,
                    shape_fn=lambda d: (d[weight_name],),
                    compute_dtype=self.param_compute_dtype,
                    master_dtype=self.param_master_dtype,
                    grad_dtype=self.param_grad_dtype,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Compute.
    # ------------------------------------------------------------------

    def _head_dim_arg(
        self, x: torch.Tensor, weights: Mapping[str, torch.Tensor] | None = None,
    ) -> int | None:
        """Return the ``head_dim`` argument to pass to the kernel.

        ``flextrain_rmsnorm_fwd`` treats ``head_dim=None`` as "normalize the
        whole last axis" (residual-stream case) and a positive integer as
        "per-head normalization" (QK-norm case).

        For per-head mode the kernel expects a 2D input ``(T, heads*D)``
        and a 1D weight ``(D,)``; head_dim is read from the weight
        vector's last dim.
        """
        if not self.per_head:
            return None
        if weights is not None and self.weight_name in weights:
            return weights[self.weight_name].shape[0]
        # Fallback: caller passed a 3D (T, H, D) tensor -- use shape[-1].
        return x.shape[-1]

    def fwd(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        rstd_out: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``y = rmsnorm(x) * w``. Writes rstd into ``rstd_out`` and
        optionally ``y`` into ``output``. Returns ``y``.

        Per-head mode: pass an ``(N, heads, head_dim)`` tensor; the kernel
        normalizes over the last axis independently per head.
        """
        y, _ = flextrain_rmsnorm_fwd(
            x,
            W=weights[self.weight_name],
            head_dim=self._head_dim_arg(x, weights),
            output=output,
            rstd=rstd_out,
            rms_norm_eps=self.eps,
        )
        return y

    def fwd_from_rstd(
        self,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        rstd: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recompute the forward output given a saved ``rstd``."""
        return flextrain_rmsnorm_fwd_recompute(
            x,
            W=weights[self.weight_name],
            rstd=rstd,
            head_dim=self._head_dim_arg(x, weights),
            output=output,
        )

    def bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        rstd: torch.Tensor,
        *,
        dx_accumulator: torch.Tensor | None = None,
        recompute_output: bool = True,
        recomputed_output_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Backward pass. Accumulates weight grad into ``grads``, input grad
        into ``dx_accumulator`` (if provided) or a fresh tensor.

        Returns ``(dx, recomputed_output)``.
        """
        # Frozen-aware: under LoRA the BufferManager skips grad
        # allocation for frozen tensors, so ``grads`` may not contain
        # ``self.grad_name``. ``dW=None`` tells flextrain_rmsnorm_bwd to
        # skip the wgrad accumulate (forward + dx still run normally).
        dx, _, recomputed = flextrain_rmsnorm_bwd(
            dy,
            x,
            W=weights[self.weight_name],
            rstd=rstd,
            head_dim=self._head_dim_arg(x, weights),
            dX=dx_accumulator,
            dW=grads.get(self.grad_name),
            recompute_output=recompute_output,
            recomputed_output_tensor=recomputed_output_tensor,
        )
        return dx, recomputed

    # ------------------------------------------------------------------
    # FLOP accounting -- negligible vs. GEMMs, matches orig's ignoring of
    # norm FLOPs in the DP solver input.
    # ------------------------------------------------------------------

    def compute_cost(
        self, num_tokens: int, dims: Mapping[str, int], max_tier: int
    ) -> ComputeCost:
        return ComputeCost(
            total_fwd_flops=0,
            avoided_recompute_flops=tuple(0 for _ in range(max_tier + 1)),
        )
