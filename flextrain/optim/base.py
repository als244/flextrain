"""Optimizer protocol + state-spec helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Protocol, runtime_checkable

import torch

from flextrain.core.layer import ParamSpec, TensorSpec


@dataclass(frozen=True)
class OptStateTensor:
    """Description of one optimizer-state tensor per parameter."""

    name: str  # e.g. "o_m" (first moment) or "o_v" (second moment)
    dtype: torch.dtype  # typically fp32; may be bf16 for memory-constrained runs


@dataclass(frozen=True)
class OptimizerStateSpec:
    """How many opt-state tensors each parameter needs, and what they are.

    :class:`AdamW` yields two: ``o_m`` (first moment), ``o_v`` (second).
    :class:`Muon` yields one: ``o_muon``.

    The engine iterates :class:`ParamSpec` x this spec to allocate host
    buffers named ``f"{prefix}{param_name}"`` for each combination.
    """

    tensors: tuple[OptStateTensor, ...]

    def byte_size_for(
        self,
        param_spec: ParamSpec,
        dims: Mapping[str, int],
    ) -> int:
        """Total bytes for this optimizer's state across all params in
        ``param_spec``, with dtypes from this opt-spec (NOT
        ``TensorSpec.opt_state_dtype`` -- the optimizer is free to decide).
        """
        per_param = sum(t.dtype.itemsize for t in self.tensors)
        total = 0
        for p in param_spec.tensors:
            total += p.numel(dims) * per_param
        return total


@runtime_checkable
class Optimizer(Protocol):
    """Minimum optimizer contract the engine consumes."""

    state_spec: OptimizerStateSpec

    def step(
        self,
        param_spec: ParamSpec,
        master: Mapping[str, torch.Tensor],
        grads: Mapping[str, torch.Tensor],
        state: MutableMapping[str, torch.Tensor],
        *,
        step_num: int,
    ) -> int:
        """Apply one update step to every parameter in ``param_spec``.

        ``master[name]`` is the master-dtype tensor on the compute device,
        ``grads[name]`` its gradient, ``state[f"{state_name}_{param_name}"]``
        the optimizer-state tensors. Returns 0 on success, nonzero on
        failure (e.g. NaN grads).

        Naming convention for state keys: ``f"o_{tensor.name}_{param_name}"``
        matches orig's ``o_m_q``, ``o_v_q`` etc. scheme.
        """
        ...


def state_key(state_tensor_name: str, param_name: str) -> str:
    """Canonical opt-state dict key: ``{state_tensor_name}_{param_name}``.

    Matches orig's naming: ``o_m_q``, ``o_v_q``, ``o_m_k``, ... See
    ``orig/awsm_transformer/dense_layer.py:344``.
    """
    # orig strips the leading ``w_`` from param names:
    #   opt key for param ``w_q`` is ``o_m_q``.
    stripped = param_name[2:] if param_name.startswith("w_") else param_name
    return f"{state_tensor_name}_{stripped}"
