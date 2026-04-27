"""AdamW wrapper over ``flextrain_adamw_step``.

Collapses the 9-unrolled step calls in
``orig/awsm_transformer/dense_layer.py:344-466`` (and its MoE twin) into a
single ``for p in param_spec: step(...)`` loop driven by :class:`ParamSpec`.

Hyperparameters match orig exactly: lr, beta1, beta2, eps, weight_decay,
step_num.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.layer import ParamSpec
from flextrain.ops import flextrain_adamw_step

from .base import Optimizer, OptStateTensor, OptimizerStateSpec, state_key


@dataclass(frozen=True)
class AdamWHyperparams:
    """Per-step hyperparameters.

    Matches the dict orig passes as ``opt_hyperparams`` (see
    ``orig/awsm_transformer/dense_layer.py:337-342``).
    """

    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.001
    check_error: bool = False


def _make_adamw_state_spec(dtype: torch.dtype) -> OptimizerStateSpec:
    return OptimizerStateSpec(
        tensors=(
            OptStateTensor(name="o_adam_m", dtype=dtype),  # first moment
            OptStateTensor(name="o_adam_v", dtype=dtype),  # second moment
        )
    )


# Default: fp32 moments. This is the safer choice for numerical
# stability in long runs. bf16 opt state is an available memory trade
# (half the opt ring size) for large models where you'd otherwise need
# host offload. User-selectable via the ``state_dtype`` constructor
# arg.
_ADAMW_STATE_SPEC_FP32 = _make_adamw_state_spec(torch.float32)


class AdamW(Optimizer):
    """AdamW optimizer.

    Usage
    -----
        opt = AdamW(hp=AdamWHyperparams(lr=3e-4))
        # or bf16 opt state to halve memory:
        opt = AdamW(hp=..., state_dtype=torch.bfloat16)
        state = {}  # engine-allocated per (param, state_tensor) pair
        opt.step(param_spec, master, grads, state, step_num=step)
    """

    def __init__(
        self,
        hp: AdamWHyperparams | None = None,
        *,
        state_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.hp = hp or AdamWHyperparams()
        self.state_spec = (
            _ADAMW_STATE_SPEC_FP32 if state_dtype is torch.float32
            else _make_adamw_state_spec(state_dtype)
        )

    def step(
        self,
        param_spec: ParamSpec,
        master: Mapping[str, torch.Tensor],
        grads: Mapping[str, torch.Tensor],
        state: MutableMapping[str, torch.Tensor],
        *,
        step_num: int,
    ) -> int:
        for p in param_spec.tensors:
            if p.frozen:
                continue
            m_key = state_key("o_adam_m", p.name)
            v_key = state_key("o_adam_v", p.name)
            grad_key = (
                "g_" + p.name[2:] if p.name.startswith("w_") else "g_" + p.name
            )
            ret = flextrain_adamw_step(
                master[p.name],
                grads[grad_key],
                state[m_key],
                state[v_key],
                lr=self.hp.lr,
                beta1=self.hp.beta1,
                beta2=self.hp.beta2,
                eps=self.hp.eps,
                weight_decay=self.hp.weight_decay,
                step=step_num,
                check_error=self.hp.check_error,
            )
            if ret != 0:
                return ret
        return 0
