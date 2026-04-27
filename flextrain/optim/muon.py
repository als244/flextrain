"""Muon wrapper over ``flextrain_muon_step``.

Muon uses one momentum buffer per parameter (no second moment). The
orthogonalization workspace is transient per-step (Newton-Schulz iterates
on a scratch tensor), so it does NOT appear in :class:`OptimizerStateSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.layer import ParamSpec
from flextrain.ops import flextrain_muon_step

from .base import Optimizer, OptStateTensor, OptimizerStateSpec, state_key


@dataclass(frozen=True)
class MuonHyperparams:
    lr: float = 1e-3
    beta: float = 0.95  # momentum coefficient
    eps: float = 1e-8
    # Newton-Schulz polynomial coefficients (Moonshot's quintic tuning)
    a: float = 3.4445
    b: float = -4.775
    c: float = 2.0315
    ns_iters: int = 5
    weight_decay: float = 0.0
    nesterov: bool = True
    check_error: bool = False


# One bf16 momentum per parameter.
_MUON_STATE_SPEC = OptimizerStateSpec(
    tensors=(OptStateTensor(name="o_muon", dtype=torch.bfloat16),)
)


class Muon(Optimizer):
    state_spec = _MUON_STATE_SPEC

    def __init__(self, hp: MuonHyperparams | None = None) -> None:
        self.hp = hp or MuonHyperparams()

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
            mom_key = state_key("o_muon", p.name)
            grad_key = (
                "g_" + p.name[2:] if p.name.startswith("w_") else "g_" + p.name
            )
            ret = flextrain_muon_step(
                master[p.name],
                grads[grad_key],
                state[mom_key],
                lr=self.hp.lr,
                beta=self.hp.beta,
                eps=self.hp.eps,
                a=self.hp.a,
                b=self.hp.b,
                c=self.hp.c,
                ns_iters=self.hp.ns_iters,
                weight_decay=self.hp.weight_decay,
                nesterov=self.hp.nesterov,
                check_error=self.hp.check_error,
            )
            if ret != 0:
                return ret
        return 0
