"""Optimizer wrappers.

Each optimizer here does THREE things:

1. Declares how much optimizer-state storage it needs per parameter, in
   terms that :class:`~flextrain.core.ParamSpec` can turn into buffer
   sizes.
2. Allocates those state tensors into caller-provided host buffers (with
   the right dtype from ``TensorSpec.opt_state_dtype``).
3. Runs the step kernel against the (master_params, grads, opt_state)
   triple.

Deriving opt-state from ParamSpec -- instead of each layer unrolling 9
``flextrain_adamw_step`` calls by hand (``orig/awsm_transformer/dense_layer.py:344-466``)
-- is the main refactor win of this module.
"""

from .adamw import AdamW, AdamWHyperparams
from .base import Optimizer, OptimizerStateSpec, state_key
from .muon import Muon, MuonHyperparams

__all__ = [
    "AdamW",
    "AdamWHyperparams",
    "Muon",
    "MuonHyperparams",
    "Optimizer",
    "OptimizerStateSpec",
    "state_key",
]
