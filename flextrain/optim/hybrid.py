"""Hybrid Muon+AdamW optimizer.

Classifies each parameter into one of two update rules:

* **Muon** — for the "structurally 2-D dense projection" weights of
  transformer blocks (attention Q/K/V/O, FFN up/down, MoE expert up/down
  when stacked). Muon orthogonalizes the update via Newton-Schulz, which
  is only meaningful on 2-D matrices.

* **AdamW** — for everything else: embeddings, LM head, RMSNorm weights,
  router logits (and router bias if present), attention biases. These
  are either 1-D or wide/rectangular in ways Muon's NS iteration doesn't
  benefit from, and empirically Muon on them harms training stability.

Per-tensor opt-state dtypes are honored (``TensorSpec.opt_state_dtype``),
so callers can keep AdamW state in fp32 but Muon momentum in bf16 (the
orig default) on a per-layer or per-tensor basis without writing two
optimizer classes.

Classification precedence:
1. ``TensorSpec.optimizer`` (``"muon"`` / ``"adamw"``) if set.
2. Auto-inference from name + rank (see ``infer_optimizer_for_param``).

The union opt-state spec declares all three tensors (``o_m``, ``o_v``,
``o_muon``), but the buffer allocator elides the two that don't
apply to any given TensorSpec — no wasted bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.layer import ParamSpec, TensorSpec
from flextrain.ops import flextrain_adamw_step, flextrain_muon_step

from .adamw import AdamWHyperparams
from .base import Optimizer, OptStateTensor, OptimizerStateSpec, state_key
from .muon import MuonHyperparams


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


# Name fragments that always take AdamW regardless of tensor rank.
_ALWAYS_ADAMW_FRAGMENTS = (
    "_norm",          # w_attn_norm, w_ffn_norm, w_final_norm, w_q_norm, w_k_norm
    "embed",          # w_tok_embeddings (some arches name it this way)
    "tok_embed",
    "head",           # w_head_proj, w_final_norm (covered above)
    "router",         # w_router (MoE gate)
    "lm_head",
)


def infer_optimizer_for_param(spec: TensorSpec, dims: Mapping[str, int]) -> str:
    """Auto-infer ``"muon"`` vs ``"adamw"`` for a tensor.

    Rules (in order):

    1. Explicit ``spec.optimizer`` setting always wins.
    2. 2-D tensors that look like dense projections → ``"muon"``.
    3. 3-D MoE expert stacks (``(E, 2F, d)`` / ``(E, d, F)``) →
       ``"muon"``. The step() loop applies Newton-Schulz per expert
       slice (each slice is a 2-D matrix).
    4. Everything else → ``"adamw"`` (1-D biases/norms, embeddings,
       head, routers, or tensors with names containing explicit
       AdamW fragments).
    """
    if spec.optimizer is not None:
        return spec.optimizer
    shape = spec.shape(dims)
    rank = len(shape)
    if rank not in (2, 3):
        return "adamw"
    lname = spec.name.lower()
    for frag in _ALWAYS_ADAMW_FRAGMENTS:
        if frag in lname:
            return "adamw"
    return "muon"


# ---------------------------------------------------------------------------
# State spec (union)
# ---------------------------------------------------------------------------


_HYBRID_STATE_SPEC = OptimizerStateSpec(
    tensors=(
        OptStateTensor(name="o_adam_m", dtype=torch.float32),      # AdamW m
        OptStateTensor(name="o_adam_v", dtype=torch.float32),      # AdamW v
        OptStateTensor(name="o_muon", dtype=torch.bfloat16),  # Muon momentum
    )
)


class HybridStateSpec(OptimizerStateSpec):
    """OptimizerStateSpec subclass that filters per-tensor state
    allocation based on ``infer_optimizer_for_param``.

    The engine's allocator calls ``byte_size_for(...)`` and enumerates
    ``tensors`` uniformly; we override both so only the applicable
    state tensors are allocated per TensorSpec (no wasted bytes for
    Muon slots on AdamW params and vice versa).
    """

    def per_param_state_tensors(
        self, param: TensorSpec, dims: Mapping[str, int]
    ) -> tuple[OptStateTensor, ...]:
        """Which state tensors apply to this specific param."""
        rule = infer_optimizer_for_param(param, dims)
        if rule == "muon":
            return (OptStateTensor(name="o_muon", dtype=param.opt_state_dtype),)
        return (
            OptStateTensor(name="o_adam_m", dtype=param.opt_state_dtype),
            OptStateTensor(name="o_adam_v", dtype=param.opt_state_dtype),
        )

    def byte_size_for(
        self,
        param_spec: ParamSpec,
        dims: Mapping[str, int],
    ) -> int:
        total = 0
        for p in param_spec.tensors:
            numel = p.numel(dims)
            for st in self.per_param_state_tensors(p, dims):
                total += numel * st.dtype.itemsize
        return total


_HYBRID_STATE_SPEC_FILTERED = HybridStateSpec(
    tensors=_HYBRID_STATE_SPEC.tensors
)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridMuonAdamWHyperparams:
    """Per-algorithm hyperparams. LR is shared (typical practice), but
    you can override AdamW-only or Muon-only via the sub-objects."""

    lr: float = 1e-3
    adamw: AdamWHyperparams = AdamWHyperparams(lr=1e-3)
    muon: MuonHyperparams = MuonHyperparams(lr=1e-3)


class HybridMuonAdamW(Optimizer):
    """Apply AdamW to 1-D/embed/head/router/norm tensors and Muon to
    2-D dense projection tensors. Classification per
    :func:`infer_optimizer_for_param`.

    Both algorithms share the same LR by default (adjustable by
    constructing custom ``AdamWHyperparams`` / ``MuonHyperparams`` via
    :class:`HybridMuonAdamWHyperparams`).
    """

    state_spec = _HYBRID_STATE_SPEC_FILTERED

    def __init__(
        self,
        hp: HybridMuonAdamWHyperparams | None = None,
    ) -> None:
        self.hp = hp or HybridMuonAdamWHyperparams()

    def step(
        self,
        param_spec: ParamSpec,
        master: Mapping[str, torch.Tensor],
        grads: Mapping[str, torch.Tensor],
        state: MutableMapping[str, torch.Tensor],
        *,
        step_num: int,
    ) -> int:
        adamw_hp = self.hp.adamw
        muon_hp = self.hp.muon
        # Per-tensor classification needs a dims-like view; we recover
        # per-tensor rule from the shape of the master copy at runtime.
        for p in param_spec.tensors:
            if p.frozen:
                continue
            m = master[p.name]
            grad_key = "g_" + p.name[2:] if p.name.startswith("w_") else "g_" + p.name
            g = grads[grad_key]
            # Classify: name fragments + rank.
            suffix = p.name[2:] if p.name.startswith("w_") else p.name
            rule = (
                p.optimizer
                if p.optimizer is not None
                else _infer_rule_from_runtime(p.name, m.dim())
            )
            if rule == "muon":
                mom_key = f"o_muon_{suffix}"
                mom = state[mom_key]
                if m.dim() == 3:
                    # MoE expert stack: apply Muon per-expert. Each
                    # slice ``m[e]`` is a 2-D ``(d_in, d_out)`` matrix
                    # the Newton-Schulz kernel expects.
                    E = m.shape[0]
                    ret = 0
                    for e in range(E):
                        ret = flextrain_muon_step(
                            m[e], g[e], mom[e],
                            lr=muon_hp.lr, beta=muon_hp.beta, eps=muon_hp.eps,
                            a=muon_hp.a, b=muon_hp.b, c=muon_hp.c,
                            ns_iters=muon_hp.ns_iters,
                            weight_decay=muon_hp.weight_decay,
                            nesterov=muon_hp.nesterov,
                            check_error=muon_hp.check_error,
                        )
                        if ret != 0:
                            return ret
                else:
                    ret = flextrain_muon_step(
                        m, g, mom,
                        lr=muon_hp.lr, beta=muon_hp.beta, eps=muon_hp.eps,
                        a=muon_hp.a, b=muon_hp.b, c=muon_hp.c,
                        ns_iters=muon_hp.ns_iters,
                        weight_decay=muon_hp.weight_decay,
                        nesterov=muon_hp.nesterov,
                        check_error=muon_hp.check_error,
                    )
            else:
                m_key = f"o_adam_m_{suffix}"
                v_key = f"o_adam_v_{suffix}"
                ret = flextrain_adamw_step(
                    m, g, state[m_key], state[v_key],
                    lr=adamw_hp.lr, beta1=adamw_hp.beta1, beta2=adamw_hp.beta2,
                    eps=adamw_hp.eps, weight_decay=adamw_hp.weight_decay,
                    step=step_num,
                )
            if ret != 0:
                return ret
        return 0


def _infer_rule_from_runtime(name: str, rank: int) -> str:
    """Runtime shim mirroring :func:`infer_optimizer_for_param` but
    without a dims map (we already have the live tensor)."""
    if rank not in (2, 3):
        return "adamw"
    lname = name.lower()
    for frag in _ALWAYS_ADAMW_FRAGMENTS:
        if frag in lname:
            return "adamw"
    return "muon"
