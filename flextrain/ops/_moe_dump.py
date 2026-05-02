"""Tiny dump helper for MoE backend parity diagnostics.

Activated by env var ``FLEXTRAIN_MOE_DUMP_DIR=<dir>``. When set, calls
to :func:`dump_tensor` save the tensor to
``<dir>/<phase>_layer<layer_id>_<name>.pt`` where ``phase`` is
"fwd" or "bwd".

Designed for SINGLE-STEP comparisons: each layer's fwd / bwd is
expected to be called once. If called multiple times, later writes
clobber earlier ones — that's fine for our use case (compare two
runs' first step end-to-end).

Used by ``routed_swiglu_moe_fwd`` and ``routed_swiglu_moe_bwd`` (see
``flextrain/ops/full_moe.py``) to capture pre-MoE inputs, router state,
post-MoE outputs, and per-layer gradients.

Zero overhead when the env var is unset (functions early-return).
"""
from __future__ import annotations

import os

import torch


_DUMP_DIR: str | None = os.environ.get("FLEXTRAIN_MOE_DUMP_DIR")
_INCLUDE_BIG: bool = os.environ.get(
    "FLEXTRAIN_MOE_DUMP_INCLUDE_BIG", "0"
).lower() in ("1", "true", "yes")
_BIG_NAMES = frozenset({"g_up", "g_down"})


def is_active() -> bool:
    return _DUMP_DIR is not None


def dump_tensor(
    name: str, tensor: torch.Tensor, *, layer_id: int, phase: str = "bwd",
) -> None:
    """Dump as CPU tensor (preserving dtype) to
    ``<dir>/<phase>_layer<layer_id>_<name>.pt``.

    Skips g_up / g_down by default (override with
    ``FLEXTRAIN_MOE_DUMP_INCLUDE_BIG=1``). Multiple calls with the
    same (phase, layer, name) overwrite — designed for single-step
    runs where each layer's fwd/bwd is called exactly once.
    """
    if _DUMP_DIR is None:
        return
    if name in _BIG_NAMES and not _INCLUDE_BIG:
        return
    os.makedirs(_DUMP_DIR, exist_ok=True)
    path = os.path.join(
        _DUMP_DIR,
        f"{phase}_layer{layer_id:03d}_{name}.pt",
    )
    # detach + cpu + clone — don't hold compute-graph aliases or
    # accumulator buffers that may be zeroed/overwritten later.
    torch.save(tensor.detach().to("cpu").clone(), path)
