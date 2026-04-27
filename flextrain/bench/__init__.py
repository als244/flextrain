"""Benchmarking + correctness-parity harnesses.

What lives here
---------------
* :mod:`flextrain.bench.parity` — loss-curve parity harness that runs
  a naive pure-PyTorch reference alongside FlexTrain across multiple
  working-set configurations, on real data, for N optimizer steps.

These are tools for confirming engine correctness after a change (e.g.
adding a new architecture, modifying the scheduler, bumping kernels).
They're separate from ``tests/`` because they're longer-running and
more exploratory.
"""

from .parity import (
    LossCurveParityConfig,
    LossCurveParityResult,
    ModelShape,
    NaiveLlamaModel,
    WorkingSetSpec,
    run_loss_curve_parity,
)

__all__ = [
    "LossCurveParityConfig",
    "LossCurveParityResult",
    "ModelShape",
    "NaiveLlamaModel",
    "WorkingSetSpec",
    "run_loss_curve_parity",
]
