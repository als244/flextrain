"""Save-level policy: the DP solver interface.

What this replaces
------------------
In ``orig/active_model.py:531-665``, ``determine_saved_levels`` builds three
parallel NumPy arrays -- ``compute_times``, ``saved_option_values``,
``saved_option_transfer_durations`` -- of shape ``(T, k)`` where ``T`` is
``num_layers * num_chunks`` and ``k`` is a GLOBAL ``num_saved_activation_levels``
pulled from ``self.model_layers[0].max_saved_activations_level + 1``. That
assumption (all layers share the same tier count) breaks the moment we support
heterogeneous backbones (GPT-OSS dense+MoE alternation, Qwen3-Next linear+full
attention -- see docs/PLAN.md "Multi-architecture strategy").

What this module gives instead
------------------------------
:class:`SaveLevel` -- a thin wrapper around an integer tier (``-1`` is the
"GPU-resident, no home slot" fast path, not a magic sentinel in a dict).

:func:`build_dp_tables` -- takes a sequence of layers with per-layer
``schema.max_tier`` and pads the DP ``(T, k_global)`` arrays with ``-inf``
values in disallowed cells so the C DP solver (the in-tree
``helpers/transmission_scheduler`` helper, built by ``pip install -e .``)
still sees a rectangular array.

:class:`SaveLevelPlan` -- the per-(layer_id, chunk_id) -> SaveLevel mapping
the engine consumes. Supports the ``n_home_act_slots == 0`` fast path
(``orig/active_model.py:546``) where every level is ``-1``.

Hardware inputs for the DP
--------------------------
The paper's §3.4 formulation needs:
    * ``ci``     : forward compute time for chunk i (ms)
    * ``delta_i,o`` : transfer time at save level o for chunk i (ms)
    * ``v_i,o``  : value = recompute time avoided in backward (ms)

We derive these from a :class:`HardwareCost` object the engine supplies
once per training run (``peak_tflops``, ``practical_efficiency_factor``,
``pcie_bw_gbps``). Mirrors ``orig/active_model.py:595``/``:615``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .activation_schema import ActivationSchema
from .layer import ChunkMeta, ComputeCost, Layer


# ---------------------------------------------------------------------------
# SaveLevel
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class SaveLevel:
    """A save-level choice for one (layer, chunk) pair.

    ``value >= 0``  -> the corresponding ``ActivationSchema`` tier.
    ``value == -1`` -> GPU-resident; NO host-pinned slot. This is the fast
                      path for when ``n_home_act_slots == 0``, i.e. the
                      activation ring on device is large enough to hold every
                      (chunk, layer) pair in the current round.
    """

    value: int

    @staticmethod
    def on_device() -> "SaveLevel":
        return SaveLevel(-1)

    @property
    def is_on_device(self) -> bool:
        return self.value == -1

    @staticmethod
    def all_tiers(schema: ActivationSchema) -> tuple["SaveLevel", ...]:
        return tuple(SaveLevel(i) for i in range(schema.max_tier + 1))


# ---------------------------------------------------------------------------
# SaveLevelPlan: the engine-consumable output of the DP.
# ---------------------------------------------------------------------------


@dataclass
class SaveLevelPlan:
    """Per-(layer_id, chunk_id) save-level mapping for one round.

    Attributes
    ----------
    choices
        ``{(layer_id, chunk_id): SaveLevel}`` for every pair in this round.
    estimated_recompute_time_ms
        Sum of avoided-recompute times actually NOT achieved (for logging /
        dashboard parity with ``orig/active_model.py:647``).
    estimated_fwd_time_ms
        Sum of forward compute times for this round.
    """

    choices: Mapping[tuple[int, int], SaveLevel]
    estimated_recompute_time_ms: float
    estimated_fwd_time_ms: float

    def level_for(self, layer_id: int, chunk_id: int) -> SaveLevel:
        return self.choices[(layer_id, chunk_id)]

    @property
    def recompute_fraction(self) -> float:
        if self.estimated_fwd_time_ms <= 0:
            return 0.0
        return self.estimated_recompute_time_ms / self.estimated_fwd_time_ms

    @classmethod
    def all_on_device(
        cls,
        layer_ids: Sequence[int],
        chunk_ids: Sequence[int],
    ) -> "SaveLevelPlan":
        """Fast-path plan: every pair lives on the GPU activation ring, no
        host slots allocated. Used when ``n_home_act_slots == 0``."""
        on_dev = SaveLevel.on_device()
        choices = {
            (lid, cid): on_dev for lid in layer_ids for cid in chunk_ids
        }
        return cls(
            choices=choices,
            estimated_recompute_time_ms=0.0,
            estimated_fwd_time_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Hardware cost model. One per training run.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareCost:
    """Practical (not peak) compute + interconnect numbers used to convert
    FLOPs and bytes into milliseconds for the DP solver."""

    # NOTE: ``orig/active_model.py`` sets ``peak_tflops_est`` from a *measured*
    # matmul benchmark (``bench_matmul``, see ``hardware_env.py``), not a
    # nameplate number -- so the measurement already bakes in kernel
    # efficiency and ``PRACTICAL_EFFICIENCY_FACTOR = 1.0`` (orig:25) is a
    # no-op. We keep the factor as an input for override-ability, but the
    # engine will feed a measured peak the same way.
    peak_tflops: float  # measured device throughput, TFLOPs (post-bench)
    pcie_bw_gbps: float  # measured unidirectional CPU<->GPU bandwidth
    practical_efficiency_factor: float = 1.0  # match orig:25 default

    @property
    def effective_flops_per_ms(self) -> float:
        return (
            self.practical_efficiency_factor
            * self.peak_tflops
            * 1e12
            / 1e3
        )

    @property
    def pcie_bytes_per_ms(self) -> float:
        return self.pcie_bw_gbps * 1e9 / 1e3

    def flops_to_ms(self, flops: float) -> float:
        if self.effective_flops_per_ms <= 0:
            return 0.0
        return flops / self.effective_flops_per_ms

    def bytes_to_ms(self, nbytes: float) -> float:
        if self.pcie_bytes_per_ms <= 0:
            return 0.0
        return nbytes / self.pcie_bytes_per_ms


# ---------------------------------------------------------------------------
# DPTables: the rectangular inputs the C solver consumes. Layers declare
# per-schema ``max_tier`` so we pad to ``max_k_global`` with -inf.
# ---------------------------------------------------------------------------


@dataclass
class DPTables:
    """Rectangular inputs for ``TransmissionScheduler.solve``.

    Attributes
    ----------
    compute_times
        Shape ``(T,)``, float64 ms. ``T = num_layers * num_chunks``.
    transfer_durations
        Shape ``(T, k_global)``, float64 ms. ``transfer_durations[t, L]`` is
        the PCIe time to offload (layer, chunk) ``t`` at level ``L``.
    values
        Shape ``(T, k_global)``, float64 ms. ``values[t, L]`` is the backward
        recompute time we avoid by saving at level ``L`` (higher is better).
    k_global
        Max ``max_tier + 1`` across all layer types in the round. DP sees
        this as ``k``.
    indexing
        ``{(layer_id, chunk_id): t}`` -- maps back to engine-side identity.
    max_tier_per_task
        Shape ``(T,)``, int. Used to reject out-of-range choices when the
        solver returns a disallowed level (can happen if we left the -inf
        sentinels too loose); the builder fills disallowed columns with -inf
        values AND sets transfer_durations to +inf so the solver never
        picks them, but we keep this for defense-in-depth.
    """

    compute_times: np.ndarray
    transfer_durations: np.ndarray
    values: np.ndarray
    k_global: int
    indexing: Mapping[tuple[int, int], int]
    max_tier_per_task: np.ndarray

    @property
    def T(self) -> int:
        return self.compute_times.shape[0]


# Sentinel value used to mark disallowed (task, level) pairs in the DP input.
# The C solver maximizes value subject to transfer-duration constraints; a
# -inf value is never chosen. We also set transfer_durations to +inf in those
# cells so that even if the solver somehow does pick a -inf cell, it would
# blow the deadline and be rejected.
_NEG_INF = -1e18
_POS_INF = 1e18


def build_dp_tables(
    layers: Sequence[Layer],
    chunk_metas: Sequence[ChunkMeta],
    dims: Mapping[str, int],
    hw: HardwareCost,
) -> DPTables:
    """Build rectangular DP-solver inputs for one gradient-accumulation round.

    The (layer, chunk) traversal order MUST match the order the engine
    performs during forward, which is "for layer in layers: for chunk in
    chunks" (``orig/active_model.py:1261,1275``). That ordering determines
    the DP's arrival-time semantics: task ``t+1`` arrives after task ``t``
    finishes computing.
    """
    T = len(layers) * len(chunk_metas)

    per_layer_k = [layer.schema.max_tier + 1 for layer in layers]
    k_global = max(per_layer_k)

    compute_times = np.zeros(T, dtype=np.float64)
    transfer_durations = np.full((T, k_global), _POS_INF, dtype=np.float64)
    values = np.full((T, k_global), _NEG_INF, dtype=np.float64)
    max_tier_per_task = np.zeros(T, dtype=np.int32)

    indexing: dict[tuple[int, int], int] = {}

    t = 0
    for layer in layers:
        k = layer.schema.max_tier + 1
        for chunk_idx, chunk in enumerate(chunk_metas):
            indexing[(layer.layer_id, chunk_idx)] = t

            cost = layer.compute_cost(chunk)
            compute_times[t] = hw.flops_to_ms(cost.total_fwd_flops)

            # One column per tier this layer actually supports; others stay at
            # their sentinel (-inf value, +inf duration).
            for L in range(k):
                nbytes = layer.schema.offloaded_bytes_at_level(
                    chunk.total_q, dims, L
                )
                transfer_durations[t, L] = hw.bytes_to_ms(nbytes)
                values[t, L] = hw.flops_to_ms(cost.avoided_recompute_flops[L])

            max_tier_per_task[t] = k - 1
            t += 1

    return DPTables(
        compute_times=compute_times,
        transfer_durations=transfer_durations,
        values=values,
        k_global=k_global,
        indexing=indexing,
        max_tier_per_task=max_tier_per_task,
    )


# ---------------------------------------------------------------------------
# Plan materialization. The engine calls solve(...) via the external
# TransmissionScheduler C extension; this module wraps the result in a
# SaveLevelPlan the engine loop can consume.
# ---------------------------------------------------------------------------


def plan_from_solution(
    tables: DPTables,
    choices_flat: np.ndarray | None,
    n_gpu_act_slots: int,
    *,
    min_required_recompute_time_ms: float,
    max_optional_recompute_time_avoided_ms: float,
) -> SaveLevelPlan:
    """Wrap a raw solver output (``choices_flat`` or ``None``) as a
    :class:`SaveLevelPlan`.

    Mirrors ``orig/active_model.py:628-661``:

    * If the solver returned no feasible schedule, every offloadable pair
      falls back to the lowest tier (``0``).
    * The last ``n_gpu_act_slots`` tasks are forced to their layer's
      ``max_tier`` -- those are the pairs whose activations live on the GPU
      ring at the end of forward and get consumed immediately in backward.
    """
    T = tables.T
    n_home = max(0, T - n_gpu_act_slots)

    if choices_flat is None:
        # Fallback: minimally saved for every task. This matches orig's
        # ``No valid DP schedule found to avoid idle time`` branch.
        # Only the FIRST n_home tasks get a tier; the tail stays on
        # device (see below).
        choices_flat = np.zeros(T, dtype=np.int32)
    else:
        choices_flat = np.asarray(choices_flat, dtype=np.int32).copy()

    # Compute headline stats against the DP-optimum choices BEFORE we
    # overwrite the tail with the -1 sentinel. Mirrors orig's
    # ``total_recompute_time = total_fwd - min_required_avoid -
    # t_optional_avoid`` computed at :644-648 (before the tail flip).
    total_fwd_time_ms = float(tables.compute_times.sum())
    # Clamp choices to each task's own max tier defensively, in case
    # the solver returned a disallowed cell on a padded-rectangular row.
    for t in range(T):
        if choices_flat[t] > tables.max_tier_per_task[t]:
            choices_flat[t] = tables.max_tier_per_task[t]
    t_optional_avoid = float(
        tables.values[np.arange(T), choices_flat].sum()
    )
    if t_optional_avoid < -1e15:
        t_optional_avoid = 0.0
    total_recompute_time_ms = max(
        0.0,
        total_fwd_time_ms - min_required_recompute_time_ms - t_optional_avoid,
    )

    # Now flip the final ``n_gpu_act_slots`` tasks to the on-device
    # sentinel (SaveLevel(-1)). Matches orig's logic at
    # ``orig/active_model.py:803-804``: the first ``n_home_act_slots``
    # pairs (in forward traversal order) get DP-chosen tiers, and
    # the last ``n_gpu_act_slots`` pairs stay resident on the GPU ring
    # (consumed by backward immediately). This is what guarantees the
    # backward pass's first iteration finds its activations already
    # on-device with no prefetch needed.
    for t in range(n_home, T):
        choices_flat[t] = -1

    choices: dict[tuple[int, int], SaveLevel] = {}
    for (lid, cid), t in tables.indexing.items():
        choices[(lid, cid)] = SaveLevel(int(choices_flat[t]))

    return SaveLevelPlan(
        choices=choices,
        estimated_recompute_time_ms=total_recompute_time_ms,
        estimated_fwd_time_ms=total_fwd_time_ms,
    )
