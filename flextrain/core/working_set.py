"""Working-set sizing: thin dataclass wrapper over the existing derivation.

Why a wrapper
-------------
``orig/working_set.py`` contains ~750 LOC of carefully-tuned heuristics that
jointly derive ``TR`` (tokens per round), ``TC`` (chunk size), ``NP``, ``NG``
(layer counts resident on GPU), and the GPU/host activation buffer sizes
from hardware + model specs. That math IS the paper (§3.2, §3.3);
reimplementing it while also landing every other piece of the v2 refactor
is an unforced risk.

So: we keep the math identical by delegating to
``orig.working_set.determine_working_set_config`` and wrap its dict output
in a typed :class:`WorkingSetConfig`. When the rest of the engine is stable
we can lift the logic module-for-module; for now the API is clean and the
math is battle-tested.

The number of GPU activation slots (``NA``) is NOT returned by orig's
working-set function -- it's computed engine-side as
``gpu_act_buffer_size // act_slot_size_bytes`` (``orig/active_model.py:453``).
We follow the same convention: a helper :func:`derive_n_gpu_act_slots` lives
here so both orig and v2 engines can use it consistently.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping


_ORIG_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "orig")
)
if _ORIG_ROOT not in sys.path:
    sys.path.insert(0, _ORIG_ROOT)


@dataclass(frozen=True)
class WorkingSetConfig:
    """Typed view of orig's working-set return tuple ``(config_dict, hw_env)``.

    Field names mirror ``working_set.py:714-732`` exactly; additions to orig
    surface here by adding a field + reading the key from ``raw``.
    """

    # Data sizing (§3.2)
    target_round_tokens: int  # TR
    max_chunk_size: int  # TC
    max_training_chunks: int
    max_total_round_tokens: int
    target_num_rounds: int

    # Memory partitioning (§3.3) -- NP, NG only. NA is derived below.
    n_gpu_layers: int  # NP
    n_gpu_grads: int  # NG
    n_gpu_opt_layers: int

    # Activation-buffer byte budgets (engine divides by act_slot_size to get NA)
    gpu_act_buffer_size: int
    host_act_buffer_size: int

    # Budgets echoed back (for logging / sanity checks)
    available_gpu_memory_bytes: int
    available_host_memory_bytes: int
    leeway_gpu_memory_bytes: int
    leeway_host_memory_bytes: int
    max_seq_len: int

    # Hardware environment orig measured to make its choice. Opaque here;
    # downstream code (e.g. HardwareCost for the DP solver) reads specific
    # keys.
    hardware_env: Mapping[str, Any]

    # The untyped original dict for anything we haven't surfaced yet.
    raw: Mapping[str, Any]


def derive_n_gpu_act_slots(
    gpu_act_buffer_size: int, act_slot_size_bytes: int
) -> int:
    """Mirror ``orig/active_model.py:453``.

    The activation-slot ring's count is ``gpu_act_buffer_size //
    act_slot_size_bytes``; the engine consults this every round to set
    ``NA``. ``act_slot_size_bytes`` is the maximum-tier size from the
    layer's schema at ``chunk_size=max_chunk_size``.
    """
    if act_slot_size_bytes <= 0:
        raise ValueError("act_slot_size_bytes must be positive")
    return gpu_act_buffer_size // act_slot_size_bytes


def determine_working_set_config(
    model_dims: Mapping[str, Any],
    max_seq_len: int,
    max_global_batch_tokens: int,
    *,
    training_config: Mapping[str, Any] | None = None,
    has_embed: bool = True,
    has_head: bool = True,
    num_local_layers: int | None = None,
    chunk_size: int | None = None,
    max_gpu_mem_bytes: int | None = None,
    max_host_mem_bytes: int | None = None,
    leeway_gpu_mem_bytes: int = 2 * (1 << 30),
    leeway_host_mem_bytes: int = 10 * (1 << 30),
    verbose: bool = False,
    device_id: int = 0,
    min_tokens_per_round_limit: int | None = None,
    max_tokens_per_round_limit: int | None = None,
    fixed_seq_len: bool = False,
    min_chunk_size: int | None = None,
    max_chunk_size: int | None = None,
) -> WorkingSetConfig:
    """Thin wrapper over orig's ``determine_working_set_config``.

    Returns one :class:`WorkingSetConfig`; orig's second return value
    (``chosen_hardware_env``) is stashed inside it as ``hardware_env``.
    """
    # Deferred import so merely importing this module doesn't require a GPU
    # (orig.working_set imports bench_matmul/bench_transfer which touch CUDA).
    from working_set import (  # type: ignore[import-not-found]
        determine_working_set_config as _orig_impl,
    )

    raw, hw_env = _orig_impl(
        model_dims=model_dims,
        max_seq_len=max_seq_len,
        max_global_batch_tokens=max_global_batch_tokens,
        training_config=training_config,
        has_embed=has_embed,
        has_head=has_head,
        num_local_layers=num_local_layers,
        chunk_size=chunk_size,
        max_gpu_mem_bytes=max_gpu_mem_bytes,
        max_host_mem_bytes=max_host_mem_bytes,
        leeway_gpu_mem_bytes=leeway_gpu_mem_bytes,
        leeway_host_mem_bytes=leeway_host_mem_bytes,
        verbose=verbose,
        device_id=device_id,
        min_tokens_per_round_limit=min_tokens_per_round_limit,
        max_tokens_per_round_limit=max_tokens_per_round_limit,
        fixed_seq_len=fixed_seq_len,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
    )

    return WorkingSetConfig(
        target_round_tokens=int(raw["target_round_tokens"]),
        max_chunk_size=int(raw["max_chunk_size"]),
        max_training_chunks=int(raw["max_training_chunks"]),
        max_total_round_tokens=int(raw["max_total_round_tokens"]),
        target_num_rounds=int(raw["target_num_rounds"]),
        n_gpu_layers=int(raw["n_gpu_layers"]),
        n_gpu_grads=int(raw["n_gpu_grads"]),
        n_gpu_opt_layers=int(raw["n_gpu_opt_layers"]),
        gpu_act_buffer_size=int(raw["gpu_act_buffer_size"]),
        host_act_buffer_size=int(raw["host_act_buffer_size"]),
        available_gpu_memory_bytes=int(raw["available_gpu_memory_bytes"]),
        available_host_memory_bytes=int(raw["available_host_memory_bytes"]),
        leeway_gpu_memory_bytes=int(raw["leeway_gpu_memory_bytes"]),
        leeway_host_memory_bytes=int(raw["leeway_host_memory_bytes"]),
        max_seq_len=int(raw["max_seq_len"]),
        hardware_env=hw_env,
        raw=raw,
    )
