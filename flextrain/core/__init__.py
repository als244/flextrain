"""Core engine primitives. No torch.nn compute lives here -- only the
abstractions the engine and layer implementations share."""

from .activation_schema import (
    ActivationField,
    ActivationSchema,
    ActivationSlot,
)
from .layer import (
    BackwardIntermediates,
    ChunkMeta,
    ComputeCost,
    InputLayer,
    Layer,
    LayerContext,
    OutputLayer,
    ParamSpec,
    TensorSpec,
)
from .save_level import (
    DPTables,
    HardwareCost,
    SaveLevel,
    SaveLevelPlan,
    build_dp_tables,
)
from .hw_probe import HardwareProbeResult, probe_hardware

__all__ = [
    "ActivationField",
    "ActivationSchema",
    "ActivationSlot",
    "BackwardIntermediates",
    "ChunkMeta",
    "ComputeCost",
    "DPTables",
    "HardwareCost",
    "HardwareProbeResult",
    "InputLayer",
    "Layer",
    "LayerContext",
    "OutputLayer",
    "ParamSpec",
    "SaveLevel",
    "SaveLevelPlan",
    "TensorSpec",
    "build_dp_tables",
    "probe_hardware",
]
