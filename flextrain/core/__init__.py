"""Core engine primitives. No torch.nn compute lives here -- only the
abstractions the engine and layer implementations share."""

from .activation_schema import (
    ActivationField,
    ActivationSchema,
    ActivationSlot,
)
from .layer import (
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
    SaveLevel,
    SaveLevelPlan,
    build_dp_tables,
)

__all__ = [
    "ActivationField",
    "ActivationSchema",
    "ActivationSlot",
    "ChunkMeta",
    "ComputeCost",
    "DPTables",
    "InputLayer",
    "Layer",
    "LayerContext",
    "OutputLayer",
    "ParamSpec",
    "SaveLevel",
    "SaveLevelPlan",
    "TensorSpec",
    "build_dp_tables",
]
