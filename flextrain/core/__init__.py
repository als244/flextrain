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
    ModalityEncoder,
    OutputLayer,
    ParamSpec,
    TensorSpec,
)
from .modality import (
    ImageEmbeddings,
    ImageGradInputs,
    ImageInputCPU,
    ImageInputs,
    InputsSummary,
    ModalityEmbeddings,
    ModalityGradInputs,
    ModalityInputCPU,
    ModalityInputs,
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
    "ImageEmbeddings",
    "ImageGradInputs",
    "ImageInputCPU",
    "ImageInputs",
    "InputLayer",
    "InputsSummary",
    "Layer",
    "LayerContext",
    "ModalityEncoder",
    "ModalityEmbeddings",
    "ModalityGradInputs",
    "ModalityInputCPU",
    "ModalityInputs",
    "OutputLayer",
    "ParamSpec",
    "SaveLevel",
    "SaveLevelPlan",
    "TensorSpec",
    "build_dp_tables",
    "probe_hardware",
]
