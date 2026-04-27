"""IO layer: HF safetensors, native checkpoints, token sources.

The guiding rule is the one from docs/PLAN.md "Multi-architecture strategy":
HuggingFace is a CONFIG + WEIGHTS + TOKENIZER source, not a compute path. We
do not instantiate ``AutoModelForCausalLM`` to run forward passes -- we
translate a ``PretrainedConfig`` into a :class:`ModelConfig` and a
safetensors shard directory into FlexTrain's pinned host buffers via a
per-architecture weight-name map.

Naming
------
Token-ingestion adapters live under :mod:`flextrain.io.sources`. We
use "source" (not "stream") because "stream" is already heavily used
for CUDA stream objects (:class:`flextrain.engine.streams.StreamBundle`).
"""

from .sequence import Sequence
from .sources import (
    CustomSchemaTokenSource,
    HFTokenSource,
    RawTokenSource,
    ShardTokenSource,
    SyntheticTokenSource,
    TokenSource,
)

__all__ = [
    "CustomSchemaTokenSource",
    "HFTokenSource",
    "RawTokenSource",
    "Sequence",
    "ShardTokenSource",
    "SyntheticTokenSource",
    "TokenSource",
]
