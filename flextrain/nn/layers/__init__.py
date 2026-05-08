"""Per-architecture (model-family) backbone blocks.

Each module in this package is named after a *released model family* and
composes a specific combination of algorithmic blocks from
:mod:`flextrain.nn.blocks`. Adding a new architecture is typically:

1. A new ``<family>.py`` file under ``nn/layers/``.
2. A matching weight-map in ``flextrain/io/arch/<family>.py`` for HF
   safetensors load / export.
3. A matching ``hf_config_to_flextrain`` adapter in the same arch module.

See docs/internal/PLAN.md "Multi-architecture strategy" for the full coverage list.
"""

from .llama import LlamaBlock, LlamaBlockConfig
from .mistral import MistralBlock, MistralBlockConfig
from .olmoe import OLMoEBlock, OLMoEBlockConfig
from .qwen2 import Qwen2Block, Qwen2BlockConfig
from .qwen3 import (
    Qwen3DenseBlock,
    Qwen3DenseBlockConfig,
    Qwen3DenseSWABlock,
    Qwen3DenseSWABlockConfig,
)

__all__ = [
    "LlamaBlock",
    "LlamaBlockConfig",
    "MistralBlock",
    "MistralBlockConfig",
    "OLMoEBlock",
    "OLMoEBlockConfig",
    "Qwen2Block",
    "Qwen2BlockConfig",
    "Qwen3DenseBlock",
    "Qwen3DenseBlockConfig",
    "Qwen3DenseSWABlock",
    "Qwen3DenseSWABlockConfig",
]
