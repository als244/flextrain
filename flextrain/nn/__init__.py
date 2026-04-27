"""nn: composable layer + block implementations.

* ``blocks/``  -- reusable pieces (norm, attention, FFN, RoPE). Each block
  declares its own activation fields and param specs.
* ``layers/``  -- per-architecture backbone blocks (Llama, Qwen, OLMoE, ...).
  Each is a thin composition of ``blocks/`` pieces and implements the
  :class:`~flextrain.core.Layer` Protocol.
* ``embed.py`` / ``head.py``  -- token-embedding and LM-head layers,
  implementing :class:`~flextrain.core.InputLayer` /
  :class:`~flextrain.core.OutputLayer`.
* ``loss.py``  -- pluggable loss objectives (CE, MSE, GRPO, ...).
  Consumed by :class:`~flextrain.nn.head.LMHead.forward_backward` inside
  its fused micro-chunk loop.
"""

from .embed import TokenEmbedConfig, TokenEmbedLayer
from .head import LMHead, LMHeadConfig
from .loss import (
    IGNORE_INDEX,
    CrossEntropyLoss,
    GRPOLoss,
    LossFn,
    MSELoss,
    TokenContext,
    TokenSlice,
)

__all__ = [
    "CrossEntropyLoss",
    "GRPOLoss",
    "IGNORE_INDEX",
    "LMHead",
    "LMHeadConfig",
    "LossFn",
    "MSELoss",
    "TokenContext",
    "TokenEmbedConfig",
    "TokenEmbedLayer",
    "TokenSlice",
]
