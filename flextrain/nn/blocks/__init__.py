"""Composable transformer sub-blocks.

Naming convention
-----------------
Files and classes in this package are named after the **algorithm** they
implement, not the model they appear in:

* :class:`GQAAttentionBlock`              -- grouped-query attention with
                                             full causal context
                                             (Llama, Qwen-dense, OLMoE)
* :class:`GQASlidingWindowAttentionBlock` -- GQA with sliding window
                                             (Mistral, Gemma-alternating,
                                             GPT-OSS subset)
* :class:`RMSNormBlock`                   -- RMSNorm
* :class:`SwiGLUFFN`                      -- SwiGLU-gated dense FFN

Future additions will be algorithm-named as well:
``LinearAttentionBlock`` (Qwen3-Next), ``MLAAttentionBlock`` (DeepSeek-V3),
``QKNormBlock`` (Qwen3-dense), ``GELUGatedFFN`` (Gemma),
``MoESwiGLUFFN`` / ``SharedExpertMoE`` etc.

Each block declares its own ``fields()``, ``param_spec()``,
``compute_cost()`` plus ``fwd``/``bwd`` methods. Model-family classes
in :mod:`flextrain.nn.layers` pick algorithmic blocks and compose them.
"""

from .attention import (
    AttentionConfig,  # back-compat alias for GQAAttentionConfig
    GQAAttentionBlock,
    GQAAttentionConfig,
    GQASlidingWindowAttentionBlock,
    GQASlidingWindowAttentionConfig,
)
from .attention_gated import (
    GQAAttentionGatedBlock,
    GQAAttentionGatedConfig,
)
from .ffn_dense import SwiGLUConfig, SwiGLUFFN
from .ffn_moe import MoESwiGLUConfig, MoESwiGLUFFN
from .ffn_moe_shared import (
    MoESwiGLUSharedExpertConfig,
    MoESwiGLUSharedExpertFFN,
)
from .norm import RMSNormBlock
from .rope import (
    apply_rope_bwd,
    apply_rope_fwd,
    apply_rope_partial_bwd,
    apply_rope_partial_fwd,
)

__all__ = [
    "AttentionConfig",
    "GQAAttentionBlock",
    "GQAAttentionConfig",
    "GQAAttentionGatedBlock",
    "GQAAttentionGatedConfig",
    "GQASlidingWindowAttentionBlock",
    "GQASlidingWindowAttentionConfig",
    "MoESwiGLUConfig",
    "MoESwiGLUFFN",
    "MoESwiGLUSharedExpertConfig",
    "MoESwiGLUSharedExpertFFN",
    "RMSNormBlock",
    "SwiGLUConfig",
    "SwiGLUFFN",
    "apply_rope_bwd",
    "apply_rope_fwd",
    "apply_rope_partial_bwd",
    "apply_rope_partial_fwd",
]
