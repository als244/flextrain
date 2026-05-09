"""FlexTrain: AdaWS-based single-GPU Transformer training engine.

See ``README.md`` and ``docs/`` for the high-level overview, and
``flextrain.api.from_pretrained`` for the recommended starting point.
"""

from .api import (  # noqa: F401  -- re-exports
    BuildContext,
    BlockBuilder,
    from_dims,
    from_pretrained,
    register_block_builder,
)

# Side-effect imports so block-builder + ArchSpec registration runs at
# package import time. Add new arches here.
from .io.arch import gemma2 as _gemma2  # noqa: F401
from .io.arch import llama as _llama  # noqa: F401
from .io.arch import mistral as _mistral  # noqa: F401
from .io.arch import olmoe as _olmoe  # noqa: F401
from .io.arch import qwen2 as _qwen2  # noqa: F401
from .io.arch import qwen3 as _qwen3  # noqa: F401
from .io.arch import qwen3_moe as _qwen3_moe  # noqa: F401


__all__ = [
    "BuildContext",
    "BlockBuilder",
    "from_dims",
    "from_pretrained",
    "register_block_builder",
]
