"""Per-architecture HF weight maps + config adapters.

Each module in this package should:

1. Declare a :class:`~flextrain.io.hf_weights.ArchSpec` naming the HF
   architectures it handles and the per-scope weight-name table.
2. Register it via :func:`~flextrain.io.hf_weights.register_arch` at
   module import time.
3. Optionally export a ``hf_config_to_flextrain(hf_config)`` function that
   produces a FlexTrain ``ModelConfig`` (or dims dict) from a HF
   ``PretrainedConfig``.

Importing this package imports all child modules as a side effect so
registration happens once.
"""

from . import gemma2  # noqa: F401
from . import gemma3  # noqa: F401
from . import llama  # noqa: F401
from . import olmoe  # noqa: F401
from . import qwen2  # noqa: F401
from . import qwen3  # noqa: F401
from . import qwen3_5  # noqa: F401
from . import qwen3_5_moe  # noqa: F401
from . import qwen3_moe  # noqa: F401
from . import qwen3_next  # noqa: F401

__all__ = [
    "gemma2", "gemma3", "llama", "olmoe", "qwen2",
    "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_moe", "qwen3_next",
]
