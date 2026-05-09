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
from . import mistral  # noqa: F401
from . import olmoe  # noqa: F401
from . import qwen2  # noqa: F401
from . import qwen3  # noqa: F401
from . import qwen3_5  # noqa: F401
from . import qwen3_5_moe  # noqa: F401
from . import qwen3_moe  # noqa: F401
from . import qwen3_next  # noqa: F401

__all__ = [
    "gemma2", "gemma3", "llama", "mistral", "olmoe", "qwen2",
    "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_moe", "qwen3_next",
    "ARCH_MODULES", "get_arch_module",
]


# Short-name registry used by ``flextrain.from_dims``. Each entry maps
# a user-facing arch name to the ``flextrain.io.arch.*`` module that
# exposes ``ARCH_NAME``, ``expand_dims``, ``default_hyperparams``, and
# ``BLOCK_BUILDER``. Hybrid-attn arches (qwen3_5, qwen3_5_moe,
# qwen3_next) are excluded — they need a per-layer ``layer_types``
# schedule that's a v2 follow-up. Gemma3 has an ArchSpec but no block
# builder registered yet, so it's also excluded.
ARCH_MODULES = {
    "llama": llama,
    "mistral": mistral,
    "gemma2": gemma2,
    "qwen2": qwen2,
    "qwen3": qwen3,
    "olmoe": olmoe,
    "qwen3_moe": qwen3_moe,
}


def get_arch_module(name: str):
    """Look up an arch module by its short name.

    Raises ``KeyError`` listing the known names if ``name`` is
    unrecognized — much friendlier than the bare ``KeyError(name)``
    a dict access would produce.
    """
    try:
        return ARCH_MODULES[name]
    except KeyError:
        known = ", ".join(sorted(ARCH_MODULES))
        raise KeyError(
            f"unknown arch {name!r}. Known: {known}"
        ) from None
