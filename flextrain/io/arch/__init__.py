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
    "ARCH_MODULES", "get_arch_module", "expand_layer_pattern",
]


# Short-name registry used by ``flextrain.from_dims``. Each entry maps
# a user-facing arch name to the ``flextrain.io.arch.*`` module that
# exposes ``ARCH_NAME``, ``expand_dims``, ``default_hyperparams``, and
# ``BLOCK_BUILDER``. Gemma3 has an ArchSpec but no block builder
# registered yet, so it's still excluded.
ARCH_MODULES = {
    "llama": llama,
    "mistral": mistral,
    "gemma2": gemma2,
    "qwen2": qwen2,
    "qwen3": qwen3,
    "olmoe": olmoe,
    "qwen3_moe": qwen3_moe,
    "qwen3_5": qwen3_5,
    "qwen3_5_moe": qwen3_5_moe,
    "qwen3_next": qwen3_next,
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


# ---------------------------------------------------------------------------
# Hybrid-attention layer-pattern shorthand. Used by Qwen3.5 / Qwen3.5-MoE /
# Qwen3-Next (and any other arch with a per-layer ``layer_types`` schedule).
# ---------------------------------------------------------------------------

_LAYER_TYPE_CODES = {
    "F": "full_attention",
    "L": "linear_attention",
    "S": "sliding_attention",
}


def expand_layer_pattern(pattern: str, n_layers: int) -> list:
    """Expand a compact layer-pattern shorthand into a length-``n_layers``
    HF ``layer_types`` list.

    Grammar: ``<count><code>(<count><code>)*``. Codes:

    * ``F`` → ``"full_attention"``
    * ``L`` → ``"linear_attention"``
    * ``S`` → ``"sliding_attention"``

    The pattern repeats to fill ``n_layers``. Examples:

    * ``"1F1L"``  with ``n_layers=4``  → FLFL  (Qwen3.5 alternating)
    * ``"1F47L"`` with ``n_layers=48`` → 1 full at start, 47 linear after
      (Qwen3-Next 47-1 hybrid)
    * ``"3L1F"``  with ``n_layers=12`` → LLLF LLLF LLLF (Qwen3.5-MoE 3:1)

    Raises ``ValueError`` on a malformed pattern (missing count, unknown
    type code, empty pattern, non-positive count).
    """
    chunks: list = []
    i = 0
    while i < len(pattern):
        j = i
        while j < len(pattern) and pattern[j].isdigit():
            j += 1
        if j == i:
            raise ValueError(
                f"layer_pattern {pattern!r}: missing count at position {i}"
            )
        if j >= len(pattern):
            raise ValueError(
                f"layer_pattern {pattern!r}: missing type code after count"
            )
        count = int(pattern[i:j])
        if count <= 0:
            raise ValueError(
                f"layer_pattern {pattern!r}: count must be positive, got {count}"
            )
        ch = pattern[j].upper()
        if ch not in _LAYER_TYPE_CODES:
            raise ValueError(
                f"layer_pattern {pattern!r}: unknown type code {ch!r} "
                f"(expected one of {','.join(sorted(_LAYER_TYPE_CODES))})"
            )
        chunks.append((count, _LAYER_TYPE_CODES[ch]))
        i = j + 1
    if not chunks:
        raise ValueError("layer_pattern is empty")
    out: list = []
    while len(out) < n_layers:
        for count, t in chunks:
            for _ in range(count):
                if len(out) >= n_layers:
                    return out
                out.append(t)
    return out
