"""RoPE block.

Thin wrapper over ``flextrain_rope_fwd`` / ``flextrain_rope_bwd``.
Operates in-place on ``(T, H, D)`` Q and K tensors, fused when both
are supplied.

The kernel takes a precomputed ``inv_freq`` array of length ``D/2``,
which lets callers pass arbitrary frequency curves: vanilla RoPE,
Llama-3.1 YARN-scaled RoPE, NTK, etc. Build it once with
:func:`build_rope_inv_freq`.

No activation fields, no parameters — RoPE is a positional rotation
driven by ``chunk.seq_positions`` and the precomputed ``inv_freq``, so
it contributes nothing to the activation schema or param spec and is
composed as a helper on the attention block.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from flextrain.ops import flextrain_rope_bwd, flextrain_rope_fwd


def _llama3_inv_freq(
    head_dim: int,
    rope_base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_pos: int,
) -> torch.Tensor:
    """Llama-3.1+ YARN-style frequency scaling, mirroring
    ``transformers.modeling_rope_utils._compute_llama3_parameters``.

    Vanilla ``inv_freq[i] = rope_base ** (-2i/D)`` is partitioned into
    three bands by wavelength ``λ_i = 2π / inv_freq[i]``:

    * High-frequency (λ < original_max_pos / high_freq_factor): unchanged.
    * Low-frequency  (λ > original_max_pos / low_freq_factor): divided by ``factor``.
    * Mid-frequency: smooth interpolation between the two.
    """
    half = head_dim // 2
    inv = 1.0 / (rope_base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim))
    low_wavelen = original_max_pos / low_freq_factor
    high_wavelen = original_max_pos / high_freq_factor
    wavelen = 2.0 * math.pi / inv

    # Split into three bands.
    inv_low = inv / factor                        # long-wavelength: scale down
    smooth = (original_max_pos / wavelen - low_freq_factor) / max(
        1e-12, (high_freq_factor - low_freq_factor)
    )
    inv_mid = (1.0 - smooth) * (inv / factor) + smooth * inv

    out = torch.where(wavelen < high_wavelen, inv, inv_mid)  # high-freq → vanilla, mid → smoothed
    out = torch.where(wavelen > low_wavelen, inv_low, out)   # low-freq → scaled
    return out


def build_rope_inv_freq(
    head_dim: int,
    rope_base: float,
    rope_scaling: Mapping | None = None,
) -> torch.Tensor:
    """Compute the ``inv_freq`` array of length ``head_dim/2`` (fp32, CPU).

    ``rope_scaling`` matches the HF config field. Currently supported
    ``rope_type`` values: ``None`` (vanilla), ``"default"`` (vanilla),
    ``"llama3"`` (Llama-3.1+ YARN). Unknown types fall back to vanilla
    with a warning.
    """
    if rope_scaling is None:
        half = head_dim // 2
        inv = 1.0 / (rope_base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim))
        return inv
    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or "default"
    if rope_type == "default":
        return build_rope_inv_freq(head_dim, rope_base, None)
    if rope_type == "linear":
        # Linear position scaling: effective positions are ``pos / factor``.
        # Equivalent to ``inv_freq / factor`` since
        # ``angle = pos * inv_freq``. Gemma-3 4B/12B use this with factor=8.
        inv = build_rope_inv_freq(head_dim, rope_base, None)
        return inv / float(rope_scaling.get("factor", 1.0))
    if rope_type == "llama3":
        return _llama3_inv_freq(
            head_dim=head_dim,
            rope_base=rope_base,
            factor=float(rope_scaling.get("factor", 8.0)),
            low_freq_factor=float(rope_scaling.get("low_freq_factor", 1.0)),
            high_freq_factor=float(rope_scaling.get("high_freq_factor", 4.0)),
            original_max_pos=int(rope_scaling.get("original_max_position_embeddings", 8192)),
        )
    import warnings
    warnings.warn(
        f"flextrain RoPE: unknown rope_type {rope_type!r}; falling back to vanilla. "
        "Cross-stack parity with HF may suffer for long-context models.",
        stacklevel=2,
    )
    return build_rope_inv_freq(head_dim, rope_base, None)


def apply_rope_fwd(
    tensors: Sequence[torch.Tensor],
    seq_positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> list[torch.Tensor]:
    """Fused RoPE fwd. ``seq_positions`` is ``(T, 1)`` int32.

    ``inv_freq`` is a 1-D tensor of length ``head_dim/2`` (fp32 CUDA).
    Build it with :func:`build_rope_inv_freq` once per attention block.
    For back-compat, a 1-element tensor holding the scalar RoPE base
    is also accepted (the kernel wrapper expands it to vanilla).
    """
    return flextrain_rope_fwd(list(tensors), seq_positions, inv_freq)


def apply_rope_bwd(
    grad_tensors: Sequence[torch.Tensor],
    seq_positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> list[torch.Tensor]:
    """Fused RoPE bwd. In-place rotation of ``dQ`` and ``dK``."""
    return flextrain_rope_bwd(list(grad_tensors), seq_positions, inv_freq)


# ---------------------------------------------------------------------------
# Partial-rotary RoPE (Qwen3-Next / Qwen3.5 / Qwen3.6).
#
# Only the first ``rot_dim = head_dim * partial_rotary_factor`` channels
# of each head are rotated; the remaining channels pass through. The
# loader's halved→pair permutation must therefore only permute the
# first ``rot_dim`` channels per head.
# ---------------------------------------------------------------------------


def build_partial_rope_inv_freq(
    rot_dim: int,
    rope_base: float,
    rope_scaling: Mapping | None = None,
) -> torch.Tensor:
    """Compute ``inv_freq`` of length ``rot_dim/2`` for partial-rotary RoPE.

    Note: divides by ``rot_dim``, NOT full ``head_dim`` — this matches HF's
    Qwen3-Next ``compute_default_rope_parameters`` which builds ``inv_freq``
    over the partial sub-dim:

        inv_freq[i] = base ** (-2i / rot_dim)        for i in [0, rot_dim/2)

    Currently Qwen3-Next/3.5/3.6 use ``rope_type='default'`` for the
    partial path, so ``rope_scaling`` is unused beyond verifying that
    the type is one of None/'default'. If a YARN-style partial variant
    appears we'll handle it here.
    """
    factor = 1.0
    if rope_scaling is not None:
        rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type") or "default"
        if rope_type == "linear":
            factor = float(rope_scaling.get("factor", 1.0))
        elif rope_type not in ("default", None):
            import warnings
            warnings.warn(
                f"build_partial_rope_inv_freq: rope_type={rope_type!r} not "
                "implemented for partial-rotary; falling back to vanilla.",
                stacklevel=2,
            )
    half = rot_dim // 2
    inv = 1.0 / (rope_base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / rot_dim))
    if factor != 1.0:
        inv = inv / factor
    return inv


def apply_rope_partial_fwd(
    tensors: Sequence[torch.Tensor],
    seq_positions: torch.Tensor,
    inv_freq: torch.Tensor,
    rot_dim: int,
) -> list[torch.Tensor]:
    """Fused partial-RoPE fwd in-place on Q (and optionally K).

    ``rot_dim`` is the rotary sub-dim (e.g. ``head_dim * 0.25 = 64``
    for Qwen3-Next). Channels ``[rot_dim:head_dim]`` pass through.

    ``inv_freq`` is length ``rot_dim/2`` (fp32 CUDA), built with
    :func:`build_partial_rope_inv_freq`. A 1-element tensor holding the
    scalar RoPE base is also accepted (vanilla inv_freq is built+cached).

    T is variable per call — ``seq_positions`` shape ``(T, 1)`` int32.
    """
    from flextrain.ops import flextrain_rope_partial_fwd
    return flextrain_rope_partial_fwd(
        list(tensors), seq_positions, inv_freq, rot_dim,
    )


def apply_rope_partial_bwd(
    grad_tensors: Sequence[torch.Tensor],
    seq_positions: torch.Tensor,
    inv_freq: torch.Tensor,
    rot_dim: int,
) -> list[torch.Tensor]:
    """Fused partial-RoPE bwd. Inverse-rotates the first ``rot_dim``
    channels of dQ/dK in-place; the rest are untouched."""
    from flextrain.ops import flextrain_rope_partial_bwd
    return flextrain_rope_partial_bwd(
        list(grad_tensors), seq_positions, inv_freq, rot_dim,
    )
