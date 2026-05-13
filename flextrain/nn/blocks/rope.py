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
    *,
    head_dim: int | None = None,
) -> torch.Tensor:
    """Compute ``inv_freq`` of length ``rot_dim/2`` for partial-rotary RoPE.

    Default convention (Qwen3-Next/3.5/3.6, rope_type ``'default'``):
    divides by ``rot_dim``, not full ``head_dim`` — matches HF's
    ``compute_default_rope_parameters`` over the partial sub-dim:

        inv_freq[i] = base ** (-2i / rot_dim)        for i in [0, rot_dim/2)

    Proportional convention (Gemma 4, rope_type ``'proportional'``):
    divides by ``head_dim`` instead of ``rot_dim``. HF's
    ``_compute_proportional_rope_parameters`` (see
    ``modeling_rope_utils.py``). Caller must pass ``head_dim`` as a kwarg:

        inv_freq[i] = base ** (-2i / head_dim)       for i in [0, rot_dim/2)

    The kernel side (``apply_rope_partial_fwd/bwd``) is unchanged; it just
    consumes whatever inv_freq curve we hand it. Channels [rot_dim:head_dim]
    pass through unrotated either way.
    """
    factor = 1.0
    rope_type = "default"
    if rope_scaling is not None:
        rope_type = (
            rope_scaling.get("rope_type")
            or rope_scaling.get("type")
            or "default"
        )
        if rope_type == "linear":
            factor = float(rope_scaling.get("factor", 1.0))
        elif rope_type not in ("default", "proportional", None):
            import warnings
            warnings.warn(
                f"build_partial_rope_inv_freq: rope_type={rope_type!r} not "
                "implemented for partial-rotary; falling back to vanilla.",
                stacklevel=2,
            )
            rope_type = "default"
    half = rot_dim // 2
    if rope_type == "proportional":
        if head_dim is None:
            raise ValueError(
                "build_partial_rope_inv_freq with rope_type='proportional' "
                "requires the head_dim kwarg (denominator differs from "
                "rot_dim under Gemma 4's proportional convention)."
            )
        inv = 1.0 / (
            rope_base
            ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / head_dim)
        )
    else:
        inv = 1.0 / (
            rope_base
            ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / rot_dim)
        )
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


# ---------------------------------------------------------------------------
# Multi-axis ("MRoPE") partial-rotary RoPE — Qwen-VL family.
#
# Qwen3.5 / Qwen3.6 / Qwen3-VL apply different position axes to different
# slices of the rotated channels. Each frequency pair is assigned to one of
# three position axes (t = temporal/text, h = image-height, w = image-width).
# For text-only sequences all three axes are equal (text position),
# numerically identical to standard partial RoPE; for multimodal sequences
# vision tokens carry distinct (t,h,w) per token so each pair rotates with
# a different effective position.
#
# Two layouts (controlled by ``mrope_interleaved`` in HF config):
#
# * ``mrope_interleaved=False`` -- contiguous: first ``mrope_section[0]``
#   pairs are t-axis, next ``mrope_section[1]`` are h-axis, last
#   ``mrope_section[2]`` are w-axis.
# * ``mrope_interleaved=True`` (Qwen3.5/3.6/3-VL default) -- HF's
#   ``apply_interleaved_mrope``: pair index k uses h-axis if
#   ``k % 3 == 1`` and ``k < 3 * mrope_section[1]``; w-axis if
#   ``k % 3 == 2`` and ``k < 3 * mrope_section[2]``; otherwise t-axis.
#   Tail indices (beyond ``3 * max(section_h, section_w)``) all stay on
#   t-axis. Reproduces HF
#   ``Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope``.
#
# Phase 1 implementation note
# ---------------------------
# The function below is a **pure-PyTorch reference impl** -- correctness
# first. It allocates short-lived cos/sin tensors of size ``(T, rot_dim/2)``
# per call. A fused Triton kernel matching the existing
# ``flextrain_rope_partial_fwd/bwd`` API is planned but deferred until
# after we have HF-parity tests in place. See
# ``docs/internal/multimodal_session_notes.md``.
# ---------------------------------------------------------------------------


_AXIS_TEMPORAL: int = 0
_AXIS_HEIGHT: int = 1
_AXIS_WIDTH: int = 2


def build_mrope_axis_assignment(
    mrope_section: Sequence[int],
    mrope_interleaved: bool,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the per-pair axis-assignment table.

    Parameters
    ----------
    mrope_section
        ``(s_t, s_h, s_w)`` -- per-axis number of frequency pairs.
        ``sum(mrope_section)`` MUST equal ``rot_dim // 2``.
    mrope_interleaved
        If True (Qwen3.5/3.6/3-VL default), the axes are interleaved
        across freq indices as in HF ``apply_interleaved_mrope``. If
        False, sections are contiguous.

    Returns
    -------
    torch.Tensor
        ``(n_pairs,) int64`` -- ``axis_assignment[k] in {0,1,2}`` is the
        position axis that freq pair k rotates with.
    """
    if len(mrope_section) != 3:
        raise ValueError(
            f"mrope_section must have length 3 (t, h, w); got {mrope_section!r}"
        )
    s_t, s_h, s_w = (int(s) for s in mrope_section)
    n_pairs = s_t + s_h + s_w
    out = torch.zeros(n_pairs, dtype=torch.int64, device=device)
    if mrope_interleaved:
        # HF apply_interleaved_mrope: walk k=0..n_pairs-1; pair k uses
        # h-axis if (k%3==1 and k < 3*s_h) else w-axis if (k%3==2 and
        # k < 3*s_w) else t-axis. Tail indices stay on t-axis.
        # Encoded as torch ops to keep it device-agnostic.
        k = torch.arange(n_pairs, dtype=torch.int64, device=device)
        is_h = (k % 3 == 1) & (k < 3 * s_h)
        is_w = (k % 3 == 2) & (k < 3 * s_w)
        out = torch.where(
            is_h,
            torch.full_like(out, _AXIS_HEIGHT),
            torch.where(
                is_w,
                torch.full_like(out, _AXIS_WIDTH),
                torch.full_like(out, _AXIS_TEMPORAL),
            ),
        )
    else:
        # Contiguous sections: t..t (s_t), h..h (s_h), w..w (s_w).
        out[s_t : s_t + s_h] = _AXIS_HEIGHT
        out[s_t + s_h : s_t + s_h + s_w] = _AXIS_WIDTH
    return out


def _apply_mrope_pair_interleaved(
    tensors: Sequence[torch.Tensor],
    seq_positions_3d: torch.Tensor,
    inv_freq: torch.Tensor,
    rot_dim: int,
    axis_assignment: torch.Tensor,
    *,
    bwd: bool,
) -> list[torch.Tensor]:
    """Common math for fwd/bwd MRoPE in pair-interleaved channel layout.

    Pair-interleaved means each tensor's last axis is laid out as
    ``(pair0_real, pair0_imag, pair1_real, pair1_imag, ..., pad)`` --
    matching what the post-load halved->pair permutation produces and
    what :func:`apply_rope_partial_fwd` already consumes.

    For each pair ``k`` with axis ``a = axis_assignment[k]``:

        pos_k(t) = seq_positions_3d[t, a]
        angle_k(t) = pos_k(t) * inv_freq[k]
        c_k(t), s_k(t) = cos(angle_k(t)), sin(angle_k(t))

    Forward rotation (per pair):

        x'_{2k}    =  x_{2k} * c  -  x_{2k+1} * s
        x'_{2k+1}  =  x_{2k} * s  +  x_{2k+1} * c

    Backward inverse:

        x_{2k}    =  x'_{2k} * c  +  x'_{2k+1} * s
        x_{2k+1}  = -x'_{2k} * s  +  x'_{2k+1} * c

    Modifies each tensor's first ``rot_dim`` channels in place.
    """
    if seq_positions_3d.dim() != 2 or seq_positions_3d.shape[-1] != 3:
        raise ValueError(
            f"seq_positions_3d must be (T, 3) int32 for MRoPE; got shape "
            f"{tuple(seq_positions_3d.shape)}"
        )
    if rot_dim % 2 != 0:
        raise ValueError(f"rot_dim must be even; got {rot_dim}")
    n_pairs = rot_dim // 2
    if inv_freq.numel() != n_pairs:
        raise ValueError(
            f"inv_freq has {inv_freq.numel()} entries but rot_dim/2 = "
            f"{n_pairs}; inv_freq must match the rotary sub-dim"
        )
    if axis_assignment.numel() != n_pairs:
        raise ValueError(
            f"axis_assignment has {axis_assignment.numel()} entries but "
            f"rot_dim/2 = {n_pairs}; mismatch implies a bad mrope_section"
        )

    # Ensure assignment lives on the same device as positions.
    if axis_assignment.device != seq_positions_3d.device:
        axis_assignment = axis_assignment.to(seq_positions_3d.device)
    if inv_freq.device != seq_positions_3d.device:
        inv_freq = inv_freq.to(seq_positions_3d.device)

    # Gather per-pair positions: (T, n_pairs) int64 then -> fp32 for the
    # angle computation.
    T = seq_positions_3d.shape[0]
    # gather requires int64 index along dim=1.
    axis_idx = axis_assignment.unsqueeze(0).expand(T, n_pairs)  # (T, n_pairs)
    pos_per_pair = seq_positions_3d.to(torch.int64).gather(1, axis_idx)
    angles = pos_per_pair.to(torch.float32) * inv_freq.to(torch.float32).unsqueeze(0)
    # cos / sin: (T, n_pairs) fp32.
    cos = angles.cos()
    sin = angles.sin()

    for x in tensors:
        if x.shape[-1] < rot_dim:
            raise ValueError(
                f"tensor last-dim {x.shape[-1]} < rot_dim {rot_dim}; cannot rotate"
            )
        # View the first rot_dim channels as (..., n_pairs, 2).
        x_rot = x[..., :rot_dim]
        leading_shape = x_rot.shape[:-1]
        x_pairs = x_rot.view(*leading_shape, n_pairs, 2)
        x_a = x_pairs[..., 0]  # (..., n_pairs)
        x_b = x_pairs[..., 1]  # (..., n_pairs)
        # cos / sin broadcast: (T, n_pairs) -> add singleton dims for any
        # axes between T and n_pairs (e.g. H for (T, H, n_pairs)).
        # Insert ones in cos/sin for each intermediate dim.
        n_extra = x_a.dim() - 2  # subtract T-axis and n_pairs-axis
        c = cos
        s = sin
        for _ in range(n_extra):
            c = c.unsqueeze(1)
            s = s.unsqueeze(1)
        # Cast cos / sin to the tensor's compute dtype right before
        # multiplication so the math runs at the storage precision and
        # the result writes back cleanly (in-place .copy_ uses the
        # destination dtype).
        c = c.to(x_a.dtype)
        s = s.to(x_a.dtype)
        if bwd:
            new_a = x_a * c + x_b * s
            new_b = -x_a * s + x_b * c
        else:
            new_a = x_a * c - x_b * s
            new_b = x_a * s + x_b * c
        # Write back in place. ``x_pairs[..., 0] = new_a`` invokes
        # in-place __setitem__ on the view, which propagates to the
        # underlying ``x`` storage. Use ``copy_`` to be explicit and to
        # avoid any chance of an out-of-place fallback.
        x_pairs[..., 0].copy_(new_a)
        x_pairs[..., 1].copy_(new_b)
    return list(tensors)


def apply_rope_mrope_fwd(
    tensors: Sequence[torch.Tensor],
    seq_positions_3d: torch.Tensor,
    inv_freq: torch.Tensor,
    rot_dim: int,
    axis_assignment: torch.Tensor,
) -> list[torch.Tensor]:
    """Multi-axis (MRoPE) partial-RoPE forward, pair-interleaved layout.

    Parameters
    ----------
    tensors
        Sequence of ``(T, H, head_dim)`` Q / K tensors. The first
        ``rot_dim`` channels of each are rotated in place; channels
        ``[rot_dim, head_dim)`` pass through untouched.
    seq_positions_3d
        ``(T, 3) int32`` -- per-token ``(t_pos, h_pos, w_pos)`` for
        MRoPE. For text-only tokens all three axes carry the same text
        position (degenerate); for vision tokens they carry the patch's
        ``(t, h, w)`` grid coordinates.
    inv_freq
        ``(rot_dim/2,) fp32`` -- frequency curve (same shape /
        provenance as the standard partial-RoPE ``inv_freq``; see
        :func:`build_partial_rope_inv_freq`).
    rot_dim
        Number of channels actually rotated (must be even).
    axis_assignment
        ``(rot_dim/2,) int64`` -- per-pair axis index in ``{0,1,2}``.
        Build with :func:`build_mrope_axis_assignment` once per block.
    """
    return _apply_mrope_pair_interleaved(
        tensors,
        seq_positions_3d,
        inv_freq,
        rot_dim,
        axis_assignment,
        bwd=False,
    )


def apply_rope_mrope_bwd(
    grad_tensors: Sequence[torch.Tensor],
    seq_positions_3d: torch.Tensor,
    inv_freq: torch.Tensor,
    rot_dim: int,
    axis_assignment: torch.Tensor,
) -> list[torch.Tensor]:
    """Inverse of :func:`apply_rope_mrope_fwd`. Modifies the first
    ``rot_dim`` channels of each dQ / dK in place.
    """
    return _apply_mrope_pair_interleaved(
        grad_tensors,
        seq_positions_3d,
        inv_freq,
        rot_dim,
        axis_assignment,
        bwd=True,
    )
