"""Modality data types for multimodal inputs.

This module defines the data shapes that flow between

* the data adapter (HF AutoImageProcessor wrapper in
  ``flextrain/io/image_processing.py``) which produces CPU-side
  per-image data,
* the engine (``flextrain/engine/active_model.py``) which gathers
  the per-round modality bundle and DMAs it to GPU,
* a :class:`~flextrain.core.layer.ModalityEncoder` which consumes
  the device-side bundle and produces per-token embeddings, and
* the splice strategies in :mod:`flextrain.nn.splices` which
  combine encoder output with text embeddings per chunk.

The engine never imports a concrete encoder. The types here are
the only shared surface.

Phase 1 covers the image modality only. Audio/video are Phase 2/3
additions — the ``Modality*`` type aliases are forward-declared as
single-arm unions so adding a new modality is purely additive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


# ---------------------------------------------------------------------------
# Image modality
# ---------------------------------------------------------------------------


@dataclass
class ImageInputCPU:
    """CPU-side per-image data attached to a :class:`Sequence`.

    Produced by the data adapter (the HF ``AutoImageProcessor``
    wrapper) at dataset-construction time and stashed on the
    Sequence. The engine moves it to GPU once per round in
    :meth:`MultimodalInputLayer.setup_round`.

    Attributes
    ----------
    pixel_values
        The processor's pre-tokenized representation of one image.
        The shape contract is owned by the
        (image_processor, ModalityEncoder) pair — the engine
        treats this as opaque bytes. For Qwen-VL it is the
        "patchified" 2-D tensor ``(grid_t * grid_h * grid_w,
        in_channels * temporal_patch_size * patch_size * patch_size)``
        on CPU.
    grid_thw
        ``(3,) int32`` -- ``(T, H_grid, W_grid)`` patch-grid sizes for
        this image *before* spatial / temporal merge. Drives both the
        number of vision tokens produced after merge and the per-token
        3-D MRoPE positions.
    placeholder_positions
        ``(n_post_merge_tokens,) int32`` -- positions in
        ``Sequence.tokens`` that this image's encoder output will
        replace. Pre-computed by the data adapter (which knows the
        chat-template's image-placeholder expansion); the engine
        relays them to ``ChunkMeta.extra`` during chunk packing.
    """

    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    placeholder_positions: torch.Tensor


@dataclass
class ImageInputs:
    """Device-side per-round bundle of images, post-DMA.

    Built once per round by :meth:`MultimodalInputLayer.setup_round`
    from the union of every Sequence's ``modality_inputs["image"]``
    in this round. Consumed by
    :meth:`ModalityEncoder.forward_round`.

    Attributes
    ----------
    pixel_values
        Concatenated along dim 0 across all images in this round:
        ``(sum_i n_patches_pre_merge_i, patch_dim)`` on GPU. Same
        per-image layout as :class:`ImageInputCPU.pixel_values`.
    pix_offsets
        ``(n_images + 1,) int32`` prefix-sum into ``pixel_values``
        rows. Image ``i`` occupies rows ``[pix_offsets[i]:
        pix_offsets[i+1]]``. Derivable from ``grid_thw`` but cached
        here so the encoder doesn't have to recompute it.
    grid_thw
        ``(n_images, 3) int32`` -- per-image
        ``(T, H_grid, W_grid)``.
    """

    pixel_values: torch.Tensor
    pix_offsets: torch.Tensor
    grid_thw: torch.Tensor


@dataclass
class ImageEmbeddings:
    """Output of :meth:`ModalityEncoder.forward_round` for the image
    modality. Cached on the input layer for the duration of one round.

    Attributes
    ----------
    embeds
        ``(sum_i n_tokens_post_merge_i, d_model)`` on GPU, ragged
        (concatenated per-image post-merge token sequences). The
        splice strategy slices into this per chunk.
    token_offsets
        ``(n_images + 1,) int32`` prefix-sum into ``embeds`` rows.
    grid_thw
        ``(n_images, 3) int32`` -- per-image post-encoder grid (which
        is the *pre-merge* grid divided by ``spatial_merge_size`` on
        the H/W axes). Used by chunk preparation to assign 3-D MRoPE
        positions to placeholder tokens. Kept here so the splice
        strategy can also reach it.
    """

    embeds: torch.Tensor
    token_offsets: torch.Tensor
    grid_thw: torch.Tensor


@dataclass
class ImageGradInputs:
    """Backward gradient accumulator for the image modality.

    Allocated by :class:`MultimodalInputLayer` at the start of
    ``_embed_backward`` (Phase 3) or skipped entirely (Phase 1,
    frozen encoder). Each chunk's splice-backward adds its
    placeholder-position ``dx`` rows into ``d_embeds`` at the right
    offsets. After the per-chunk backward loop completes,
    :meth:`ModalityEncoder.backward_round` consumes the accumulated
    ``d_embeds`` to produce encoder param grads.

    Shape mirrors :class:`ImageEmbeddings`. In Phase 1 (frozen
    encoders) this dataclass is unused and not allocated.
    """

    d_embeds: torch.Tensor


# ---------------------------------------------------------------------------
# Stats payloads -- used by sizing/cost queries that don't need the actual
# tensors (working-set planner, FLOP accounting).
# ---------------------------------------------------------------------------


@dataclass
class InputsSummary:
    """Stats describing a hypothetical round's modality inputs, used by

    * :meth:`ModalityEncoder.peak_workspace_bytes` -- working-set planner
      asks "how much GPU peak will your forward need if you process N
      images of average shape S?" before any real data is on hand.
    * :meth:`ModalityEncoder.compute_cost_round` -- aggregate FLOPs for
      logging / TFLOPS reporting (not in the DP problem).

    For the image modality:

    * ``n_inputs`` -- number of images.
    * ``max_patches_pre_merge`` -- worst-case ``grid_t * grid_h *
      grid_w`` for a single image (drives per-image attention size).
    * ``sum_patches_pre_merge`` -- total pre-merge patches across all
      images (drives batched matmul sizes).
    * ``patch_dim`` -- second axis of ``ImageInputs.pixel_values``.
    """

    n_inputs: int
    max_patches_pre_merge: int
    sum_patches_pre_merge: int
    patch_dim: int


# ---------------------------------------------------------------------------
# Modality-keyed unions -- forward-extensible to audio / video.
# Phase 1 has a single arm; Phase 2/3 will expand the union.
# ---------------------------------------------------------------------------


ModalityInputs = Union[ImageInputs]
ModalityEmbeddings = Union[ImageEmbeddings]
ModalityGradInputs = Union[ImageGradInputs]
ModalityInputCPU = Union[ImageInputCPU]
