"""HF AutoImageProcessor wrapper -> :class:`ImageInputCPU`.

Phase 1 data-adapter helper. Wraps HF's ``AutoImageProcessor`` (the
source of truth for per-model image preprocessing -- resize / normalize
/ patch-flatten) and converts the output into flextrain's CPU-side
:class:`~flextrain.core.modality.ImageInputCPU` bundles. Also exposes a
``build_multimodal_sequence`` helper that handles the chat-template
expansion of image placeholders into N copies of ``image_token_id``
per image's post-merge token count, and computes the per-image
``placeholder_positions`` that :class:`ConcatSplice` consumes.

Phase 1 supports only the Qwen-VL family (Qwen3.5 / Qwen3.6 /
Qwen3-VL). Gemma3/4 substitution-style preprocessing is Phase 2.

Dependencies
------------
* ``transformers`` (HF). Required at runtime for AutoProcessor /
  AutoTokenizer / AutoImageProcessor. flextrain already lists it as a
  soft dependency for parity tests, so this is not a new import for
  the flextrain envs we ship with.
* ``PIL`` (Pillow). Required to open image files. Bundled with HF
  transformers' image-processing extras.

The wrapper does NOT load the entire HF model. Just the processors
(tens of MB at most).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from flextrain.core.modality import ImageInputCPU


@dataclass
class MultimodalProcessorBundle:
    """Cached HF processor + tokenizer for one model path.

    Construction is moderately expensive (loads JSON config files +
    instantiates HF processor objects); reuse this across many
    sequences if possible.
    """

    model_path: str
    processor: Any        # transformers.ProcessorMixin -- the combined processor that
                          # handles text+image and does the ``<|image_pad|>`` -> N copies
                          # expansion internally.
    image_processor: Any  # transformers.image_processing_utils.BaseImageProcessor
    tokenizer: Any        # transformers.PreTrainedTokenizerBase
    # Cached config bits we need on the hot path. Pulled from
    # ``processor.image_processor`` and ``processor.tokenizer`` so we
    # don't re-read JSON every call.
    image_token_id: int
    vision_start_token_id: int | None
    vision_end_token_id: int | None
    spatial_merge_size: int
    patch_size: int
    temporal_patch_size: int

    @classmethod
    def from_pretrained(cls, model_path: str) -> "MultimodalProcessorBundle":
        """Load HF AutoProcessor + AutoTokenizer for the given path."""
        from transformers import AutoProcessor, AutoTokenizer

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        image_processor = getattr(processor, "image_processor", processor)

        # Sniff HF config for the constants we need.
        import json
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path) as f:
            hf_config = json.load(f)
        image_token_id = int(
            hf_config.get("image_token_id")
            or hf_config.get("image_token_index")
            or 0
        )
        vision_start = hf_config.get("vision_start_token_id")
        vision_end = hf_config.get("vision_end_token_id")
        vc = hf_config.get("vision_config", {}) or {}
        spatial_merge_size = int(vc.get("spatial_merge_size", 2))
        patch_size = int(vc.get("patch_size", 16))
        temporal_patch_size = int(vc.get("temporal_patch_size", 2))

        return cls(
            model_path=model_path,
            processor=processor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_token_id=image_token_id,
            vision_start_token_id=(
                int(vision_start) if vision_start is not None else None
            ),
            vision_end_token_id=(
                int(vision_end) if vision_end is not None else None
            ),
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )


def preprocess_images(
    images: Sequence,
    bundle: MultimodalProcessorBundle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run HF's image processor over a sequence of PIL images / paths.

    Returns ``(pixel_values, image_grid_thw)``:

    * ``pixel_values`` -- ``(sum_i n_patches_pre_merge_i, patch_dim)``
      on CPU, dtype matching the processor's default
      (typically fp32; the encoder casts to compute_dtype).
    * ``image_grid_thw`` -- ``(n_images, 3) int64``: per-image
      ``(T, H_grid, W_grid)`` patch-grid sizes (pre-merge).

    Both are exactly what HF Qwen-VL processor produces for the same
    image list (we delegate -- no reimplementation).
    """
    from PIL import Image as _Image

    # Normalize each item to a PIL.Image.
    pil_images = []
    for item in images:
        if isinstance(item, str):
            pil_images.append(_Image.open(item).convert("RGB"))
        else:
            pil_images.append(item)
    out = bundle.image_processor(images=pil_images, return_tensors="pt")
    # HF returns dict-like: ``pixel_values`` (sum_patches, patch_dim),
    # ``image_grid_thw`` (n_images, 3).
    pix = out["pixel_values"]
    grid_thw = out["image_grid_thw"]
    return pix.cpu(), grid_thw.cpu().to(torch.int64)


def build_multimodal_sequence(
    text: str,
    images: Sequence,
    bundle: MultimodalProcessorBundle,
    *,
    targets: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
):
    """Build a :class:`~flextrain.io.sequence.Sequence` with images.

    ``text`` is the chat-template-rendered prompt with a single
    ``<|image_pad|>`` per image (and the surrounding
    ``<|vision_start|>`` / ``<|vision_end|>`` markers). Most callers
    produce ``text`` via
    ``bundle.tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=False)``.

    The function delegates to ``bundle.processor`` (HF
    ``Qwen3VLProcessor`` for Qwen-VL family) which:

    * runs the image processor to get ``pixel_values`` +
      ``image_grid_thw``,
    * expands each ``<|image_pad|>`` placeholder in ``text`` to
      ``N = (T * H * W) / spatial_merge_size**2`` copies, then
    * tokenizes the expanded text to ``input_ids``.

    We then walk the resulting ``input_ids``, locate each image's
    contiguous run of placeholder tokens, and pack everything into
    a flextrain ``Sequence`` with
    ``modality_inputs={"image": [ImageInputCPU, ...]}``.

    Parameters
    ----------
    text
        Chat-templated text containing a single ``<|image_pad|>`` per
        image (per HF chat-template convention).
    images
        List of PIL.Image objects or string paths.
    bundle
        :class:`MultimodalProcessorBundle` previously loaded for this
        model.
    targets
        Optional next-token targets (defaults to ``tokens`` rolled by
        -1 inside :class:`Sequence.__init__`).
    loss_mask
        Optional ``(N,)`` bool mask to override per-position loss
        weighting (e.g. mask out prompt tokens).
    """
    from PIL import Image as _Image
    from flextrain.io.sequence import Sequence

    # Normalize input images to PIL objects (HF processor accepts both
    # but we want to be explicit).
    pil_images = []
    for item in images:
        if isinstance(item, str):
            pil_images.append(_Image.open(item).convert("RGB"))
        else:
            pil_images.append(item)

    # Run the combined HF processor on text + images. This produces
    # the post-expansion ``input_ids`` along with ``pixel_values`` +
    # ``image_grid_thw``. The processor handles the
    # ``<|image_pad|>`` -> N copies expansion internally (see
    # ``Qwen3VLProcessor.__call__`` for the reference).
    enc = bundle.processor(
        text=[text],
        images=pil_images,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"][0].to(torch.int64)
    pixel_values = enc["pixel_values"].cpu()
    grid_thw = enc["image_grid_thw"].cpu().to(torch.int64)
    n_images = grid_thw.shape[0]
    if n_images != len(images):
        raise ValueError(
            f"image processor returned grid_thw with {n_images} rows but "
            f"got {len(images)} input images"
        )

    # Compute per-image post-merge token counts and pre-merge offsets.
    merge_unit = bundle.spatial_merge_size * bundle.spatial_merge_size
    post_merge_per_image: list[int] = []
    pre_offsets = [0]
    for i in range(n_images):
        t = int(grid_thw[i, 0].item())
        h = int(grid_thw[i, 1].item())
        w = int(grid_thw[i, 2].item())
        n_pre = t * h * w
        n_post = n_pre // merge_unit
        post_merge_per_image.append(n_post)
        pre_offsets.append(pre_offsets[-1] + n_pre)

    # Locate placeholder positions in the (post-expansion) input_ids.
    image_id = bundle.image_token_id
    placeholder_mask = input_ids == image_id
    all_placeholder_positions = placeholder_mask.nonzero(as_tuple=False).squeeze(-1)
    total_placeholders = int(all_placeholder_positions.numel())
    expected_total = sum(post_merge_per_image)
    if total_placeholders != expected_total:
        raise ValueError(
            f"image_token_id={image_id} appears {total_placeholders} times in "
            f"the processor's expanded input_ids but expected {expected_total} "
            f"based on grid_thw ({n_images} images, sum post-merge tokens = "
            f"{expected_total}). The processor's expansion mismatched the "
            "image grid -- check that the tokenizer / image_processor / "
            "spatial_merge_size are consistent."
        )

    # Slice per image.
    image_inputs: list[ImageInputCPU] = []
    cursor = 0
    for i in range(n_images):
        n_post = post_merge_per_image[i]
        pos_slice = all_placeholder_positions[cursor : cursor + n_post]
        cursor += n_post
        lo, hi = pre_offsets[i], pre_offsets[i + 1]
        pv = pixel_values[lo:hi].contiguous()
        image_inputs.append(
            ImageInputCPU(
                pixel_values=pv,
                grid_thw=grid_thw[i].contiguous(),
                placeholder_positions=pos_slice.to(torch.int32),
            )
        )

    return Sequence(
        tokens=input_ids,
        targets=targets,
        loss_mask=loss_mask,
        modality_inputs={"image": image_inputs},
    )
