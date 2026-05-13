"""End-to-end smoke test: load Qwen3.5-2B with multimodal enabled,
run one fwd_bwd on a fixed `(prompt, image)` example, and assert no
crash.

This exercises the complete Phase 1 + 1.5 plumbing in one shot:

* ``from_pretrained(enable_multimodal=True)`` -- ArchSpec dispatch +
  multimodal input layer construction + vision encoder weights loaded
  from the safetensors via ``vision_embed`` / ``vision_layer``.
* ``MultimodalProcessorBundle.build_multimodal_sequence`` -- HF
  AutoProcessor for image preprocessing + chat-template image
  placeholder expansion + per-image ``placeholder_positions`` packed
  into a flextrain ``Sequence``.
* ``ActiveModel.fwd_bwd`` -- engine round setup with
  ``MultimodalInputLayer.setup_round`` (vision encoder runs once per
  round), chunk preparation populating ``mm_*`` extras + 3-D MRoPE
  positions, per-chunk forward through ConcatSplice + LM backbone
  with MRoPE block dispatch.

This test does NOT validate output correctness vs HF -- that's the
separate forward-parity test (next milestone). The goal here is to
catch integration crashes BEFORE wading into per-token cosine debug.

Run: ``./run_with_env.sh python tests/test_qwen3_5_mm_smoke.py``
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-2B")


def _make_image():
    """Generate a deterministic synthetic image for the smoke test."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_from_pretrained_multimodal_smoke() -> None:
    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.io.image_processing import (
        MultimodalProcessorBundle,
        build_multimodal_sequence,
    )

    print("[setup] loading processor bundle...")
    bundle = MultimodalProcessorBundle.from_pretrained(MODEL_PATH)
    print(f"[setup] image_token_id={bundle.image_token_id}, "
          f"spatial_merge_size={bundle.spatial_merge_size}, "
          f"patch_size={bundle.patch_size}")

    # Build a chat-templated prompt with an image. Use the tokenizer's
    # chat template to get the right vision-placeholder expansion.
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }]
    chat_text = bundle.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    print(f"[setup] chat-templated prompt (first 200 chars): {chat_text[:200]!r}...")

    image = _make_image()
    seq = build_multimodal_sequence(chat_text, [image], bundle)
    print(f"[setup] Sequence: tokens.shape={tuple(seq.tokens.shape)}, "
          f"n_images={len(seq.modality_inputs.get('image', []))}, "
          f"image_placeholders={int((seq.tokens == bundle.image_token_id).sum())}")

    # Build the ActiveModel with multimodal enabled.
    # Use generous memory caps because this is a one-off forward pass.
    print("[setup] building ActiveModel via from_pretrained (this can take a while)...")
    am = from_pretrained(
        MODEL_PATH,
        optimizer=AdamW(AdamWHyperparams(lr=3e-5)),
        max_seq_len=2048,
        max_global_batch_tokens=4096,
        max_gpu_mem_bytes=30 * (1 << 30),
        max_host_mem_bytes=120 * (1 << 30),
        leeway_gpu_mem_bytes=3 * (1 << 30),
        enable_multimodal=True,
        freeze_modality_encoders=True,
        lora_targets="all",  # LoRA to keep memory low for this smoke test
        verbose=False,
    )
    print(f"[setup] ActiveModel built; backbone has {len(am.backbone)} layers; "
          f"embed has num_vision_layers={getattr(am.embed, 'num_vision_layers', 0)}")

    # Run one forward+backward.
    print("[run] running am.fwd_bwd([seq])...")
    stats = am.fwd_bwd([seq])
    print(f"[run] fwd_bwd complete: total_tokens={stats.total_tokens}, "
          f"total_loss={stats.total_loss:.4f}")

    assert torch.isfinite(torch.tensor(stats.total_loss)), (
        f"total_loss is non-finite: {stats.total_loss}"
    )
    print("[OK] multimodal smoke test passed.")


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available.")
        return
    if not os.path.isdir(MODEL_PATH):
        print(f"SKIP: {MODEL_PATH} not present.")
        return
    try:
        import transformers  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as e:
        print(f"SKIP: missing dep: {e}")
        return
    test_from_pretrained_multimodal_smoke()


if __name__ == "__main__":
    main()
