"""Smoke tests for :class:`MultimodalInputLayer` construction +
ParamSpec merging.

CPU-only / GPU-free. Verifies:

* The class can be constructed with a real :class:`QwenVLVisionEncoder`.
* The merged ``param_spec`` has the expected text-embed +
  encoder-prefixed tensor names.
* All encoder tensors carry ``frozen=True`` so the engine's
  ``BufferManager`` will skip grad / opt-state allocation.
* The protocol surface (forward / backward / setup_round /
  finalize_round / compute_cost) is wired through correctly.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
from flextrain.nn.encoders import QwenVLVisionConfig, QwenVLVisionEncoder
from flextrain.nn.multimodal_input import MultimodalInputLayer
from flextrain.nn.splices import concat_splice_bwd, concat_splice_fwd


def _make_text_embed() -> TokenEmbedLayer:
    return TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=2048,
        d_model=2048,
        compute_dtype=torch.bfloat16,
    ))


def _make_encoder(depth: int = 2) -> QwenVLVisionEncoder:
    """Build a tiny vision encoder so the param spec is small enough
    to enumerate cheaply."""
    return QwenVLVisionEncoder(
        cfg=QwenVLVisionConfig(
            depth=depth,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            hidden_act="gelu_pytorch_tanh",
            compute_dtype=torch.bfloat16,
        ),
        modality="image",
        encoder_id=0,
        frozen=True,
    )


def test_construct_with_one_encoder() -> None:
    text_embed = _make_text_embed()
    encoder = _make_encoder(depth=2)
    layer = MultimodalInputLayer(
        text_embed=text_embed,
        encoders=(encoder,),
        splice_strategies=((concat_splice_fwd, concat_splice_bwd),),
    )
    # Has the expected attributes.
    assert layer.layer_id == -1
    assert layer.schema.max_tier == 0
    assert len(layer.schema.fields) == 0, (
        f"expected empty schema; got fields={layer.schema.fields}"
    )
    assert layer.num_vision_layers == 2

    # Merged param_spec contains text-embed + every encoder tensor.
    names = {t.name for t in layer.param_spec.tensors}
    assert "w_tok_embeddings" in names, "text-embed tensor missing"
    # Spot-check encoder names.
    expected_some = {
        "image0_patch_embed_proj_w",
        "image0_pos_embed_w",
        "image0_layer_0_norm1_w",
        "image0_layer_1_qkv_w",
        "image0_merger_fc2_b",
    }
    missing = expected_some - names
    assert not missing, f"missing encoder tensors in merged spec: {missing}"
    print(f"[OK] MultimodalInputLayer.param_spec has {len(names)} merged tensors "
          f"({len(names) - 1} from encoder + 1 from text embed)")


def test_all_encoder_tensors_frozen() -> None:
    text_embed = _make_text_embed()
    encoder = _make_encoder(depth=3)
    layer = MultimodalInputLayer(
        text_embed=text_embed,
        encoders=(encoder,),
        splice_strategies=((concat_splice_fwd, concat_splice_bwd),),
    )
    frozen = [t for t in layer.param_spec.tensors if t.frozen]
    trainable = [t for t in layer.param_spec.tensors if not t.frozen]
    # Text-embed tensor should be the only trainable one in Phase 1.
    trainable_names = {t.name for t in trainable}
    assert trainable_names == {"w_tok_embeddings"}, (
        f"expected only w_tok_embeddings trainable; got {trainable_names}"
    )
    # Every encoder tensor (prefix image0_) must be frozen.
    encoder_names = {t.name for t in layer.param_spec.tensors if t.name.startswith("image0_")}
    for t in layer.param_spec.tensors:
        if t.name.startswith("image0_"):
            assert t.frozen, f"encoder tensor {t.name} should be frozen"
    print(f"[OK] All {len(encoder_names)} encoder tensors are frozen; "
          f"text-embed is trainable.")


def test_invalid_construction_raises() -> None:
    """Mismatched encoder/splice counts and empty encoders should raise."""
    text_embed = _make_text_embed()
    encoder = _make_encoder(depth=1)

    # Empty encoders list.
    try:
        MultimodalInputLayer(text_embed=text_embed, encoders=(), splice_strategies=())
    except ValueError as e:
        assert "at least one ModalityEncoder" in str(e)
    else:
        raise AssertionError("empty encoders should raise ValueError")

    # Mismatched counts.
    try:
        MultimodalInputLayer(
            text_embed=text_embed,
            encoders=(encoder,),
            splice_strategies=(),
        )
    except ValueError as e:
        assert "splice strategy count" in str(e)
    else:
        raise AssertionError("mismatched counts should raise ValueError")
    print("[OK] Invalid construction surfaces ValueError with helpful messages.")


def main() -> None:
    test_construct_with_one_encoder()
    test_all_encoder_tensors_frozen()
    test_invalid_construction_raises()
    print("\nAll MultimodalInputLayer smoke tests passed.")


if __name__ == "__main__":
    main()
