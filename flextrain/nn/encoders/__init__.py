"""Modality-encoder implementations.

Each module here exposes a concrete :class:`~flextrain.core.layer.ModalityEncoder`
that turns raw modality data into ``d_model`` embeddings. Encoders are
invoked once per round from
:meth:`~flextrain.nn.multimodal_input.MultimodalInputLayer.setup_round`
and their output is cached for the duration of the round.

Phase 1: only the Qwen-VL family vision encoder is implemented.
"""

from .qwen_vl_vit import (
    QwenVLVisionConfig,
    QwenVLVisionEncoder,
    qwen_vl_vit_param_spec,
)

__all__ = [
    "QwenVLVisionConfig",
    "QwenVLVisionEncoder",
    "qwen_vl_vit_param_spec",
]
