"""MistralBlock: one Mistral-family transformer block.

Identical to :class:`~flextrain.nn.layers.llama.LlamaBlock` except it
picks :class:`GQASlidingWindowAttentionBlock` instead of
:class:`GQAAttentionBlock`. Demonstrates the composition pattern:
a model-family class lives in ``nn/layers/`` and picks specific
algorithmic blocks from ``nn/blocks/``.

Covers: Mistral-7B (window_size=4096), Mistral-NeMo variants, and any
sliding-window-only layer in a heterogeneous backbone.

For alternating full + SWA (Gemma2, GPT-OSS), instantiate a backbone
mixing :class:`LlamaBlock` and :class:`MistralBlock` -- the engine sees
both as :class:`~flextrain.core.Layer` Protocol instances and handles the
mix without any layer-type branching.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flextrain.nn.blocks import (
    GQASlidingWindowAttentionBlock,
    GQASlidingWindowAttentionConfig,
    RMSNormBlock,
    SwiGLUConfig,
    SwiGLUFFN,
)

from .llama import LlamaBlock


@dataclass(frozen=True)
class MistralBlockConfig:
    """Per-layer config for a Mistral-family layer.

    Sliding-window GQA with ``window_size_left`` tokens of prior context.
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int
    window_size_left: int = 4096
    rms_norm_eps: float = 1e-5
    rope_base: float = 10000.0  # Mistral-7B default
    rope_scaling: object | None = None  # see LlamaBlockConfig
    is_causal: bool = True

    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None
    norm_grad_dtype: torch.dtype = torch.float32
    norm_master_dtype: torch.dtype = torch.float32

    def dims(self) -> dict[str, int]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "expert_dim": self.expert_dim,
        }


class MistralBlock(LlamaBlock):
    """One Mistral-family layer. Inherits all the composition logic from
    :class:`LlamaBlock`; only the attention block differs.

    (Inheriting avoids duplicating the ~130-line composition / schema /
    forward / backward code. The attention-kernel swap happens in
    ``__init__`` via ``self.attn = ...`` before the schema is assembled.)
    """

    def __init__(self, layer_id: int, cfg: MistralBlockConfig) -> None:
        # Store Mistral's own config.
        self.layer_id = layer_id
        self.cfg = cfg  # type: ignore[assignment]
        self._dims = cfg.dims()

        # Build blocks. Same composition pattern as LlamaBlock, with the
        # swap for sliding-window attention.
        self.attn_norm = RMSNormBlock(
            prefix="attn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.attn = GQASlidingWindowAttentionBlock(
            GQASlidingWindowAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_base=cfg.rope_base,
                rope_scaling=cfg.rope_scaling,
                is_causal=cfg.is_causal,
                window_size_left=cfg.window_size_left,
                window_size_right=0,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            )
        )
        self.ffn_norm = RMSNormBlock(
            prefix="ffn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.ffn = SwiGLUFFN(
            SwiGLUConfig(
                d_model=cfg.d_model,
                expert_dim=cfg.expert_dim,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            )
        )

        # Reuse LlamaBlock's schema-assembly logic. We can't call
        # super().__init__ cleanly because it rebuilds attn as full-context
        # GQA; do the schema assembly inline instead (same 15 lines).
        from flextrain.core.activation_schema import (
            ActivationField,
            ActivationSchema,
            concat_fields,
        )
        from flextrain.core.layer import ParamSpec

        x_inp_field = ActivationField(
            "x_inp",
            lambda n, d: (n, cfg.d_model),
            cfg.compute_dtype,
            tier=0,
        )
        self.schema = ActivationSchema(
            fields=concat_fields(
                [
                    self.attn_norm.fields(),
                    (x_inp_field,),
                    self.attn.fields(),
                    self.ffn_norm.fields(),
                    self.ffn.fields(),
                ]
            ),
            max_tier=3,
        )
        self.param_spec = ParamSpec.merge(
            [
                self.attn_norm.param_spec(),
                self.attn.param_spec(),
                self.ffn_norm.param_spec(),
                self.ffn.param_spec(),
            ]
        )
