"""Qwen2 / Qwen2.5 dense family.

Qwen2 is architecturally Llama + Q/K/V biases on attention. No
QK-norm (that's Qwen3). Defaults:

* RMSNorm eps = 1e-6.
* RoPE base = 1e6 (Qwen2 default; Qwen2 tiny variants may differ —
  read from hf config).
* GQA attention with Q/K/V biases (``cfg.qkv_bias=True``).
* SwiGLU FFN, same as Llama.

Compose the block like::

    cfg = Qwen2BlockConfig(d_model=..., n_heads=..., ...)
    block = Qwen2Block(layer_id=0, cfg=cfg)

Forward / backward are inherited structurally from
:class:`LlamaBlock` — the bias path is handled inside
:class:`GQAAttentionBlock` when ``cfg.qkv_bias=True``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flextrain.nn.blocks import GQAAttentionBlock, GQAAttentionConfig
from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig


@dataclass(frozen=True)
class Qwen2BlockConfig(LlamaBlockConfig):
    """Per-layer config for Qwen2 / Qwen2.5.

    Same fields as :class:`LlamaBlockConfig` — Qwen2's bias path is
    enabled implicitly (it's the only differentiator from LlamaBlock
    at the layer level, and it's hardwired on). Override
    :attr:`rope_base` / :attr:`rms_norm_eps` at construction time.
    """

    # Qwen2 defaults.
    rms_norm_eps: float = 1e-6
    rope_base: float = 1_000_000.0


class Qwen2Block(LlamaBlock):
    """Qwen2 dense layer.

    Composition identical to :class:`LlamaBlock` except the attention
    block is constructed with ``cfg.qkv_bias=True``, so w_q/w_k/w_v
    projections carry per-head bias vectors. Forward / backward
    bodies are inherited unchanged — the bias path lives inside
    :class:`GQAAttentionBlock`.
    """

    def __init__(self, layer_id: int, cfg: Qwen2BlockConfig) -> None:
        # Piggyback on LlamaBlock's __init__, but replace the attention
        # block it built with a qkv_bias-enabled one. Simplest: bypass
        # super().__init__ and mirror its structure but flip the flag.
        from flextrain.core.activation_schema import (
            ActivationField,
            ActivationSchema,
            concat_fields,
        )
        from flextrain.core.layer import ParamSpec
        from flextrain.nn.blocks import (
            RMSNormBlock,
            SwiGLUConfig,
            SwiGLUFFN,
        )

        self.layer_id = layer_id
        self.cfg = cfg
        self._dims = cfg.dims()

        self.attn_norm = RMSNormBlock(
            prefix="attn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.attn = GQAAttentionBlock(
            GQAAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_base=cfg.rope_base,
                rope_scaling=cfg.rope_scaling,
                is_causal=cfg.is_causal,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
                qkv_bias=True,
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
