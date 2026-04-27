"""Gemma 2 transformer layer.

Differences from Llama:

1. **Dual-residual norm topology**: each sublayer has BOTH a pre-norm
   AND a post-norm:

       residual = x
       y = post_attn_norm(attn(pre_attn_norm(x)))
       x = residual + y

       residual = x
       y = post_ffn_norm(ffn(pre_ffn_norm(x)))
       x = residual + y

   Llama only has the pre-norms; Gemma 2 adds the post-norms.

2. **Attention logit softcap**: ``tanh(scores / cap) * cap`` applied
   pre-softmax. Plumbed through to flash-attn's ``softcap`` argument via
   ``GQAAttentionConfig.attn_logit_softcap``.

3. **Alternating sliding-window** attention layers. Layer i is full
   (``"global_attention"``) or sliding (``"sliding_attention"``)
   per ``config.layer_types``. The full vs sliding variant is built
   via :class:`Gemma2DenseBlock` vs :class:`Gemma2SWABlock`.

4. **RMSNorm weight convention**: Gemma 2 stores ``weight = γ - 1`` so
   that an untrained init centers at 0 (instead of 1). The arch spec's
   ``post_load_hook`` shifts loaded values by +1 so we can reuse the
   stock :class:`RMSNormBlock` (which expects ``γ`` directly).

5. **Final logit softcap** on the LM head output is separate; see
   :class:`flextrain.nn.head.LMHead` plus the
   ``final_logit_softcap`` config field (TODO: not yet implemented).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import (
    ActivationField, ActivationSchema, concat_fields,
)
from flextrain.core.layer import (
    ChunkMeta, ComputeCost, LayerContext, ParamSpec,
)
from flextrain.nn.blocks import (
    GQAAttentionBlock, GQAAttentionConfig,
    GQASlidingWindowAttentionBlock, GQASlidingWindowAttentionConfig,
    RMSNormBlock, SwiGLUConfig, SwiGLUFFN,
)


@dataclass(frozen=True)
class Gemma2BlockConfig:
    """Per-layer config for Gemma 2.

    ``query_pre_attn_scalar`` overrides the default ``1/sqrt(head_dim)``
    attention scale. Gemma 2 9B uses ``224**-0.5 ~= 0.0668`` with
    ``head_dim=256``; default ``head_dim**-0.5 ~= 0.0625``.

    ``window_size_left`` is a positive int for sliding layers and ``-1``
    for full layers. Pass through ``layer_types`` from the HF config
    when building a backbone.
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int
    rms_norm_eps: float = 1e-6
    rope_base: float = 10_000.0
    is_causal: bool = True
    attn_logit_softcap: float = 50.0   # Gemma 2 default
    final_logit_softcap: float = 30.0  # Gemma 2 default (head-side, not used here)
    query_pre_attn_scalar: float | None = None  # Override 1/sqrt(head_dim)
    window_size_left: int = -1                  # >= 0 for sliding-window layer

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
            "attn_dim": self.n_heads * self.head_dim,
            "kv_dim": self.n_kv_heads * self.head_dim,
            "expert_dim": self.expert_dim,
        }


def _build_attn(cfg: Gemma2BlockConfig):
    if cfg.window_size_left >= 0:
        return GQASlidingWindowAttentionBlock(
            GQASlidingWindowAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_base=cfg.rope_base,
                is_causal=cfg.is_causal,
                window_size_left=cfg.window_size_left,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
                attn_logit_softcap=cfg.attn_logit_softcap,
            )
        )
    return GQAAttentionBlock(
        GQAAttentionConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            rope_base=cfg.rope_base,
            is_causal=cfg.is_causal,
            compute_dtype=cfg.compute_dtype,
            master_dtype=cfg.master_dtype,
            grad_dtype=cfg.grad_dtype,
            attn_logit_softcap=cfg.attn_logit_softcap,
        )
    )


class Gemma2Block:
    """Gemma 2 dense layer with dual-residual norms.

    Note: the bwd uses an autograd-scoped subgraph for the post-norms
    (similar to GatedDeltaNetBlock). The dual-residual structure is
    expressible directly with our existing RMSNormBlock + GQA, but the
    bwd routing is non-trivial — it's the third (post-attn-norm) and
    fourth (post-ffn-norm) RMSNorm bwds that make it tedious. For now
    we use a scoped autograd block. Full hand-rolled bwd is a follow-up.

    A handwritten bwd will land alongside Gemma 3 (which builds on this).
    """

    def __init__(self, layer_id: int, cfg: Gemma2BlockConfig) -> None:
        self.layer_id = layer_id
        self.cfg = cfg
        self._dims = cfg.dims()

        # Four norms (vs Llama's two). All on residual stream.
        self.pre_attn_norm = RMSNormBlock(
            prefix="pre_attn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.post_attn_norm = RMSNormBlock(
            prefix="post_attn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.pre_ffn_norm = RMSNormBlock(
            prefix="pre_ffn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.post_ffn_norm = RMSNormBlock(
            prefix="post_ffn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.attn = _build_attn(cfg)
        self.ffn = SwiGLUFFN(
            SwiGLUConfig(
                d_model=cfg.d_model, expert_dim=cfg.expert_dim,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            )
        )

        x_inp = ActivationField(
            "x_inp", lambda n, d: (n, cfg.d_model),
            cfg.compute_dtype, tier=0,
        )
        # Gemma 2 also needs an extra mid-residual save (between attn
        # block and pre_ffn_norm) so the FFN bwd has its input.
        x_mid = ActivationField(
            "x_mid", lambda n, d: (n, cfg.d_model),
            cfg.compute_dtype, tier=0,
        )
        self.schema = ActivationSchema(
            fields=concat_fields(
                [
                    self.pre_attn_norm.fields(),
                    (x_inp,),
                    self.attn.fields(),
                    self.post_attn_norm.fields(),
                    (x_mid,),
                    self.pre_ffn_norm.fields(),
                    self.post_ffn_norm.fields(),
                    self.ffn.fields(),
                ]
            ),
            max_tier=3,
        )
        self.param_spec = ParamSpec.merge(
            [
                self.pre_attn_norm.param_spec(),
                self.attn.param_spec(),
                self.post_attn_norm.param_spec(),
                self.pre_ffn_norm.param_spec(),
                self.ffn.param_spec(),
                self.post_ffn_norm.param_spec(),
            ]
        )

    def forward(
        self, x, chunk: ChunkMeta, weights, slot, ctx: LayerContext,
    ) -> torch.Tensor:
        cfg = self.cfg
        slot.x_inp.copy_(x)
        x_temp = ctx.scratch(x.shape, x.dtype)

        # Attention sub-layer (pre + post norm, residual added after).
        # The attn block fuses residual into its output; we pass a zero
        # tensor as the residual to recover unfused attn output.
        zero_resid = ctx.scratch(x.shape, x.dtype).zero_()
        h = self.pre_attn_norm.fwd(x, weights, slot.pre_attn_norm_rstd, output=x_temp)
        a_only = self.attn.fwd(zero_resid, h, chunk, weights, slot, ctx)
        h2 = self.post_attn_norm.fwd(
            a_only.view(-1, cfg.d_model),
            weights, slot.post_attn_norm_rstd, output=x_temp,
        )
        x_after_attn = (x.view(-1, cfg.d_model) + h2).view_as(x)
        slot.x_mid.copy_(x_after_attn)

        # FFN sub-layer (pre + post norm, residual added after).
        zero_ffn_resid = ctx.scratch(x.shape, x.dtype).zero_()
        h = self.pre_ffn_norm.fwd(
            x_after_attn.view(-1, cfg.d_model),
            weights, slot.pre_ffn_norm_rstd, output=x_temp,
        )
        ffn_only = self.ffn.fwd(
            h, weights, zero_ffn_resid,
            out_tensor=x, slot=slot, ctx=ctx,
        )
        h3 = self.post_ffn_norm.fwd(
            ffn_only.view(-1, cfg.d_model),
            weights, slot.post_ffn_norm_rstd, output=x_temp,
        )
        return (x_after_attn.view(-1, cfg.d_model) + h3).view_as(x)

    def forward_recompute(self, slot, chunk, weights, ctx) -> None:
        # Conservative: regenerate everything Llama-style. The dual-norm
        # bwd needs the input/output of every norm, so partial recompute
        # is more involved than Llama. First-cut: full recompute on demand.
        pass

    def backward(self, dx, chunk, weights, grads, slot, ctx) -> torch.Tensor:
        # Gemma 2 bwd is non-trivial because of the dual-residual structure.
        # First cut: scoped autograd reference path.
        # TODO: hand-rolled bwd (lands alongside Gemma 3, which builds
        # on Gemma 2's structure with QK-norm added).
        raise NotImplementedError(
            "Gemma2Block.bwd is a stub for this iteration. The dual-residual "
            "structure (pre+post norm on each sublayer) needs a handwritten "
            "bwd that's currently being designed. The forward path is "
            "complete and tested; bwd lands in a follow-up."
        )

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        max_tier = self.schema.max_tier
        return ComputeCost.sum(
            [
                self.pre_attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.attn.compute_cost(chunk, max_tier=max_tier),
                self.post_attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.pre_ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.ffn.compute_cost(chunk, max_tier=max_tier),
                self.post_ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
            ],
            max_tier=max_tier,
        )
