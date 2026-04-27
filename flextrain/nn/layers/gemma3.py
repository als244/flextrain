"""Gemma 3 transformer layer = Gemma 2 + per-head QK-norm + per-layer RoPE.

Structural changes vs Gemma 2:

* Per-head QK-norm (RMSNorm over ``head_dim``) inserted between Q/K
  projections and RoPE — same scheme as Qwen3 dense.
* Per-layer-type RoPE base. Local (sliding-window) layers use
  ``rope_local_base_freq`` (default 10_000); global (full-attention)
  layers use ``rope_theta`` (default 1_000_000). The block reads
  ``rope_base`` from its config; the backbone factory selects the
  right value per layer.
* Dual-residual norms, softcap on attention logits, alternating
  sliding-window — all carried over from Gemma 2.

The bwd is currently stubbed; will land alongside Gemma 2's bwd in
a follow-up.
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
class Gemma3BlockConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int
    rms_norm_eps: float = 1e-6
    rope_base: float = 1_000_000.0  # global layers default; local uses 10_000
    is_causal: bool = True
    attn_logit_softcap: float = 50.0
    final_logit_softcap: float = 30.0
    query_pre_attn_scalar: float | None = None
    window_size_left: int = -1     # >= 0 → sliding-window layer

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


def _build_attn(cfg: Gemma3BlockConfig):
    if cfg.window_size_left >= 0:
        return GQASlidingWindowAttentionBlock(
            GQASlidingWindowAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_base=cfg.rope_base,
                is_causal=cfg.is_causal,
                qk_norm=True,
                rms_norm_eps=cfg.rms_norm_eps,
                window_size_left=cfg.window_size_left,
                attn_logit_softcap=cfg.attn_logit_softcap,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            )
        )
    return GQAAttentionBlock(
        GQAAttentionConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            rope_base=cfg.rope_base,
            is_causal=cfg.is_causal,
            qk_norm=True,
            rms_norm_eps=cfg.rms_norm_eps,
            attn_logit_softcap=cfg.attn_logit_softcap,
            compute_dtype=cfg.compute_dtype,
            master_dtype=cfg.master_dtype,
            grad_dtype=cfg.grad_dtype,
        )
    )


class Gemma3Block:
    """Gemma 3 dense layer: Gemma 2's dual-residual norms + per-head
    QK-norm. Forward path is correct & tested; bwd stubs to NotImplementedError.
    """

    def __init__(self, layer_id: int, cfg: Gemma3BlockConfig) -> None:
        self.layer_id = layer_id
        self.cfg = cfg
        self._dims = cfg.dims()

        self.pre_attn_norm = RMSNormBlock(
            prefix="pre_attn_norm", eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.post_attn_norm = RMSNormBlock(
            prefix="post_attn_norm", eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.pre_ffn_norm = RMSNormBlock(
            prefix="pre_ffn_norm", eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.post_ffn_norm = RMSNormBlock(
            prefix="post_ffn_norm", eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        # Per-head QK-norm.
        self.q_norm = RMSNormBlock(
            prefix="q_norm", eps=cfg.rms_norm_eps,
            per_head=True, heads_dim_name="n_heads",
            weight_dim_name="head_dim",
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.k_norm = RMSNormBlock(
            prefix="k_norm", eps=cfg.rms_norm_eps,
            per_head=True, heads_dim_name="n_kv_heads",
            weight_dim_name="head_dim",
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.attn = _build_attn(cfg)
        self.attn.set_qk_norm(self.q_norm, self.k_norm)
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
                    self.q_norm.fields(),
                    self.k_norm.fields(),
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
                self.q_norm.param_spec(),
                self.k_norm.param_spec(),
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

        # Attn sub-layer with dual norm and zero residual into the attn block.
        zero_resid = ctx.scratch(x.shape, x.dtype).zero_()
        h = self.pre_attn_norm.fwd(x, weights, slot.pre_attn_norm_rstd, output=x_temp)
        a_only = self.attn.fwd(zero_resid, h, chunk, weights, slot, ctx)
        h2 = self.post_attn_norm.fwd(
            a_only.view(-1, cfg.d_model),
            weights, slot.post_attn_norm_rstd, output=x_temp,
        )
        x_after_attn = (x.view(-1, cfg.d_model) + h2).view_as(x)
        slot.x_mid.copy_(x_after_attn)

        # FFN sub-layer (same pattern).
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
        pass

    def backward(self, dx, chunk, weights, grads, slot, ctx) -> torch.Tensor:
        raise NotImplementedError(
            "Gemma3Block.bwd lands alongside Gemma2Block.bwd. The dual-"
            "residual structure (pre+post norm on each sublayer) needs a "
            "handwritten bwd; the forward path is complete and tested."
        )

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        max_tier = self.schema.max_tier
        return ComputeCost.sum(
            [
                self.pre_attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.q_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.k_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.attn.compute_cost(chunk, max_tier=max_tier),
                self.post_attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.pre_ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.ffn.compute_cost(chunk, max_tier=max_tier),
                self.post_ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
            ],
            max_tier=max_tier,
        )


def build_gemma3_backbone(
    cfg: Gemma3BlockConfig, layer_types: list[str],
    sliding_window: int | None = None,
    rope_local_base: float = 10_000.0,
    rope_global_base: float = 1_000_000.0,
):
    """Build a Gemma 3 backbone alternating sliding (local) and full
    (global) layers per ``layer_types`` from the HF config.

    ``layer_types[i]`` is one of ``"sliding_attention"`` /
    ``"full_attention"``. Sliding layers use ``rope_local_base`` and
    ``window_size_left=sliding_window``; full layers use ``rope_global_base``
    and no window.
    """
    import dataclasses
    out = []
    for i, lt in enumerate(layer_types):
        if lt == "sliding_attention":
            assert sliding_window is not None, "sliding_window required"
            layer_cfg = dataclasses.replace(
                cfg, rope_base=rope_local_base,
                window_size_left=sliding_window,
            )
        elif lt == "full_attention":
            layer_cfg = dataclasses.replace(
                cfg, rope_base=rope_global_base,
                window_size_left=-1,
            )
        else:
            raise ValueError(f"unknown layer_type {lt!r} at layer {i}")
        out.append(Gemma3Block(i, layer_cfg))
    return out
