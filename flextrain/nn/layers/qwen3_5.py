"""Qwen3.5 family: hybrid linear-attention + full-attention with DENSE MLPs.

Differences vs Qwen3-Next:

* **Dense MLPs everywhere** (no MoE, no shared experts). HF stores
  ``mlp.gate_proj`` / ``mlp.up_proj`` / ``mlp.down_proj`` per layer.
* Same gated GQA on full-attn layers (doubled ``q_proj``, sigmoid output
  gate, partial-rotary 0.25, per-head QK-norm).
* Same gated DeltaNet on linear-attn layers, but HF stores its
  projections as 4 split tensors (``in_proj_qkv``, ``in_proj_z``,
  ``in_proj_b``, ``in_proj_a``) instead of bundled ``qkvz`` / ``ba``.
  The arch loader concatenates them into our bundled layout.
* Tied word embeddings (no separate ``lm_head.weight``).
* Multimodal wrapper: text weights live under
  ``model.language_model.layers.{i}.*``.

This module exports:

* :class:`Qwen3_5LayerConfig` — per-layer config
* :class:`Qwen3_5FullLayer` — full-attention dense layer
* :class:`Qwen3_5LinearLayer` — linear-attention dense layer
* :func:`build_qwen3_5_backbone` — driven by ``layer_types`` list
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
    GQAAttentionGatedBlock, GQAAttentionGatedConfig, RMSNormBlock,
)
from flextrain.nn.blocks.ffn_dense import SwiGLUConfig, SwiGLUFFN
from flextrain.nn.blocks.linear_attn import (
    GatedDeltaNetBlock, GatedDeltaNetConfig,
)


@dataclass(frozen=True)
class Qwen3_5LayerConfig:
    """Per-layer config for Qwen3.5 (linear- and full-attn variants share fields)."""

    # Common
    d_model: int
    expert_dim: int                 # dense MLP intermediate (HF intermediate_size)
    # Full-attn-only
    n_heads: int
    n_kv_heads: int
    head_dim: int
    # Linear-attn-only
    linear_num_v_heads: int = 16
    linear_num_k_heads: int = 16
    linear_head_k_dim: int = 128
    linear_head_v_dim: int = 128
    linear_conv_kernel: int = 4

    rms_norm_eps: float = 1e-6
    rope_base: float = 10_000_000.0
    is_causal: bool = True
    partial_rotary_factor: float = 0.25

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
            "num_v_heads": self.linear_num_v_heads,
            "num_k_heads": self.linear_num_k_heads,
            "head_k_dim": self.linear_head_k_dim,
            "head_v_dim": self.linear_head_v_dim,
            "key_dim": self.linear_num_k_heads * self.linear_head_k_dim,
            "value_dim": self.linear_num_v_heads * self.linear_head_v_dim,
            "conv_dim": (
                2 * self.linear_num_k_heads * self.linear_head_k_dim
                + self.linear_num_v_heads * self.linear_head_v_dim
            ),
            "proj_qkvz_dim": (
                2 * self.linear_num_k_heads * self.linear_head_k_dim
                + 2 * self.linear_num_v_heads * self.linear_head_v_dim
            ),
            "proj_ba_dim": 2 * self.linear_num_v_heads,
            "conv_kernel_size": self.linear_conv_kernel,
        }


# ---------------------------------------------------------------------------
# Full-attention layer (gated GQA + dense SwiGLU MLP)
# ---------------------------------------------------------------------------


class Qwen3_5FullLayer:
    """Qwen3.5 full-attention layer: GQAAttentionGatedBlock + dense SwiGLU."""

    def __init__(self, layer_id: int, cfg: Qwen3_5LayerConfig) -> None:
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
        self.attn = GQAAttentionGatedBlock(
            GQAAttentionGatedConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_base=cfg.rope_base,
                is_causal=cfg.is_causal,
                qk_norm=True,
                rms_norm_eps=cfg.rms_norm_eps,
                qk_norm_master_dtype=cfg.norm_master_dtype,
                qk_norm_grad_dtype=cfg.norm_grad_dtype,
                partial_rotary_factor=cfg.partial_rotary_factor,
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

    # ------------------------------------------------------------------
    # Layer Protocol
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        slot.x_inp.copy_(x)
        x_temp = ctx.scratch(x.shape, x.dtype)
        attn_norm_output = self.attn_norm.fwd(
            x, weights, slot.attn_norm_rstd, output=x_temp,
        )
        attn_output_with_residual = self.attn.fwd(
            x, attn_norm_output, chunk, weights, slot, ctx,
        )
        ffn_norm_output = self.ffn_norm.fwd(
            attn_output_with_residual.view(-1, self.cfg.d_model),
            weights, slot.ffn_norm_rstd, output=x_temp,
        )
        layer_output = self.ffn.fwd(
            ffn_norm_output, weights, attn_output_with_residual,
            out_tensor=x, slot=slot, ctx=ctx,
        )
        return layer_output

    def forward_recompute(
        self, slot, chunk: ChunkMeta, weights, ctx: LayerContext,
    ) -> None:
        cfg = self.cfg
        x_inp = slot.x_inp
        if not slot.has("xq"):
            attn_norm_output = self.attn_norm.fwd_from_rstd(
                x_inp, weights, slot.attn_norm_rstd,
            )
            self.attn.fwd_recompute_qo(
                attn_norm_output, chunk, weights, slot, x_inp,
            )
            slot.aux["recompute_attn_norm_output"] = attn_norm_output
        if not slot.has("attn_result"):
            self.attn.fwd_recompute_attn(chunk, slot, ctx)
        if not slot.has("xo"):
            self.attn.fwd_recompute_o(x_inp, weights, slot)
        recompute_x1 = not slot.has("x1")
        recompute_x3 = not slot.has("x3")
        if recompute_x1 or recompute_x3:
            ffn_norm_output = self.ffn_norm.fwd_from_rstd(
                slot.xo.view(-1, cfg.d_model), weights, slot.ffn_norm_rstd,
            )
            self.ffn.fwd_recompute_x1x3(
                ffn_norm_output, weights, slot,
                recompute_x1=recompute_x1, recompute_x3=recompute_x3,
            )
            slot.aux["recompute_ffn_norm_output"] = ffn_norm_output

    def backward(
        self, dx, chunk: ChunkMeta, weights, grads, slot, ctx: LayerContext,
    ) -> torch.Tensor:
        upstream_dx, intermediates = self.backward_dgrad(
            dx, chunk, weights, grads, slot, ctx,
        )
        self.backward_wgrad(intermediates, weights, grads, slot, ctx)
        return upstream_dx

    def backward_dgrad(
        self, dx, chunk: ChunkMeta, weights, grads, slot, ctx: LayerContext,
        *, skip_target_names: frozenset[str] = frozenset(),
    ):
        from flextrain.core.layer import BackwardIntermediates
        cfg = self.cfg
        skip_g_inline: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_o", "w_2")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_inline else None
        )

        dx_ffn_norm_up = self.ffn.bwd(
            dx, weights, grads, slot,
            skip_grads=skip_g_inline, capture_xy=capture_xy,
        )
        ffn_norm_fwd_output_hint = slot.aux.pop(
            "recompute_ffn_norm_output", None
        )
        dx, ffn_norm_fwd_output = self.ffn_norm.bwd(
            dx_ffn_norm_up,
            slot.xo.view(-1, cfg.d_model),
            weights, grads, slot.ffn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=ffn_norm_fwd_output_hint is None,
            recomputed_output_tensor=None,
        )
        if ffn_norm_fwd_output_hint is not None:
            ffn_norm_fwd_output = ffn_norm_fwd_output_hint

        attn_norm_fwd_output_hint = slot.aux.pop(
            "recompute_attn_norm_output", None
        )
        if attn_norm_fwd_output_hint is None:
            attn_norm_fwd_output = self.attn_norm.fwd_from_rstd(
                slot.x_inp, weights, slot.attn_norm_rstd,
            )
        else:
            attn_norm_fwd_output = attn_norm_fwd_output_hint
        dx_attn_norm_up = self.attn.bwd(
            dx, chunk, weights, grads, slot, ctx,
            attn_norm_output=attn_norm_fwd_output,
            skip_grads=skip_g_inline, capture_xy=capture_xy,
        )
        dx, _ = self.attn_norm.bwd(
            dx_attn_norm_up,
            slot.x_inp,
            weights, grads, slot.attn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=False,
            recomputed_output_tensor=None,
        )
        intermediates = BackwardIntermediates(
            proj_inputs_and_grads={},
            aux={
                "ffn_norm_fwd_output": ffn_norm_fwd_output,
                "attn_norm_fwd_output": attn_norm_fwd_output,
            },
        )
        if capture_xy is not None:
            for g_name, xy in capture_xy.items():
                w_name = "w_" + g_name[2:]
                intermediates.proj_inputs_and_grads[w_name] = xy
        return dx, intermediates

    def backward_wgrad(
        self, intermediates, weights, grads, slot, ctx,
        *, skip_target_names: frozenset[str] = frozenset(),
    ) -> None:
        del ctx, weights
        ffn_norm_fwd_output = intermediates.aux["ffn_norm_fwd_output"]
        attn_norm_fwd_output = intermediates.aux["attn_norm_fwd_output"]
        skip_g_names = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_q", "w_k", "w_v", "w_1", "w_3")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_names else None
        )
        self.ffn.bwd_accumulate_w1_w3_grads(
            ffn_norm_fwd_output, grads, slot,
            skip_grads=skip_g_names, capture_xy=capture_xy,
        )
        self.attn.bwd_accumulate_qkv_grads(
            attn_norm_fwd_output, grads, slot,
            skip_grads=skip_g_names, capture_xy=capture_xy,
        )
        if capture_xy is not None:
            for g_name, xy in capture_xy.items():
                w_name = "w_" + g_name[2:]
                intermediates.proj_inputs_and_grads[w_name] = xy

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        max_tier = self.schema.max_tier
        return ComputeCost.sum(
            [
                self.attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.attn.compute_cost(chunk, max_tier=max_tier),
                self.ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.ffn.compute_cost(chunk, max_tier=max_tier),
            ],
            max_tier=max_tier,
        )


# ---------------------------------------------------------------------------
# Linear-attention layer (DeltaNet + dense SwiGLU MLP)
# ---------------------------------------------------------------------------


class Qwen3_5LinearLayer:
    """Qwen3.5 linear-attention layer: GatedDeltaNetBlock + dense SwiGLU.

    Same DeltaNet block as Qwen3-Next; the HF storage difference (split
    in_proj_qkv/z/b/a vs bundled qkvz/ba) is handled by the arch loader.
    """

    def __init__(self, layer_id: int, cfg: Qwen3_5LayerConfig) -> None:
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
        self.lin_attn = GatedDeltaNetBlock(
            GatedDeltaNetConfig(
                d_model=cfg.d_model,
                num_v_heads=cfg.linear_num_v_heads,
                num_k_heads=cfg.linear_num_k_heads,
                head_k_dim=cfg.linear_head_k_dim,
                head_v_dim=cfg.linear_head_v_dim,
                conv_kernel_size=cfg.linear_conv_kernel,
                rms_norm_eps=cfg.rms_norm_eps,
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
                    self.lin_attn.fields(),
                    self.ffn_norm.fields(),
                    self.ffn.fields(),
                ]
            ),
            max_tier=3,
        )
        self.param_spec = ParamSpec.merge(
            [
                self.attn_norm.param_spec(),
                self.lin_attn.param_spec(),
                self.ffn_norm.param_spec(),
                self.ffn.param_spec(),
            ]
        )

    def forward(
        self, x, chunk: ChunkMeta, weights, slot, ctx: LayerContext,
    ) -> torch.Tensor:
        slot.x_inp.copy_(x)
        x_temp = ctx.scratch(x.shape, x.dtype)
        attn_norm_output = self.attn_norm.fwd(
            x, weights, slot.attn_norm_rstd, output=x_temp,
        )
        lin_out = self.lin_attn.fwd(attn_norm_output, weights, slot, ctx)
        attn_output_with_residual = x + lin_out
        ffn_norm_output = self.ffn_norm.fwd(
            attn_output_with_residual.view(-1, self.cfg.d_model),
            weights, slot.ffn_norm_rstd, output=x_temp,
        )
        layer_output = self.ffn.fwd(
            ffn_norm_output, weights, attn_output_with_residual,
            out_tensor=x, slot=slot, ctx=ctx,
        )
        return layer_output

    def forward_recompute(
        self, slot, chunk: ChunkMeta, weights, ctx: LayerContext,
    ) -> None:
        cfg = self.cfg
        if not slot.has("lin_post_conv_pre_silu"):
            attn_norm_output = self.attn_norm.fwd_from_rstd(
                slot.x_inp, weights, slot.attn_norm_rstd,
            )
            self.lin_attn.fwd_recompute_post_conv(
                attn_norm_output, weights, slot,
            )
        if not slot.has("lin_q"):
            self.lin_attn.fwd_recompute_qkv_heads(slot)
        if not slot.has("lin_core_out") or not slot.has("lin_A_int"):
            self.lin_attn.fwd_recompute_fla(weights, slot)
        recompute_x1 = not slot.has("x1")
        recompute_x3 = not slot.has("x3")
        if recompute_x1 or recompute_x3:
            ffn_norm_output = self.ffn_norm.fwd_from_rstd(
                slot.x_inp, weights, slot.ffn_norm_rstd,
            )
            self.ffn.fwd_recompute_x1x3(
                ffn_norm_output, weights, slot,
                recompute_x1=recompute_x1, recompute_x3=recompute_x3,
            )
            slot.aux["recompute_ffn_norm_output"] = ffn_norm_output

    def backward(
        self, dx, chunk: ChunkMeta, weights, grads, slot, ctx: LayerContext,
    ) -> torch.Tensor:
        upstream_dx, intermediates = self.backward_dgrad(
            dx, chunk, weights, grads, slot, ctx,
        )
        self.backward_wgrad(intermediates, weights, grads, slot, ctx)
        return upstream_dx

    def backward_dgrad(
        self, dx, chunk: ChunkMeta, weights, grads, slot, ctx: LayerContext,
        *, skip_target_names: frozenset[str] = frozenset(),
    ):
        from flextrain.core.layer import BackwardIntermediates
        cfg = self.cfg
        # Inline-Wgrad gates: w_2 (FFN inline), w_lin_out / w_lin_qkvz /
        # w_lin_ba (linear-attn inline). Note: w_o is N/A (no attention).
        skip_g_inline: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_2", "w_lin_out", "w_lin_qkvz", "w_lin_ba")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_inline else None
        )

        dx_ffn_norm_up = self.ffn.bwd(
            dx, weights, grads, slot,
            skip_grads=skip_g_inline, capture_xy=capture_xy,
        )
        ffn_norm_fwd_output_hint = slot.aux.pop(
            "recompute_ffn_norm_output", None
        )
        dx, ffn_norm_fwd_output = self.ffn_norm.bwd(
            dx_ffn_norm_up,
            slot.x_inp.view(-1, cfg.d_model),
            weights, grads, slot.ffn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=ffn_norm_fwd_output_hint is None,
            recomputed_output_tensor=None,
        )
        if ffn_norm_fwd_output_hint is not None:
            ffn_norm_fwd_output = ffn_norm_fwd_output_hint

        # Linear-attn bwd. Skip-able for w_lin_out / w_lin_qkvz / w_lin_ba.
        dx_lin = self.lin_attn.bwd(
            dx, weights, grads, slot, ctx,
            skip_grads=skip_g_inline, capture_xy=capture_xy,
        )
        dx, _ = self.attn_norm.bwd(
            dx_lin,
            slot.x_inp,
            weights, grads, slot.attn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=False,
            recomputed_output_tensor=None,
        )

        intermediates = BackwardIntermediates(
            proj_inputs_and_grads={},
            aux={"ffn_norm_fwd_output": ffn_norm_fwd_output},
        )
        if capture_xy is not None:
            for g_name, xy in capture_xy.items():
                w_name = "w_" + g_name[2:]
                intermediates.proj_inputs_and_grads[w_name] = xy
        return dx, intermediates

    def backward_wgrad(
        self, intermediates, weights, grads, slot, ctx,
        *, skip_target_names: frozenset[str] = frozenset(),
    ) -> None:
        del ctx, weights
        ffn_norm_fwd_output = intermediates.aux["ffn_norm_fwd_output"]
        skip_g_names = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_1", "w_3")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_names else None
        )
        self.ffn.bwd_accumulate_w1_w3_grads(
            ffn_norm_fwd_output, grads, slot,
            skip_grads=skip_g_names, capture_xy=capture_xy,
        )
        if capture_xy is not None:
            for g_name, xy in capture_xy.items():
                w_name = "w_" + g_name[2:]
                intermediates.proj_inputs_and_grads[w_name] = xy

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        max_tier = self.schema.max_tier
        return ComputeCost.sum(
            [
                self.attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.lin_attn.compute_cost(chunk, max_tier=max_tier),
                self.ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.ffn.compute_cost(chunk, max_tier=max_tier),
            ],
            max_tier=max_tier,
        )


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------


def build_qwen3_5_backbone(
    cfg: Qwen3_5LayerConfig, layer_types: list[str],
) -> list:
    """Build a Qwen3.5 backbone from ``layer_types`` (one entry per layer)."""
    out = []
    for i, lt in enumerate(layer_types):
        if lt == "linear_attention":
            out.append(Qwen3_5LinearLayer(i, cfg))
        elif lt == "full_attention":
            out.append(Qwen3_5FullLayer(i, cfg))
        else:
            raise ValueError(f"unknown layer_type at layer {i}: {lt!r}")
    return out
