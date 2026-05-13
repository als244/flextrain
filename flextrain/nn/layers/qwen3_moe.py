"""Qwen3-MoE layer (e.g. Qwen3-30B-A3B, Qwen3-235B-A22B).

Architectural ingredients (from HF ``Qwen3MoeDecoderLayer``):
* GQA attention with **per-head** QK-norm (same as Qwen3-dense).
* SwiGLU MoE FFN: ``num_experts`` routed experts, top-K per token,
  ``norm_topk_prob=True`` by default (renormalized top-K weights).
* No shared experts (unlike OLMoE / DeepSeek-V2).
* RMSNorm (attn_norm, ffn_norm) eps usually ``1e-6``; RoPE theta
  ``1_000_000.0``; all standard Qwen3 dense defaults.

Construction mirrors :class:`OLMoEBlock` but with per-head QK-norm
plumbing copied from :class:`Qwen3DenseBlock`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import (
    ActivationField,
    ActivationSchema,
    concat_fields,
)
from flextrain.core.layer import (
    ChunkMeta,
    ComputeCost,
    LayerContext,
    ParamSpec,
)
from flextrain.nn.blocks import (
    GQAAttentionBlock,
    GQAAttentionConfig,
    RMSNormBlock,
)
from flextrain.nn.blocks.ffn_moe import MoESwiGLUConfig, MoESwiGLUFFN


@dataclass(frozen=True)
class Qwen3MoEBlockConfig:
    """Per-layer config for a Qwen3-MoE layer."""

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int
    num_experts: int
    top_k: int
    rms_norm_eps: float = 1e-6
    rope_base: float = 1_000_000.0
    is_causal: bool = True
    # Qwen3-MoE default: aux-loss coefficient 0.001 (vs. OLMoE's 0.01).
    load_balance_coef: float = 0.001
    # Qwen3-MoE config.json field ``norm_topk_prob`` (default True) →
    # softmax over K after top-K selection, weights sum to 1.
    routing_mode: str = "topk_then_softmax"

    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None
    norm_grad_dtype: torch.dtype = torch.float32
    norm_master_dtype: torch.dtype = torch.float32
    norm_compute_dtype: torch.dtype = torch.float32  # fp32 throughout for RMSNorm weights -- the (1+w) storage convention pushes them into the bf16 magnitude-1 regime where AdamW lr*sign(g) is below ULP.

    def dims(self) -> dict[str, int]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attn_dim": self.n_heads * self.head_dim,
            "kv_dim": self.n_kv_heads * self.head_dim,
            "expert_dim": self.expert_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }


class Qwen3MoEBlock:
    # Allocate a secondary CUDA compute stream so the MoE expert loop
    # can overlap per-expert matmuls (see ffn_moe.py).
    uses_secondary_stream = True

    """Qwen3-MoE full-context layer.

    Per-head QK-norm (like Qwen3-dense) + MoE FFN (like OLMoE), with
    Qwen3-MoE's ``topk_then_softmax`` router by default.
    """

    def __init__(
        self, layer_id: int, cfg: Qwen3MoEBlockConfig,
        *,
        expert_compute=None,  # MoEExpertCompute | None
    ) -> None:
        self.layer_id = layer_id
        self.cfg = cfg
        self._dims = cfg.dims()

        self.attn_norm = RMSNormBlock(
            prefix="attn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.norm_compute_dtype,
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
                is_causal=cfg.is_causal,
                qk_norm=True,
                rms_norm_eps=cfg.rms_norm_eps,
                qk_norm_master_dtype=cfg.norm_master_dtype,
                qk_norm_compute_dtype=cfg.norm_compute_dtype,
                qk_norm_grad_dtype=cfg.norm_grad_dtype,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            )
        )

        self.ffn_norm = RMSNormBlock(
            prefix="ffn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.norm_compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.ffn = MoESwiGLUFFN(
            MoESwiGLUConfig(
                d_model=cfg.d_model,
                expert_dim=cfg.expert_dim,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                load_balance_coef=cfg.load_balance_coef,
                routing_mode=cfg.routing_mode,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            expert_compute=expert_compute,
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
            out_tensor=x, slot=slot, ctx=ctx, chunk=chunk,
            layer_id=self.layer_id,
        )
        return layer_output

    def forward_recompute(
        self,
        slot,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        ctx: LayerContext,
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

        if not slot.has("x_up"):
            ffn_norm_output = self.ffn_norm.fwd_from_rstd(
                slot.xo.view(-1, cfg.d_model), weights, slot.ffn_norm_rstd,
            )
            self.ffn.fwd_recompute_x_up(
                ffn_norm_output, weights, slot, chunk, ctx,
                layer_id=self.layer_id,
            )
            slot.aux["recompute_ffn_norm_output"] = ffn_norm_output

    def backward(
        self,
        dx: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        upstream_dx, intermediates = self.backward_dgrad(
            dx, chunk, weights, grads, slot, ctx,
        )
        self.backward_wgrad(intermediates, weights, grads, slot, ctx)
        return upstream_dx

    def backward_dgrad(
        self,
        dx, chunk: ChunkMeta, weights, grads, slot, ctx: LayerContext,
        *, skip_target_names: frozenset[str] = frozenset(),
    ):
        from flextrain.core.layer import BackwardIntermediates
        cfg = self.cfg
        skip_g_inline: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_o",)
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_inline else None
        )

        if "recompute_ffn_norm_output" not in slot.aux:
            slot.aux["recompute_ffn_norm_output"] = self.ffn_norm.fwd_from_rstd(
                slot.xo.view(-1, cfg.d_model), weights, slot.ffn_norm_rstd,
            )

        # MoE LoRA capture (deferred-wgrad path).
        moe_capture = slot.aux.get("__lora_moe_capture__")

        ffn_norm_upstream = self.ffn.bwd(
            dx, weights, grads, slot, ctx, chunk,
            layer_id=self.layer_id,
            lora_capture=moe_capture,
        )

        ffn_norm_fwd_output = slot.aux.pop("recompute_ffn_norm_output")
        dx, _ = self.ffn_norm.bwd(
            ffn_norm_upstream,
            slot.xo.view(-1, cfg.d_model),
            weights, grads, slot.ffn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=False,
            recomputed_output_tensor=None,
        )

        attn_norm_fwd_output = slot.aux.pop(
            "recompute_attn_norm_output", None
        )
        if attn_norm_fwd_output is None:
            attn_norm_fwd_output = self.attn_norm.fwd_from_rstd(
                slot.x_inp, weights, slot.attn_norm_rstd,
            )

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
            aux={"attn_norm_fwd_output": attn_norm_fwd_output},
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
        attn_norm_fwd_output = intermediates.aux["attn_norm_fwd_output"]
        skip_g_names = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_q", "w_k", "w_v")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_names else None
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
                self.attn_norm.compute_cost(
                    chunk.total_q, self._dims, max_tier,
                ),
                self.attn.compute_cost(chunk, max_tier=max_tier),
                self.ffn_norm.compute_cost(
                    chunk.total_q, self._dims, max_tier,
                ),
                self.ffn.compute_cost(chunk, max_tier=max_tier),
            ],
            max_tier=max_tier,
        )
