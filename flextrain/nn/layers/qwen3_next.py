"""Qwen3-Next family: alternating linear-attention and full-attention layers.

Per HF ``Qwen3NextConfig.layer_types`` (a ``list[str]`` indexed by
layer position), each layer is either:

* ``"linear_attention"`` — :class:`Qwen3NextLinearLayer` (uses the
  :class:`GatedDeltaNetBlock` instead of softmax attention).
* ``"full_attention"``   — :class:`Qwen3NextFullLayer` (uses
  :class:`GQAAttentionGatedBlock` with per-head QK-norm, partial
  rotary, and a sigmoid output gate; plus a shared-expert MoE FFN).

Building a Qwen3-Next backbone is a list comprehension over
``layer_types`` — the engine's heterogeneous-backbone path takes care
of mixing layer types in one model.

Hybrid models (Qwen 3.5 / 3.6) reuse this same scaffolding — only the
layer_types pattern changes per release.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import (
    ActivationField, ActivationSchema, concat_fields,
)
from flextrain.core.layer import (
    ChunkMeta, ComputeCost, LayerContext, MoEChunkConfig, ParamSpec,
)
from flextrain.nn.blocks import (
    GQAAttentionGatedBlock, GQAAttentionGatedConfig, RMSNormBlock,
)
from flextrain.nn.blocks.ffn_moe_shared import (
    MoESwiGLUSharedExpertConfig, MoESwiGLUSharedExpertFFN,
)
from flextrain.nn.blocks.linear_attn import (
    GatedDeltaNetBlock, GatedDeltaNetConfig,
)


@dataclass(frozen=True)
class Qwen3NextLayerConfig:
    """Per-layer config; same fields used by both linear and full layers."""

    # Common
    d_model: int
    n_heads: int                    # full-attn only
    n_kv_heads: int                 # full-attn only
    head_dim: int                   # full-attn head_dim
    expert_dim: int
    num_experts: int
    top_k: int

    # Linear-attn-specific (ignored by full layers)
    linear_num_v_heads: int = 16
    linear_num_k_heads: int = 16
    linear_head_k_dim: int = 128
    linear_head_v_dim: int = 128
    linear_conv_kernel: int = 4

    # Full-attn-specific (ignored by linear layers).
    # Qwen3-Next default per HF Qwen3NextConfig: 0.25 → rotary on first
    # head_dim/4 channels only, rest pass through.
    partial_rotary_factor: float = 0.25

    # Shared-expert MoE (used by full-attn layers' FFN; Qwen3-Next has
    # exactly 1 shared expert with intermediate size 512).
    num_shared_experts: int = 1
    shared_expert_dim: int = 512

    # Common knobs
    rms_norm_eps: float = 1e-6
    rope_base: float = 10_000_000.0  # Qwen3-Next default
    is_causal: bool = True
    load_balance_coef: float = 0.001
    routing_mode: str = "topk_then_softmax"

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
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "num_shared_experts": self.num_shared_experts,
            "shared_expert_dim": self.shared_expert_dim,
            # Linear-attn dims.
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
# Linear-attention layer
# ---------------------------------------------------------------------------


class Qwen3NextLinearLayer:
    """Qwen3-Next layer with gated-DeltaNet linear attention + MoE FFN."""

    def __init__(self, layer_id: int, cfg: Qwen3NextLayerConfig) -> None:
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
        # Qwen3-Next: every layer (linear- and full-attn) has a shared
        # expert in addition to routed experts.
        self.ffn = MoESwiGLUSharedExpertFFN(
            MoESwiGLUSharedExpertConfig(
                d_model=cfg.d_model,
                expert_dim=cfg.expert_dim,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                num_shared_experts=cfg.num_shared_experts,
                shared_expert_dim=cfg.shared_expert_dim,
                load_balance_coef=cfg.load_balance_coef,
                routing_mode=cfg.routing_mode,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            )
        )
        self.moe_chunk_config = MoEChunkConfig(
            num_experts=cfg.num_experts, top_k=cfg.top_k,
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
        # Linear attention is residual-aware via its caller. The block
        # returns ``(T, d_model)`` un-residualed; we add the residual.
        lin_out = self.lin_attn.fwd(attn_norm_output, weights, slot, ctx)
        attn_output_with_residual = x + lin_out
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
        self, slot, chunk: ChunkMeta, weights, ctx: LayerContext,
    ) -> None:
        """Fill in fields that weren't saved at this slot's ``level``.

        Mirrors :meth:`flextrain.nn.layers.llama.LlamaBlock.forward_recompute`'s
        partial-tier-recompute pattern — each upstream stage is only
        re-run if its outputs aren't present.
        """
        cfg = self.cfg

        # Linear-attn block stages, in order of dependency depth:
        # Tier 3 ⊃ Tier 2 ⊃ Tier 1 ⊃ Tier 0.
        # ``lin_q`` is tier 2 — present iff the slot's level >= 2.
        # ``lin_post_conv_pre_silu`` is tier 3 — present iff level >= 3.
        # If post-conv is missing we have to re-run from x_inp (stage 1+2).
        post_conv = None
        if not slot.has("lin_post_conv_pre_silu"):
            attn_norm_output = self.attn_norm.fwd_from_rstd(
                slot.x_inp, weights, slot.attn_norm_rstd,
            )
            post_conv = self.lin_attn.fwd_recompute_post_conv(
                attn_norm_output, weights, slot,
            )

        if not slot.has("lin_q"):
            # Re-derive Q/K/V from (saved or just-recomputed)
            # post_conv. fwd_recompute_qkv_heads reads
            # lin_post_conv_pre_silu from slot.
            self.lin_attn.fwd_recompute_qkv_heads(slot)

        if not slot.has("lin_core_out") or not slot.has("lin_A_int"):
            # Re-run FLA fwd from saved q/k/v/g/b.
            self.lin_attn.fwd_recompute_fla(weights, slot)

        # MoE tier-3 recompute (x_up).
        if not slot.has("x_up"):
            ffn_norm_output = self.ffn_norm.fwd_from_rstd(
                slot.x_inp, weights, slot.ffn_norm_rstd,
            )
            self.ffn.fwd_recompute_x_up(
                ffn_norm_output, weights, slot, chunk,
                layer_id=self.layer_id,
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
            if n in ("w_lin_out", "w_lin_qkvz", "w_lin_ba")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_inline else None
        )
        skip_g_moe: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_up", "w_down", "w_router",
                     "w_shared_up", "w_shared_down", "w_shared_expert_gate")
        )
        moe_callback = None
        if skip_g_moe:
            moe_callback = slot.aux.pop("__lora_moe_callback__", None)
            if moe_callback is None:
                skip_g_moe = frozenset()

        if "recompute_ffn_norm_output" not in slot.aux:
            slot.aux["recompute_ffn_norm_output"] = self.ffn_norm.fwd_from_rstd(
                slot.x_inp, weights, slot.ffn_norm_rstd,
            )

        ffn_norm_upstream = self.ffn.bwd(
            dx, weights, grads, slot, ctx, chunk,
            layer_id=self.layer_id,
            skip_grads=skip_g_moe,
            lora_per_expert_callback=moe_callback,
        )
        ffn_norm_fwd_output = slot.aux.pop("recompute_ffn_norm_output")
        dx, _ = self.ffn_norm.bwd(
            ffn_norm_upstream,
            slot.x_inp.view(-1, cfg.d_model),
            weights, grads, slot.ffn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=False,
            recomputed_output_tensor=None,
        )

        attn_norm_fwd_output = self.attn_norm.fwd_from_rstd(
            slot.x_inp, weights, slot.attn_norm_rstd,
        )
        dx_lin = self.lin_attn.bwd(
            dx, weights, grads, slot, ctx,
            skip_grads=skip_g_inline, capture_xy=capture_xy,
        )
        dx_attn_norm_up = dx_lin
        dx, _ = self.attn_norm.bwd(
            dx_attn_norm_up,
            slot.x_inp,
            weights, grads, slot.attn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=False,
            recomputed_output_tensor=None,
        )

        intermediates = BackwardIntermediates(
            proj_inputs_and_grads={}, aux={},
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
        # No deferred Wgrads in linear-attention layer (everything
        # inline in lin_attn.bwd / ffn.bwd).
        del intermediates, weights, grads, slot, ctx, skip_target_names

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        max_tier = self.schema.max_tier
        return ComputeCost.sum(
            [
                self.attn_norm.compute_cost(
                    chunk.total_q, self._dims, max_tier,
                ),
                self.lin_attn.compute_cost(chunk, max_tier=max_tier),
                self.ffn_norm.compute_cost(
                    chunk.total_q, self._dims, max_tier,
                ),
                self.ffn.compute_cost(chunk, max_tier=max_tier),
            ],
            max_tier=max_tier,
        )


# ---------------------------------------------------------------------------
# Full-attention layer
# ---------------------------------------------------------------------------


class Qwen3NextFullLayer:
    """Qwen3-Next full-attention layer.

    Composition:

    * RMSNorm(attn_norm)
    * GQAAttentionGatedBlock — w_q is doubled (Q + sigmoid gate), per-head
      QK-norm, partial-rotary RoPE (factor 0.25 by default).
    * RMSNorm(ffn_norm)
    * MoESwiGLUSharedExpertFFN — routed top-K experts + 1 always-on
      shared expert with its own per-token sigmoid gate.

    Used by Qwen3-Next, Qwen3.5, and Qwen3.6 ``"full_attention"`` layer
    types in ``layer_types``.
    """

    def __init__(self, layer_id: int, cfg: Qwen3NextLayerConfig) -> None:
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
        self.ffn = MoESwiGLUSharedExpertFFN(
            MoESwiGLUSharedExpertConfig(
                d_model=cfg.d_model,
                expert_dim=cfg.expert_dim,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                num_shared_experts=cfg.num_shared_experts,
                shared_expert_dim=cfg.shared_expert_dim,
                load_balance_coef=cfg.load_balance_coef,
                routing_mode=cfg.routing_mode,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            )
        )

        self.moe_chunk_config = MoEChunkConfig(
            num_experts=cfg.num_experts, top_k=cfg.top_k,
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
                ffn_norm_output, weights, slot, chunk,
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
        self, dx, chunk: ChunkMeta, weights, grads, slot, ctx,
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
        skip_g_moe: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_up", "w_down", "w_router",
                     "w_shared_up", "w_shared_down", "w_shared_expert_gate")
        )
        moe_callback = None
        if skip_g_moe:
            moe_callback = slot.aux.pop("__lora_moe_callback__", None)
            if moe_callback is None:
                skip_g_moe = frozenset()

        if "recompute_ffn_norm_output" not in slot.aux:
            slot.aux["recompute_ffn_norm_output"] = self.ffn_norm.fwd_from_rstd(
                slot.xo.view(-1, cfg.d_model), weights, slot.ffn_norm_rstd,
            )

        ffn_norm_upstream = self.ffn.bwd(
            dx, weights, grads, slot, ctx, chunk,
            layer_id=self.layer_id,
            skip_grads=skip_g_moe,
            lora_per_expert_callback=moe_callback,
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


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------


def build_qwen3_next_backbone(
    cfg: Qwen3NextLayerConfig, layer_types: list[str],
) -> list:
    """Build a heterogeneous Qwen3-Next backbone driven by ``layer_types``.

    Args:
        cfg: per-layer config (shared by all layer types in the backbone).
        layer_types: list of length ``n_layers`` whose entries are
            ``"linear_attention"`` or ``"full_attention"``.

    Returns:
        list of layer instances; pass directly as ``backbone=...`` to
        :class:`ActiveModel`.
    """
    out = []
    for i, lt in enumerate(layer_types):
        if lt == "linear_attention":
            out.append(Qwen3NextLinearLayer(i, cfg))
        elif lt == "full_attention":
            out.append(Qwen3NextFullLayer(i, cfg))
        else:
            raise ValueError(f"unknown layer_type at layer {i}: {lt!r}")
    return out
