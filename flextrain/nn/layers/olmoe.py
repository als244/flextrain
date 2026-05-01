"""OLMoE layer: Llama-style attention + MoE SwiGLU FFN.

Matches the OLMoE-1B-7B architecture (AllenAI):
* GQA attention (16 heads, 16 kv_heads → no grouping actually in
  OLMoE; n_heads == n_kv_heads).
* No QK-norm, no biases, causal.
* SwiGLU MoE FFN with num_experts=64, top_k=8.
* RoPE base 10,000; rms_norm_eps=1e-5.

The forward / backward / forward_recompute bodies follow LlamaBlock
very closely — the only swap is ``self.ffn`` being MoESwiGLUFFN
instead of SwiGLUFFN. Because MoE fwd/bwd have extra dependencies
(per-chunk token_index_mapping, expert_counts_host), we plumb
``chunk`` through to the FFN calls.
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
    BackwardIntermediates,
    ChunkMeta,
    ComputeCost,
    LayerContext,
    ParamSpec,
)
from flextrain.core.activation_schema import ActivationSlot
from flextrain.nn.blocks import (
    GQAAttentionBlock,
    GQAAttentionConfig,
    RMSNormBlock,
)
from flextrain.nn.blocks.ffn_moe import MoESwiGLUConfig, MoESwiGLUFFN


@dataclass(frozen=True)
class OLMoEBlockConfig:
    """Per-layer config for OLMoE (and OLMoE-like dense-attn + MoE-FFN)."""

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int
    num_experts: int
    top_k: int
    rms_norm_eps: float = 1e-5
    rope_base: float = 10_000.0
    is_causal: bool = True
    load_balance_coef: float = 0.01  # OLMoE default
    # OLMoE-1B-7B: norm_topk_prob=False → softmax_then_topk (default).
    # If a future OLMoE variant sets norm_topk_prob=True, caller should
    # pass routing_mode="topk_then_softmax".
    routing_mode: str = "softmax_then_topk"

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
        }


class OLMoEBlock:
    # Tells ``ActiveModel.__post_init__`` to allocate a secondary
    # CUDA compute stream so ``MoESwiGLUFFN.fwd`` / ``bwd`` can
    # alternate per-expert matmuls between primary and secondary
    # streams (see flextrain/nn/blocks/ffn_moe.py).
    uses_secondary_stream = True

    """OLMoE-style MoE transformer layer.

    Activation schema (max_tier=3)
    -------------------------------
    Tier 0  attn_norm_rstd, ffn_norm_rstd, x_inp, xk, xv,
            x_router, expert_counts, router_weights, chosen_experts,
            scattered_router_weights
    Tier 1  attn_result, softmax_lse
    Tier 2  xq, xo
    Tier 3  x_up (flat (T*K, 2F), expert-sorted)
    """

    def __init__(self, layer_id: int, cfg: OLMoEBlockConfig) -> None:
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
        # OLMoE Q/K RMSNorm is FULL-ROW (across attn_dim / kv_dim, not
        # per-head like Qwen3). Weight vectors are sized (attn_dim,) and
        # (kv_dim,); shape is independent of head partition. The attention
        # block owns the q_norm/k_norm RMSNormBlocks internally when
        # qk_norm=True; qk_norm_per_head=False selects the OLMoE layout.
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
                qk_norm_grad_dtype=cfg.norm_grad_dtype,
                qk_norm_per_head=False,
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
        """Delegate to ``backward_dgrad`` + ``backward_wgrad`` -- same
        FLOPs, same kernels, same order. The split lets LoRA's fast
        path (rank-r dA/dB on attention deferred Wgrads) skip the
        per-projection Wgrad addmm. See ``docs/lora_fast_backward.md``.
        """
        upstream_dx, intermediates = self.backward_dgrad(
            dx, chunk, weights, grads, slot, ctx,
        )
        self.backward_wgrad(intermediates, weights, grads, slot, ctx)
        return upstream_dx

    def backward_dgrad(
        self,
        dx: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot: ActivationSlot,
        ctx: LayerContext,
        *,
        skip_target_names: frozenset[str] = frozenset(),
    ) -> tuple[torch.Tensor, BackwardIntermediates]:
        """First half of OLMoE backward: dx + recomputed RMSNorm
        outputs into intermediates payload.

        ``skip_target_names`` honors ``w_o`` (attention output proj
        inline Wgrad) here; ``w_q/w_k/w_v`` are honored in
        :meth:`backward_wgrad`. MoE FFN expert projections
        (``w_up, w_down, w_router``) are always inline and currently
        go through the slow scratch-`dW` path -- the wrapper handles
        that internally.
        """
        cfg = self.cfg

        # Translate w_o (only inline-Wgrad target on this layer) ->
        # skip_grads for attn.bwd.
        skip_g_inline: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names if n in ("w_o",)
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_inline else None
        )

        # MoE FFN: gate g_up/g_down/g_router. The wrapper provides a
        # per-expert callback via intermediates.aux on the way in --
        # passed positionally below.
        skip_g_moe: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_up", "w_down", "w_router")
        )
        moe_callback = None  # populated by wrapper via slot.aux below
        if skip_g_moe:
            moe_callback = slot.aux.pop("__lora_moe_callback__", None)
            if moe_callback is None:
                # Wrapper didn't install one -- fall back to slow path
                # (don't actually skip).
                skip_g_moe = frozenset()

        # --- MoE FFN backward (monolithic; expert Wgrads always inline) ---
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
        dx_xo, _ = self.ffn_norm.bwd(
            ffn_norm_upstream,
            slot.xo.view(-1, cfg.d_model),
            weights, grads, slot.ffn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=False,
            recomputed_output_tensor=None,
        )
        dx = dx_xo

        # --- Attention backward ---
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
            aux={
                "attn_norm_fwd_output": attn_norm_fwd_output,
            },
        )
        if capture_xy is not None:
            for g_name, xy in capture_xy.items():
                w_name = "w_" + g_name[2:]
                intermediates.proj_inputs_and_grads[w_name] = xy
        return dx, intermediates

    def backward_wgrad(
        self,
        intermediates: BackwardIntermediates,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot: ActivationSlot,
        ctx: LayerContext,
        *,
        skip_target_names: frozenset[str] = frozenset(),
    ) -> None:
        """Second half: deferred attention Wgrads (``g_q, g_k, g_v``).
        MoE FFN Wgrads are inline in ``ffn.bwd`` and not skip-able yet."""
        del ctx, weights

        attn_norm_fwd_output = intermediates.aux["attn_norm_fwd_output"]

        # Translate w_* skip names -> g_* skip names. Only attention
        # deferred Wgrads are gateable in this Phase.
        skip_g_names: frozenset[str] = frozenset(
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
