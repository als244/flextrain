"""Qwen3 dense family.

Qwen3-dense is architecturally Llama + QK-norm + optional per-layer SWA:

* RMSNorm (attn_norm + ffn_norm) -- same as Llama.
* GQA attention -- same algorithm as Llama, but Q and K are each
  per-head RMSNorm'd before RoPE. Two extra ``RMSNormBlock(per_head=True)``
  instances at the Q/K-projection boundary.
* SwiGLU FFN -- same as Llama.
* Per-layer attention type: full-context OR sliding window, based on
  Qwen3's ``max_window_layers`` config (first N layers full, rest SWA).

This module therefore ships TWO model-family classes:

* :class:`Qwen3DenseBlock`          -- full-context QK-normed GQA
* :class:`Qwen3DenseSWABlock`       -- sliding-window QK-normed GQA

Build a Qwen3-style backbone by mixing them per ``max_window_layers``::

    backbone = [
        Qwen3DenseBlock(i, cfg) for i in range(max_window_layers)
    ] + [
        Qwen3DenseSWABlock(i, cfg_swa)
        for i in range(max_window_layers, n_layers)
    ]

The engine iterates the list with zero layer-type branches -- exactly the
pattern docs/PLAN.md "Multi-architecture strategy" describes.

Qwen3 MoE is a separate (forthcoming) layer that composes with
:class:`MoESwiGLUFFN` instead of :class:`SwiGLUFFN`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import (
    ActivationField,
    ActivationSchema,
    ActivationSlot,
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
    GQASlidingWindowAttentionBlock,
    GQASlidingWindowAttentionConfig,
    RMSNormBlock,
    SwiGLUConfig,
    SwiGLUFFN,
)


@dataclass(frozen=True)
class Qwen3DenseBlockConfig:
    """Per-layer config for a Qwen3-dense layer.

    Matches Qwen3's ``config.json`` fields (``head_dim``,
    ``num_key_value_heads``, ``rms_norm_eps``, ``rope_theta``, etc.).
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int
    rms_norm_eps: float = 1e-6  # Qwen3 default
    rope_base: float = 1_000_000.0  # Qwen3 default; long-context friendly
    is_causal: bool = True
    # For sliding-window layers only; use Qwen3DenseSWABlockConfig for those.

    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None
    norm_grad_dtype: torch.dtype = torch.float32
    # RMSNorm (and QK-norm) weight vectors are tiny; keep master fp32
    # so small-lr updates don't round to zero in bf16.
    norm_master_dtype: torch.dtype = torch.float32

    def dims(self) -> dict[str, int]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "expert_dim": self.expert_dim,
        }


@dataclass(frozen=True)
class Qwen3DenseSWABlockConfig(Qwen3DenseBlockConfig):
    """Qwen3-dense layer with sliding-window attention.

    Used for layers ``>= max_window_layers`` in Qwen3's config.
    """

    window_size_left: int = 4096


# ---------------------------------------------------------------------------
# Shared assembly helper so both variants share ~90% of __init__.
# ---------------------------------------------------------------------------


def _assemble_qwen3_block(
    self,
    layer_id: int,
    cfg: Qwen3DenseBlockConfig,
    *,
    attn_block,
    attn_cfg,
) -> None:
    """Common __init__ path for both Qwen3Dense variants.

    ``attn_block`` is :class:`GQAAttentionBlock` or
    :class:`GQASlidingWindowAttentionBlock`; ``attn_cfg`` the matching
    config. Keeps the composition in one place so the two subclasses
    only differ in which attention variant they pick.
    """
    self.layer_id = layer_id
    self.cfg = cfg
    self._dims = cfg.dims()

    # Residual-stream norms (same as Llama).
    self.attn_norm = RMSNormBlock(
        prefix="attn_norm",
        eps=cfg.rms_norm_eps,
        param_compute_dtype=cfg.compute_dtype,
        param_master_dtype=cfg.norm_master_dtype,
        param_grad_dtype=cfg.norm_grad_dtype,
    )
    self.ffn_norm = RMSNormBlock(
        prefix="ffn_norm",
        eps=cfg.rms_norm_eps,
        param_compute_dtype=cfg.compute_dtype,
        param_master_dtype=cfg.norm_master_dtype,
        param_grad_dtype=cfg.norm_grad_dtype,
    )

    # Per-head QK-norms: RMSNormBlock in per_head mode. Same kernel, just a
    # different ``head_dim`` arg and weight-vector dim.
    self.q_norm = RMSNormBlock(
        prefix="q_norm",
        eps=cfg.rms_norm_eps,
        per_head=True,
        heads_dim_name="n_heads",
        weight_dim_name="head_dim",
        param_compute_dtype=cfg.compute_dtype,
        param_master_dtype=cfg.norm_master_dtype,
        param_grad_dtype=cfg.norm_grad_dtype,
    )
    self.k_norm = RMSNormBlock(
        prefix="k_norm",
        eps=cfg.rms_norm_eps,
        per_head=True,
        heads_dim_name="n_kv_heads",
        weight_dim_name="head_dim",
        param_compute_dtype=cfg.compute_dtype,
        param_master_dtype=cfg.norm_master_dtype,
        param_grad_dtype=cfg.norm_grad_dtype,
    )

    # Attention + FFN.
    self.attn = attn_block(attn_cfg)
    self.ffn = SwiGLUFFN(
        SwiGLUConfig(
            d_model=cfg.d_model,
            expert_dim=cfg.expert_dim,
            compute_dtype=cfg.compute_dtype,
            master_dtype=cfg.master_dtype,
            grad_dtype=cfg.grad_dtype,
        )
    )

    # Assemble schema. Tier-0 x_inp goes right after attn_norm_rstd for the
    # same memory layout convention as LlamaBlock.
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
                self.q_norm.fields(),
                self.k_norm.fields(),
                self.ffn_norm.fields(),
                self.ffn.fields(),
            ]
        ),
        max_tier=3,
    )
    self.param_spec = ParamSpec.merge(
        [
            self.attn_norm.param_spec(),
            self.q_norm.param_spec(),
            self.k_norm.param_spec(),
            self.attn.param_spec(),
            self.ffn_norm.param_spec(),
            self.ffn.param_spec(),
        ]
    )


# ---------------------------------------------------------------------------
# Qwen3DenseBlock (full context)
# ---------------------------------------------------------------------------


class Qwen3DenseBlock:
    """Qwen3-dense full-context layer (layers ``< max_window_layers``).

    Forward / backward are the same as :class:`LlamaBlock` except we
    interpose per-head ``q_norm`` / ``k_norm`` (RMSNorm) between the Q/K
    projections and RoPE. The QK-norm hook is wired directly into
    :class:`GQAAttentionBlock` via ``cfg.qk_norm=True`` + ``set_qk_norm``.
    """

    def __init__(self, layer_id: int, cfg: Qwen3DenseBlockConfig) -> None:
        _assemble_qwen3_block(
            self,
            layer_id,
            cfg,
            attn_block=GQAAttentionBlock,
            attn_cfg=GQAAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_base=cfg.rope_base,
                is_causal=cfg.is_causal,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
                qk_norm=True,
                rms_norm_eps=cfg.rms_norm_eps,
            ),
        )
        self.attn.set_qk_norm(self.q_norm, self.k_norm)

    # ------------------------------------------------------------------
    # Layer Protocol (mirrors LlamaBlock, with QK-norm hook in attn)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        slot,  # ActivationSlot
        ctx: LayerContext,
    ) -> torch.Tensor:
        slot.x_inp.copy_(x)
        x_temp = ctx.scratch(x.shape, x.dtype)
        attn_norm_output = self.attn_norm.fwd(
            x, weights, slot.attn_norm_rstd, output=x_temp
        )
        # attn.fwd applies Q/K/V projections, QK-norm (per qk_norm=True),
        # RoPE, flash-attn, fused O-proj + residual. Writes rstd_q / rstd_k
        # into slot via the RMSNormBlock fields declared on the layer.
        attn_output_with_residual = self.attn.fwd(
            x, attn_norm_output, chunk, weights, slot, ctx
        )
        ffn_norm_output = self.ffn_norm.fwd(
            attn_output_with_residual.view(-1, self.cfg.d_model),
            weights,
            slot.ffn_norm_rstd,
            output=x_temp,
        )
        layer_output = self.ffn.fwd(
            ffn_norm_output,
            weights,
            attn_output_with_residual,
            out_tensor=x,
            slot=slot,
            ctx=ctx,
        )
        return layer_output

    def forward_recompute(
        self,
        slot,  # ActivationSlot
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> None:
        cfg = self.cfg
        x_inp = slot.x_inp

        if not slot.has("xq"):
            attn_norm_output = self.attn_norm.fwd_from_rstd(
                x_inp, weights, slot.attn_norm_rstd
            )
            # fwd_recompute_qo re-applies QK-norm on Q using the saved rstd_q.
            self.attn.fwd_recompute_qo(
                attn_norm_output, chunk, weights, slot, x_inp
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
                slot.xo.view(-1, cfg.d_model), weights, slot.ffn_norm_rstd
            )
            self.ffn.fwd_recompute_x1x3(
                ffn_norm_output,
                weights,
                slot,
                recompute_x1=recompute_x1,
                recompute_x3=recompute_x3,
            )
            slot.aux["recompute_ffn_norm_output"] = ffn_norm_output

    def backward(
        self,
        dx: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,  # ActivationSlot
        ctx: LayerContext,
    ) -> torch.Tensor:
        cfg = self.cfg

        # --- FFN backward ---
        dx_ffn_norm_up = self.ffn.bwd(dx, weights, grads, slot)
        ffn_norm_fwd_output_hint = slot.aux.pop(
            "recompute_ffn_norm_output", None
        )
        dx, ffn_norm_fwd_output = self.ffn_norm.bwd(
            dx_ffn_norm_up,
            slot.xo.view(-1, cfg.d_model),
            weights,
            grads,
            slot.ffn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=ffn_norm_fwd_output_hint is None,
            recomputed_output_tensor=None,
        )
        if ffn_norm_fwd_output_hint is not None:
            ffn_norm_fwd_output = ffn_norm_fwd_output_hint

        self.ffn.bwd_accumulate_w1_w3_grads(
            ffn_norm_fwd_output, grads, slot
        )
        del ffn_norm_fwd_output

        # --- Attention backward ---
        # Unlike LlamaBlock, attn.bwd needs attn_norm_output (to recompute
        # pre-QK-norm Q/K for RMSNorm bwd). Recompute it now from x_inp +
        # saved attn_norm_rstd, and reuse it for bwd_accumulate_qkv_grads.
        attn_norm_fwd_output = self.attn_norm.fwd_from_rstd(
            slot.x_inp, weights, slot.attn_norm_rstd
        )
        dx_attn_norm_up = self.attn.bwd(
            dx, chunk, weights, grads, slot, ctx,
            attn_norm_output=attn_norm_fwd_output,
        )

        # RMSNorm bwd for attn_norm. Pass recomputed output as a hint so we
        # don't redo the norm inside the kernel.
        dx, _ = self.attn_norm.bwd(
            dx_attn_norm_up,
            slot.x_inp,
            weights,
            grads,
            slot.attn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=False,
            recomputed_output_tensor=None,
        )

        self.attn.bwd_accumulate_qkv_grads(
            attn_norm_fwd_output, grads, slot
        )
        del attn_norm_fwd_output
        return dx

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        max_tier = self.schema.max_tier
        return ComputeCost.sum(
            [
                self.attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.attn.compute_cost(chunk, max_tier=max_tier),
                self.q_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.k_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.ffn.compute_cost(chunk, max_tier=max_tier),
            ],
            max_tier=max_tier,
        )


# ---------------------------------------------------------------------------
# Qwen3DenseSWABlock (sliding-window)
# ---------------------------------------------------------------------------


class Qwen3DenseSWABlock(Qwen3DenseBlock):
    """Qwen3-dense sliding-window variant (layers ``>= max_window_layers``).

    Same composition, with :class:`GQASlidingWindowAttentionBlock` in place
    of :class:`GQAAttentionBlock`.
    """

    def __init__(self, layer_id: int, cfg: Qwen3DenseSWABlockConfig) -> None:
        _assemble_qwen3_block(
            self,
            layer_id,
            cfg,
            attn_block=GQASlidingWindowAttentionBlock,
            attn_cfg=GQASlidingWindowAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_base=cfg.rope_base,
                is_causal=cfg.is_causal,
                window_size_left=cfg.window_size_left,
                window_size_right=0,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
                qk_norm=True,
                rms_norm_eps=cfg.rms_norm_eps,
            ),
        )
        self.attn.set_qk_norm(self.q_norm, self.k_norm)
