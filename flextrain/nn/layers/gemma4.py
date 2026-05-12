"""Gemma 4 transformer layer = Gemma 3 + V-RMSNorm + k_eq_v on global +
per-layer-type head_dim + proportional rope + layer_scalar tail.

Structural changes vs :class:`flextrain.nn.layers.gemma3.Gemma3Block`
(text-only path of Gemma-4-31B-Instruct only):

1. Uses :class:`Gemma4AttentionBlock` (forked) instead of
   :class:`GQAAttentionBlock`. The fork carries V-RMSNorm,
   ``k_eq_v`` for global layers, and proportional partial-rope.

2. Per-layer-type head shapes: sliding layers use ``head_dim=256`` and
   ``num_key_value_heads=16`` (31B); global layers use
   ``global_head_dim=512`` and ``num_global_key_value_heads=4``. The
   layer config holds the resolved per-layer values; the backbone
   factory selects them per layer index from ``layer_types``.

3. ``layer_scalar`` buffer (HF: ``register_buffer("layer_scalar",
   torch.ones(1))``). Multiplied at the very end of the forward path.
   Stored as a layer-owned scalar; not a learnable parameter. Loaded
   from the checkpoint by the arch loader; defaults to 1.0.

4. ``rope_scaling`` carries the per-layer-type rope_type:
   ``"proportional"`` for global layers (with ``partial_rotary_factor=0.25``,
   ``rope_theta=1e6``), ``"default"`` for sliding layers
   (``rope_theta=10000.0``, full rotation over the sliding head_dim).

Backward path is currently stubbed — the dual-residual + V-norm +
k_eq_v + proportional-rope-bwd derivation is documented in
``docs/internal/gemma4_status.md`` and will land in a follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from flextrain.core.activation_schema import (
    ActivationField, ActivationSchema, concat_fields,
)
from flextrain.core.layer import (
    BackwardIntermediates, ChunkMeta, ComputeCost, LayerContext, ParamSpec,
)
from flextrain.nn.blocks import RMSNormBlock, SwiGLUConfig, SwiGLUFFN
from flextrain.nn.blocks.attention_gemma4 import (
    Gemma4AttentionBlock,
    Gemma4AttentionConfig,
    Gemma4SlidingWindowAttentionBlock,
    Gemma4SlidingWindowAttentionConfig,
)


@dataclass(frozen=True)
class Gemma4BlockConfig:
    """Per-layer Gemma 4 config.

    Most fields mirror :class:`flextrain.nn.layers.gemma3.Gemma3BlockConfig`.
    The Gemma-4 deltas are flagged in the comments.
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int                                    # Per-layer: 256 sliding / 512 global on 31B.
    expert_dim: int
    rms_norm_eps: float = 1e-6
    rope_base: float = 1_000_000.0
    rope_scaling: object | None = None
    is_causal: bool = True
    # Gemma 4 disables attn_logit_softcap (Gemma-2 vintage); final-logit
    # softcap is 30.0 (applied head-side, like Gemma 2).
    attn_logit_softcap: float | None = None
    final_logit_softcap: float | None = 30.0
    query_pre_attn_scalar: float | None = None
    window_size_left: int = -1
    # Gemma-4-specific:
    v_norm: bool = True                              # Always on for Gemma 4.
    k_eq_v: bool = False                             # True only on global layers.
    partial_rotary_factor: float = 1.0               # 0.25 on global, 1.0 on sliding.
    layer_scalar_init: float = 1.0                   # Loaded from checkpoint at runtime.

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


def _build_attn(cfg: Gemma4BlockConfig):
    # Gemma 4 disables attn_logit_softcap (per HF config). Map None → 0.0 at
    # the boundary so the underlying typed-float field stays untouched.
    softcap = 0.0 if cfg.attn_logit_softcap is None else float(cfg.attn_logit_softcap)
    common = dict(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        rope_base=cfg.rope_base,
        rope_scaling=cfg.rope_scaling,
        is_causal=cfg.is_causal,
        qk_norm=True,
        rms_norm_eps=cfg.rms_norm_eps,
        qk_norm_master_dtype=cfg.norm_master_dtype,
        qk_norm_grad_dtype=cfg.norm_grad_dtype,
        attn_logit_softcap=softcap,
        v_norm=cfg.v_norm,
        k_eq_v=cfg.k_eq_v,
        partial_rotary_factor=cfg.partial_rotary_factor,
        compute_dtype=cfg.compute_dtype,
        master_dtype=cfg.master_dtype,
        grad_dtype=cfg.grad_dtype,
    )
    if cfg.window_size_left >= 0:
        return Gemma4SlidingWindowAttentionBlock(
            Gemma4SlidingWindowAttentionConfig(
                window_size_left=cfg.window_size_left, **common,
            )
        )
    return Gemma4AttentionBlock(Gemma4AttentionConfig(**common))


class Gemma4Block:
    """Gemma 4 text decoder layer (text-only path).

    Dual-residual norm topology identical to Gemma 2/3; the body of the
    layer uses :class:`Gemma4AttentionBlock` (with V-norm and optional
    ``k_eq_v``) and a SwiGLU FFN with gelu_pytorch_tanh activation
    (Gemma family default — same as Gemma 2/3).

    Output is multiplied by a per-layer scalar (HF: ``layer_scalar``)
    before being returned. The scalar is stored on the layer instance
    (not in ParamSpec) and defaults to 1.0.
    """

    def __init__(self, layer_id: int, cfg: Gemma4BlockConfig) -> None:
        self.layer_id = layer_id
        self.cfg = cfg
        self._dims = cfg.dims()

        # Non-learnable per-layer scalar. Loader can mutate via
        # ``set_layer_scalar``. Stored as a Python float for simplicity;
        # the multiply at the end of forward is a single broadcasted op.
        self.layer_scalar: float = float(cfg.layer_scalar_init)

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
        self.attn = _build_attn(cfg)
        self.ffn = SwiGLUFFN(
            SwiGLUConfig(
                d_model=cfg.d_model, expert_dim=cfg.expert_dim,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
                activation="gelu_tanh",   # Gemma family: gelu_pytorch_tanh.
            )
        )

        # Activation schema. Same shape as Gemma3Block (x_inp / a_only /
        # x_mid / ffn_only) plus whatever the forked attention adds
        # (v_norm_rstd).
        x_inp = ActivationField(
            "x_inp", lambda n, d: (n, cfg.d_model),
            cfg.compute_dtype, tier=0,
        )
        x_mid = ActivationField(
            "x_mid", lambda n, d: (n, cfg.d_model),
            cfg.compute_dtype, tier=0,
        )
        a_only = ActivationField(
            "a_only", lambda n, d: (n, cfg.d_model),
            cfg.compute_dtype, tier=0,
        )
        ffn_only = ActivationField(
            "ffn_only", lambda n, d: (n, cfg.d_model),
            cfg.compute_dtype, tier=0,
        )
        self.schema = ActivationSchema(
            fields=concat_fields(
                [
                    self.pre_attn_norm.fields(),
                    (x_inp,),
                    (a_only,),
                    self.attn.fields(),
                    self.post_attn_norm.fields(),
                    (x_mid,),
                    (ffn_only,),
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

    # ------------------------------------------------------------------
    # Layer protocol
    # ------------------------------------------------------------------

    def set_layer_scalar(self, value: float) -> None:
        """Loader entry point. Called by arch.gemma4's post_load hook
        when the safetensor contains a non-1.0 ``layer_scalar``."""
        self.layer_scalar = float(value)

    def forward(
        self, x, chunk: ChunkMeta, weights, slot, ctx: LayerContext,
    ) -> torch.Tensor:
        cfg = self.cfg
        slot.x_inp.copy_(x)
        x_temp = ctx.scratch(x.shape, x.dtype)

        # Attention sub-layer (pre + post norm, residual added after).
        zero_resid = ctx.scratch(x.shape, x.dtype).zero_()
        h = self.pre_attn_norm.fwd(x, weights, slot.pre_attn_norm_rstd, output=x_temp)
        a_only = self.attn.fwd(zero_resid, h, chunk, weights, slot, ctx)
        slot.a_only.copy_(a_only.view(-1, cfg.d_model))
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
        slot.ffn_only.copy_(ffn_only.view(-1, cfg.d_model))
        h3 = self.post_ffn_norm.fwd(
            ffn_only.view(-1, cfg.d_model),
            weights, slot.post_ffn_norm_rstd, output=x_temp,
        )
        layer_out = (x_after_attn.view(-1, cfg.d_model) + h3).view_as(x)
        if self.layer_scalar != 1.0:
            layer_out = layer_out * self.layer_scalar
        return layer_out

    def forward_recompute(self, slot, chunk, weights, ctx) -> None:
        """Refill fields whose ``tier > slot.level``. Same ladder as
        Gemma3Block.forward_recompute (xq/xo/x1/x3 reachable from saved
        x_inp / x_mid / KV / rstds)."""
        cfg = self.cfg
        del cfg

        if not slot.has("xq"):
            pre_attn_norm_fwd_output = self.pre_attn_norm.fwd_from_rstd(
                slot.x_inp, weights, slot.pre_attn_norm_rstd,
            )
            self.attn.fwd_recompute_qo(
                pre_attn_norm_fwd_output, chunk, weights, slot,
                x_resid=ctx.scratch(slot.x_inp.shape, slot.x_inp.dtype).zero_(),
            )
            slot.aux["recompute_pre_attn_norm_output"] = pre_attn_norm_fwd_output

        if not slot.has("attn_result"):
            self.attn.fwd_recompute_attn(chunk, slot, ctx)

        if not slot.has("xo"):
            zero_resid = ctx.scratch(slot.x_inp.shape, slot.x_inp.dtype).zero_()
            self.attn.fwd_recompute_o(zero_resid, weights, slot)

        recompute_x1 = not slot.has("x1")
        recompute_x3 = not slot.has("x3")
        if recompute_x1 or recompute_x3:
            pre_ffn_norm_fwd_output = self.pre_ffn_norm.fwd_from_rstd(
                slot.x_mid, weights, slot.pre_ffn_norm_rstd,
            )
            self.ffn.fwd_recompute_x1x3(
                pre_ffn_norm_fwd_output, weights, slot,
                recompute_x1=recompute_x1, recompute_x3=recompute_x3,
            )
            slot.aux["recompute_pre_ffn_norm_output"] = pre_ffn_norm_fwd_output

    # ------------------------------------------------------------------
    # Backward — split form (dgrad + wgrad), mirroring Gemma3Block.
    #
    # The dual-residual chain rule is identical to Gemma 3; the
    # Gemma-4-specific gradient routes (V-norm bwd, k_eq_v fold) all
    # live inside Gemma4AttentionBlock.bwd. The layer just threads
    # the recomputed pre_attn_norm output through and applies the
    # outer-residual chain rule.
    #
    # ``layer_scalar`` handling: the forward multiplies the layer
    # output by ``self.layer_scalar``. By chain rule the incoming dx
    # must be scaled by the same factor before the bwd chain. For the
    # common case (``layer_scalar == 1.0``) the multiply is skipped.
    # ------------------------------------------------------------------

    def backward(
        self, dx, chunk, weights, grads, slot, ctx,
    ) -> torch.Tensor:
        upstream_dx, intermediates = self.backward_dgrad(
            dx, chunk, weights, grads, slot, ctx,
        )
        self.backward_wgrad(intermediates, weights, grads, slot, ctx)
        return upstream_dx

    def backward_dgrad(
        self,
        dx: torch.Tensor,
        chunk: ChunkMeta,
        weights,
        grads,
        slot,
        ctx: LayerContext,
        *,
        skip_target_names: frozenset[str] = frozenset(),
    ) -> tuple[torch.Tensor, BackwardIntermediates]:
        cfg = self.cfg
        d_model = cfg.d_model

        skip_g_inline: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names if n in ("w_o", "w_2")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_inline else None
        )

        # layer_scalar pre-multiply: dx_eff = dx * layer_scalar.
        # Cheap copy if scalar != 1; otherwise pass-through.
        if self.layer_scalar != 1.0:
            dx = dx * self.layer_scalar
        dy = dx.view(-1, d_model)

        # --- Outer FFN residual ---
        # post_ffn_norm.bwd reads dy but doesn't mutate it; ffn.bwd
        # leaves dy alone; we can use dy as the pre_ffn_norm.bwd
        # accumulator in-place (same pattern as Gemma3Block).
        dffn_only, _ = self.post_ffn_norm.bwd(
            dy, slot.ffn_only, weights, grads, slot.post_ffn_norm_rstd,
            dx_accumulator=None, recompute_output=False,
        )
        dpre_ffn_norm_h = self.ffn.bwd(
            dffn_only, weights, grads, slot,
            skip_grads=skip_g_inline, capture_xy=capture_xy,
        )
        dx_mid, pre_ffn_norm_fwd_output = self.pre_ffn_norm.bwd(
            dpre_ffn_norm_h, slot.x_mid, weights, grads,
            slot.pre_ffn_norm_rstd,
            dx_accumulator=dy,
            recompute_output=True,
            recomputed_output_tensor=None,
        )

        # --- Outer attn residual --- recompute pre_attn_norm output here
        # so attn.bwd can use it (it's the left operand for the V-norm,
        # QK-norm, and Q/K/V wgrad matmuls).
        pre_attn_norm_fwd_output = self.pre_attn_norm.fwd_from_rstd(
            slot.x_inp, weights, slot.pre_attn_norm_rstd,
        )
        da_only, _ = self.post_attn_norm.bwd(
            dx_mid, slot.a_only, weights, grads, slot.post_attn_norm_rstd,
            dx_accumulator=None, recompute_output=False,
        )
        dpre_attn_h = self.attn.bwd(
            da_only, chunk, weights, grads, slot, ctx,
            attn_norm_output=pre_attn_norm_fwd_output,
            skip_grads=skip_g_inline, capture_xy=capture_xy,
        )
        dx_final, _ = self.pre_attn_norm.bwd(
            dpre_attn_h, slot.x_inp, weights, grads,
            slot.pre_attn_norm_rstd,
            dx_accumulator=dx_mid,
            recompute_output=True,
            recomputed_output_tensor=pre_attn_norm_fwd_output,
        )

        intermediates = BackwardIntermediates(
            proj_inputs_and_grads={},
            aux={
                "pre_ffn_norm_fwd_output": pre_ffn_norm_fwd_output,
                "pre_attn_norm_fwd_output": pre_attn_norm_fwd_output,
            },
        )
        if capture_xy is not None:
            for g_name, xy in capture_xy.items():
                w_name = "w_" + g_name[2:]
                intermediates.proj_inputs_and_grads[w_name] = xy
        return dx_final.view_as(dx), intermediates

    def backward_wgrad(
        self,
        intermediates: BackwardIntermediates,
        weights,
        grads,
        slot,
        ctx: LayerContext,
        *,
        skip_target_names: frozenset[str] = frozenset(),
    ) -> None:
        del ctx, weights
        pre_ffn_norm_fwd_output = intermediates.aux["pre_ffn_norm_fwd_output"]
        pre_attn_norm_fwd_output = intermediates.aux["pre_attn_norm_fwd_output"]

        # For Gemma 4 global layers (k_eq_v=True) the param ``w_v`` is
        # absent; ``g_v`` is not in skip_g_names regardless.
        skip_g_names: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_q", "w_k", "w_v", "w_1", "w_3")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_names else None
        )

        self.ffn.bwd_accumulate_w1_w3_grads(
            pre_ffn_norm_fwd_output, grads, slot,
            skip_grads=skip_g_names, capture_xy=capture_xy,
        )
        self.attn.bwd_accumulate_qkv_grads(
            pre_attn_norm_fwd_output, grads, slot,
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
                self.pre_attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.attn.compute_cost(chunk, max_tier=max_tier),
                self.post_attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.pre_ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
                self.ffn.compute_cost(chunk, max_tier=max_tier),
                self.post_ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
            ],
            max_tier=max_tier,
        )


def build_gemma4_backbone(
    base_cfg: Gemma4BlockConfig,
    layer_types: list[str],
    *,
    sliding_window: int,
    sliding_head_dim: int,
    sliding_n_kv_heads: int,
    global_head_dim: int,
    global_n_kv_heads: int,
    rope_local_base: float = 10_000.0,
    rope_global_base: float = 1_000_000.0,
    global_partial_rotary_factor: float = 0.25,
    global_attention_k_eq_v: bool = True,
) -> list[Gemma4Block]:
    """Build a Gemma 4 backbone alternating sliding (local) and full
    (global) layers per ``layer_types`` from the HF config.

    Each layer gets a freshly-replaced :class:`Gemma4BlockConfig` with
    layer-type-specific head_dim, n_kv_heads, rope_base, partial_rotary
    factor, k_eq_v, and window size. Use this from the arch loader's
    block builder.

    ``base_cfg`` provides the shared / non-per-layer fields (d_model,
    n_heads, expert_dim, dtypes, eps). Its ``head_dim`` /
    ``n_kv_heads`` / ``rope_base`` / etc. fields are overwritten per
    layer.
    """
    import dataclasses

    out: list[Gemma4Block] = []
    for i, lt in enumerate(layer_types):
        if lt == "sliding_attention":
            layer_cfg = dataclasses.replace(
                base_cfg,
                head_dim=sliding_head_dim,
                n_kv_heads=sliding_n_kv_heads,
                rope_base=rope_local_base,
                rope_scaling=None,                # default rope on sliding
                window_size_left=sliding_window,
                v_norm=True,
                k_eq_v=False,
                partial_rotary_factor=1.0,
            )
        elif lt == "full_attention":
            layer_cfg = dataclasses.replace(
                base_cfg,
                head_dim=global_head_dim,
                n_kv_heads=global_n_kv_heads,
                rope_base=rope_global_base,
                rope_scaling={"rope_type": "proportional"},
                window_size_left=-1,
                v_norm=True,
                k_eq_v=global_attention_k_eq_v,
                partial_rotary_factor=global_partial_rotary_factor,
            )
        else:
            raise ValueError(f"unknown layer_type {lt!r} at layer {i}")
        out.append(Gemma4Block(i, layer_cfg))
    return out
