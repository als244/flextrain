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
    BackwardIntermediates, ChunkMeta, ComputeCost, LayerContext, ParamSpec,
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
            param_compute_dtype=cfg.norm_compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.post_attn_norm = RMSNormBlock(
            prefix="post_attn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.norm_compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.pre_ffn_norm = RMSNormBlock(
            prefix="pre_ffn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.norm_compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.post_ffn_norm = RMSNormBlock(
            prefix="post_ffn_norm",
            eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.norm_compute_dtype,
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
                # Gemma 2 uses gated-GELU (tanh approximation), not SiLU.
                activation="gelu_tanh",
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
        # Dual-residual extras: ``a_only`` and ``ffn_only`` are the
        # unfused sublayer outputs (attn / ffn called with zero residual),
        # which feed the post_*_norm.bwd as the pre-norm input. We can't
        # recompute them without re-running the sublayer, so they ride
        # at tier 0. See docs/internal/gemma3_status.md §"Design
        # decision 1" for why these go on the Gemma layer's schema
        # rather than on the shared attn/ffn schemas.
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
        return (x_after_attn.view(-1, cfg.d_model) + h3).view_as(x)

    def forward_recompute(self, slot, chunk, weights, ctx) -> None:
        """Fill in fields whose ``tier > slot.level``.

        Tier ladder for Gemma 2:
          * Tier 0 (always saved): x_inp, x_mid, a_only, ffn_only, xk, xv,
                                    all four norm rstds.
          * Tier 1: attn_result, softmax_lse.
          * Tier 2: xq, xo.
          * Tier 3: x1, x3.

        ``a_only`` and ``ffn_only`` are tier-0 so they're always present
        and never recomputed here. The recompute paths for ``xq``,
        ``xo``, ``x1``, ``x3``, and ``attn_result`` mirror Llama with
        two adjustments:

        * ``xo`` is reconstructed with ZERO residual (Gemma's attn fwd
          passes zero into ``attn.fwd``; xo = ``attn_result @ w_o``).
        * The pre_ffn_norm input is ``slot.x_mid`` (the post-attn
          residual sum), not ``slot.xo`` as in Llama.

        Stashes the recomputed pre-norm fwd outputs into ``slot.aux``
        so ``backward_dgrad`` can reuse them without a second recompute.
        """
        cfg = self.cfg

        if not slot.has("xq"):
            pre_attn_norm_fwd_output = self.pre_attn_norm.fwd_from_rstd(
                slot.x_inp, weights, slot.pre_attn_norm_rstd,
            )
            # x_resid=ZERO so the fused addmm in fwd_recompute_o (called
            # below) reproduces the gemma "no residual" attn output.
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
    # Backward — split form (dgrad + wgrad).
    # ------------------------------------------------------------------
    #
    # Dual-residual chain rule (out = x_mid + post_ffn_norm(ffn_only),
    # x_mid = x_inp + post_attn_norm(a_only)):
    #
    #   dffn_only, g_post_ffn_norm += post_ffn_norm.bwd(dout, slot.ffn_only)
    #   dpre_ffn_norm_h             = ffn.bwd(dffn_only)
    #                                  inline: g_2
    #                                  deferred: g_1, g_3
    #   dx_mid, g_pre_ffn_norm     += pre_ffn_norm.bwd(dpre_ffn_norm_h,
    #                                                  slot.x_mid,
    #                                                  dx_accumulator=dout)
    #
    #   da_only, g_post_attn_norm  += post_attn_norm.bwd(dx_mid, slot.a_only)
    #   dpre_attn_h                = attn.bwd(da_only)
    #                                  inline: g_o
    #                                  deferred: g_q, g_k, g_v
    #   dx, g_pre_attn_norm        += pre_attn_norm.bwd(dpre_attn_h,
    #                                                   slot.x_inp,
    #                                                   dx_accumulator=dx_mid)
    #
    # The four RMSNorm γ grads accumulate inline (1-D, no recompute
    # needed). g_o / g_2 accumulate inline in attn.bwd / ffn.bwd.
    # g_q / g_k / g_v / g_1 / g_3 are deferred to wgrad because they
    # need the recomputed ``pre_{attn,ffn}_norm_fwd_output`` as the
    # left operand for X^T @ dY. Those outputs ride via
    # ``intermediates.aux``, same trick as Llama uses.
    #
    # Mirrors the split form in ``flextrain/nn/layers/llama.py``.

    def backward(
        self, dx, chunk, weights, grads, slot, ctx,
    ) -> torch.Tensor:
        """Delegating shim. See ``backward_dgrad`` + ``backward_wgrad``."""
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
        """Returns ``(upstream_dx, intermediates)`` — the gradient w.r.t.
        the layer input plus a payload of recomputed pre-norm outputs
        the deferred wgrad pass needs.

        ``skip_target_names`` ⊆ ``{w_q, w_k, w_v, w_o, w_1, w_2, w_3}``
        skips the corresponding inline Wgrad addmm (LoRA fast path).
        Inline cases honored here: ``w_o`` and ``w_2``. Deferred cases
        (``w_q/w_k/w_v/w_1/w_3``) are honored in
        ``backward_wgrad``.
        """
        cfg = self.cfg
        d_model = cfg.d_model

        # Translate w_* skip targets → g_* skip names for the inline
        # Wgrads (g_o in attn.bwd, g_2 in ffn.bwd).
        skip_g_inline: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names if n in ("w_o", "w_2")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_inline else None
        )

        dy = dx.view(-1, d_model)

        # --- Outer FFN residual ---
        # post_ffn_norm.bwd reads ``dy`` (the incoming layer-output grad)
        # but doesn't mutate it; ffn.bwd takes the post-norm dx and
        # doesn't touch ``dy``. So ``dy`` can serve as the
        # pre_ffn_norm.bwd accumulator directly — no extra clone needed
        # (same pattern as LlamaBlock).
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
            dx_accumulator=dy,  # in-place: dy now holds dx_mid_total
            recompute_output=True,
            recomputed_output_tensor=None,
        )

        # --- Outer attn residual ---
        da_only, _ = self.post_attn_norm.bwd(
            dx_mid, slot.a_only, weights, grads, slot.post_attn_norm_rstd,
            dx_accumulator=None, recompute_output=False,
        )
        dpre_attn_h = self.attn.bwd(
            da_only, chunk, weights, grads, slot, ctx,
            attn_norm_output=None,  # type: ignore[arg-type]
            skip_grads=skip_g_inline, capture_xy=capture_xy,
        )
        dx_final, pre_attn_norm_fwd_output = self.pre_attn_norm.bwd(
            dpre_attn_h, slot.x_inp, weights, grads,
            slot.pre_attn_norm_rstd,
            dx_accumulator=dx_mid,  # in-place
            recompute_output=True,
            recomputed_output_tensor=None,
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
        """Deferred Wgrads (``g_q/g_k/g_v/g_1/g_3``) — each needs the
        recomputed pre-norm forward output as its left operand. Reads
        the cached outputs from ``intermediates.aux``."""
        del ctx, weights
        pre_ffn_norm_fwd_output = intermediates.aux["pre_ffn_norm_fwd_output"]
        pre_attn_norm_fwd_output = intermediates.aux["pre_attn_norm_fwd_output"]

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
