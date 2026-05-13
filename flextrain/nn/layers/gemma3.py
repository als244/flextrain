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
    BackwardIntermediates, ChunkMeta, ComputeCost, LayerContext, ParamSpec,
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
    rope_scaling: object | None = None  # e.g. {"rope_type": "linear", "factor": 8.0} for 4B/12B
    is_causal: bool = True
    # Gemma 3 disables both softcaps by default (HF configs ship null).
    # Gemma 2 left both on (50.0 / 30.0); pass explicit values when
    # constructing a Gemma 2 layer that uses this config.
    attn_logit_softcap: float | None = None
    final_logit_softcap: float | None = None
    query_pre_attn_scalar: float | None = None
    window_size_left: int = -1     # >= 0 → sliding-window layer

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


def _build_attn(cfg: Gemma3BlockConfig):
    # GQAAttentionConfig.attn_logit_softcap is typed ``float`` and treats
    # ``>0`` as enabled; ``None`` (HF Gemma 3 default) maps to ``0.0``.
    softcap = 0.0 if cfg.attn_logit_softcap is None else float(cfg.attn_logit_softcap)
    if cfg.window_size_left >= 0:
        return GQASlidingWindowAttentionBlock(
            GQASlidingWindowAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_base=cfg.rope_base,
                rope_scaling=cfg.rope_scaling,
                is_causal=cfg.is_causal,
                qk_norm=True,
                rms_norm_eps=cfg.rms_norm_eps,
                qk_norm_master_dtype=cfg.norm_master_dtype,
                qk_norm_compute_dtype=cfg.norm_compute_dtype,
                qk_norm_grad_dtype=cfg.norm_grad_dtype,
                window_size_left=cfg.window_size_left,
                attn_logit_softcap=softcap,
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
            rope_scaling=cfg.rope_scaling,
            is_causal=cfg.is_causal,
            qk_norm=True,
            rms_norm_eps=cfg.rms_norm_eps,
            qk_norm_master_dtype=cfg.norm_master_dtype,
            qk_norm_compute_dtype=cfg.norm_compute_dtype,
            qk_norm_grad_dtype=cfg.norm_grad_dtype,
            attn_logit_softcap=softcap,
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
            param_compute_dtype=cfg.norm_compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.post_attn_norm = RMSNormBlock(
            prefix="post_attn_norm", eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.norm_compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.pre_ffn_norm = RMSNormBlock(
            prefix="pre_ffn_norm", eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.norm_compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.post_ffn_norm = RMSNormBlock(
            prefix="post_ffn_norm", eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.norm_compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        # Per-head QK-norm is owned by the attention block when
        # cfg.qk_norm=True (rolled up via attn.fields() / param_spec()).
        self.attn = _build_attn(cfg)
        self.ffn = SwiGLUFFN(
            SwiGLUConfig(
                d_model=cfg.d_model, expert_dim=cfg.expert_dim,
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
                # Gemma 3 uses gated-GELU (tanh approximation), not SiLU.
                activation="gelu_tanh",
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
        # Dual-residual extras: ``a_only`` / ``ffn_only`` are the
        # unfused sublayer outputs (attn / ffn called with zero
        # residual), which feed post_*_norm.bwd. Same rationale as
        # Gemma2Block — see docs/internal/gemma3_status.md §"Design
        # decision 1". Cost: 2 × T × d_model bf16 per layer per chunk.
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

        # Attn sub-layer with dual norm and zero residual into the attn block.
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
        slot.ffn_only.copy_(ffn_only.view(-1, cfg.d_model))
        h3 = self.post_ffn_norm.fwd(
            ffn_only.view(-1, cfg.d_model),
            weights, slot.post_ffn_norm_rstd, output=x_temp,
        )
        return (x_after_attn.view(-1, cfg.d_model) + h3).view_as(x)

    def forward_recompute(self, slot, chunk, weights, ctx) -> None:
        """Same recompute ladder as Gemma2Block.forward_recompute. QK-norm
        is handled inside ``attn.fwd_recompute_qo`` automatically when
        ``cfg.qk_norm=True`` (uses the saved per-head rstd_q / rstd_k).
        """
        cfg = self.cfg
        del cfg  # not consumed; readability comment lives here

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
    # Backward — identical structure to Gemma2Block, with the QK-norm
    # wrinkle: GQAAttentionBlock.bwd requires ``attn_norm_output`` when
    # ``cfg.qk_norm=True`` (it recomputes xq/xk pre-norm from this to
    # run the per-head RMSNorm bwd). We recompute pre_attn_norm fwd
    # output once and pass it both to attn.bwd AND to pre_attn_norm.bwd
    # (which then skips its own recompute via ``recomputed_output_tensor=``).
    # ------------------------------------------------------------------
    #
    # PARITY VALIDATION (kept here as a reference so future debuggers
    # don't redo this). The Gemma 3 backward was validated at four
    # increasing levels of integration:
    #
    #   * Block-level (tests/test_gemma3_block_parity.py): per-tensor
    #     weight grad vs a hand-rolled torch.autograd oracle on the
    #     same math. {gemma2,gemma3} × {full,sliding} × tier{0..3} =
    #     16 cases — cos > 0.998, sign-match ≥ 0.95, rel-L2 ≤ 8e-2.
    #
    #   * Full-model fwd (tests/test_gemma3_full_forward_parity.py):
    #     1B/4B/12B vs HF transformers — cos > 0.9998 on final-layer
    #     hidden state and on logits.
    #
    #   * Engine fwd+bwd (tests/test_engine_fwd_bwd_parity.py): one
    #     ``am.fwd_bwd`` step on a fixed prompt vs HF
    #     ``loss.backward()``, all per-block weight grads compared
    #     (cos > 0.99, rel-L2 < 20%); loss within 1%. 1B + 4B pass;
    #     12B skipped on 32 GiB cards (HF backward OOM).
    #
    #   * 5-step trajectory (tests/test_arch_parity.py): same five
    #     MathInstruct sequences on both stacks with bf16 AdamW.
    #     1B/2B/4B match within 3% relative on max |Δloss|; 12B LoRA
    #     drifts to ~10% by step 1.
    #
    # SKEPTICISM ABOUT 12B LORA TRAJECTORY (2026-05-11). The 12B drift
    # was bigger than the smaller models. We investigated:
    #
    #   (a) LoRA target module audit: flextrain LoRA-wraps the same 7
    #       projections per layer as HF PEFT (q,k,v,o + gate,up,down),
    #       182 = 26·7 adapters on 1B, identical counts on 12B. NO
    #       missing or mis-routed targets.
    #
    #   (b) lr=0 baseline (same 5 MathInstruct sequences, no optimizer
    #       updates): max |Δloss| = 0.017 over 5 steps for 12B LoRA —
    #       same order as the 1B baseline. This confirms FORWARD
    #       parity is tight on 12B; the 0.11 drift at lr=1e-4 is
    #       OPTIMIZER-SIDE bf16-noise-through-AdamW, not a routing
    #       or math bug. HF's gradient checkpointing (enabled for
    #       12B to fit 32 GiB) also slightly perturbs bwd op
    #       ordering vs flextrain's save-level tiering.
    #
    # If 12B LoRA trajectory ever regresses past these baselines,
    # re-run lr=0 + LoRA-audit first to triage: it's almost
    # certainly a forward-path bug if forward parity loosens, and a
    # routing bug if the LoRA target list changes.
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

        dy = dx.view(-1, d_model)

        # --- Outer FFN residual ---
        # post_ffn_norm.bwd reads ``dy`` (the incoming layer-output grad)
        # but doesn't mutate it; ffn.bwd takes the post-norm dx but
        # leaves ``dy`` alone. So we can hand ``dy`` straight to
        # pre_ffn_norm.bwd as the accumulator and avoid a clone (same
        # zero-clone pattern as LlamaBlock).
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

        # --- Outer attn residual (QK-norm path: recompute pre-attn-norm
        #     fwd output here so attn.bwd can run QK-norm bwd) ---
        pre_attn_norm_fwd_output = self.pre_attn_norm.fwd_from_rstd(
            slot.x_inp, weights, slot.pre_attn_norm_rstd,
        )
        # Same zero-clone reasoning: post_attn_norm.bwd reads dx_mid,
        # attn.bwd never touches it, so dx_mid can serve as the
        # pre_attn_norm.bwd accumulator directly.
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
            dx_accumulator=dx_mid,  # in-place
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
