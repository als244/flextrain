"""LlamaBlock: one Llama-family transformer block.

Covers: Llama 2, 3, 3.1, 3.2, 3.3 (+ the RoPE-base difference between
Llama2 and Llama3+). Composes algorithmic blocks::

    RMSNormBlock (attn_norm)
    GQAAttentionBlock                <-- full causal context
    RMSNormBlock (ffn_norm)
    SwiGLUFFN

Model-family classes in :mod:`flextrain.nn.layers` are named after the
released model; they compose algorithmic blocks (from
:mod:`flextrain.nn.blocks`) in a specific combination that matches what
the published architecture actually does.

See :mod:`flextrain.nn.layers.mistral` for the sliding-window sibling
(same composition, swap attention block).
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
    BackwardIntermediates,
    ChunkMeta,
    ComputeCost,
    LayerContext,
    ParamSpec,
)
from flextrain.nn.blocks import (
    GQAAttentionBlock,
    GQAAttentionConfig,
    RMSNormBlock,
    SwiGLUConfig,
    SwiGLUFFN,
)


@dataclass(frozen=True)
class LlamaBlockConfig:
    """Per-layer config for a Llama-family layer.

    Full-context GQA (no window). Use
    :class:`~flextrain.nn.layers.mistral.MistralBlockConfig` for the
    sliding-window sibling.
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int  # FFN intermediate dim
    rms_norm_eps: float = 1e-5
    rope_base: float = 500000.0  # Llama3 default; Llama2 uses 10000.0
    # Optional RoPE frequency scaling — set to the HF config's
    # ``rope_scaling`` dict (e.g. ``{"rope_type": "llama3", "factor": 8.0,
    # "low_freq_factor": 1.0, "high_freq_factor": 4.0,
    # "original_max_position_embeddings": 8192}``) for Llama-3.1+.
    # ``None`` (default) means vanilla RoPE.
    rope_scaling: object | None = None
    is_causal: bool = True

    # Per-role dtypes (default bf16 compute + bf16 master, matching orig).
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None
    norm_grad_dtype: torch.dtype = torch.float32  # flextrain_rmsnorm_bwd requires fp32
    # RMSNorm weight vectors are tiny (d_model per norm) so fp32 master
    # costs essentially nothing but avoids update-rounding-to-zero at
    # small lrs. Default to fp32 even when the rest of the layer uses
    # bf16 master weights.
    norm_master_dtype: torch.dtype = torch.float32

    def dims(self) -> dict[str, int]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "expert_dim": self.expert_dim,
        }


class LlamaBlock:
    """One Llama-family transformer layer.

    Activation schema (max_tier=3)
    -------------------------------
    Tier 0  attn_norm_rstd, ffn_norm_rstd, x_inp, xk, xv
    Tier 1  attn_result, softmax_lse
    Tier 2  xq, xo
    Tier 3  x1, x3
    """

    def __init__(
        self,
        layer_id: int,
        cfg: LlamaBlockConfig,
    ) -> None:
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
        self.attn = GQAAttentionBlock(
            GQAAttentionConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                rope_base=cfg.rope_base,
                rope_scaling=cfg.rope_scaling,
                is_causal=cfg.is_causal,
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

        # Insert x_inp (the tier-0 residual-stream save) right after
        # attn_norm_rstd so field ordering matches orig's slot layout.
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
        slot: ActivationSlot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        """Mirror ``orig/dense_layer.py:23-110`` in block-composed form."""
        # Save tier-0 input (the ``act_slot["x_inp"].copy_(X)`` line at orig:39).
        slot.x_inp.copy_(x)

        # Scratch for norm outputs -- both are immediately consumed by the
        # following matmuls, so they don't need slot storage.
        x_temp = ctx.scratch(x.shape, x.dtype)

        # attn-norm -> Q/K/V -> RoPE -> flash-attn -> fused O-proj + residual
        attn_norm_output = self.attn_norm.fwd(
            x, weights, slot.attn_norm_rstd, output=x_temp
        )
        attn_output_with_residual = self.attn.fwd(
            x, attn_norm_output, chunk, weights, slot, ctx
        )

        # ffn-norm -> x1/x3 -> SwiGLU -> W_2 (addmm into x)
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
        slot: ActivationSlot,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> None:
        """Fill in fields that weren't saved at this slot's ``level``.

        Mirrors ``orig/dense_layer.py:115-180``.
        """
        cfg = self.cfg
        x_inp = slot.x_inp

        # Tier < 2: recompute Q (and re-RoPE) from attn_norm_output.
        if not slot.has("xq"):
            attn_norm_output = self.attn_norm.fwd_from_rstd(
                x_inp, weights, slot.attn_norm_rstd
            )
            self.attn.fwd_recompute_qo(
                attn_norm_output, chunk, weights, slot, x_inp
            )
            slot.aux["recompute_attn_norm_output"] = attn_norm_output

        # Tier < 1: recompute attention (attn_result + softmax_lse).
        if not slot.has("attn_result"):
            self.attn.fwd_recompute_attn(chunk, slot, ctx)

        # Tier < 2: recompute xo (fused O-proj + residual addmm).
        if not slot.has("xo"):
            self.attn.fwd_recompute_o(x_inp, weights, slot)

        # Tier < 3: recompute x1 / x3 from ffn_norm_output.
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
        slot: ActivationSlot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        """Mirror ``orig/dense_layer.py:183-330``.

        Implemented as a delegating shim over :meth:`backward_dgrad` +
        :meth:`backward_wgrad` -- same FLOPs, same kernels, same order.
        The split is what enables LoRA's fast-path (skip the per-
        projection Wgrad matmul on frozen base weights). See
        ``docs/lora_fast_backward.md``.
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
        """First half of backward: produce ``upstream_dx`` and stash the
        recomputed RMSNorm outputs into the intermediates payload.

        ``skip_target_names`` is honored for ALL 7 LoRA-targetable
        projections. Inline cases (``g_o`` in ``attn.bwd``, ``g_2`` in
        ``ffn.bwd``) honor it here in dgrad; the deferred cases
        (``g_q, g_k, g_v, g_1, g_3``) honor it in :meth:`backward_wgrad`.
        Captured ``(X, dY)`` for skipped projections is written to
        ``intermediates.proj_inputs_and_grads`` keyed by the ``w_*``
        target name.
        """
        cfg = self.cfg

        # Translate w_* skip target names -> g_* skip names for the
        # inline-Wgrad addmms here (g_o, g_2 only).
        skip_g_inline: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_o", "w_2")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_inline else None
        )

        # --- FFN backward (dgrad path; accumulates g_2 unless skipped) ---
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
            weights,
            grads,
            slot.ffn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=ffn_norm_fwd_output_hint is None,
            recomputed_output_tensor=None,
        )
        if ffn_norm_fwd_output_hint is not None:
            ffn_norm_fwd_output = ffn_norm_fwd_output_hint

        # --- Attention backward (dgrad path; accumulates g_o unless skipped) ---
        dx_attn_norm_up = self.attn.bwd(
            dx, chunk, weights, grads, slot, ctx,
            attn_norm_output=None,  # type: ignore[arg-type]
            skip_grads=skip_g_inline, capture_xy=capture_xy,
        )

        attn_norm_fwd_output_hint = slot.aux.pop(
            "recompute_attn_norm_output", None
        )
        dx, attn_norm_fwd_output = self.attn_norm.bwd(
            dx_attn_norm_up,
            slot.x_inp,
            weights,
            grads,
            slot.attn_norm_rstd,
            dx_accumulator=dx,
            recompute_output=attn_norm_fwd_output_hint is None,
            recomputed_output_tensor=None,
        )
        if attn_norm_fwd_output_hint is not None:
            attn_norm_fwd_output = attn_norm_fwd_output_hint

        intermediates = BackwardIntermediates(
            proj_inputs_and_grads={},
            aux={
                "ffn_norm_fwd_output": ffn_norm_fwd_output,
                "attn_norm_fwd_output": attn_norm_fwd_output,
            },
        )
        # Inline-skipped (X, dY) go into intermediates so the wrapper
        # picks them up alongside the deferred ones.
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
        """Second half of backward: consume the recomputed RMSNorm
        outputs from ``intermediates.aux`` and run the deferred
        Wgrad matmuls (``g_1, g_3, g_q, g_k, g_v``).

        ``skip_target_names`` is a set of base parameter names (e.g.
        ``"w_q"``) whose Wgrad ``addmm`` should be skipped. For each
        skipped projection we instead populate
        ``intermediates.proj_inputs_and_grads[name] = (X, dY)`` so a
        LoRA wrapper can compute ``dA, dB`` via rank-r matmuls
        directly without ever materializing ``dW``. Default empty set
        means "compute every Wgrad" -- bit-identical to today's
        behavior.

        Note: only the deferred Wgrads (``g_1, g_3, g_q, g_k, g_v``)
        are skip-able in this Phase-2 version. The inline ``g_o`` and
        ``g_2`` addmms (computed inside ``attn.bwd`` / ``ffn.bwd``
        during ``backward_dgrad``) are not skip-able yet -- LoRA
        targeting ``w_o`` or ``w_2`` falls back to the slow path
        (materialize-then-decompose ``dW``). Skipping those would
        require modifying the block-level ``bwd`` itself; deferred to
        a follow-up.
        """
        del ctx  # not consumed by the deferred Wgrad addmms
        del weights  # unused here -- LoRA wrapper reads A/B itself

        ffn_norm_fwd_output = intermediates.aux["ffn_norm_fwd_output"]
        attn_norm_fwd_output = intermediates.aux["attn_norm_fwd_output"]

        # Translate w_* skip target names to g_* grad names. Only the
        # deferred Wgrads are gateable in Phase 2.
        skip_g_names: frozenset[str] = frozenset(
            f"g_{n[2:]}" for n in skip_target_names
            if n in ("w_q", "w_k", "w_v", "w_1", "w_3")
        )
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = (
            {} if skip_g_names else None
        )

        # FFN Wgrads that need ffn_norm_fwd_output as the left operand.
        self.ffn.bwd_accumulate_w1_w3_grads(
            ffn_norm_fwd_output, grads, slot,
            skip_grads=skip_g_names, capture_xy=capture_xy,
        )

        # Attention Wgrads that need attn_norm_fwd_output as the left operand.
        self.attn.bwd_accumulate_qkv_grads(
            attn_norm_fwd_output, grads, slot,
            skip_grads=skip_g_names, capture_xy=capture_xy,
        )

        # Hand captured (X, dY) pairs back to the wrapper via
        # intermediates.proj_inputs_and_grads, keyed by the w_* name
        # the wrapper expects.
        if capture_xy is not None:
            for g_name, xy in capture_xy.items():
                w_name = "w_" + g_name[2:]
                intermediates.proj_inputs_and_grads[w_name] = xy

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        max_tier = self.schema.max_tier
        return ComputeCost.sum(
            [
                self.attn_norm.compute_cost(
                    num_tokens=chunk.total_q,
                    dims=self._dims,
                    max_tier=max_tier,
                ),
                self.attn.compute_cost(chunk, max_tier=max_tier),
                self.ffn_norm.compute_cost(
                    num_tokens=chunk.total_q,
                    dims=self._dims,
                    max_tier=max_tier,
                ),
                self.ffn.compute_cost(chunk, max_tier=max_tier),
            ],
            max_tier=max_tier,
        )
