"""Dense FFN blocks.

:class:`SwiGLUFFN` -- Llama / Qwen-dense / Mistral / OLMoE variants of the
dense feedforward block: ``W_2 @ (SiLU(W_1 @ x) * (W_3 @ x))``.

Activation fields
-----------------
* ``x1``  tier 3 -- the ``W_1`` (gate) projection output.
* ``x3``  tier 3 -- the ``W_3`` (up) projection output.

When saved at tier 3, backward skips re-projecting x1/x3 entirely. At
lower tiers we recompute them from the saved ``ffn_norm_rstd`` + the
tier-0 ``x_inp``.

Ports attention-adjacent pieces of ``orig/dense_layer.py:94-105`` (fwd)
and ``:193-226`` (bwd).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import ActivationField
from flextrain.core.layer import (
    ChunkMeta,
    ComputeCost,
    LayerContext,
    ParamSpec,
    TensorSpec,
)
from flextrain.ops import flextrain_swiglu_bwd, flextrain_swiglu_fwd


@dataclass(frozen=True)
class SwiGLUConfig:
    d_model: int
    expert_dim: int
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None


class SwiGLUFFN:
    """SwiGLU dense FFN: gate/up/down projections + Swish-gated activation."""

    def __init__(self, cfg: SwiGLUConfig) -> None:
        self.cfg = cfg

    def fields(self) -> tuple[ActivationField, ...]:
        cfg = self.cfg
        bf = cfg.compute_dtype
        return (
            ActivationField(
                "x1",
                lambda n, d: (n, cfg.expert_dim),
                bf,
                tier=3,
            ),
            ActivationField(
                "x3",
                lambda n, d: (n, cfg.expert_dim),
                bf,
                tier=3,
            ),
        )

    def param_spec(self) -> ParamSpec:
        cfg = self.cfg
        return ParamSpec(
            tensors=(
                TensorSpec(
                    "w_1",
                    lambda d: (cfg.d_model, cfg.expert_dim),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "w_2",
                    lambda d: (cfg.expert_dim, cfg.d_model),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "w_3",
                    lambda d: (cfg.d_model, cfg.expert_dim),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Compute.
    # ------------------------------------------------------------------

    def fwd(
        self,
        ffn_norm_output: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        attn_output_with_residual: torch.Tensor,
        out_tensor: torch.Tensor,
        slot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        """Compute ``layer_output = attn_output_with_residual + W_2 @ SwiGLU(W_1 @ h, W_3 @ h)``
        where ``h = ffn_norm_output``. Writes into ``slot.x1 / slot.x3``.

        ``out_tensor`` is the caller-owned residual buffer written by the
        final ``addmm`` (orig passes ``X`` back in to avoid a DtoD copy --
        see ``dense_layer.py:105``).
        """
        torch.matmul(ffn_norm_output, weights["w_1"], out=slot.x1)
        torch.matmul(ffn_norm_output, weights["w_3"], out=slot.x3)

        swiglu_scratch = ctx.scratch(slot.x1.shape, slot.x1.dtype)
        swiglu_result = flextrain_swiglu_fwd(slot.x1, slot.x3, out=swiglu_scratch)

        layer_output = torch.addmm(
            attn_output_with_residual,
            swiglu_result,
            weights["w_2"],
            out=out_tensor,
        )
        return layer_output

    def fwd_recompute_x1x3(
        self,
        ffn_norm_output: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
        *,
        recompute_x1: bool,
        recompute_x3: bool,
    ) -> None:
        """Recompute x1 / x3 during backward if tier < 3."""
        if recompute_x1:
            torch.matmul(ffn_norm_output, weights["w_1"], out=slot.x1)
        if recompute_x3:
            torch.matmul(ffn_norm_output, weights["w_3"], out=slot.x3)

    def bwd(
        self,
        dy_resid: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,
        *,
        skip_grads: frozenset[str] = frozenset(),
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """FFN backward. Accumulates ``g_1 / g_2 / g_3`` and returns
        ``dx_ffn_norm_up`` (gradient w.r.t. the FFN-norm output, so callers
        can pass it to RMSNorm backward).

        ``skip_grads`` can include ``"g_2"`` to gate the inline SwiGLU
        down-projection Wgrad addmm (LoRA fast path). ``g_1/g_3`` skip
        is in :meth:`bwd_accumulate_w1_w3_grads`.

        Mirrors ``orig/dense_layer.py:193-226``.
        """
        # 1. dx_up_act = dy @ w_2^T
        dx_up_act = torch.matmul(dy_resid, weights["w_2"].T)

        # 2. SwiGLU bwd -- also returns the recomputed SwiGLU forward output
        #    (the matmul input for w_2's grad).
        dx1_up, dx3_up, fwd_act_swiglu = flextrain_swiglu_bwd(
            slot.x1, slot.x3, dx_up_act, store_activations=True
        )

        # 3. g_2 += SwiGLU(x1, x3)^T @ dy
        if "g_2" in skip_grads:
            if capture_xy is not None:
                # Clone fwd_act_swiglu and dy_resid: both might be reused or
                # mutated downstream. fwd_act_swiglu lives only inside this
                # function's scope (built fresh each call), but dy_resid is
                # the layer's residual upstream and gets mutated by
                # ffn_norm.bwd's dx_accumulator path -- without cloning the
                # captured tensor would track that mutation.
                capture_xy["g_2"] = (fwd_act_swiglu.clone(), dy_resid.clone())
        else:
            torch.addmm(
                grads["g_2"], fwd_act_swiglu.T, dy_resid,
                alpha=1.0, beta=1.0, out=grads["g_2"],
            )

        # 4. dx_ffn_norm_up = dx1_up @ w_1^T + dx3_up @ w_3^T
        dx_ffn_norm_up = torch.matmul(dx1_up, weights["w_1"].T)
        torch.addmm(
            dx_ffn_norm_up, dx3_up, weights["w_3"].T,
            alpha=1.0, beta=1.0, out=dx_ffn_norm_up,
        )

        # Stash for the post-RMSNorm weight-grad accumulations.
        slot.aux["bwd_dx1_up"] = dx1_up
        slot.aux["bwd_dx3_up"] = dx3_up

        return dx_ffn_norm_up

    def bwd_accumulate_w1_w3_grads(
        self,
        ffn_norm_output: torch.Tensor,
        grads: MutableMapping[str, torch.Tensor],
        slot,
        *,
        skip_grads: frozenset[str] = frozenset(),
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        """Weight grads for w_1 and w_3 after RMSNorm bwd (which needs the
        recomputed norm output as its left operand).

        ``skip_grads`` / ``capture_xy``: see
        :meth:`GQAAttentionBlock.bwd_accumulate_qkv_grads`. Same LoRA
        fast-path contract -- skipped names get their ``(X, dY)``
        handed back via ``capture_xy`` instead of an addmm.

        Mirrors ``orig/dense_layer.py:221-222``.
        """
        for name, dy_key in (
            ("g_1", "bwd_dx1_up"),
            ("g_3", "bwd_dx3_up"),
        ):
            dy = slot.aux[dy_key]
            if name in skip_grads:
                if capture_xy is not None:
                    capture_xy[name] = (ffn_norm_output, dy)
            else:
                torch.addmm(
                    grads[name], ffn_norm_output.T, dy,
                    alpha=1.0, beta=1.0, out=grads[name],
                )
        del slot.aux["bwd_dx1_up"]
        del slot.aux["bwd_dx3_up"]

    # ------------------------------------------------------------------
    # FLOP accounting -- mirrors ``orig/dense_layer.py:1057-1075``.
    # ------------------------------------------------------------------

    def compute_cost(
        self, chunk: ChunkMeta, max_tier: int
    ) -> ComputeCost:
        cfg = self.cfg
        avoided = [0] * (max_tier + 1)
        total = 0

        for seq_len in chunk.seq_lens_host:
            # Up/gate projections -- tier 3 avoids recompute.
            up_gate = 2 * seq_len * cfg.d_model * cfg.expert_dim * 2
            total += up_gate
            if max_tier >= 3:
                avoided[3] += up_gate

            # Down projection -- never saved, always "free" (no recompute
            # value since it's only a single matmul in backward).
            down = 2 * seq_len * cfg.expert_dim * cfg.d_model
            total += down

        return ComputeCost(
            total_fwd_flops=total,
            avoided_recompute_flops=tuple(avoided),
        )
