"""LMHead: the ``OutputLayer`` for Transformer architectures.

Ports ``orig/awsm_transformer/head.py`` onto the :class:`OutputLayer`
Protocol from :mod:`flextrain.core.layer`, with a pluggable
:class:`LossFn` so the same head can drive SFT (cross-entropy),
RL (GRPO/PPO/DPO), distillation (MSE), etc.

Fused forward + loss + backward (memory contract)
-------------------------------------------------
The single biggest VRAM cost in a naive head implementation is the
full ``(num_tokens, vocab)`` logits tensor. AdaWS's whole point is
tight memory budgets, so we never materialize it.

Instead we micro-chunk along the token axis: for each slice of
``head_chunk_size`` tokens we compute RMSNorm-fwd → head-proj matmul →
loss (softmax + objective-specific) → weight grads → dX → RMSNorm-bwd
all inside one inner iteration. The ``(T', V)`` logits buffer (and
``probs``) lives only for the span of one iteration and gets freed
before the next; peak logits VRAM is therefore bounded by
``head_chunk_size * vocab_size * dtype_size`` (default
``1024 * V * 2`` bytes).

No tensor of size ``(total_chunk_tokens, V)`` is ever materialized.
This is a hard invariant — breaking it defeats the system's memory
model.

Pluggable loss
--------------
The inner loop calls ``loss_fn.compute(logits, token_slice, ...)``,
which returns ``dZ`` in the same shape as ``logits`` (ideally in
place). The head then runs the standard backward matmuls
unchanged. This is why separating loss from head works: every loss
objective we care about (CE, GRPO, DPO, MSE, ...) ultimately
produces a ``(T', V)`` grad w.r.t. logits, and the head doesn't care
how it was computed.

See :mod:`flextrain.nn.loss` for built-in loss fns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import ActivationSchema
from flextrain.core.layer import (
    ChunkMeta,
    ComputeCost,
    LayerContext,
    LossStats,
    ParamSpec,
    TensorSpec,
)
from flextrain.ops import flextrain_rmsnorm_bwd, flextrain_rmsnorm_fwd

from .loss import CrossEntropyLoss, LossFn, TokenContext


@dataclass(frozen=True)
class LMHeadConfig:
    """LMHead config.

    Covers Llama / Qwen / Mistral / OLMoE (all use final RMSNorm + a
    linear LM-head). Tied embeddings (``lm_head.weight ==
    embed_tokens.weight``) are a future extension — when needed, add a
    ``tied: bool`` and branch inside __init__.
    """

    d_model: int
    vocab_size: int
    rms_norm_eps: float = 1e-5
    # Micro-chunk size along the token axis. Bounds the peak (T', V)
    # logits buffer. Orig default = 1024 (see head.py:110).
    head_chunk_size: int = 1024

    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None  # defaults to compute_dtype
    grad_dtype: torch.dtype | None = None  # defaults to compute_dtype
    # flextrain_rmsnorm_bwd atomic-adds into dW in fp32; match in the spec.
    norm_grad_dtype: torch.dtype = torch.float32
    # fp32 master for the final_norm weight: tiny cost, avoids rounding.
    norm_master_dtype: torch.dtype = torch.float32


class LMHead:
    """LM-head :class:`OutputLayer`.

    Runs RMSNorm → linear head → loss → backward micro-chunked along
    the token axis. The loss function is supplied per-call via
    :meth:`forward_backward`, keeping the head arch-generic and the
    loss objective-generic.

    Attributes
    ----------
    schema
        Empty :class:`ActivationSchema` (max_tier=0). The head owns no
        per-chunk activation slot — all intermediates live and die in
        the inner loop. See [DECISION 11] in docs/NOTES.md.
    param_spec
        Two tensors: ``w_final_norm (d_model,)`` and ``w_head_proj
        (d_model, vocab_size)``.
    """

    # Distinct from embed's -1 for debug prints; engine doesn't use.
    layer_id: int = -2

    def __init__(self, cfg: LMHeadConfig) -> None:
        self.cfg = cfg
        self.schema = ActivationSchema(fields=(), max_tier=0)

        def _norm_shape(dims: Mapping[str, int]) -> tuple[int, ...]:
            return (dims.get("d_model", cfg.d_model),)

        def _head_shape(dims: Mapping[str, int]) -> tuple[int, ...]:
            return (
                dims.get("d_model", cfg.d_model),
                dims.get("vocab_size", cfg.vocab_size),
            )

        self.param_spec = ParamSpec(
            tensors=(
                TensorSpec(
                    name="w_final_norm",
                    shape_fn=_norm_shape,
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.norm_master_dtype,
                    grad_dtype=cfg.norm_grad_dtype,
                ),
                TensorSpec(
                    name="w_head_proj",
                    shape_fn=_head_shape,
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
            )
        )

    # ------------------------------------------------------------------
    # OutputLayer Protocol
    # ------------------------------------------------------------------

    def forward_backward(
        self,
        x: torch.Tensor,
        token_ctx: TokenContext,
        chunk: ChunkMeta,  # noqa: ARG002  — kept for Protocol symmetry
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        ctx: LayerContext,  # noqa: ARG002  — kept for Protocol symmetry
        *,
        loss_scale: float = 1.0,
        loss_fn: LossFn | None = None,
    ) -> tuple[torch.Tensor, LossStats]:
        """One training-chunk fused forward + loss + backward.

        Parameters
        ----------
        x
            ``(num_tokens, d_model)`` residual-stream input from the last
            backbone layer. Overwritten in place with ``dx`` as we walk
            micro-chunks (matches orig's head.py:169 contract).
        token_ctx
            :class:`TokenContext` holding per-token signals the loss fn
            consumes (``labels``, ``advantages``, ``ref_logprobs``,
            ``extra``). Length must match ``num_tokens``.
        weights, grads
            The head's two-tensor param + grad buffers. Grads accumulate
            in place across micro-chunks (weight-grad addmm with
            ``beta=1.0``).
        loss_scale
            Folded into ``dZ`` inside the loss fn, which multiplies it
            in so the head's weight-grad matmul can use
            ``alpha=1.0`` with no double scaling. Orig's convention:
            pass ``1.0 / total_tokens_per_step`` to get "sum ≡ mean"
            gradients across the whole optimization step.
        loss_fn
            Pluggable :class:`LossFn`. Defaults to
            :class:`CrossEntropyLoss` for backward-compat with the SFT
            use case. RL and distillation callers pass their own.

        Returns
        -------
        (dx, stats)
            ``dx is x`` (aliased -- orig semantics). ``stats`` is a
            :class:`LossStats` with per-token loss tensors and any
            per-micro-chunk diagnostics the loss returned concatenated
            along the token axis.

        Memory invariant (do not break)
        -------------------------------
        Peak extra VRAM here is ``head_chunk_size * vocab_size *
        dtype_size`` for the logits buffer (+ the same for probs inside
        CE). A full ``(num_tokens, vocab)`` logits tensor is NEVER
        materialized.
        """
        if loss_fn is None:
            loss_fn = CrossEntropyLoss()

        num_tokens, d_model = x.shape
        assert d_model == self.cfg.d_model, (
            f"x has d_model={d_model}, head expects {self.cfg.d_model}"
        )
        if token_ctx.total_tokens != num_tokens:
            raise ValueError(
                f"token_ctx covers {token_ctx.total_tokens} tokens, "
                f"x has {num_tokens}"
            )

        device = x.device
        dtype = x.dtype
        cfg = self.cfg
        head_chunk_size = cfg.head_chunk_size

        # Per-chunk output tensors (engine reads via LossStats).
        per_token_loss = torch.empty(
            num_tokens, dtype=torch.float32, device=device
        )
        # Populated ONLY if the chosen loss fn reports these in aux.
        # Pre-allocated so the engine can always read them.
        next_prediction = torch.empty(
            num_tokens, dtype=torch.int64, device=device
        )
        next_prediction_prob = torch.empty(
            num_tokens, dtype=torch.float32, device=device
        )
        any_next_prediction = False

        w_final_norm = weights["w_final_norm"]
        w_head_proj = weights["w_head_proj"]
        # Frozen-aware: under LoRA the BufferManager skips grad
        # allocation for frozen tensors. ``None`` means "skip the
        # wgrad accumulate" (forward + dx still run normally).
        g_final_norm = grads.get("g_final_norm")
        g_head_proj = grads.get("g_head_proj")

        # For extras (e.g. KL for RL), collect per-chunk and let the
        # engine / caller concatenate. We don't want to force a schema
        # on arbitrary loss-aux keys, so we keep them as a list and
        # let the user reduce.
        aux_chunks: list[dict] = []

        offset = 0
        while offset < num_tokens:
            t_prime = min(head_chunk_size, num_tokens - offset)
            x_slice = x[offset : offset + t_prime]

            # ---- fwd: RMSNorm ----
            head_proj_in, final_norm_rstd = flextrain_rmsnorm_fwd(
                x_slice, W=w_final_norm, rms_norm_eps=cfg.rms_norm_eps
            )

            # ---- fwd: head projection matmul -> (T', V) logits ----
            logits = torch.mm(head_proj_in, w_head_proj)

            # ---- loss ----
            token_slice = token_ctx.slice_for(offset, t_prime)
            dZ, aux = loss_fn.compute(
                logits,
                token_slice,
                loss_scale=loss_scale,
                per_token_loss_out=per_token_loss[offset : offset + t_prime],
            )
            # logits buffer may be the same as dZ (CE reuses it). Either
            # way, drop the local name so it can be freed next iter.
            del logits

            # Populate the built-in LossStats diagnostic slots if the
            # loss fn reported them. Losses that don't compute argmax
            # can omit; we leave those slots uninitialized (documented).
            if "next_prediction" in aux:
                next_prediction[offset : offset + t_prime].copy_(
                    aux.pop("next_prediction")
                )
                next_prediction_prob[offset : offset + t_prime].copy_(
                    aux.pop("next_prediction_prob")
                )
                any_next_prediction = True
            if aux:
                aux_chunks.append(aux)

            # ---- bwd: accumulate w_head_proj grad ----
            #   g_head_proj += head_proj_in.T @ dZ   (loss_scale already
            #   folded into dZ by the loss fn, so alpha=1.0 here)
            if g_head_proj is not None:
                torch.addmm(
                    g_head_proj,
                    head_proj_in.T,
                    dZ,
                    alpha=1.0,
                    beta=1.0,
                    out=g_head_proj,
                )

            # ---- bwd: dX_head_in = dZ @ w_head_proj.T ----
            dX_head_in = torch.empty(
                head_proj_in.shape, dtype=dtype, device=device
            )
            del head_proj_in
            torch.addmm(
                dX_head_in,
                dZ,
                w_head_proj.T,
                alpha=1.0,
                beta=0.0,
                out=dX_head_in,
            )
            del dZ  # if dZ aliased logits, the buffer is now free

            # ---- bwd: RMSNorm bwd (dW accumulated in g_final_norm) ----
            # ``dW=None`` tells flextrain_rmsnorm_bwd to skip the
            # wgrad accumulate (frozen final-norm).
            dX_slice, _dW_norm, _ = flextrain_rmsnorm_bwd(
                dX_head_in,
                x_slice,
                w_final_norm,
                final_norm_rstd,
                dW=g_final_norm,
            )
            del dX_head_in

            # Copy dX back into x (orig contract; head.py:169).
            x_slice.copy_(dX_slice)
            del dX_slice

            offset += t_prime

        stats = LossStats(
            per_token_loss=per_token_loss,
            next_prediction=next_prediction if any_next_prediction else per_token_loss.new_empty(0, dtype=torch.int64),
            next_prediction_prob=next_prediction_prob if any_next_prediction else per_token_loss.new_empty(0),
            token_count=num_tokens,
        )
        # Stash per-micro-chunk aux for callers that want them.
        # The engine does not consume this; RL callers will.
        if aux_chunks:
            stats.aux_chunks = aux_chunks  # type: ignore[attr-defined]
        return x, stats

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        """FLOP estimate for the DP solver.

        Dominant term is the head projection matmul (fwd = T·d·V,
        bwd = 2·T·d·V for the weight grad and dX matmuls). Softmax /
        loss / RMSNorm are O(T·V) / O(T·d) -- negligible.
        Factor-of-2 for MAC → FLOP.

        No recompute tiers (schema is empty), so
        ``avoided_recompute_flops = (0,)``.
        """
        cfg = self.cfg
        T = chunk.total_q
        fwd = 2 * T * cfg.d_model * cfg.vocab_size
        bwd = 2 * fwd
        return ComputeCost(
            total_fwd_flops=fwd + bwd, avoided_recompute_flops=(0,)
        )
