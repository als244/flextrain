"""GQA attention block with in-place KV-cache ring.

Composable unit owning:
  * Q / K / V / O projections + gradients
  * activation fields at tiers 0..2 (``xk``, ``xv``, ``attn_result``,
    ``softmax_lse``, ``xq``, ``xo``)
  * RoPE application on Q and K before attention
  * flash-attn varlen fwd / bwd with causal + sliding-window support
  * fused O-projection with residual add (via the orig ``CublasLtDispatcher``
    shim -- folds ``X + attn_result @ W_O`` into one GEMM, preserving
    orig's kernel fusion)

This is a faithful port of the attention portion of
``orig/awsm_transformer/dense_layer.py:43-91`` (fwd),
``:183-330`` (bwd), and ``:115-180`` (forward_recompute) rolled into
standalone methods the :class:`~flextrain.nn.layers.llama.LlamaBlock` (and
its cousins) compose.

The block is DTYPE-AGNOSTIC at declaration time: callers pass compute /
master / grad dtypes via :class:`AttentionConfig`; defaults match orig
(bf16 everywhere, fp32 weight-grad accumulators for norm blocks -- but
attention grads are bf16).
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
from flextrain.ops import (
    flextrain_attention_bwd,
    flextrain_attention_fwd,
    dispatcher,
)

from .rope import apply_rope_bwd, apply_rope_fwd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GQAAttentionConfig:
    """Per-instance parameters for ``GQAAttentionBlock`` (full-context GQA).

    ``d_model`` / ``n_heads`` / ``n_kv_heads`` / ``head_dim`` define GQA shape.
    ``is_causal`` is the flash-attn causal flag (True for decoder-style
    models, False for encoders). ``rope_base`` is the RoPE theta (Llama2:
    10000.0, Llama3: 500000.0).

    Dtypes default to bf16 compute + bf16 master; override for mixed
    precision.

    Subclass :class:`GQASlidingWindowAttentionConfig` adds a
    ``window_size_left`` for sliding-window variants (Mistral, Gemma
    alternating, GPT-OSS subset).

    ``qk_norm`` (default False) toggles Qwen3-style per-head RMSNorm on Q
    and K between projection and RoPE. When True, the block instantiates
    its own per-head ``q_norm`` / ``k_norm`` :class:`RMSNormBlock` instances
    internally, contributing their fields/params/compute_cost into this
    block's. The weights dict passed to ``fwd``/``bwd`` must therefore
    include ``w_q_norm`` / ``w_k_norm``.

    ``qk_norm_master_dtype`` / ``qk_norm_grad_dtype`` control the QK-norm
    weight master and grad dtypes (default fp32 — RMSNorm weight vectors
    are tiny, so we keep them fp32 to avoid losing tiny SGD updates and
    to match the kernel's atomic-fp32 wgrad accumulator).
    """

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_base: float = 10000.0
    # Optional RoPE frequency scaling. Supported variants:
    #   * ``None`` — vanilla ``inv_freq[i] = rope_base ** (-2i/D)``.
    #   * ``{"rope_type": "llama3", "factor": <f>, "low_freq_factor": <l>,
    #      "high_freq_factor": <h>, "original_max_position_embeddings": <m>}``
    #      — Llama-3.1+ frequency-band scaling. Required for cross-stack
    #      parity on Llama-3.1 / 3.2 / 3.3 — without it the low-frequency
    #      RoPE bands diverge by up to factor× from HF and you get a
    #      systematic forward-pass bias.
    # Build the corresponding ``inv_freq`` array via
    # :func:`flextrain.nn.blocks.rope.build_rope_inv_freq`.
    rope_scaling: object | None = None
    is_causal: bool = True
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None
    qk_norm: bool = False
    rms_norm_eps: float = 1e-6  # Used only when qk_norm=True
    qk_norm_master_dtype: torch.dtype = torch.float32
    qk_norm_grad_dtype: torch.dtype = torch.float32
    # Default Qwen3-style PER-HEAD RMSNorm on Q/K (weight vector sized
    # head_dim, broadcast across heads, rstd shape (T, heads)). OLMoE
    # uses FULL-ROW RMSNorm (per_head=False; weight sized attn_dim/kv_dim,
    # rstd shape (T, 1)). Set to False for OLMoE-style.
    qk_norm_per_head: bool = True
    # Qwen2 family adds biases on Q/K/V projections (o_proj has no bias).
    # When True, w_q/w_k/w_v ParamSpec adds matching b_q/b_k/b_v tensors
    # with shape (attn_dim,) / (kv_dim,) / (kv_dim,).
    qkv_bias: bool = False
    # Gemma 2 / 3 attention logit softcap. ``tanh(scores / cap) * cap``
    # applied pre-softmax. 0.0 disables (default). Plumbed straight into
    # flash-attn's ``softcap`` argument.
    attn_logit_softcap: float = 0.0

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    # Base GQA is full-context (no window).
    @property
    def window_size_left(self) -> int:  # pragma: no cover -- constant
        return -1

    @property
    def window_size_right(self) -> int:  # pragma: no cover -- constant
        return 0


@dataclass(frozen=True)
class GQASlidingWindowAttentionConfig(GQAAttentionConfig):
    """Sliding-window GQA (Mistral, Gemma alternating, GPT-OSS).

    ``window_size_left`` is the number of prior tokens attended to
    (inclusive of the current position); ``window_size_right`` stays 0
    (causal) for all mainstream LLM architectures.

    Usage: ``GQASlidingWindowAttentionConfig(..., window_size_left=4096)``.
    """

    # Override the base properties with real dataclass fields.
    window_size_left: int = 4096  # type: ignore[assignment]
    window_size_right: int = 0  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.window_size_left < 0:
            raise ValueError(
                "GQASlidingWindowAttentionConfig requires "
                "window_size_left >= 0; use GQAAttentionConfig for full "
                "context."
            )


# Back-compat alias. Left here because the first port of the attention
# block named the config ``AttentionConfig``. New code should use
# :class:`GQAAttentionConfig`.
AttentionConfig = GQAAttentionConfig


class GQAAttentionBlock:
    """Grouped-query attention with RoPE + flash-attn varlen kernel.

    Algorithmic unit -- one of several attention variants:

    * :class:`GQAAttentionBlock`              -- full causal context
                                                 (Llama, Qwen-dense, OLMoE)
    * :class:`GQASlidingWindowAttentionBlock` -- sliding-window variant
                                                 (Mistral, Gemma-alt, GPT-OSS
                                                 subset)

    Future: ``LinearAttentionBlock`` (Qwen3-Next), ``MLAAttentionBlock``
    (DeepSeek-V3). All variants share the same activation schema shape so
    they are interchangeable from the scheduler's point of view.
    """

    def __init__(self, cfg: GQAAttentionConfig) -> None:
        self.cfg = cfg
        self._rope_inv_freq_cache: torch.Tensor | None = None
        # Qwen3-style per-head QK-norm. Owned internally so the modeling
        # layer doesn't need to wire up parallel RMSNormBlocks. Fields,
        # params, and compute_cost roll up via this block's methods.
        if cfg.qk_norm:
            from .norm import RMSNormBlock
            if cfg.qk_norm_per_head:
                # Qwen3-style: weight (head_dim,), rstd (T, heads).
                q_weight_dim, k_weight_dim = "head_dim", "head_dim"
            else:
                # OLMoE-style: weight (attn_dim,) / (kv_dim,), rstd (T, 1).
                q_weight_dim, k_weight_dim = "attn_dim", "kv_dim"
            self.q_norm = RMSNormBlock(
                prefix="q_norm",
                eps=cfg.rms_norm_eps,
                per_head=cfg.qk_norm_per_head,
                heads_dim_name="n_heads",
                weight_dim_name=q_weight_dim,
                param_compute_dtype=cfg.compute_dtype,
                param_master_dtype=cfg.qk_norm_master_dtype,
                param_grad_dtype=cfg.qk_norm_grad_dtype,
            )
            self.k_norm = RMSNormBlock(
                prefix="k_norm",
                eps=cfg.rms_norm_eps,
                per_head=cfg.qk_norm_per_head,
                heads_dim_name="n_kv_heads",
                weight_dim_name=k_weight_dim,
                param_compute_dtype=cfg.compute_dtype,
                param_master_dtype=cfg.qk_norm_master_dtype,
                param_grad_dtype=cfg.qk_norm_grad_dtype,
            )
        else:
            self.q_norm = None
            self.k_norm = None

    # ------------------------------------------------------------------
    # Declarations consumed by the layer / engine.
    # ------------------------------------------------------------------

    def fields(self) -> tuple[ActivationField, ...]:
        """Activation fields this block owns.

        Tiers
        -----
        * ``xk``           tier 0 (always saved; local per-chunk K projection)
        * ``xv``           tier 0
        * ``attn_result``  tier 1
        * ``softmax_lse``  tier 1 (token_axis=1, shape ``(n_heads, T)``)
        * ``xq``           tier 2
        * ``xo``           tier 2

        When ``cfg.qk_norm=True``, the per-head q_norm/k_norm rstd fields
        are appended as well (tier 0 each).
        """
        cfg = self.cfg
        bf = cfg.compute_dtype
        out = (
            ActivationField(
                "xk",
                lambda n, d: (n, cfg.n_kv_heads, cfg.head_dim),
                bf,
                tier=0,
            ),
            ActivationField(
                "xv",
                lambda n, d: (n, cfg.n_kv_heads, cfg.head_dim),
                bf,
                tier=0,
            ),
            ActivationField(
                "attn_result",
                lambda n, d: (n, cfg.n_heads, cfg.head_dim),
                bf,
                tier=1,
            ),
            ActivationField(
                "softmax_lse",
                lambda n, d: (cfg.n_heads, n),
                torch.float32,
                tier=1,
                token_axis=1,
            ),
            ActivationField(
                "xq",
                lambda n, d: (n, cfg.n_heads, cfg.head_dim),
                bf,
                tier=2,
            ),
            ActivationField(
                "xo",
                lambda n, d: (n, cfg.d_model),
                bf,
                tier=2,
            ),
        )
        if cfg.qk_norm:
            out = out + self.q_norm.fields() + self.k_norm.fields()
        return out

    def param_spec(self) -> ParamSpec:
        """``w_q``, ``w_k``, ``w_v``, ``w_o`` as ``(in, out)`` matrices
        (transposed from HF's ``(out, in)``; :mod:`flextrain.io.arch.llama`
        applies the transpose on load / export).

        When ``cfg.qkv_bias=True`` (Qwen2 family), also declares
        ``b_q``, ``b_k``, ``b_v`` bias vectors. ``w_o`` has no bias in
        any of the architectures we target.
        """
        cfg = self.cfg
        tensors: list[TensorSpec] = [
            TensorSpec(
                "w_q",
                lambda d: (cfg.d_model, cfg.attn_dim),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            TensorSpec(
                "w_k",
                lambda d: (cfg.d_model, cfg.kv_dim),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            TensorSpec(
                "w_v",
                lambda d: (cfg.d_model, cfg.kv_dim),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            TensorSpec(
                "w_o",
                lambda d: (cfg.attn_dim, cfg.d_model),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
        ]
        if cfg.qkv_bias:
            tensors.extend([
                TensorSpec(
                    "b_q",
                    lambda d: (cfg.attn_dim,),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "b_k",
                    lambda d: (cfg.kv_dim,),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "b_v",
                    lambda d: (cfg.kv_dim,),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
            ])
        spec = ParamSpec(tensors=tuple(tensors))
        if cfg.qk_norm:
            spec = ParamSpec.merge([
                spec,
                self.q_norm.param_spec(),
                self.k_norm.param_spec(),
            ])
        return spec

    # ------------------------------------------------------------------
    # Compute. Called by the enclosing layer's forward / backward.
    # ------------------------------------------------------------------

    def _rope_theta(self, device: torch.device) -> torch.Tensor:
        """Return the precomputed ``inv_freq`` array (length head_dim/2,
        fp32) for this attention's RoPE config. Built lazily and cached
        per device. Method name kept for back-compat with call sites."""
        if (
            self._rope_inv_freq_cache is None
            or self._rope_inv_freq_cache.device != device
        ):
            from .rope import build_rope_inv_freq
            inv_freq_cpu = build_rope_inv_freq(
                head_dim=self.cfg.head_dim,
                rope_base=self.cfg.rope_base,
                rope_scaling=self.cfg.rope_scaling,
            )
            self._rope_inv_freq_cache = inv_freq_cpu.to(
                device=device, dtype=torch.float32,
            )
        return self._rope_inv_freq_cache

    def fwd(
        self,
        x_resid: torch.Tensor,
        attn_norm_output: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        slot,  # ActivationSlot; type avoided for circular import
        ctx: LayerContext,
    ) -> torch.Tensor:
        """Full attention-block forward.

        Writes into ``slot.xk``, ``slot.xv``, ``slot.xq``, ``slot.attn_result``,
        ``slot.softmax_lse``, ``slot.xo``. Updates the KV context window via
        ``ctx.kv_cache``. Returns the residual-added output
        ``X + attn_result @ W_O`` (written into ``slot.xo``).

        Parameters
        ----------
        x_resid
            The residual-stream input ``(T, d_model)``. Also used as the
            ``C`` operand in the fused O-proj + residual GEMM.
        attn_norm_output
            The RMSNorm output -- left operand of Q/K/V projections. Separate
            argument (not read from ``slot``) because orig writes it into a
            scratch ``x_temp`` that is NOT saved at any tier.
        """
        cfg = self.cfg
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        total_q = chunk.total_q
        total_k = chunk.total_k

        # Q / K / V projections (into the slot's preallocated views).
        torch.matmul(
            attn_norm_output, weights["w_q"],
            out=slot.xq.view(-1, cfg.attn_dim),
        )
        torch.matmul(
            attn_norm_output, weights["w_k"],
            out=slot.xk.view(-1, cfg.kv_dim),
        )
        torch.matmul(
            attn_norm_output, weights["w_v"],
            out=slot.xv.view(-1, cfg.kv_dim),
        )

        # Qwen2-style Q/K/V biases (added post-projection, pre-norm/RoPE).
        if cfg.qkv_bias:
            slot.xq.view(-1, cfg.attn_dim).add_(weights["b_q"])
            slot.xk.view(-1, cfg.kv_dim).add_(weights["b_k"])
            slot.xv.view(-1, cfg.kv_dim).add_(weights["b_v"])

        # Qwen3-style QK-norm: per-head RMSNorm on Q and K between
        # projection and RoPE. In-place rewrites of slot.xq / slot.xk.
        # The flextrain_rmsnorm_fwd kernel expects 2D (T, heads*head_dim)
        # with head_dim passed via weight shape; pass the 2D view.
        if cfg.qk_norm:
            xq2d = slot.xq.view(-1, cfg.attn_dim)
            xk2d = slot.xk.view(-1, cfg.kv_dim)
            self.q_norm.fwd(
                xq2d, weights, getattr(slot, self.q_norm.rstd_name),
                output=xq2d,
            )
            self.k_norm.fwd(
                xk2d, weights, getattr(slot, self.k_norm.rstd_name),
                output=xk2d,
            )

        # RoPE (in-place on xq / xk)
        rope_theta = self._rope_theta(x_resid.device)
        apply_rope_fwd([slot.xq, slot.xk], chunk.seq_positions, rope_theta)

        # KV cache write
        kv = ctx.kv_cache
        start = (
            chunk.prior_seq_offsets_host[0] + chunk.prior_seq_lens_host[0]
        )
        kv.k[start : start + total_q, :].copy_(slot.xk)
        kv.v[start : start + total_q, :].copy_(slot.xv)

        # Flash-attn varlen fwd
        flextrain_attention_fwd(
            slot.xq.view(-1, n_heads, head_dim),
            kv.k[:total_k, :],
            kv.v[:total_k, :],
            slot.attn_result,
            slot.softmax_lse,
            chunk.q_seq_offsets,
            chunk.k_seq_offsets,
            chunk.q_seq_lens,
            chunk.k_seq_lens,
            chunk.max_seqlen_q,
            chunk.max_seqlen_k,
            causal=cfg.is_causal,
            window_size=(cfg.window_size_left, cfg.window_size_right),
            softcap=cfg.attn_logit_softcap,
        )

        # Fused O-projection + residual (orig's dispatcher path): replaces
        # two separate matmul + add kernels with a single cublasLt call.
        # See ``orig/awsm_transformer/dense_layer.py:91``.
        stream_ptr = torch.cuda.current_stream().cuda_stream
        return dispatcher.matmul(
            stream_ptr,
            A=slot.attn_result.view(-1, cfg.attn_dim),
            B=weights["w_o"],
            C=x_resid,
            D=slot.xo,
            alpha=1.0, beta=1.0,
        )

    def fwd_recompute_qo(
        self,
        attn_norm_output: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        slot,
        x_resid: torch.Tensor,
    ) -> torch.Tensor:
        """Tier<2 recompute path: Q was not saved, XO was not saved, so we
        recompute both from ``attn_norm_output`` and fold the residual back
        into XO via the dispatcher fused GEMM. Also recomputes attn_result
        if tier<1.

        Mirrors ``orig/dense_layer.py:131-160``.
        """
        cfg = self.cfg
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

        # Re-project Q and re-apply Q bias (Qwen2) + QK-norm (Qwen3) + RoPE
        # (K / V already saved at tier 0; K was saved post-norm+post-RoPE).
        torch.matmul(
            attn_norm_output, weights["w_q"], out=slot.xq.view(-1, cfg.attn_dim)
        )
        if cfg.qkv_bias:
            slot.xq.view(-1, cfg.attn_dim).add_(weights["b_q"])
        if cfg.qk_norm:
            # Reuse the saved rstd_q (tier 0 — always available). Pass 2D.
            xq2d = slot.xq.view(-1, cfg.attn_dim)
            self.q_norm.fwd_from_rstd(
                xq2d, weights, getattr(slot, self.q_norm.rstd_name),
                output=xq2d,
            )
        rope_theta = self._rope_theta(attn_norm_output.device)
        apply_rope_fwd([slot.xq], chunk.seq_positions, rope_theta)
        return slot.xq

    def fwd_recompute_attn(
        self,
        chunk: ChunkMeta,
        slot,
        ctx: LayerContext,
    ) -> None:
        """Tier<1 recompute: attn_result + softmax_lse from saved Q, K/V in
        kv cache. Mirrors ``orig/dense_layer.py:140-150``."""
        cfg = self.cfg
        n_heads, head_dim = cfg.n_heads, cfg.head_dim
        total_k = chunk.total_k
        kv = ctx.kv_cache
        flextrain_attention_fwd(
            slot.xq.view(-1, n_heads, head_dim),
            kv.k[:total_k, :],
            kv.v[:total_k, :],
            slot.attn_result,
            slot.softmax_lse,
            chunk.q_seq_offsets,
            chunk.k_seq_offsets,
            chunk.q_seq_lens,
            chunk.k_seq_lens,
            chunk.max_seqlen_q,
            chunk.max_seqlen_k,
            causal=cfg.is_causal,
            window_size=(cfg.window_size_left, cfg.window_size_right),
            softcap=cfg.attn_logit_softcap,
        )

    def fwd_recompute_o(
        self,
        x_resid: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
    ) -> torch.Tensor:
        """Tier<2 xo recompute: addmm from attn_result + residual."""
        cfg = self.cfg
        n_heads, head_dim = cfg.n_heads, cfg.head_dim
        return torch.addmm(
            x_resid,
            slot.attn_result.view(-1, cfg.attn_dim),
            weights["w_o"],
            out=slot.xo,
        )

    def bwd(
        self,
        dx_resid: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
        attn_norm_output: torch.Tensor,
        *,
        skip_grads: frozenset[str] = frozenset(),
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Attention backward. Accumulates ``g_o / g_q / g_k / g_v`` into
        ``grads`` and returns ``dx_attn_norm_up`` (gradient w.r.t. the input
        of attn-norm, so callers can pass it to RMSNorm backward).

        ``skip_grads`` can include ``"g_o"`` to gate the inline output-
        projection Wgrad addmm (LoRA fast path). ``g_q/g_k/g_v`` skip is
        handled in :meth:`bwd_accumulate_qkv_grads`.

        Does NOT accumulate into the residual stream -- the caller holds
        ``dx_resid`` and merges the attn-norm downstream gradient itself,
        matching ``orig/dense_layer.py:311-315``.
        """
        cfg = self.cfg
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        num_tokens = dx_resid.shape[0]
        total_k = chunk.total_k
        total_q = chunk.total_q

        attn_result = slot.attn_result.view(num_tokens, -1)

        # 1. g_o += attn_result^T @ dx_resid
        if "g_o" in skip_grads:
            if capture_xy is not None:
                # Clone dx_resid because the caller will overwrite it
                # downstream with attn-norm-upstream gradient and we want
                # the LoRA wrapper to see the value at this point.
                capture_xy["g_o"] = (attn_result, dx_resid.clone())
        else:
            torch.addmm(
                grads["g_o"], attn_result.T, dx_resid,
                alpha=1.0, beta=1.0, out=grads["g_o"],
            )

        # 2. dx_up_attn = dx_resid @ w_o^T
        dx_up_attn = torch.matmul(dx_resid, weights["w_o"].T)
        dx_up_attn = dx_up_attn.view(num_tokens, n_heads, head_dim)

        # 3. flash-attn bwd -- writes into dq and into the running dk/dv
        #    context ring.
        #
        # Cross-chunk dK/dV accumulation: ``flextrain_attention_bwd``
        # OVERWRITES the dk/dv tensors it's given (per the note in
        # ``flextrain/ops/_kernels/attention.py``). For multi-chunk
        # seqs where later-fwd chunks have ALREADY run their bwd in
        # the reverse traversal, those chunks accumulated cross-chunk
        # dK/dV contributions into ``ctx.kv_cache.dk/dv`` at this
        # chunk's positions. Writing flash_attn_bwd's output directly
        # into the kv-cache window would clobber them.
        #
        # Detect via ``chunk.has_more_chunks_host`` (set by
        # ``_pack_sequences._emit_large`` for non-final chunks of a
        # long seq). When True, write to scratch + add. Otherwise
        # write directly to the kv-cache window (legacy fast path).
        dq = ctx.scratch(dx_up_attn.shape, dx_up_attn.dtype)
        dq.zero_()
        need_dkv_accum = any(chunk.has_more_chunks_host)
        if need_dkv_accum:
            dk_target = ctx.scratch(
                (total_k, n_kv, head_dim), ctx.kv_cache.dk.dtype,
            )
            dv_target = ctx.scratch(
                (total_k, n_kv, head_dim), ctx.kv_cache.dv.dtype,
            )
        else:
            dk_target = ctx.kv_cache.dk[:total_k, :]
            dv_target = ctx.kv_cache.dv[:total_k, :]
        flextrain_attention_bwd(
            dx_up_attn,
            slot.xq.view(-1, n_heads, head_dim),
            ctx.kv_cache.k[:total_k, :],
            ctx.kv_cache.v[:total_k, :],
            slot.attn_result,
            slot.softmax_lse,
            dq,
            dk_target,
            dv_target,
            chunk.q_seq_offsets,
            chunk.k_seq_offsets,
            chunk.q_seq_lens,
            chunk.k_seq_lens,
            chunk.max_seqlen_q,
            chunk.max_seqlen_k,
            causal=cfg.is_causal,
            window_size=(cfg.window_size_left, cfg.window_size_right),
            softcap=cfg.attn_logit_softcap,
        )
        if need_dkv_accum:
            ctx.kv_cache.dk[:total_k, :].add_(dk_target)
            ctx.kv_cache.dv[:total_k, :].add_(dv_target)

        # 4. Pull out this chunk's local dK / dV from the backward ring and
        #    zero those positions so the previous chunk doesn't double-count.
        start = (
            chunk.prior_seq_offsets_host[0] + chunk.prior_seq_lens_host[0]
        )
        local_dk = ctx.scratch(slot.xk.shape, slot.xk.dtype)
        local_dv = ctx.scratch(slot.xv.shape, slot.xv.dtype)
        local_dk.copy_(ctx.kv_cache.dk[start : start + total_q, :])
        local_dv.copy_(ctx.kv_cache.dv[start : start + total_q, :])
        ctx.kv_cache.dk[start : start + total_q, :].zero_()
        ctx.kv_cache.dv[start : start + total_q, :].zero_()

        # 5. RoPE bwd (in-place on dq and local_dk). After this call the
        #    gradients are w.r.t. pre-RoPE Q/K — which for Qwen3 is still
        #    post-QK-norm; we'll undo the norm next. For Llama/Mistral
        #    there is no QK-norm, so this step completes the chain.
        rope_theta = self._rope_theta(dx_resid.device)
        dq_view, local_dk_view = apply_rope_bwd(
            [dq.view(-1, n_heads, head_dim), local_dk.view(-1, n_kv, head_dim)],
            chunk.seq_positions,
            rope_theta,
        )

        # 5b. Qwen3 QK-norm bwd. We need the pre-norm xq / xk as inputs,
        # which we recompute from attn_norm_output + w_q / w_k. The layer
        # must pass a non-None attn_norm_output when cfg.qk_norm=True.
        # Kernel expects 2D (T, heads*head_dim) with head_dim via weight.
        if cfg.qk_norm:
            if attn_norm_output is None:
                raise ValueError(
                    "GQAAttentionBlock.bwd requires attn_norm_output when "
                    "cfg.qk_norm=True (need pre-norm Q/K for RMSNorm bwd)"
                )
            xq_pre_norm_2d = torch.matmul(
                attn_norm_output, weights["w_q"]
            ).contiguous()  # (T, attn_dim)
            xk_pre_norm_2d = torch.matmul(
                attn_norm_output, weights["w_k"]
            ).contiguous()  # (T, kv_dim)
            dq_pre_norm_2d, _ = self.q_norm.bwd(
                dq_view.view(num_tokens, -1),
                xq_pre_norm_2d,
                weights,
                grads,
                getattr(slot, self.q_norm.rstd_name),
                dx_accumulator=None,
                recompute_output=False,
            )
            dk_pre_norm_2d, _ = self.k_norm.bwd(
                local_dk_view.view(num_tokens, -1),
                xk_pre_norm_2d,
                weights,
                grads,
                getattr(slot, self.k_norm.rstd_name),
                dx_accumulator=None,
                recompute_output=False,
            )
            dq_view = dq_pre_norm_2d.view(num_tokens, n_heads, head_dim)
            local_dk_view = dk_pre_norm_2d.view(num_tokens, n_kv, head_dim)

        # 5c. Qwen2 QKV-bias grad: bias adds post-projection, so
        # dL/db = sum(dL/d_post-bias, dim=0). At this point dq_view /
        # local_dk_view / local_dv are exactly pre-projection = post-bias.
        if cfg.qkv_bias:
            grads["g_b_q"].add_(dq_view.view(num_tokens, -1).sum(dim=0))
            grads["g_b_k"].add_(local_dk_view.view(num_tokens, -1).sum(dim=0))
            grads["g_b_v"].add_(local_dv.view(num_tokens, -1).sum(dim=0))

        # 6. dx_attn_norm_up = dq @ w_q.T + local_dk @ w_k.T + local_dv @ w_v.T
        dx_attn_norm_up = torch.matmul(
            dq_view.view(num_tokens, -1), weights["w_q"].T
        )
        torch.addmm(
            dx_attn_norm_up, local_dk_view.view(num_tokens, -1),
            weights["w_k"].T, alpha=1.0, beta=1.0, out=dx_attn_norm_up,
        )
        torch.addmm(
            dx_attn_norm_up, local_dv.view(num_tokens, -1),
            weights["w_v"].T, alpha=1.0, beta=1.0, out=dx_attn_norm_up,
        )

        # Return stash; caller combines with RMSNorm bwd. Also we stash
        # (dq, local_dk, local_dv) on the slot so the weight-grad matmuls
        # AFTER rmsnorm_bwd (which needs the recomputed norm output as its
        # left operand) can pick them up. The layer routes them.
        slot.aux["bwd_dq"] = dq_view.view(num_tokens, -1)
        slot.aux["bwd_local_dk"] = local_dk_view.view(num_tokens, -1)
        slot.aux["bwd_local_dv"] = local_dv.view(num_tokens, -1)

        return dx_attn_norm_up

    def bwd_accumulate_qkv_grads(
        self,
        attn_norm_output: torch.Tensor,
        grads: MutableMapping[str, torch.Tensor],
        slot,
        *,
        skip_grads: frozenset[str] = frozenset(),
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        """Second half of backward: once the caller has recomputed
        ``attn_norm_output`` via RMSNorm bwd, we use it as the left operand
        for the w_q / w_k / w_v weight-grad matmuls.

        ``skip_grads`` -- names from {``g_q``, ``g_k``, ``g_v``} whose
        addmm should be skipped (LoRA fast-path: the wrapper computes
        rank-r ``dA, dB`` from ``(X, dY)`` directly and never needs the
        full ``dW``). Default empty -- behavior identical to before.

        ``capture_xy`` -- if provided, the ``(X, dY)`` pair for each
        skipped projection is written into this dict by name. The
        wrapper consumes them via the rank-r matmul. Caller is
        responsible for cloning if it needs them past this call;
        we hand back the same tensors the addmm would have used.

        Mirrors ``orig/dense_layer.py:318-320``.
        """
        for name, dy_key in (
            ("g_v", "bwd_local_dv"),
            ("g_k", "bwd_local_dk"),
            ("g_q", "bwd_dq"),
        ):
            dy = slot.aux[dy_key]
            if name in skip_grads:
                if capture_xy is not None:
                    capture_xy[name] = (attn_norm_output, dy)
            else:
                torch.addmm(
                    grads[name], attn_norm_output.T, dy,
                    alpha=1.0, beta=1.0, out=grads[name],
                )
        del slot.aux["bwd_dq"]
        del slot.aux["bwd_local_dk"]
        del slot.aux["bwd_local_dv"]

    # ------------------------------------------------------------------
    # FLOP accounting -- mirrors ``orig/dense_layer.py:1003-1053``.
    # ------------------------------------------------------------------

    def compute_cost(
        self, chunk: ChunkMeta, max_tier: int
    ) -> ComputeCost:
        cfg = self.cfg
        avoided = [0] * (max_tier + 1)
        total = 0

        for seq_len, prior_len in zip(
            chunk.seq_lens_host, chunk.prior_seq_lens_host
        ):
            # KV projections (tier 0, never recomputed once saved)
            kv_proj = 2 * seq_len * cfg.d_model * cfg.kv_dim * 2
            total += kv_proj

            # Q + O projections (tier >= 2 avoids recompute)
            qo = 2 * (2 * seq_len * cfg.d_model * cfg.attn_dim)
            total += qo
            if max_tier >= 2:
                for L in range(2, max_tier + 1):
                    avoided[L] += qo

            # Flash attention FLOPs (tier >= 1 avoids recompute)
            attn_prior = 4 * seq_len * prior_len * cfg.attn_dim
            if cfg.is_causal:
                attn_current = 2 * seq_len * seq_len * cfg.attn_dim
            else:
                attn_current = 4 * seq_len * seq_len * cfg.attn_dim
            attn = attn_prior + attn_current
            total += attn
            if max_tier >= 1:
                for L in range(1, max_tier + 1):
                    avoided[L] += attn

        return ComputeCost(
            total_fwd_flops=total,
            avoided_recompute_flops=tuple(avoided),
        )


class GQASlidingWindowAttentionBlock(GQAAttentionBlock):
    """Sliding-window variant of :class:`GQAAttentionBlock`.

    Identical compute path -- only the ``window_size`` argument passed to
    ``flextrain_attention_fwd`` / ``flextrain_attention_bwd`` differs. Accepts only
    :class:`GQASlidingWindowAttentionConfig` (which enforces
    ``window_size_left >= 0``).

    Kept as a distinct class so model-family compositions (Mistral, Gemma,
    GPT-OSS) name the algorithm they actually use -- swapping attention
    types at the layer level is what makes a "Mistral layer" different
    from a "Llama layer".
    """

    def __init__(self, cfg: GQASlidingWindowAttentionConfig) -> None:
        if not isinstance(cfg, GQASlidingWindowAttentionConfig):
            raise TypeError(
                "GQASlidingWindowAttentionBlock requires "
                "GQASlidingWindowAttentionConfig; got "
                f"{type(cfg).__name__}"
            )
        super().__init__(cfg)
