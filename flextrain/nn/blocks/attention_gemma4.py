"""Gemma 4 attention block — fork of :class:`GQAAttentionBlock` for the
text-only path of Gemma-4-31B-Instruct.

Structural deltas vs :class:`flextrain.nn.blocks.attention.GQAAttentionBlock`
(see ``docs/internal/gemma4_status.md`` for the design notes):

1. **V-RMSNorm everywhere** (`modular_gemma4.py:947`). After the V
   projection, V passes through a per-head RMSNorm with NO learnable γ
   (HF's `with_scale=False`). We reuse the standard RMSNorm kernel by
   handing it a constant ``ones(head_dim)`` weight allocated once per
   device. The rstd is saved as a tier-0 ActivationField (``v_norm_rstd``)
   declared on THIS forked block's schema; no shared-block schema delta.

2. **``k_eq_v`` mode (full-attention layers only)**
   (`modular_gemma4.py:915, 952-956, 991`). When enabled:
   * No ``w_v`` weight (config asserts this; param spec omits it).
   * V is the **pre-K-norm, pre-rope** output of ``W_k @ x`` run through
     ``v_norm``.
   * K is the same tensor run through ``k_norm`` then RoPE.
   Backward (deferred to a follow-up): the V path and K path both flow
   back to ``xk_pre_norm`` and sum.

3. **Proportional partial-rope** (full-attention layers,
   `rope_type="proportional"`). Same kernel as Qwen3-Next's partial rope
   but with a different ``inv_freq`` curve: divides by ``head_dim`` not
   ``rot_dim`` in the exponent. Built via
   :func:`flextrain.nn.blocks.rope.build_partial_rope_inv_freq` with the
   new ``"proportional"`` branch.

4. **Per-instance head dimensions are already supported by
   ``GQAAttentionConfig``** — Gemma 4 sliding layers use head_dim=256,
   global layers use head_dim=512. The layer config drives this and the
   block is just per-instance.

Forward path is implemented end-to-end. Backward is currently stubbed
to ``NotImplementedError`` matching the way the original Gemma2/Gemma3
blocks landed (forward first, hand-rolled bwd in a follow-up session).
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
    flextrain_rmsnorm_bwd,
    flextrain_rmsnorm_fwd,
    flextrain_rmsnorm_fwd_recompute,
    dispatcher,
)

from .attention import GQAAttentionConfig
from .rope import (
    apply_rope_bwd,
    apply_rope_fwd,
    apply_rope_partial_bwd,
    apply_rope_partial_fwd,
    build_partial_rope_inv_freq,
    build_rope_inv_freq,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gemma4AttentionConfig(GQAAttentionConfig):
    """Per-instance Gemma 4 attention config.

    Adds three Gemma-4-specific knobs on top of :class:`GQAAttentionConfig`:

    * ``v_norm`` (default True for Gemma 4) — apply per-head RMSNorm with
      no learnable γ to V before the flash-attn call. Saves a tier-0
      ``v_norm_rstd`` ActivationField. Set False to disable for tests /
      legacy compatibility.
    * ``k_eq_v`` (default False) — when True, no ``w_v`` projection; V is
      the K projection output (pre-K-norm, pre-rope) run through
      ``v_norm``. Requires ``v_norm=True``.
    * ``partial_rotary_factor`` (default 1.0) — fraction of ``head_dim``
      that receives RoPE. Combined with the upstream ``rope_scaling``
      kwarg this drives the proportional-rope curve when
      ``rope_scaling={"rope_type": "proportional"}``.
    """

    v_norm: bool = True
    k_eq_v: bool = False
    partial_rotary_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.k_eq_v and not self.v_norm:
            raise ValueError(
                "Gemma4AttentionConfig: k_eq_v=True requires v_norm=True "
                "(K-projection output flows through v_norm to produce V)."
            )
        if self.k_eq_v and self.qkv_bias:
            # HF Gemma 4 has attention_bias=false. Defensive: V comes from
            # W_k @ x and there's no b_v to add. Wire this up only if a
            # future config needs it.
            raise NotImplementedError(
                "Gemma4AttentionConfig: k_eq_v + qkv_bias unimplemented"
            )


@dataclass(frozen=True)
class Gemma4SlidingWindowAttentionConfig(Gemma4AttentionConfig):
    """Sliding-window Gemma 4 attention. Inherits Gemma4AttentionConfig's
    new knobs. Sliding layers have ``k_eq_v=False`` (only full layers
    enable it), but the config allows either."""

    window_size_left: int = 1024  # type: ignore[assignment]
    window_size_right: int = 0  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class Gemma4AttentionBlock:
    """Gemma 4 GQA attention. Forward complete; backward stubbed.

    Algorithmic pipeline (forward):
    ::

        xq = w_q @ attn_norm_output                # (T, attn_dim)
        xk = w_k @ attn_norm_output                # (T, kv_dim) — saved pre-norm if k_eq_v
        if k_eq_v:
            xv = v_norm(xk)                        # share W_k output, separate per-head RMSNorm
        else:
            xv = w_v @ attn_norm_output
            xv = v_norm(xv)                        # always applied for Gemma 4
        xq = q_norm(xq); xk = k_norm(xk)           # per-head RMSNorm, learnable γ
        rope(xq, xk)                               # partial / proportional, in-place
        attn_result = flash_attn_varlen(xq, kv.k, kv.v, ...)
        xo = attn_result @ w_o + x_resid           # fused via dispatcher

    Saved fields (added vs base GQAAttentionBlock): ``v_norm_rstd``
    (tier 0). Inherits xk / xv / attn_result / softmax_lse / xq / xo and
    the q_norm / k_norm rstds.

    Backward is deferred — raises NotImplementedError. The dual-path
    grad math for ``k_eq_v=True`` (V path + K path both flow back to
    ``xk_pre_norm``) is documented in ``docs/internal/gemma4_status.md``
    so the next session can land it directly.
    """

    # Flash-attn's standard kernel supports head_dim up to 256. Gemma 4
    # global layers use head_dim=512 (``global_head_dim``); for those
    # layers we fall back to torch SDPA, which handles arbitrary head_dim
    # at the cost of materializing the T×T score matrix (no on-line
    # softmax). For T < a few hundred this is fine; production-fast
    # large-T runs need a real head_dim=512 kernel (out of scope).
    _FLASH_HEAD_DIM_MAX: int = 256

    def __init__(self, cfg: Gemma4AttentionConfig) -> None:
        self.cfg = cfg
        self._rope_inv_freq_cache: torch.Tensor | None = None
        # V-norm uses the standard RMSNorm kernel with a constant
        # ones(head_dim) weight. Allocated lazily on first forward
        # (device-aware). γ-free RMSNorm matches HF's with_scale=False.
        self._v_norm_ones_weight: torch.Tensor | None = None
        # Eager-SDPA fallback when flash can't take this head_dim.
        self._use_sdpa_attn: bool = cfg.head_dim > self._FLASH_HEAD_DIM_MAX

        # Per-head learnable QK-norm reuses the base attention pattern.
        if cfg.qk_norm:
            from .norm import RMSNormBlock
            if cfg.qk_norm_per_head:
                q_weight_dim, k_weight_dim = "head_dim", "head_dim"
            else:
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
        """Activation fields owned by this block.

        Identical to :meth:`GQAAttentionBlock.fields` plus ``v_norm_rstd``
        (tier 0, per-head fp32) when ``cfg.v_norm=True``.
        """
        cfg = self.cfg
        bf = cfg.compute_dtype
        out: tuple[ActivationField, ...] = (
            ActivationField(
                "xk",
                lambda n, d: (n, cfg.n_kv_heads, cfg.head_dim),
                bf, tier=0,
            ),
            ActivationField(
                "xv",
                lambda n, d: (n, cfg.n_kv_heads, cfg.head_dim),
                bf, tier=0,
            ),
            ActivationField(
                "attn_result",
                lambda n, d: (n, cfg.n_heads, cfg.head_dim),
                bf, tier=1,
            ),
            ActivationField(
                "softmax_lse",
                lambda n, d: (cfg.n_heads, n),
                torch.float32, tier=1, token_axis=1,
            ),
            ActivationField(
                "xq",
                lambda n, d: (n, cfg.n_heads, cfg.head_dim),
                bf, tier=2,
            ),
            ActivationField(
                "xo",
                lambda n, d: (n, cfg.d_model),
                bf, tier=2,
            ),
        )
        if cfg.qk_norm:
            out = out + self.q_norm.fields() + self.k_norm.fields()
        if cfg.v_norm:
            # Per-head V-RMSNorm rstd, shape (T, n_kv_heads), fp32.
            out = out + (
                ActivationField(
                    "v_norm_rstd",
                    lambda n, d: (n, cfg.n_kv_heads),
                    torch.float32, tier=0,
                ),
            )
        return out

    def param_spec(self) -> ParamSpec:
        """``w_q``, ``w_k``, optional ``w_v`` (omitted when ``k_eq_v=True``),
        ``w_o``. Per-head QK-norm γ when ``qk_norm=True``. V-norm has no
        γ (HF ``with_scale=False``).
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
        ]
        if not cfg.k_eq_v:
            tensors.append(
                TensorSpec(
                    "w_v",
                    lambda d: (cfg.d_model, cfg.kv_dim),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                )
            )
        tensors.append(
            TensorSpec(
                "w_o",
                lambda d: (cfg.attn_dim, cfg.d_model),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            )
        )
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
            ])
            if not cfg.k_eq_v:
                tensors.append(
                    TensorSpec(
                        "b_v",
                        lambda d: (cfg.kv_dim,),
                        compute_dtype=cfg.compute_dtype,
                        master_dtype=cfg.master_dtype,
                        grad_dtype=cfg.grad_dtype,
                    )
                )
        spec = ParamSpec(tensors=tuple(tensors))
        if cfg.qk_norm:
            spec = ParamSpec.merge([
                spec,
                self.q_norm.param_spec(),
                self.k_norm.param_spec(),
            ])
        return spec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _is_partial_rotary(self) -> bool:
        return self.cfg.partial_rotary_factor != 1.0

    @property
    def _rot_dim(self) -> int:
        """Number of head channels that receive RoPE rotation. Equal to
        ``head_dim`` for full rotary; ``head_dim * partial_rotary_factor``
        otherwise (Gemma 4 global layers: 0.25 × 512 = 128)."""
        prf = self.cfg.partial_rotary_factor
        rot_dim = int(self.cfg.head_dim * prf)
        if rot_dim % 2 != 0:
            raise ValueError(
                f"Gemma4AttentionBlock: partial_rotary_factor={prf} × "
                f"head_dim={self.cfg.head_dim} = {rot_dim}, must be even."
            )
        return rot_dim

    def _rope_theta(self, device: torch.device) -> torch.Tensor:
        if (
            self._rope_inv_freq_cache is None
            or self._rope_inv_freq_cache.device != device
        ):
            cfg = self.cfg
            rope_type = "default"
            if cfg.rope_scaling is not None:
                rope_type = (
                    cfg.rope_scaling.get("rope_type")
                    or cfg.rope_scaling.get("type")
                    or "default"
                )
            if self._is_partial_rotary or rope_type == "proportional":
                # Proportional rope uses the partial-rope kernel with a
                # head_dim-based inv_freq curve. The rope.py branch
                # handles both default-partial and proportional.
                inv_freq_cpu = build_partial_rope_inv_freq(
                    rot_dim=self._rot_dim,
                    rope_base=cfg.rope_base,
                    rope_scaling=cfg.rope_scaling,
                    head_dim=cfg.head_dim if rope_type == "proportional" else None,
                )
            else:
                inv_freq_cpu = build_rope_inv_freq(
                    head_dim=cfg.head_dim,
                    rope_base=cfg.rope_base,
                    rope_scaling=cfg.rope_scaling,
                )
            self._rope_inv_freq_cache = inv_freq_cpu.to(
                device=device, dtype=torch.float32,
            )
        return self._rope_inv_freq_cache

    def _rope_fwd(self, tensors, seq_positions, rope_theta):
        if self._is_partial_rotary:
            return apply_rope_partial_fwd(
                tensors, seq_positions, rope_theta, self._rot_dim,
            )
        return apply_rope_fwd(tensors, seq_positions, rope_theta)

    def _rope_bwd(self, grad_tensors, seq_positions, rope_theta):
        if self._is_partial_rotary:
            return apply_rope_partial_bwd(
                grad_tensors, seq_positions, rope_theta, self._rot_dim,
            )
        return apply_rope_bwd(grad_tensors, seq_positions, rope_theta)

    def _attn_fwd_sdpa(
        self,
        chunk: ChunkMeta,
        slot,
        ctx: LayerContext,
    ) -> None:
        """Eager-SDPA attention forward, used when ``head_dim > 256``
        exceeds flash-attn's kernel limit (Gemma 4 global layers).

        Reads slot.xq (post-norm, post-rope) and kv.k / kv.v (post-norm,
        K post-rope, V post-v_norm). Writes slot.attn_result and
        zero-fills slot.softmax_lse (unused on the fwd side; the bwd
        path raises NotImplementedError when this branch was taken).

        Iterates over chunk sequences (single-seq for the parity tests;
        the loop is still cheap for multi-seq prefill). No sliding-
        window or softcap support — those features only apply to layers
        with head_dim <= 256, which take the flash branch.
        """
        cfg = self.cfg
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        kv = ctx.kv_cache
        xq3 = slot.xq.view(-1, n_heads, head_dim)
        seq_lens = chunk.seq_lens_host
        prior_seq_lens = chunk.prior_seq_lens_host
        prior_seq_offsets = chunk.prior_seq_offsets_host
        q_cursor = 0
        for s, (q_len, prior_len, prior_off) in enumerate(
            zip(seq_lens, prior_seq_lens, prior_seq_offsets)
        ):
            qa, qb = q_cursor, q_cursor + int(q_len)
            q_cursor = qb
            ka = int(prior_off)
            kb = ka + int(prior_len) + int(q_len)
            if qb <= qa:
                continue
            seq_q = xq3[qa:qb].transpose(0, 1).unsqueeze(0).contiguous()
            seq_k = kv.k[ka:kb].transpose(0, 1).unsqueeze(0).contiguous()
            seq_v = kv.v[ka:kb].transpose(0, 1).unsqueeze(0).contiguous()
            # HF Gemma 4 sets ``self.scaling = 1.0`` on attention, which
            # OVERRIDES the eager-default 1/sqrt(head_dim) scaling
            # (modular_gemma4.py:920, eager_attention_forward in
            # modeling_gemma4.py:779). q_norm/k_norm pre-scale Q/K via
            # their γ vectors instead. Pass ``scale=1.0`` explicitly so
            # SDPA doesn't insert its own 1/sqrt(d) factor.
            out = torch.nn.functional.scaled_dot_product_attention(
                seq_q, seq_k, seq_v,
                is_causal=cfg.is_causal,
                enable_gqa=(n_kv != n_heads),
                scale=1.0,
            )
            seg = out.squeeze(0).transpose(0, 1).contiguous()
            slot.attn_result[qa:qb].copy_(seg)
        slot.softmax_lse.zero_()

    def _v_norm_weight(self, device: torch.device) -> torch.Tensor:
        """Constant ones(head_dim) tensor used as the V-norm 'weight'.

        HF Gemma 4 declares ``v_norm`` with ``with_scale=False`` (no
        learnable γ), but our shared RMSNorm kernel requires a weight.
        We hand it ones-of-the-right-dtype so the multiply is a no-op
        numerically. Allocated lazily per device; cached on the block.
        """
        cfg = self.cfg
        if (
            self._v_norm_ones_weight is None
            or self._v_norm_ones_weight.device != device
        ):
            self._v_norm_ones_weight = torch.ones(
                cfg.head_dim, dtype=cfg.compute_dtype, device=device,
            )
        return self._v_norm_ones_weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def fwd(
        self,
        x_resid: torch.Tensor,
        attn_norm_output: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        """Gemma 4 attention-block forward. See class docstring for the
        algorithmic pipeline. Writes into the same slot fields as
        GQAAttentionBlock plus ``slot.v_norm_rstd``. Returns the
        residual-added output (written into ``slot.xo``).

        When ``cfg.k_eq_v=True`` the W_v matmul is skipped; V is the
        K-projection output (pre-K-norm) run through v_norm. The K path
        still applies k_norm + rope as usual.
        """
        cfg = self.cfg
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        total_q = chunk.total_q
        total_k = chunk.total_k

        # Q projection.
        torch.matmul(
            attn_norm_output, weights["w_q"],
            out=slot.xq.view(-1, cfg.attn_dim),
        )
        if cfg.qkv_bias:
            slot.xq.view(-1, cfg.attn_dim).add_(weights["b_q"])

        # K projection.
        torch.matmul(
            attn_norm_output, weights["w_k"],
            out=slot.xk.view(-1, cfg.kv_dim),
        )
        if cfg.qkv_bias:
            slot.xk.view(-1, cfg.kv_dim).add_(weights["b_k"])

        # V projection (or copy from K when k_eq_v).
        if cfg.k_eq_v:
            # V := pre-K-norm, pre-rope output of W_k @ x. v_norm runs
            # on this; the K path applies k_norm + rope on the *same*
            # slot.xk tensor afterwards.
            slot.xv.copy_(slot.xk)
        else:
            torch.matmul(
                attn_norm_output, weights["w_v"],
                out=slot.xv.view(-1, cfg.kv_dim),
            )
            if cfg.qkv_bias:
                slot.xv.view(-1, cfg.kv_dim).add_(weights["b_v"])

        # V-norm (per-head RMSNorm, no learnable γ). In-place on slot.xv.
        if cfg.v_norm:
            v_weight = self._v_norm_weight(x_resid.device)
            xv_2d = slot.xv.view(-1, cfg.kv_dim)
            flextrain_rmsnorm_fwd(
                xv_2d,
                W=v_weight,
                head_dim=head_dim,
                output=xv_2d,
                rstd=slot.v_norm_rstd,
                rms_norm_eps=cfg.rms_norm_eps,
            )

        # Q/K-norm (per-head RMSNorm, learnable γ). In-place on slot.xq / slot.xk.
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

        # RoPE on Q and K (full or partial / proportional per config).
        # V does NOT receive RoPE — it's the post-v_norm tensor already.
        rope_theta = self._rope_theta(x_resid.device)
        self._rope_fwd([slot.xq, slot.xk], chunk.seq_positions, rope_theta)

        # KV cache write.
        kv = ctx.kv_cache
        start = (
            chunk.prior_seq_offsets_host[0] + chunk.prior_seq_lens_host[0]
        )
        kv.k[start : start + total_q, :].copy_(slot.xk)
        kv.v[start : start + total_q, :].copy_(slot.xv)

        # Attention: flash for head_dim <= 256, eager SDPA otherwise
        # (Gemma 4 global layers have head_dim=512 — flash kernel limit).
        if self._use_sdpa_attn:
            self._attn_fwd_sdpa(chunk, slot, ctx)
        else:
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

        # Fused O-projection + residual: xo = attn_result @ w_o + x_resid.
        stream_ptr = torch.cuda.current_stream().cuda_stream
        return dispatcher.matmul(
            stream_ptr,
            A=slot.attn_result.view(-1, cfg.attn_dim),
            B=weights["w_o"],
            C=x_resid,
            D=slot.xo,
            alpha=1.0, beta=1.0,
        )

    # ------------------------------------------------------------------
    # Forward-recompute helpers
    # ------------------------------------------------------------------

    def fwd_recompute_qo(
        self,
        attn_norm_output: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        slot,
        x_resid: torch.Tensor,
    ) -> torch.Tensor:
        """Tier<2: recompute Q (re-projection + q_norm + rope). xv / xk are
        already saved at tier 0 so no V-norm recompute is needed.
        """
        cfg = self.cfg
        torch.matmul(
            attn_norm_output, weights["w_q"],
            out=slot.xq.view(-1, cfg.attn_dim),
        )
        if cfg.qkv_bias:
            slot.xq.view(-1, cfg.attn_dim).add_(weights["b_q"])
        if cfg.qk_norm:
            xq2d = slot.xq.view(-1, cfg.attn_dim)
            self.q_norm.fwd_from_rstd(
                xq2d, weights, getattr(slot, self.q_norm.rstd_name),
                output=xq2d,
            )
        rope_theta = self._rope_theta(attn_norm_output.device)
        self._rope_fwd([slot.xq], chunk.seq_positions, rope_theta)
        return slot.xq

    def fwd_recompute_attn(
        self,
        chunk: ChunkMeta,
        slot,
        ctx: LayerContext,
    ) -> None:
        """Tier<1: recompute attn_result + softmax_lse from saved Q + KV
        cache. Branches between flash and SDPA on head_dim like ``fwd``."""
        cfg = self.cfg
        n_heads, head_dim = cfg.n_heads, cfg.head_dim
        total_k = chunk.total_k
        kv = ctx.kv_cache
        if self._use_sdpa_attn:
            self._attn_fwd_sdpa(chunk, slot, ctx)
            return
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
        """Tier<2: xo = attn_result @ w_o + x_resid."""
        cfg = self.cfg
        return torch.addmm(
            x_resid,
            slot.attn_result.view(-1, cfg.attn_dim),
            weights["w_o"],
            out=slot.xo,
        )

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    #
    # Structure mirrors :class:`GQAAttentionBlock.bwd` /
    # :class:`GQAAttentionGatedBlock.bwd`. Gemma-4 deltas:
    #
    #  * V-RMSNorm bwd (no γ wgrad): inserted between flash-attn's
    #    ``local_dv`` output and the wgrad-pass left operand. Uses the
    #    standard rmsnorm kernel with ``dW=None`` and our private
    #    ``ones(head_dim)`` weight buffer.
    #
    #  * ``k_eq_v=True`` fold: V grad path and K grad path both flow
    #    back to ``xk_pre_norm`` (the W_k @ x output). They sum into a
    #    single dL/d(xk_pre_norm). The deferred wgrad pass therefore
    #    only accumulates ``g_k`` for this layer (no ``g_v`` exists).
    #
    #  * Recomputed left operand for ``v_norm.bwd``: the pre-V-norm
    #    tensor. For ``k_eq_v=False`` it's a fresh ``attn_norm_output @
    #    W_v``. For ``k_eq_v=True`` it's exactly the recomputed
    #    ``xk_pre_norm = attn_norm_output @ W_k`` (no extra matmul).
    #
    # See ``docs/internal/gemma4_status.md`` §"Backward derivation".

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
        cfg = self.cfg
        if self._use_sdpa_attn:
            raise NotImplementedError(
                "Gemma4AttentionBlock.bwd does not yet support the SDPA "
                "fallback (head_dim > 256). The flash-attn bwd handles "
                "softmax_lse / dQ / dK / dV in one kernel; the SDPA "
                "equivalent needs an autograd-graph stitch which is a "
                "Stage 3 (full-grad parity) item. Block parity tests at "
                "small head_dim still cover the V-norm + k_eq_v fold "
                "math via the flash path."
            )
        n_heads, n_kv, head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        num_tokens = dx_resid.shape[0]
        total_k = chunk.total_k
        total_q = chunk.total_q

        attn_result = slot.attn_result.view(num_tokens, -1)

        # 1. g_o += attn_result^T @ dx_resid (inline Wgrad).
        if "g_o" in skip_grads:
            if capture_xy is not None:
                capture_xy["g_o"] = (attn_result, dx_resid.clone())
        else:
            torch.addmm(
                grads["g_o"], attn_result.T, dx_resid,
                alpha=1.0, beta=1.0, out=grads["g_o"],
            )

        # 2. d_attn_result = dx_resid @ w_o^T.
        dx_up_attn = torch.matmul(dx_resid, weights["w_o"].T)
        dx_up_attn = dx_up_attn.view(num_tokens, n_heads, head_dim)

        # 3. flash-attn bwd → dq + (local_dk + local_dv via kv-cache).
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

        # 4. Pull this chunk's local dK / dV from the bwd ring; zero
        #    consumed positions so a prior chunk doesn't double-count.
        start = (
            chunk.prior_seq_offsets_host[0] + chunk.prior_seq_lens_host[0]
        )
        local_dk = ctx.scratch(slot.xk.shape, slot.xk.dtype)
        local_dv = ctx.scratch(slot.xv.shape, slot.xv.dtype)
        local_dk.copy_(ctx.kv_cache.dk[start : start + total_q, :])
        local_dv.copy_(ctx.kv_cache.dv[start : start + total_q, :])
        ctx.kv_cache.dk[start : start + total_q, :].zero_()
        ctx.kv_cache.dv[start : start + total_q, :].zero_()

        # 5. RoPE bwd on dq + local_dk (partial / proportional or full).
        #    V is not RoPE'd in Gemma 4, so local_dv is untouched here.
        rope_theta = self._rope_theta(dx_resid.device)
        dq_view, local_dk_view = self._rope_bwd(
            [
                dq.view(-1, n_heads, head_dim),
                local_dk.view(-1, n_kv, head_dim),
            ],
            chunk.seq_positions,
            rope_theta,
        )

        # 5b. Recompute pre-norm Q/K (left operand for QK-norm bwd; also
        #     left operand for V-norm bwd when k_eq_v=True).
        if attn_norm_output is None:
            raise ValueError(
                "Gemma4AttentionBlock.bwd requires attn_norm_output "
                "(recomputed pre_attn_norm output); pass it from the "
                "enclosing layer's backward_dgrad."
            )
        xq_pre_norm_2d = torch.matmul(
            attn_norm_output, weights["w_q"],
        ).contiguous()
        xk_pre_norm_2d = torch.matmul(
            attn_norm_output, weights["w_k"],
        ).contiguous()

        # 6. QK-norm bwd (per-head, learnable γ). Inline g_q_norm /
        #    g_k_norm wgrads. dq / local_dk are 2D (T, attn_dim) /
        #    (T, kv_dim) for the kernel.
        if cfg.qk_norm:
            dq_pre_norm_2d, _, _ = flextrain_rmsnorm_bwd(
                dY=dq_view.view(num_tokens, -1),
                X=xq_pre_norm_2d,
                W=weights["w_q_norm"],
                rstd=getattr(slot, self.q_norm.rstd_name),
                head_dim=head_dim,
                dX=None,
                dW=grads.get(self.q_norm.grad_name),
                recompute_output=False,
            )
            dk_pre_norm_2d, _, _ = flextrain_rmsnorm_bwd(
                dY=local_dk_view.view(num_tokens, -1),
                X=xk_pre_norm_2d,
                W=weights["w_k_norm"],
                rstd=getattr(slot, self.k_norm.rstd_name),
                head_dim=head_dim,
                dX=None,
                dW=grads.get(self.k_norm.grad_name),
                recompute_output=False,
            )
            dq_view = dq_pre_norm_2d.view(num_tokens, n_heads, head_dim)
            local_dk_view = dk_pre_norm_2d.view(num_tokens, n_kv, head_dim)

        # 7. V-norm bwd (per-head, NO γ wgrad). For k_eq_v=True, the
        #    pre-V-norm tensor IS xk_pre_norm_2d. For k_eq_v=False, it's
        #    a fresh attn_norm_output @ W_v.
        if cfg.v_norm:
            if cfg.k_eq_v:
                xv_pre_v_norm_2d = xk_pre_norm_2d
            else:
                xv_pre_v_norm_2d = torch.matmul(
                    attn_norm_output, weights["w_v"],
                ).contiguous()
                if cfg.qkv_bias:
                    xv_pre_v_norm_2d = xv_pre_v_norm_2d + weights["b_v"]
            v_weight_ones = self._v_norm_weight(dx_resid.device)
            d_v_pre_v_norm_2d, _, _ = flextrain_rmsnorm_bwd(
                dY=local_dv.view(num_tokens, -1),
                X=xv_pre_v_norm_2d,
                W=v_weight_ones,
                rstd=slot.v_norm_rstd,
                head_dim=head_dim,
                dX=None,
                dW=None,   # γ-free; no g_v_norm tensor exists.
                recompute_output=False,
            )
        else:
            d_v_pre_v_norm_2d = local_dv.view(num_tokens, -1)

        # 8. k_eq_v fold: V grad and K grad both target xk_pre_norm.
        if cfg.k_eq_v:
            # Sum K-path and V-path grads into local_dk_view; no
            # separate d_xv to track.
            local_dk_view = (
                local_dk_view.view(num_tokens, -1) + d_v_pre_v_norm_2d
            ).view(num_tokens, n_kv, head_dim)
            d_xv_for_wgrad = None
        else:
            d_xv_for_wgrad = d_v_pre_v_norm_2d.view(num_tokens, n_kv, head_dim)

        # 8b. Qwen2-style QKV-bias grads (only present if cfg.qkv_bias;
        #     never the case for Gemma 4 itself but we keep the path for
        #     future use). Bias adds post-projection; bias-grad = sum over T.
        if cfg.qkv_bias:
            grads["g_b_q"].add_(dq_view.view(num_tokens, -1).sum(dim=0))
            grads["g_b_k"].add_(local_dk_view.view(num_tokens, -1).sum(dim=0))
            if not cfg.k_eq_v and "g_b_v" in grads:
                grads["g_b_v"].add_(
                    d_xv_for_wgrad.view(num_tokens, -1).sum(dim=0)
                )

        # 9. dx_attn_norm_up = dq @ w_q^T + local_dk @ w_k^T
        #    (+ d_xv @ w_v^T if not k_eq_v)
        dx_attn_norm_up = torch.matmul(
            dq_view.reshape(num_tokens, -1), weights["w_q"].T,
        )
        torch.addmm(
            dx_attn_norm_up, local_dk_view.reshape(num_tokens, -1),
            weights["w_k"].T, alpha=1.0, beta=1.0, out=dx_attn_norm_up,
        )
        if not cfg.k_eq_v:
            torch.addmm(
                dx_attn_norm_up, d_xv_for_wgrad.reshape(num_tokens, -1),
                weights["w_v"].T, alpha=1.0, beta=1.0, out=dx_attn_norm_up,
            )

        # 10. Stash dq / dk / dv pre-projection grads for the deferred
        #     wgrad pass. The layer reads slot.aux after recomputing
        #     attn_norm_output (the left operand for the matmuls).
        slot.aux["bwd_dq"] = dq_view.view(num_tokens, -1)
        slot.aux["bwd_local_dk"] = local_dk_view.view(num_tokens, -1)
        if not cfg.k_eq_v:
            slot.aux["bwd_local_dv"] = d_xv_for_wgrad.view(num_tokens, -1)

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
        """Deferred ``g_q / g_k`` (and ``g_v`` when ``k_eq_v=False``)
        wgrads: ``g_X += attn_norm_output^T @ dY``.

        For ``k_eq_v=True`` global layers there is no ``g_v`` — the V
        gradient is already folded into ``bwd_local_dk`` upstream.
        """
        cfg = self.cfg
        pairs: list[tuple[str, str]] = []
        if not cfg.k_eq_v:
            pairs.append(("g_v", "bwd_local_dv"))
        pairs.extend([("g_k", "bwd_local_dk"), ("g_q", "bwd_dq")])
        for name, dy_key in pairs:
            dy = slot.aux[dy_key]
            if name in skip_grads:
                if capture_xy is not None:
                    capture_xy[name] = (attn_norm_output, dy)
            else:
                torch.addmm(
                    grads[name], attn_norm_output.T, dy,
                    alpha=1.0, beta=1.0, out=grads[name],
                )
        for k in ("bwd_dq", "bwd_local_dk", "bwd_local_dv"):
            slot.aux.pop(k, None)

    # ------------------------------------------------------------------
    # FLOP accounting -- mirrors GQAAttentionBlock; V-projection cost is
    # zero when k_eq_v.
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
            # K projection (always present, tier 0).
            kv_proj = 2 * seq_len * cfg.d_model * cfg.kv_dim
            if not cfg.k_eq_v:
                # V projection too.
                kv_proj += 2 * seq_len * cfg.d_model * cfg.kv_dim
            total += kv_proj

            # Q + O projections (tier >= 2 avoids recompute).
            qo = 2 * (2 * seq_len * cfg.d_model * cfg.attn_dim)
            total += qo
            if max_tier >= 2:
                for L in range(2, max_tier + 1):
                    avoided[L] += qo

            # Flash attention FLOPs (tier >= 1 avoids recompute).
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


class Gemma4SlidingWindowAttentionBlock(Gemma4AttentionBlock):
    """Sliding-window variant. Same compute path, only ``window_size`` differs
    in the flash-attn call. Accepts only :class:`Gemma4SlidingWindowAttentionConfig`."""

    def __init__(self, cfg: Gemma4SlidingWindowAttentionConfig) -> None:
        if not isinstance(cfg, Gemma4SlidingWindowAttentionConfig):
            raise TypeError(
                "Gemma4SlidingWindowAttentionBlock requires "
                "Gemma4SlidingWindowAttentionConfig; got "
                f"{type(cfg).__name__}"
            )
        super().__init__(cfg)
