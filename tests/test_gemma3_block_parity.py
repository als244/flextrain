"""Block-level forward + backward parity for Gemma 2 / Gemma 3 dense.

This is the autograd-reference oracle that validates the dual-residual
math in :class:`Gemma2Block` and :class:`Gemma3Block` against a pure-
torch reference. It is the test that the hand-rolled bwd implementation
will be developed against — every gradient drops in grad-by-grad with
the corresponding xfail flipping to PASS.

Phase A (what this file delivers initially): forward parity is asserted
unconditionally; backward parity is marked ``xfail(strict=False)``
because ``Gemma{2,3}Block.backward`` currently raises
``NotImplementedError``. As Phase B implements the split backward, the
xfail flips to a real assertion.

Why a hand-rolled torch reference instead of HF ``Gemma{2,3}DecoderLayer``:
the block oracle exists to validate the dual-residual chain rule in
isolation. HF's machinery (rotary-embedding layout, position-id
plumbing, kv-cache layouts, softcap conventions) would add confounds.
HF-vs-flextrain parity is verified separately at the full-model level
(test_gemma3_1b_parity.py / test_gemma3_multimodal_parity.py — future
work in this same effort).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F

# Make the repo root importable when pytest is invoked from elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import _rmsnorm, _rope_pair_interleave
from flextrain.core.activation_schema import ActivationSlot
from flextrain.core.layer import ChunkMeta, LayerContext
from flextrain.engine.buffers import ScratchPool
from flextrain.nn.layers.gemma2 import Gemma2Block, Gemma2BlockConfig
from flextrain.nn.layers.gemma3 import Gemma3Block, Gemma3BlockConfig


# Block-test config: small but realistic GQA shape that exercises every
# code path (head splitting, kv-head ratio, sliding window). Matches the
# numbers in docs/internal/gemma3_status.md §"Recommended next-session
# execution order".
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
D_MODEL = 128
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 32
EXPERT_DIM = 256
T = 64
RMS_NORM_EPS = 1e-6
ROPE_BASE = 10_000.0
SLIDING_WINDOW = 16  # well under T so the mask actually clips

# bf16 noise floor on a 64-token, four-matmul, four-norm dual-residual
# layer: max|Δ| empirically ~0.06 against output max ~6 (i.e. ~1%
# relative). Element-wise atol/rtol are reported but NOT used as the
# primary pass criterion — they blow up on near-zero outliers without
# indicating a real math error. The robust criteria are cosine
# similarity, sign agreement, and L2 relative error (see _compare).
FWD_COS_TOL = 0.9995       # direction alignment
FWD_SIGN_TOL = 0.99        # ≥99% of elements have the same sign as ref
FWD_REL_L2_TOL = 5e-2      # ||a - b||_2 / ||b||_2 ≤ 5%

# Backward grads accumulate more bf16 noise (more matmuls in the chain)
# so use slightly looser thresholds. The sign threshold is set at 0.95
# because tiny γ vectors (e.g. per-head w_q_norm with head_dim=32) can
# have a handful of near-zero entries whose sign flips under bf16
# quantization even when the gradient direction is otherwise correct
# (cos > 0.9999, rel_l2 < 1%). The cosine + L2 thresholds catch real
# math errors; sign-match is a robustness signal, not a correctness gate.
BWD_COS_TOL = 0.998
BWD_SIGN_TOL = 0.95
BWD_REL_L2_TOL = 8e-2


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def _diffstats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Compute robust correctness metrics between ``a`` and reference ``b``.

    Returns:
      ``cos``        cosine similarity (1.0 = perfect alignment of direction);
                     bf16 quantization preserves direction so this stays
                     ≥0.9995 for non-buggy paths.
      ``sign_match`` fraction of elements where ``sign(a) == sign(b)``.
                     Near-zero elements may flip; <99% is a real signal.
      ``rel_l2``     ``||a - b||_2 / ||b||_2`` — overall L2 relative error.
                     Insensitive to near-zero outliers, unlike per-element
                     relative error.
      ``max_rel``    per-element max ``|a - b| / max(|b|, eps)``. Reported
                     for diagnosis only; can be huge on near-zero outliers
                     even when ``a`` and ``b`` are otherwise identical.
      ``max_abs``    ``max|a - b|``. Diagnostic.
      ``mean_abs``   ``mean|a - b|``. Diagnostic.
      ``ref_scale``  ``mean|b|``. Useful denominator when the test fails.
    """
    a_f = a.detach().float().flatten()
    b_f = b.detach().float().flatten()
    diff = a_f - b_f
    abs_diff = diff.abs()
    cos = torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0), dim=-1,
    ).item()
    # Sign-match: treat exact zeros as matching (any sign-of-zero
    # convention is fine). Compare nonzero signs.
    sa = torch.sign(a_f)
    sb = torch.sign(b_f)
    sign_match = (sa == sb).float().mean().item()
    ref_l2 = b_f.norm().item()
    rel_l2 = (diff.norm().item() / ref_l2) if ref_l2 > 0 else 0.0
    eps = 1e-12
    max_rel = (abs_diff / b_f.abs().clamp_min(eps)).max().item()
    return {
        "cos": cos,
        "sign_match": sign_match,
        "rel_l2": rel_l2,
        "max_rel": max_rel,
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "ref_scale": b_f.abs().mean().item(),
    }


def _compare(
    label: str,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    cos_tol: float,
    sign_tol: float,
    rel_l2_tol: float,
) -> None:
    """Assert ``a`` matches reference ``b`` on cosine, sign agreement, and
    L2 relative error. Raises ``AssertionError`` listing every failing
    metric (not the first; surface all simultaneously so a wrong grad
    is diagnosed in one report).
    """
    s = _diffstats(a, b)
    failures = []
    if s["cos"] < cos_tol:
        failures.append(f"cos={s['cos']:.6f} < {cos_tol}")
    if s["sign_match"] < sign_tol:
        failures.append(f"sign_match={s['sign_match']:.4f} < {sign_tol}")
    if s["rel_l2"] > rel_l2_tol:
        failures.append(f"rel_l2={s['rel_l2']:.3e} > {rel_l2_tol}")
    if failures:
        raise AssertionError(
            f"{label}: " + "; ".join(failures)
            + f" | stats: cos={s['cos']:.6f} "
            f"sign={s['sign_match']:.4f} rel_l2={s['rel_l2']:.3e} "
            f"max_rel={s['max_rel']:.3e} max_abs={s['max_abs']:.3e} "
            f"mean_abs={s['mean_abs']:.3e} ref_scale={s['ref_scale']:.3e}"
        )


# ---------------------------------------------------------------------------
# Reference module (pure torch; autograd handles the chain rule)
# ---------------------------------------------------------------------------


def _per_head_rmsnorm(
    x: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    """Per-head RMSNorm for Gemma 3 QK-norm. ``x`` is ``(T, H, D)``;
    ``w`` is ``(D,)`` shared across heads. Normalizes the last axis."""
    x_fp = x.float()
    rstd = torch.rsqrt(x_fp.pow(2).mean(-1, keepdim=True) + eps)
    return (x_fp * rstd).to(x.dtype) * w


class NaiveGemmaBlock(torch.nn.Module):
    """Pure-torch reference replicating Gemma 2 / 3 dual-residual math.

    Set ``qk_norm=True`` for the Gemma 3 variant (per-head RMSNorm on
    Q/K after projection, before RoPE). ``window_size_left=-1`` for
    full attention; non-negative for sliding-window.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        expert_dim: int,
        rms_norm_eps: float,
        rope_base: float,
        qk_norm: bool,
        window_size_left: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.qk_norm = qk_norm
        self.window_size_left = window_size_left

        attn_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        z = lambda *s: torch.nn.Parameter(torch.zeros(*s, dtype=DTYPE))
        o = lambda *s: torch.nn.Parameter(torch.ones(*s, dtype=DTYPE))

        self.w_pre_attn_norm = o(d_model)
        self.w_post_attn_norm = o(d_model)
        self.w_pre_ffn_norm = o(d_model)
        self.w_post_ffn_norm = o(d_model)
        self.w_q = z(d_model, attn_dim)
        self.w_k = z(d_model, kv_dim)
        self.w_v = z(d_model, kv_dim)
        self.w_o = z(attn_dim, d_model)
        self.w_1 = z(d_model, expert_dim)
        self.w_2 = z(expert_dim, d_model)
        self.w_3 = z(d_model, expert_dim)
        if qk_norm:
            self.w_q_norm = o(head_dim)
            self.w_k_norm = o(head_dim)

    def forward(
        self, x: torch.Tensor, seq_positions: torch.Tensor
    ) -> torch.Tensor:
        # --- Attention sublayer (dual norm, residual added after) ---
        h = _rmsnorm(x, self.w_pre_attn_norm, self.rms_norm_eps)
        xq = (h @ self.w_q).view(-1, self.n_heads, self.head_dim)
        xk = (h @ self.w_k).view(-1, self.n_kv_heads, self.head_dim)
        xv = (h @ self.w_v).view(-1, self.n_kv_heads, self.head_dim)
        if self.qk_norm:
            xq = _per_head_rmsnorm(xq, self.w_q_norm, self.rms_norm_eps)
            xk = _per_head_rmsnorm(xk, self.w_k_norm, self.rms_norm_eps)
        rope_q = _rope_pair_interleave(xq, seq_positions, self.rope_base)
        rope_k = _rope_pair_interleave(xk, seq_positions, self.rope_base)

        T_, H, D = rope_q.shape
        if self.n_kv_heads != H:
            rep = H // self.n_kv_heads
            rope_k = rope_k.repeat_interleave(rep, dim=1)
            xv = xv.repeat_interleave(rep, dim=1)

        q_ = rope_q.transpose(0, 1).float()
        k_ = rope_k.transpose(0, 1).float()
        v_ = xv.transpose(0, 1).float()
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale

        # Causal mask + (optional) sliding-window restriction. flash-attn
        # semantics: with window_size=(W, 0), position i attends to
        # positions in [i-W, i] inclusive. So forbid j > i (causal) AND
        # i - j > W (window).
        idx = torch.arange(T_, device=x.device)
        delta = idx[:, None] - idx[None, :]  # i - j
        block = delta < 0  # j > i (causal)
        if self.window_size_left >= 0:
            block = block | (delta > self.window_size_left)
        mask = torch.where(
            block, torch.tensor(float("-inf"), device=x.device),
            torch.tensor(0.0, device=x.device),
        )
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        attn_out = (
            torch.matmul(probs, v_).transpose(0, 1).to(x.dtype).contiguous()
        )
        attn_flat = attn_out.reshape(T_, -1)
        a_only = attn_flat @ self.w_o
        post_attn = _rmsnorm(a_only, self.w_post_attn_norm, self.rms_norm_eps)
        x_mid = x + post_attn

        # --- FFN sublayer (dual norm, residual added after) ---
        h2 = _rmsnorm(x_mid, self.w_pre_ffn_norm, self.rms_norm_eps)
        x1 = h2 @ self.w_1
        x3 = h2 @ self.w_3
        # Gemma 2/3 use gated-GELU (tanh approximation), NOT SiLU.
        gated = F.gelu(x1.float(), approximate="tanh").to(x1.dtype) * x3
        swiglu_out = gated @ self.w_2
        post_ffn = _rmsnorm(swiglu_out, self.w_post_ffn_norm, self.rms_norm_eps)
        return x_mid + post_ffn


# ---------------------------------------------------------------------------
# Harness for standalone block invocation (no engine).
# ---------------------------------------------------------------------------


class _MiniKV:
    """Minimal KV ring for a single full-prefill chunk."""

    def __init__(self, *, max_t: int, n_kv_heads: int, head_dim: int) -> None:
        shp = (max_t, n_kv_heads, head_dim)
        self.k = torch.zeros(shp, dtype=DTYPE, device=DEVICE)
        self.v = torch.zeros(shp, dtype=DTYPE, device=DEVICE)
        self.dk = torch.zeros(shp, dtype=DTYPE, device=DEVICE)
        self.dv = torch.zeros(shp, dtype=DTYPE, device=DEVICE)


def _dims_for(cfg) -> Dict[str, int]:
    return {
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "attn_dim": cfg.n_heads * cfg.head_dim,
        "kv_dim": cfg.n_kv_heads * cfg.head_dim,
        "expert_dim": cfg.expert_dim,
    }


def _random_weights(
    block, dims: Dict[str, int], *, gen: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Allocate every entry in ``block.param_spec``. RMSNorm γ tensors are
    initialized to ``1 + 0.02 * N(0,1)`` to match the canonical post-load
    Gemma γ regime (the HF loader applies a ``+1`` shift; here we
    construct γ already in the canonical range).
    """
    weights: Dict[str, torch.Tensor] = {}
    for spec in block.param_spec.tensors:
        shape = spec.shape(dims)
        if "norm" in spec.name:
            base = torch.ones(*shape, dtype=spec.compute_dtype, device=DEVICE)
            noise = torch.randn(
                *shape, generator=gen, dtype=torch.float32, device=DEVICE,
            ) * 0.02
            w = base + noise.to(spec.compute_dtype)
        else:
            w = (
                torch.randn(
                    *shape, generator=gen, dtype=torch.float32, device=DEVICE,
                ) * 0.02
            ).to(spec.compute_dtype)
        weights[spec.name] = w
    return weights


_REF_PARAM_NAMES_BASE = (
    "w_pre_attn_norm", "w_post_attn_norm",
    "w_pre_ffn_norm", "w_post_ffn_norm",
    "w_q", "w_k", "w_v", "w_o",
    "w_1", "w_2", "w_3",
)
_REF_PARAM_NAMES_QK_NORM = ("w_q_norm", "w_k_norm")


def _build_reference(
    cfg, *, qk_norm: bool, weights: Dict[str, torch.Tensor],
) -> Tuple[NaiveGemmaBlock, Dict[str, torch.nn.Parameter]]:
    """Construct the autograd reference module and tie its parameters
    (as detached clones with ``requires_grad_(True)``) to the flextrain
    weights. Returns ``(module, name_to_ref_param)``."""
    ref = NaiveGemmaBlock(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        expert_dim=cfg.expert_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_base=cfg.rope_base,
        qk_norm=qk_norm,
        window_size_left=cfg.window_size_left,
    ).to(DEVICE)

    name_to_ref_param: Dict[str, torch.nn.Parameter] = {}
    names = list(_REF_PARAM_NAMES_BASE)
    if qk_norm:
        names += list(_REF_PARAM_NAMES_QK_NORM)
    for name in names:
        if name not in weights:
            raise KeyError(
                f"weights missing {name!r}; have {sorted(weights)}"
            )
        ref_param = getattr(ref, name)
        with torch.no_grad():
            ref_param.copy_(weights[name])
        name_to_ref_param[name] = ref_param
    return ref, name_to_ref_param


def _make_chunk(t: int) -> ChunkMeta:
    return ChunkMeta.build(
        seq_lens=[t],
        seq_positions=list(range(t)),
        prior_seq_lens=[0],
        prior_seq_offsets=[0],
        device=DEVICE,
    )


def _make_ctx(*, max_t: int, n_kv_heads: int, head_dim: int) -> LayerContext:
    kv = _MiniKV(max_t=max_t, n_kv_heads=n_kv_heads, head_dim=head_dim)
    return LayerContext(
        scratch=ScratchPool(device=DEVICE),
        kv_cache=kv,
        stream=torch.cuda.current_stream(),
        secondary_stream=None,
        total_tokens_per_step=max_t,
    )


def _allocate_slot(block, t: int, dims: Dict[str, int], level: int) -> ActivationSlot:
    """Allocate every declared field (GPU-ring style) and wrap with
    ``ActivationSlot(level=level)`` so ``slot.has()`` reports the save
    level correctly. Higher-tier fields are still backed by storage —
    the schema invariant is that the ring is sized at ``max_tier`` and
    ``has()`` gates visibility, not allocation."""
    tensors = {
        f.name: torch.empty(
            f.shape_fn(t, dims), dtype=f.dtype, device=DEVICE,
        )
        for f in block.schema.fields
    }
    return ActivationSlot(schema=block.schema, level=level, tensors=tensors)


# ---------------------------------------------------------------------------
# Parameterization
# ---------------------------------------------------------------------------


def _build_block_and_cfg(variant: str, attn_kind: str):
    """Construct the flextrain block + matching config. ``attn_kind`` is
    ``"full"`` (window_size_left=-1) or ``"sliding"`` (=SLIDING_WINDOW)."""
    window = -1 if attn_kind == "full" else SLIDING_WINDOW
    if variant == "gemma2":
        cfg = Gemma2BlockConfig(
            d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM, expert_dim=EXPERT_DIM,
            rms_norm_eps=RMS_NORM_EPS, rope_base=ROPE_BASE,
            is_causal=True, attn_logit_softcap=0.0,
            final_logit_softcap=0.0, window_size_left=window,
            compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        )
        return Gemma2Block(layer_id=0, cfg=cfg), cfg, False
    if variant == "gemma3":
        cfg = Gemma3BlockConfig(
            d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM, expert_dim=EXPERT_DIM,
            rms_norm_eps=RMS_NORM_EPS, rope_base=ROPE_BASE,
            is_causal=True, attn_logit_softcap=None,
            final_logit_softcap=None, window_size_left=window,
            compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        )
        return Gemma3Block(layer_id=0, cfg=cfg), cfg, True
    raise ValueError(variant)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="flextrain block parity requires CUDA",
)


@pytest.mark.parametrize("save_tier", [0, 1, 2, 3])
@pytest.mark.parametrize("attn_kind", ["full", "sliding"])
@pytest.mark.parametrize("variant", ["gemma2", "gemma3"])
def test_gemma_block_forward_parity(
    variant: str, attn_kind: str, save_tier: int,
) -> None:
    """Forward parity: flextrain block ≈ pure-torch autograd reference.

    Save tier doesn't affect forward output (forward always populates
    every field; the save level only matters for what backward
    recomputes). We still parameterize over it to keep the test matrix
    aligned with the backward test below.
    """
    torch.manual_seed(0xA5 ^ hash((variant, attn_kind, save_tier)) & 0xFFFFFFFF)
    gen = torch.Generator(device=DEVICE).manual_seed(7)

    block, cfg, qk_norm = _build_block_and_cfg(variant, attn_kind)
    dims = _dims_for(cfg)
    weights = _random_weights(block, dims, gen=gen)
    ref, _ = _build_reference(cfg, qk_norm=qk_norm, weights=weights)

    x_ref = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)
    # FT's gemma forward overwrites the input buffer (uses x as
    # SwiGLU's out_tensor). Hand it a clone so the reference's x stays
    # pristine.
    x_ft = x_ref.clone()

    chunk = _make_chunk(T)
    ctx = _make_ctx(max_t=T, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim)
    slot = _allocate_slot(block, T, dims, level=save_tier)

    out_ft = block.forward(x_ft, chunk, weights, slot, ctx)
    out_ref = ref(x_ref, chunk.seq_positions)

    _compare(
        f"out[{variant}-{attn_kind}-tier{save_tier}]",
        out_ft, out_ref,
        cos_tol=FWD_COS_TOL,
        sign_tol=FWD_SIGN_TOL,
        rel_l2_tol=FWD_REL_L2_TOL,
    )


@pytest.mark.parametrize("save_tier", [0, 1, 2, 3])
@pytest.mark.parametrize("attn_kind", ["full", "sliding"])
@pytest.mark.parametrize("variant", ["gemma2", "gemma3"])
def test_gemma_block_recompute_then_backward_parity(
    variant: str, attn_kind: str, save_tier: int,
) -> None:
    """``forward + (zero higher-tier fields) + forward_recompute + backward``
    ≈ autograd reference for every parameter gradient.

    This is the engine's actual backward flow at a lower save tier:
    the host slot holds only fields with ``tier <= save_tier``;
    higher-tier fields must be regenerated by ``forward_recompute``
    before backward. We simulate offload by explicitly zeroing the
    storage of higher-tier fields after forward — if recompute is
    wrong, backward will read zeros and produce wildly wrong grads.
    """
    torch.manual_seed(0xBEEF ^ hash((variant, attn_kind, save_tier)) & 0xFFFFFFFF)
    gen = torch.Generator(device=DEVICE).manual_seed(17)

    block, cfg, qk_norm = _build_block_and_cfg(variant, attn_kind)
    dims = _dims_for(cfg)
    weights = _random_weights(block, dims, gen=gen)
    ref, ref_param_for = _build_reference(
        cfg, qk_norm=qk_norm, weights=weights,
    )

    x_ref = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)
    x_ft = x_ref.clone()
    dout = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)

    chunk = _make_chunk(T)
    ctx = _make_ctx(max_t=T, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim)
    # Engine-style: slot.level = save_tier (so slot.has() reports
    # accordingly), even though every field's storage is allocated.
    slot = _allocate_slot(block, T, dims, level=save_tier)

    out_ft = block.forward(x_ft, chunk, weights, slot, ctx)
    out_ref = ref(x_ref, chunk.seq_positions)

    # Simulate offload: zero the storage of every field above
    # ``save_tier`` so recompute can't accidentally read stale data.
    # Tier-0 fields (rstds, x_inp, x_mid, a_only, ffn_only, xk, xv) stay.
    for f in block.schema.fields:
        if f.tier > save_tier:
            getattr(slot, f.name).zero_()

    block.forward_recompute(slot, chunk, weights, ctx)

    # Grad buffers and reference grads.
    grads: Dict[str, torch.Tensor] = {}
    for spec in block.param_spec.tensors:
        grads["g_" + spec.name[2:]] = torch.zeros(
            spec.shape(dims), dtype=spec.grad_dtype, device=DEVICE,
        )
    out_ref.backward(dout.float().to(out_ref.dtype))
    block.backward(dout, chunk, weights, grads, slot, ctx)

    names = list(_REF_PARAM_NAMES_BASE)
    if qk_norm:
        names += list(_REF_PARAM_NAMES_QK_NORM)
    failures: list[str] = []
    for name in names:
        ref_grad = ref_param_for[name].grad
        if ref_grad is None:
            failures.append(f"{name}: reference grad is None")
            continue
        ft_grad = grads["g_" + name[2:]]
        try:
            _compare(
                name, ft_grad, ref_grad,
                cos_tol=BWD_COS_TOL,
                sign_tol=BWD_SIGN_TOL,
                rel_l2_tol=BWD_REL_L2_TOL,
            )
        except AssertionError as e:
            failures.append(str(e))
    if failures:
        msg = "\n".join(failures)
        raise AssertionError(f"gradient mismatches:\n{msg}")


@pytest.mark.parametrize("attn_kind", ["full", "sliding"])
@pytest.mark.parametrize("variant", ["gemma2", "gemma3"])
def test_gemma_block_backward_parity(variant: str, attn_kind: str) -> None:
    """Backward parity: every weight gradient matches autograd reference.

    Validates every gradient in the layer's ``param_spec``:
      - Gemma 2: 11 grads (w_pre/post_{attn,ffn}_norm, w_{q,k,v,o,1,2,3}).
      - Gemma 3: 13 grads (+ w_q_norm, w_k_norm).
    """
    torch.manual_seed(0xC0FE ^ hash((variant, attn_kind)) & 0xFFFFFFFF)
    gen = torch.Generator(device=DEVICE).manual_seed(13)

    block, cfg, qk_norm = _build_block_and_cfg(variant, attn_kind)
    dims = _dims_for(cfg)
    weights = _random_weights(block, dims, gen=gen)
    ref, ref_param_for = _build_reference(
        cfg, qk_norm=qk_norm, weights=weights,
    )

    x_ref = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)
    x_ft = x_ref.clone()
    dout = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)

    chunk = _make_chunk(T)
    ctx = _make_ctx(max_t=T, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim)
    # Backward needs every activation present → save_tier = max_tier so
    # forward_recompute is never required (Phase A stubs it to pass).
    slot = _allocate_slot(block, T, dims, level=block.schema.max_tier)

    # Forward both sides.
    out_ft = block.forward(x_ft, chunk, weights, slot, ctx)
    out_ref = ref(x_ref, chunk.seq_positions)

    # Allocate per-param grad buffers matching ParamSpec.grad_dtype.
    grads: Dict[str, torch.Tensor] = {}
    for spec in block.param_spec.tensors:
        grads["g_" + spec.name[2:]] = torch.zeros(
            spec.shape(dims), dtype=spec.grad_dtype, device=DEVICE,
        )

    # Reference backward via autograd.
    out_ref.backward(dout.float().to(out_ref.dtype))

    # FT backward — currently raises NotImplementedError (Phase A).
    block.backward(dout, chunk, weights, grads, slot, ctx)

    # Compare every parameter gradient on cosine / sign-match / rel-L2.
    names = list(_REF_PARAM_NAMES_BASE)
    if qk_norm:
        names += list(_REF_PARAM_NAMES_QK_NORM)
    failures: list[str] = []
    for name in names:
        ref_grad = ref_param_for[name].grad
        if ref_grad is None:
            failures.append(f"{name}: reference grad is None")
            continue
        ft_grad = grads["g_" + name[2:]]
        try:
            _compare(
                name, ft_grad, ref_grad,
                cos_tol=BWD_COS_TOL,
                sign_tol=BWD_SIGN_TOL,
                rel_l2_tol=BWD_REL_L2_TOL,
            )
        except AssertionError as e:
            failures.append(str(e))
    if failures:
        msg = "\n".join(failures)
        raise AssertionError(f"gradient mismatches:\n{msg}")
