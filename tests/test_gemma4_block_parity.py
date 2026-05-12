"""Block-level forward parity for Gemma 4 dense (text-only path).

Validates :class:`flextrain.nn.layers.gemma4.Gemma4Block` against a
pure-torch autograd reference. Two variants exercised:

* ``sliding`` — sliding-window attention, head_dim=32, full RoPE,
  separate W_v, V-RMSNorm everywhere.
* ``full_k_eq_v`` — full attention, larger head_dim=64
  (``global_head_dim``), proportional partial RoPE (rot_dim=16 out of
  head_dim=64), NO W_v projection (V is the K-projection output run
  through V-norm), per-head QK-norm.

Backward parity is xfailed because ``Gemma4Block.backward`` currently
raises ``NotImplementedError`` — the dual-residual + V-norm + k_eq_v
derivation is documented in ``docs/internal/gemma4_status.md`` and lands
in a follow-up session.

Borrows the parity machinery (``_diffstats``, ``_compare``, ``_MiniKV``,
``_make_chunk``, ``_allocate_slot``) from
``tests/test_gemma3_block_parity.py`` so the assertion thresholds and
slot construction stay aligned with the Gemma 2 / 3 tests.
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import _rmsnorm, _rope_pair_interleave
from flextrain.core.activation_schema import ActivationSlot
from flextrain.core.layer import ChunkMeta, LayerContext
from flextrain.engine.buffers import ScratchPool
from flextrain.nn.layers.gemma4 import Gemma4Block, Gemma4BlockConfig

from tests.test_gemma3_block_parity import (
    _diffstats, _compare, _MiniKV, _make_chunk, _allocate_slot,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Gemma 4 block parity requires CUDA (uses flash-attn + Triton kernels)",
)


# ---------------------------------------------------------------------------
# Test shapes — small but exercise every code path.
# ---------------------------------------------------------------------------

DTYPE = torch.bfloat16
DEVICE = "cuda:0"

D_MODEL = 128
N_HEADS = 4
N_KV_SLIDING = 2
N_KV_GLOBAL = 1     # Mimics 31B's 32:4 (sliding) → 32:1 (global) widening.
HEAD_DIM_SLIDING = 32
HEAD_DIM_GLOBAL = 64  # ~ "global_head_dim", doubled vs sliding (mimics 31B's 256→512).
EXPERT_DIM = 256
T = 48
SLIDING_WINDOW = 16
RMS_NORM_EPS = 1e-6
ROPE_LOCAL_BASE = 10_000.0
ROPE_GLOBAL_BASE = 1_000_000.0
GLOBAL_PARTIAL_ROTARY_FACTOR = 0.25  # → rot_dim = 16


FWD_COS_TOL = 0.9995
FWD_SIGN_TOL = 0.99
FWD_REL_L2_TOL = 5e-2

BWD_COS_TOL = 0.998
BWD_SIGN_TOL = 0.95
BWD_REL_L2_TOL = 8e-2


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _per_head_rmsnorm_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm with no learnable γ (HF's ``with_scale=False``).
    ``x`` shape ``(T, H, D)``; normalizes the last axis per head.
    """
    x_fp = x.float()
    rstd = torch.rsqrt(x_fp.pow(2).mean(-1, keepdim=True) + eps)
    return (x_fp * rstd).to(x.dtype)


def _per_head_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp = x.float()
    rstd = torch.rsqrt(x_fp.pow(2).mean(-1, keepdim=True) + eps)
    return (x_fp * rstd).to(x.dtype) * w


def _rope_partial_proportional_pair(
    x: torch.Tensor,
    seq_positions: torch.Tensor,
    rope_base: float,
    head_dim: int,
    rot_dim: int,
) -> torch.Tensor:
    """Proportional partial-rope in pair-interleave layout (matches
    flextrain's kernel convention; halved→pair load permute already
    applied to weights upstream).

    Channels [0, rot_dim) rotate with inv_freq[i] = base ** (-2i/head_dim)
    (note: head_dim, NOT rot_dim). Channels [rot_dim, head_dim) pass
    through unchanged.
    """
    T_, H, D = x.shape
    assert D == head_dim
    assert rot_dim % 2 == 0
    pair_count = rot_dim // 2
    inv_freq = 1.0 / (
        rope_base
        ** (torch.arange(0, pair_count, device=x.device, dtype=torch.float32)
            * 2.0 / head_dim)
    )
    pos = seq_positions.view(-1).float()
    angles = pos.unsqueeze(-1) * inv_freq               # (T, pair_count)
    cos = angles.cos().to(x.dtype).unsqueeze(1)          # (T, 1, pair_count)
    sin = angles.sin().to(x.dtype).unsqueeze(1)
    x_fp = x.float()
    out = x_fp.clone()
    even = x_fp[..., 0:rot_dim:2]                        # (T, H, pair_count)
    odd = x_fp[..., 1:rot_dim:2]
    rot_even = even * cos.float() - odd * sin.float()
    rot_odd = even * sin.float() + odd * cos.float()
    out[..., 0:rot_dim:2] = rot_even
    out[..., 1:rot_dim:2] = rot_odd
    return out.to(x.dtype)


# ---------------------------------------------------------------------------
# Reference module (pure torch)
# ---------------------------------------------------------------------------


class NaiveGemma4Block(torch.nn.Module):
    """Pure-torch reference replicating Gemma 4 dual-residual + V-norm
    (+ optional k_eq_v) + proportional partial-rope (when prf < 1).

    ``layer_scalar`` is left at 1.0 (default) — the loader would override
    it from the checkpoint, but for block-parity testing it's a constant
    multiplier and we keep it at the identity.
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
        window_size_left: int,
        k_eq_v: bool,
        partial_rotary_factor: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.window_size_left = window_size_left
        self.k_eq_v = k_eq_v
        self.partial_rotary_factor = partial_rotary_factor

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
        if not k_eq_v:
            self.w_v = z(d_model, kv_dim)
        self.w_o = z(attn_dim, d_model)
        self.w_q_norm = o(head_dim)
        self.w_k_norm = o(head_dim)
        # V-norm has no γ (with_scale=False). Stored as a non-parameter for clarity.
        self.w_1 = z(d_model, expert_dim)
        self.w_2 = z(expert_dim, d_model)
        self.w_3 = z(d_model, expert_dim)

    def _rope(self, x: torch.Tensor, seq_positions: torch.Tensor) -> torch.Tensor:
        if self.partial_rotary_factor < 1.0:
            rot_dim = int(self.head_dim * self.partial_rotary_factor)
            return _rope_partial_proportional_pair(
                x, seq_positions, self.rope_base, self.head_dim, rot_dim,
            )
        return _rope_pair_interleave(x, seq_positions, self.rope_base)

    def forward(
        self, x: torch.Tensor, seq_positions: torch.Tensor
    ) -> torch.Tensor:
        # Attention sublayer.
        h = _rmsnorm(x, self.w_pre_attn_norm, self.rms_norm_eps)
        xq = (h @ self.w_q).view(-1, self.n_heads, self.head_dim)
        xk = (h @ self.w_k).view(-1, self.n_kv_heads, self.head_dim)
        if self.k_eq_v:
            # V := pre-K-norm, pre-rope output of W_k @ x (a copy of xk).
            xv = xk.clone()
        else:
            xv = (h @ self.w_v).view(-1, self.n_kv_heads, self.head_dim)

        # V-norm (no γ). Same xv tensor whether k_eq_v or not.
        xv = _per_head_rmsnorm_no_scale(xv, self.rms_norm_eps)
        # Q/K-norm (with γ).
        xq = _per_head_rmsnorm(xq, self.w_q_norm, self.rms_norm_eps)
        xk = _per_head_rmsnorm(xk, self.w_k_norm, self.rms_norm_eps)
        # RoPE on Q and K (partial / proportional or full).
        rope_q = self._rope(xq, seq_positions)
        rope_k = self._rope(xk, seq_positions)

        T_, H, D = rope_q.shape
        H_kv = rope_k.shape[1]
        if H_kv != H:
            rep = H // H_kv
            rope_k = rope_k.repeat_interleave(rep, dim=1)
            xv = xv.repeat_interleave(rep, dim=1)
        q_ = rope_q.transpose(0, 1).float()
        k_ = rope_k.transpose(0, 1).float()
        v_ = xv.transpose(0, 1).float()
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale

        idx = torch.arange(T_, device=x.device)
        delta = idx[:, None] - idx[None, :]
        block = delta < 0
        if self.window_size_left >= 0:
            block = block | (delta > self.window_size_left)
        mask = torch.where(
            block,
            torch.tensor(float("-inf"), device=x.device),
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

        # FFN sublayer.
        h2 = _rmsnorm(x_mid, self.w_pre_ffn_norm, self.rms_norm_eps)
        x1 = h2 @ self.w_1
        x3 = h2 @ self.w_3
        gated = F.gelu(x1.float(), approximate="tanh").to(x1.dtype) * x3
        swiglu_out = gated @ self.w_2
        post_ffn = _rmsnorm(swiglu_out, self.w_post_ffn_norm, self.rms_norm_eps)
        return x_mid + post_ffn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(*, max_t: int, n_kv_heads: int, head_dim: int) -> LayerContext:
    kv = _MiniKV(max_t=max_t, n_kv_heads=n_kv_heads, head_dim=head_dim)
    return LayerContext(
        scratch=ScratchPool(device=DEVICE),
        kv_cache=kv,
        stream=torch.cuda.current_stream(),
        secondary_stream=None,
        total_tokens_per_step=max_t,
    )


def _dims_for(cfg: Gemma4BlockConfig) -> Dict[str, int]:
    return {
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "attn_dim": cfg.n_heads * cfg.head_dim,
        "kv_dim": cfg.n_kv_heads * cfg.head_dim,
        "expert_dim": cfg.expert_dim,
    }


def _random_weights(block, dims, *, gen) -> Dict[str, torch.Tensor]:
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


def _copy_weights_to_reference(
    ref: NaiveGemma4Block,
    weights: Dict[str, torch.Tensor],
    *,
    k_eq_v: bool,
) -> None:
    """Copy flextrain weights into the reference module's parameters.
    W_v is absent on k_eq_v=True (skip it)."""
    names = [
        "w_pre_attn_norm", "w_post_attn_norm",
        "w_pre_ffn_norm", "w_post_ffn_norm",
        "w_q", "w_k", "w_o",
        "w_q_norm", "w_k_norm",
        "w_1", "w_2", "w_3",
    ]
    if not k_eq_v:
        names.append("w_v")
    for name in names:
        ref_param = getattr(ref, name)
        with torch.no_grad():
            ref_param.copy_(weights[name])


def _build_block_and_cfg(variant: str) -> tuple[Gemma4Block, Gemma4BlockConfig]:
    if variant == "sliding":
        cfg = Gemma4BlockConfig(
            d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_KV_SLIDING,
            head_dim=HEAD_DIM_SLIDING, expert_dim=EXPERT_DIM,
            rms_norm_eps=RMS_NORM_EPS,
            rope_base=ROPE_LOCAL_BASE,
            rope_scaling=None,
            window_size_left=SLIDING_WINDOW,
            attn_logit_softcap=None,
            v_norm=True,
            k_eq_v=False,
            partial_rotary_factor=1.0,
            compute_dtype=DTYPE,
        )
    elif variant == "full_k_eq_v":
        cfg = Gemma4BlockConfig(
            d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_KV_GLOBAL,
            head_dim=HEAD_DIM_GLOBAL, expert_dim=EXPERT_DIM,
            rms_norm_eps=RMS_NORM_EPS,
            rope_base=ROPE_GLOBAL_BASE,
            rope_scaling={"rope_type": "proportional"},
            window_size_left=-1,
            attn_logit_softcap=None,
            v_norm=True,
            k_eq_v=True,
            partial_rotary_factor=GLOBAL_PARTIAL_ROTARY_FACTOR,
            compute_dtype=DTYPE,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return Gemma4Block(0, cfg), cfg


# ---------------------------------------------------------------------------
# Forward parity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["sliding", "full_k_eq_v"])
def test_gemma4_block_forward_parity(variant: str) -> None:
    """Forward parity vs hand-rolled torch reference.

    The two variants exercise:
    * ``sliding``: standard W_v path + V-norm + sliding window + full rope.
    * ``full_k_eq_v``: no W_v + V-norm-of-W_k-output + full attention +
      proportional partial rope.
    """
    block, cfg = _build_block_and_cfg(variant)
    dims = _dims_for(cfg)
    gen = torch.Generator(device=DEVICE).manual_seed(0xc0ffee + hash(variant) % 9999)

    weights = _random_weights(block, dims, gen=gen)
    ref = NaiveGemma4Block(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim, expert_dim=cfg.expert_dim,
        rms_norm_eps=cfg.rms_norm_eps, rope_base=cfg.rope_base,
        window_size_left=cfg.window_size_left,
        k_eq_v=cfg.k_eq_v,
        partial_rotary_factor=cfg.partial_rotary_factor,
    ).to(DEVICE)
    _copy_weights_to_reference(ref, weights, k_eq_v=cfg.k_eq_v)

    # Random input.
    x = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)
    chunk = _make_chunk(T)
    seq_positions = torch.arange(T, device=DEVICE, dtype=torch.int32).view(-1, 1)

    # Reference forward (autograd graph kept off for memory).
    with torch.no_grad():
        y_ref = ref(x, seq_positions)

    # Flextrain forward.
    ctx = _make_ctx(max_t=T, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim)
    slot = _allocate_slot(block, T, dims, level=block.schema.max_tier)
    y_ft = block.forward(x, chunk, weights, slot, ctx)

    _compare(
        f"gemma4/{variant}/forward",
        y_ft, y_ref,
        cos_tol=FWD_COS_TOL,
        sign_tol=FWD_SIGN_TOL,
        rel_l2_tol=FWD_REL_L2_TOL,
    )


# ---------------------------------------------------------------------------
# Backward parity tests
# ---------------------------------------------------------------------------


# Param names per variant. ``full_k_eq_v`` has NO ``w_v`` and NO
# ``g_v`` (V is the W_k output run through v_norm; gradient folds into
# g_k upstream of this comparison). V-norm has no γ so no ``w_v_norm``
# exists in either variant.
_PARAMS_SLIDING = (
    "w_pre_attn_norm", "w_post_attn_norm",
    "w_pre_ffn_norm", "w_post_ffn_norm",
    "w_q", "w_k", "w_v", "w_o",
    "w_q_norm", "w_k_norm",
    "w_1", "w_2", "w_3",
)
_PARAMS_FULL_K_EQ_V = (
    "w_pre_attn_norm", "w_post_attn_norm",
    "w_pre_ffn_norm", "w_post_ffn_norm",
    "w_q", "w_k", "w_o",
    "w_q_norm", "w_k_norm",
    "w_1", "w_2", "w_3",
)


def _ref_param_for(variant: str) -> tuple[str, ...]:
    return _PARAMS_SLIDING if variant == "sliding" else _PARAMS_FULL_K_EQ_V


@pytest.mark.parametrize("variant", ["sliding", "full_k_eq_v"])
def test_gemma4_block_backward_parity(variant: str) -> None:
    """Backward parity: every weight gradient matches the torch
    autograd reference for the given variant. Save tier = max_tier so
    forward_recompute is unused (covered by the recompute test below).
    """
    block, cfg = _build_block_and_cfg(variant)
    dims = _dims_for(cfg)
    gen = torch.Generator(device=DEVICE).manual_seed(0xfeed + hash(variant) % 9999)
    weights = _random_weights(block, dims, gen=gen)

    ref = NaiveGemma4Block(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim, expert_dim=cfg.expert_dim,
        rms_norm_eps=cfg.rms_norm_eps, rope_base=cfg.rope_base,
        window_size_left=cfg.window_size_left,
        k_eq_v=cfg.k_eq_v,
        partial_rotary_factor=cfg.partial_rotary_factor,
    ).to(DEVICE)
    _copy_weights_to_reference(ref, weights, k_eq_v=cfg.k_eq_v)

    x_ref = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)
    x_ft = x_ref.clone()
    dout = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)

    chunk = _make_chunk(T)
    seq_positions = torch.arange(T, device=DEVICE, dtype=torch.int32).view(-1, 1)
    ctx = _make_ctx(max_t=T, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim)
    slot = _allocate_slot(block, T, dims, level=block.schema.max_tier)

    # Forward both.
    out_ft = block.forward(x_ft, chunk, weights, slot, ctx)
    out_ref = ref(x_ref, seq_positions)

    # Reference backward via autograd.
    out_ref.backward(dout.float().to(out_ref.dtype))

    # FT backward into per-param grad buffers.
    grads: Dict[str, torch.Tensor] = {}
    for spec in block.param_spec.tensors:
        grads["g_" + spec.name[2:]] = torch.zeros(
            spec.shape(dims), dtype=spec.grad_dtype or spec.compute_dtype,
            device=DEVICE,
        )
    block.backward(dout, chunk, weights, grads, slot, ctx)

    failures: list[str] = []
    for name in _ref_param_for(variant):
        ref_param = getattr(ref, name)
        ref_grad = ref_param.grad
        if ref_grad is None:
            failures.append(f"{name}: reference grad is None")
            continue
        ft_grad = grads["g_" + name[2:]]
        try:
            _compare(
                f"gemma4/{variant}/{name}",
                ft_grad, ref_grad,
                cos_tol=BWD_COS_TOL,
                sign_tol=BWD_SIGN_TOL,
                rel_l2_tol=BWD_REL_L2_TOL,
            )
        except AssertionError as e:
            failures.append(str(e))
    if failures:
        msg = "\n".join(failures)
        raise AssertionError(f"gradient mismatches:\n{msg}")


@pytest.mark.parametrize("save_tier", [0, 1, 2, 3])
@pytest.mark.parametrize("variant", ["sliding", "full_k_eq_v"])
def test_gemma4_block_recompute_then_backward_parity(
    variant: str, save_tier: int,
) -> None:
    """``forward + (zero higher-tier fields) + forward_recompute +
    backward`` ≈ autograd reference for every weight gradient. Mirrors
    the Gemma 3 recompute-then-bwd test in
    ``tests/test_gemma3_block_parity.py``.
    """
    block, cfg = _build_block_and_cfg(variant)
    dims = _dims_for(cfg)
    gen = torch.Generator(device=DEVICE).manual_seed(
        0xface ^ ((hash((variant, save_tier))) & 0xFFFFFFFF)
    )
    weights = _random_weights(block, dims, gen=gen)

    ref = NaiveGemma4Block(
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim, expert_dim=cfg.expert_dim,
        rms_norm_eps=cfg.rms_norm_eps, rope_base=cfg.rope_base,
        window_size_left=cfg.window_size_left,
        k_eq_v=cfg.k_eq_v,
        partial_rotary_factor=cfg.partial_rotary_factor,
    ).to(DEVICE)
    _copy_weights_to_reference(ref, weights, k_eq_v=cfg.k_eq_v)

    x_ref = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)
    x_ft = x_ref.clone()
    dout = torch.randn(
        T, cfg.d_model, generator=gen, dtype=torch.float32, device=DEVICE,
    ).to(DTYPE)

    chunk = _make_chunk(T)
    seq_positions = torch.arange(T, device=DEVICE, dtype=torch.int32).view(-1, 1)
    ctx = _make_ctx(max_t=T, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim)
    slot = _allocate_slot(block, T, dims, level=save_tier)

    block.forward(x_ft, chunk, weights, slot, ctx)
    out_ref = ref(x_ref, seq_positions)

    # Simulate offload: zero every field with tier > save_tier.
    for f in block.schema.fields:
        if f.tier > save_tier:
            getattr(slot, f.name).zero_()
    block.forward_recompute(slot, chunk, weights, ctx)

    out_ref.backward(dout.float().to(out_ref.dtype))

    grads: Dict[str, torch.Tensor] = {}
    for spec in block.param_spec.tensors:
        grads["g_" + spec.name[2:]] = torch.zeros(
            spec.shape(dims), dtype=spec.grad_dtype or spec.compute_dtype,
            device=DEVICE,
        )
    block.backward(dout, chunk, weights, grads, slot, ctx)

    failures: list[str] = []
    for name in _ref_param_for(variant):
        ref_param = getattr(ref, name)
        ref_grad = ref_param.grad
        if ref_grad is None:
            failures.append(f"{name}: reference grad is None")
            continue
        ft_grad = grads["g_" + name[2:]]
        try:
            _compare(
                f"gemma4/{variant}/tier{save_tier}/{name}",
                ft_grad, ref_grad,
                cos_tol=BWD_COS_TOL,
                sign_tol=BWD_SIGN_TOL,
                rel_l2_tol=BWD_REL_L2_TOL,
            )
        except AssertionError as e:
            failures.append(str(e))
    if failures:
        msg = "\n".join(failures)
        raise AssertionError(f"gradient mismatches:\n{msg}")
