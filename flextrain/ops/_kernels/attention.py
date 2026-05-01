"""Flash-attention varlen wrappers.

Public entry points: :func:`flextrain_attention_fwd` and
:func:`flextrain_attention_bwd`. Each dispatches to the best available
backend at runtime:

  1. flash-attn 4 (CUTE DSL, sm_90+, fastest where supported — checks
     for ``flash_attn.cute.interface._flash_attn_fwd``)
  2. flash-attn 3 (Hopper-only, ``flash_attn_interface._flash_attn_forward``)
  3. flash-attn 2 (sm_75+ varlen, ``flash_attn.flash_attn_interface``)
  4. eager-PyTorch fallback (always available; slow but correct).

This module calls the **public** ``_flash_attn_*_forward / _backward``
entry points exported by upstream's various ``*_interface`` modules
rather than the raw C-extension symbols (``flash_attn_2_cuda.varlen_fwd``
etc.). The public API is the supported, no-autograd path the upstream
autograd Functions use internally; it does contiguous-fixup, default-
arg derivation, and stride checks for us.

Backend selection: by default this module auto-picks the highest-
priority available backend (fa4 > fa3 > fa2 > eager) at each call.
Callers can pin to a specific backend via :func:`set_attention_backend`
("fa4" / "fa3" / "fa2" / "eager"), which raises if the chosen backend
isn't available.

Backend interface contract (kept identical across backends):

* ``flextrain_attention_fwd(q, k, v, out, softmax_lse, ...)``:
  caller supplies ``out`` (overwritten in place) and ``softmax_lse``
  (overwritten in place). Shapes:
    q              (total_q, n_q_heads, head_dim)
    k              (total_k, n_kv_heads, head_dim)
    v              (total_k, n_kv_heads, head_dim)
    out            (total_q, n_q_heads, head_dim)
    softmax_lse    (n_q_heads, total_q)  fp32

* ``flextrain_attention_bwd(dout, ..., dq, dk, dv, ...)``: caller
  supplies ``dq/dk/dv`` (OVERWRITTEN — not accumulated). For
  cross-chunk dK/dV accumulation, callers pass scratch buffers and
  do their own accumulation. See ``GQAAttentionBlock.bwd``.

Both wrappers use bf16 q/k/v/dout/out and fp32 softmax_lse, varlen
indexed via cu_seqlens. No alibi, no dropout, no rotary fold-in
(those happen upstream of these calls).
"""
from __future__ import annotations

import torch


class FlashAttentionNotAvailableError(Exception):
    """Raised when no attention backend is available for this call."""
    pass


# Selected backend, set via :func:`set_attention_backend`. ``None``
# means auto-pick (highest-priority available at call time). Otherwise
# one of "fa4", "fa3", "fa2", "eager".
_SELECTED_BACKEND: str | None = None
_BACKEND_NAMES = ("fa4", "fa3", "fa2", "eager")


# ---------------------------------------------------------------------------
# Backend availability detection (import time).
#
# We probe each backend's public API at import time:
#   * flash-attn 4 (CUTE DSL): ``flash_attn.cute.interface._flash_attn_fwd / _bwd``
#   * flash-attn 3 (Hopper):   ``flash_attn_interface._flash_attn_forward / _backward``
#   * flash-attn 2 (varlen):   ``flash_attn.flash_attn_interface._flash_attn_varlen_forward / _backward``
#
# Any probe failing → that backend is unavailable. The dispatcher
# falls through to the next. We do NOT raise at import time even if
# all are unavailable; defer the error to the first call so importing
# this module on a non-CUDA dev machine doesn't hard-fail.
# ---------------------------------------------------------------------------

try:
    # flash-attn 4 CUTE DSL path. The package's __init__ only re-exports
    # the autograd Functions; the underscore-prefixed no-autograd
    # entry points live in the .interface submodule.
    from flash_attn.cute.interface import (
        _flash_attn_fwd as _f4_fwd,
        _flash_attn_bwd as _f4_bwd,
    )
    FLASH_ATTN_4_AVAILABLE = True
except Exception:
    _f4_fwd = None
    _f4_bwd = None
    FLASH_ATTN_4_AVAILABLE = False

try:
    # flash-attn 3 hopper path. The package layout is a single
    # top-level module ``flash_attn_interface`` that exposes both the
    # raw C extension (``flash_attn_3_cuda``) and the Python wrappers
    # (``_flash_attn_forward / _flash_attn_backward``).
    from flash_attn_interface import (
        _flash_attn_forward as _f3_fwd,
        _flash_attn_backward as _f3_bwd,
    )
    FLASH_ATTN_3_AVAILABLE = True
except Exception:
    _f3_fwd = None
    _f3_bwd = None
    FLASH_ATTN_3_AVAILABLE = False

try:
    from flash_attn.flash_attn_interface import (
        _flash_attn_varlen_forward as _f2_fwd,
        _flash_attn_varlen_backward as _f2_bwd,
    )
    FLASH_ATTN_2_AVAILABLE = True
except Exception:
    _f2_fwd = None
    _f2_bwd = None
    FLASH_ATTN_2_AVAILABLE = False


# Eager fallback is always available (pure torch). It's slow — only
# used when no flash-attn backend is installed. Useful for dev work
# on machines without flash-attn (e.g., when libcudart versions don't
# match) or for unit tests that don't care about perf.
EAGER_AVAILABLE = True


_BACKEND_AVAILABLE = {
    "fa4": lambda: FLASH_ATTN_4_AVAILABLE,
    "fa3": lambda: FLASH_ATTN_3_AVAILABLE,
    "fa2": lambda: FLASH_ATTN_2_AVAILABLE,
    "eager": lambda: EAGER_AVAILABLE,
}


def set_attention_backend(name: str | None) -> None:
    """Pin the attention backend used by the public dispatchers.

    ``name=None`` restores auto-pick (fa4 → fa3 → fa2 → eager). Any
    other value must be one of ``"fa4" / "fa3" / "fa2" / "eager"`` and
    must be currently available; otherwise raises
    :class:`FlashAttentionNotAvailableError`.
    """
    global _SELECTED_BACKEND
    if name is None:
        _SELECTED_BACKEND = None
        return
    if name not in _BACKEND_NAMES:
        raise ValueError(
            f"unknown attention backend {name!r}; expected one of "
            f"{_BACKEND_NAMES} or None for auto"
        )
    if not _BACKEND_AVAILABLE[name]():
        raise FlashAttentionNotAvailableError(
            f"requested attention backend {name!r} is not available in "
            f"this environment. Available: "
            f"{[n for n in _BACKEND_NAMES if _BACKEND_AVAILABLE[n]()]}"
        )
    _SELECTED_BACKEND = name


def get_attention_backend() -> str | None:
    """Currently pinned backend, or ``None`` if auto-pick is active."""
    return _SELECTED_BACKEND


# ---------------------------------------------------------------------------
# Backend wrappers: each takes the same arg names/shapes as the
# top-level ``flextrain_attention_*`` and adapts to the upstream
# public API's expectations.
# ---------------------------------------------------------------------------


def _window_for_flash4(window_size):
    """flash-attn 4 uses ``None`` to mean "no windowing" (the upstream
    ``_resolve_causal_local_window`` checks for None to decide whether
    to apply a sliding-window mask), while flash-attn 2/3 use ``-1``.
    Convert our (-1, -1) sentinel to (None, None)."""
    left, right = window_size
    return (left if left >= 0 else None, right if right >= 0 else None)


def _flash4_fwd(
    q, k, v, out, softmax_lse,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    sm_margin=0, softcap=0.0,
):
    """flash-attn 4 (CUTE DSL) fwd via the public API.

    Upstream's ``_flash_attn_fwd`` accepts a caller-allocated
    ``out=`` and ``lse=`` and returns ``(out, lse)``. Use ``return_lse=True``
    so we get the softmax_lse back. ``softcap=None`` and
    ``window_size_left/right=None`` are upstream's "disabled" sentinels.
    """
    softmax_scale = q.shape[-1] ** -0.5
    win_left, win_right = _window_for_flash4(window_size)
    out_ret, lse_ret = _f4_fwd(
        q, k, v,
        cu_seqlens_q=q_seq_offsets,
        cu_seqlens_k=k_seq_offsets,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap if softcap > 0 else None,
        window_size_left=win_left,
        window_size_right=win_right,
        return_lse=True,
        out=out,
        lse=softmax_lse,
    )
    if out_ret is not out:
        out.copy_(out_ret)
    if lse_ret is not softmax_lse:
        softmax_lse.copy_(lse_ret)
    # ``sm_margin`` is not exposed in flash4's public API; ignored.
    return out, softmax_lse


def _flash4_bwd(
    dout, q, k, v, out, softmax_lse,
    dq, dk, dv,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    deterministic=True, sm_margin=0, softcap=0.0,
):
    """flash-attn 4 (CUTE DSL) bwd via the public API.

    The bwd asserts compute capability ∈ {9.x, 10.x, 11.x, 12.x} —
    Hopper through Blackwell. Caller-supplied ``dq/dk/dv`` are
    overwritten in place. ``sm_margin`` is not exposed; ignored.
    softmax_lse is read directly; copy if non-contiguous.
    """
    softmax_scale = q.shape[-1] ** -0.5
    softmax_lse_contig = softmax_lse.contiguous()
    win_left, win_right = _window_for_flash4(window_size)
    _f4_bwd(
        q, k, v, out, dout, softmax_lse_contig,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size_left=win_left,
        window_size_right=win_right,
        cu_seqlens_q=q_seq_offsets,
        cu_seqlens_k=k_seq_offsets,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        deterministic=deterministic,
        dq=dq, dk=dk, dv=dv,
    )
    return dq, dk, dv


def _flash3_fwd(
    q, k, v, out, softmax_lse,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    sm_margin=0, softcap=0.0,
):
    """flash-attn 3 fwd via the public API.

    Upstream's ``_flash_attn_forward`` accepts a caller-allocated
    ``out_=`` buffer and writes into it directly. softmax_lse is
    returned by upstream — we copy into the caller's buffer.
    """
    softmax_scale = q.shape[-1] ** -0.5
    out_ret, lse_ret, _out_accum, _lse_accum = _f3_fwd(
        q, k, v,
        out_=out,
        cu_seqlens_q=q_seq_offsets,
        cu_seqlens_k=k_seq_offsets,
        seqused_q=None, seqused_k=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        sm_margin=sm_margin,
    )
    # The upstream API copies into ``out_`` in place, but defensively
    # mirror — if it ever decides to allocate fresh, our caller's
    # buffer still gets the result.
    if out_ret is not out:
        out.copy_(out_ret)
    softmax_lse.copy_(lse_ret)
    return out, softmax_lse


def _flash3_bwd(
    dout, q, k, v, out, softmax_lse,
    dq, dk, dv,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    deterministic=True, sm_margin=0, softcap=0.0,
):
    """flash-attn 3 bwd via the public API.

    Upstream signature note: bwd takes ``is_causal`` (not ``causal``)
    and ``sequed_q / sequed_k`` (sic — typo in upstream).
    ``_flash_attn_backward`` mutates ``dq / dk / dv`` in place.

    Upstream's bwd kernel reads softmax_lse via raw pointers and is
    sensitive to non-contiguous strides — copy if needed.
    """
    softmax_scale = q.shape[-1] ** -0.5
    softmax_lse_contig = softmax_lse.contiguous()
    _f3_bwd(
        dout, q, k, v, out, softmax_lse_contig,
        cu_seqlens_q=q_seq_offsets,
        cu_seqlens_k=k_seq_offsets,
        sequed_q=None, sequed_k=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dq=dq, dk=dk, dv=dv,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        deterministic=deterministic,
        sm_margin=sm_margin,
    )
    return dq, dk, dv


def _flash2_fwd(
    q, k, v, out, softmax_lse,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    softcap=0.0,
):
    """flash-attn 2 varlen fwd via the public API.

    Upstream's ``_flash_attn_varlen_forward`` does NOT accept a caller-
    allocated out buffer (the ``out`` slot in its signature is hidden:
    the C extension's varlen_fwd takes one but the Python wrapper
    passes None). It returns a fresh ``out`` tensor; we copy into the
    caller's buffer for interface uniformity.
    """
    softmax_scale = q.shape[-1] ** -0.5
    out_ret, lse_ret, _S_dmask, _rng_state = _f2_fwd(
        q, k, v,
        cu_seqlens_q=q_seq_offsets,
        cu_seqlens_k=k_seq_offsets,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        alibi_slopes=None,
        return_softmax=False,
    )
    out.copy_(out_ret)
    softmax_lse.copy_(lse_ret)
    return out, softmax_lse


def _flash2_bwd(
    dout, q, k, v, out, softmax_lse,
    dq, dk, dv,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    deterministic=True, softcap=0.0,
):
    """flash-attn 2 varlen bwd via the public API.

    ``_flash_attn_varlen_backward`` mutates ``dq / dk / dv`` in place
    (declared via ``mutates_args=("dq","dk","dv")`` on the custom-op
    decorator). Caller-supplied buffers are overwritten — caller is
    responsible for any cross-chunk accumulation.
    """
    softmax_scale = q.shape[-1] ** -0.5
    softmax_lse_contig = softmax_lse.contiguous()
    _f2_bwd(
        dout, q, k, v, out, softmax_lse_contig,
        dq=dq, dk=dk, dv=dv,
        cu_seqlens_q=q_seq_offsets,
        cu_seqlens_k=k_seq_offsets,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        softcap=softcap,
        alibi_slopes=None,
        deterministic=deterministic,
    )
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Eager-PyTorch fallback. Pure torch ops, per-sequence loop. Slow but
# correct, no external deps. Same I/O contract as the flash backends:
# OVERWRITES out/softmax_lse/dq/dk/dv in place.
#
# GQA replication: when n_q_heads != n_kv_heads, per-sequence we
# repeat_interleave k/v along the head dim.
#
# Mask convention:
#   * causal=True: q[i] attends to k[j] iff j <= i (per-sequence).
#   * window_size=(left, right): additionally restrict to
#     i - left <= j <= i + right (when both left and right >= 0).
#     A negative value disables that side. Matches flash2/3.
#
# softcap: scores = softcap * tanh(scores / softcap), elementwise.
# softmax_scale = 1/sqrt(head_dim).
#
# softmax_lse output shape: (n_q_heads, total_q) fp32, holding the
# natural log of the per-row partition function (so that
# row-attn = exp(scores - lse[None, :])).
# ---------------------------------------------------------------------------


def _build_attn_mask(Lq, Lk, *, causal, window_size, device):
    """Bool mask of shape (Lq, Lk). True = position is allowed.

    For decoder-style varlen, ``Lq <= Lk``: queries are the rightmost
    ``Lq`` positions of the key sequence. Causal aligns: q[i] sees
    k[j] iff j <= (Lk - Lq + i). When Lq == Lk this reduces to j <= i.
    """
    win_left, win_right = window_size
    q_pos = torch.arange(Lq, device=device)[:, None]
    k_pos = torch.arange(Lk, device=device)[None, :]
    # Right-anchor query positions inside the (possibly longer) key seq.
    q_in_k = q_pos + (Lk - Lq)
    mask = torch.ones(Lq, Lk, dtype=torch.bool, device=device)
    if causal:
        mask = mask & (k_pos <= q_in_k)
    if win_left >= 0:
        mask = mask & (k_pos >= q_in_k - win_left)
    if win_right >= 0:
        mask = mask & (k_pos <= q_in_k + win_right)
    return mask


def _eager_attention_fwd(
    q, k, v, out, softmax_lse,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    sm_margin=0, softcap=0.0,
):
    """Pure-PyTorch varlen GQA attention fwd.

    Loops over sequences. Slow at scale but only used as a fallback
    when no flash backend is available. ``sm_margin`` is ignored
    (kernel-only knob).
    """
    total_q, n_heads, head_dim = q.shape
    n_kv = k.shape[1]
    softmax_scale = head_dim ** -0.5
    rep = n_heads // n_kv
    # CPU views of the cu_seqlens for the python loop.
    q_off_h = q_seq_offsets.tolist()
    k_off_h = k_seq_offsets.tolist()
    num_seqs = len(q_off_h) - 1

    for s in range(num_seqs):
        q_lo, q_hi = q_off_h[s], q_off_h[s + 1]
        k_lo, k_hi = k_off_h[s], k_off_h[s + 1]
        Lq = q_hi - q_lo
        Lk = k_hi - k_lo
        if Lq == 0 or Lk == 0:
            continue
        q_s = q[q_lo:q_hi]            # (Lq, n_heads, D)
        k_s = k[k_lo:k_hi]            # (Lk, n_kv, D)
        v_s = v[k_lo:k_hi]
        if rep > 1:
            k_s = k_s.repeat_interleave(rep, dim=1)
            v_s = v_s.repeat_interleave(rep, dim=1)
        # (n_heads, Lq, D) @ (n_heads, D, Lk) -> (n_heads, Lq, Lk)
        q_t = q_s.transpose(0, 1).contiguous().float()  # promote to fp32 for stability
        k_t = k_s.transpose(0, 1).contiguous().float()
        v_t = v_s.transpose(0, 1).contiguous().float()
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale
        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        mask = _build_attn_mask(
            Lq, Lk, causal=causal, window_size=window_size, device=q.device,
        )
        scores = scores.masked_fill(~mask, float("-inf"))
        # numerically-stable softmax with saved lse.
        m = scores.max(dim=-1, keepdim=True).values  # (n_heads, Lq, 1)
        # Replace -inf rows (no valid positions) with 0 in m so we
        # don't propagate NaNs; output for those rows will be 0.
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        exp_scores = (scores - m).exp()
        # If all positions were masked out, exp_scores is all-zero;
        # set z=1 to keep the softmax well-defined (all-zero output).
        z = exp_scores.sum(dim=-1, keepdim=True)
        z_safe = torch.where(z > 0, z, torch.ones_like(z))
        p = exp_scores / z_safe
        out_s = torch.matmul(p, v_t)  # (n_heads, Lq, D)
        # Cast back to caller's dtype on copy.
        out[q_lo:q_hi].copy_(out_s.transpose(0, 1).to(out.dtype))
        # softmax_lse: (n_heads, total_q)  fp32. lse[h, i] = log z + m
        # (with m=0 when row was fully masked → lse = log(1) = 0; the
        # caller's bwd checks for finite lse so this is acceptable).
        lse_s = (m.squeeze(-1) + z_safe.squeeze(-1).log())  # (n_heads, Lq)
        softmax_lse[:, q_lo:q_hi].copy_(lse_s)
    return out, softmax_lse


def _eager_attention_bwd(
    dout, q, k, v, out, softmax_lse,
    dq, dk, dv,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    deterministic=True, sm_margin=0, softcap=0.0,
):
    """Pure-PyTorch varlen GQA attention bwd.

    OVERWRITES dq/dk/dv (matches flash backend semantics). Loops over
    sequences and computes analytical gradients of softmax(scores) @ V.
    """
    total_q, n_heads, head_dim = q.shape
    n_kv = k.shape[1]
    softmax_scale = head_dim ** -0.5
    rep = n_heads // n_kv

    # Zero outputs (we OVERWRITE — caller handles cross-chunk accum).
    dq.zero_()
    dk.zero_()
    dv.zero_()

    q_off_h = q_seq_offsets.tolist()
    k_off_h = k_seq_offsets.tolist()
    num_seqs = len(q_off_h) - 1

    for s in range(num_seqs):
        q_lo, q_hi = q_off_h[s], q_off_h[s + 1]
        k_lo, k_hi = k_off_h[s], k_off_h[s + 1]
        Lq = q_hi - q_lo
        Lk = k_hi - k_lo
        if Lq == 0 or Lk == 0:
            continue
        q_s = q[q_lo:q_hi]
        k_s = k[k_lo:k_hi]
        v_s = v[k_lo:k_hi]
        dout_s = dout[q_lo:q_hi]
        lse_s = softmax_lse[:, q_lo:q_hi]  # (n_heads, Lq)

        if rep > 1:
            k_s_rep = k_s.repeat_interleave(rep, dim=1)
            v_s_rep = v_s.repeat_interleave(rep, dim=1)
        else:
            k_s_rep = k_s
            v_s_rep = v_s

        q_t = q_s.transpose(0, 1).contiguous().float()      # (n_heads, Lq, D)
        k_t = k_s_rep.transpose(0, 1).contiguous().float()  # (n_heads, Lk, D)
        v_t = v_s_rep.transpose(0, 1).contiguous().float()
        dout_t = dout_s.transpose(0, 1).contiguous().float()

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale
        if softcap > 0:
            pre_softcap = scores / softcap
            scores = softcap * torch.tanh(pre_softcap)
        mask = _build_attn_mask(
            Lq, Lk, causal=causal, window_size=window_size, device=q.device,
        )
        scores = scores.masked_fill(~mask, float("-inf"))
        # Reconstruct softmax probs from saved lse.
        # p = exp(scores - lse[..., None])
        p = (scores - lse_s.unsqueeze(-1)).exp()
        # Mask any NaNs from the exp(-inf - 0) = 0 (well-defined) but
        # also any rows where lse was 0 (fully-masked seqs); set to 0.
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        # dV = p^T @ dout                 (n_heads, Lk, D)
        dv_t = torch.matmul(p.transpose(-2, -1), dout_t)

        # dp = dout @ v^T                 (n_heads, Lq, Lk)
        dp = torch.matmul(dout_t, v_t.transpose(-2, -1))

        # ds = p * (dp - sum(dp * p, dim=-1, keepdim=True))    softmax bwd
        ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
        if softcap > 0:
            # d(softcap * tanh(x / softcap))/dx = 1 - tanh(x/softcap)^2
            ds = ds * (1.0 - torch.tanh(pre_softcap).pow(2))
        ds = ds * softmax_scale
        # Zero out gradient at masked positions (defensive — these were
        # already zero in p, but ds via dp may have leaked).
        ds = ds.masked_fill(~mask, 0.0)

        # dQ = ds @ k                     (n_heads, Lq, D)
        dq_t = torch.matmul(ds, k_t)
        # dK = ds^T @ q                   (n_heads, Lk, D)
        dk_t_rep = torch.matmul(ds.transpose(-2, -1), q_t)

        dq[q_lo:q_hi].copy_(dq_t.transpose(0, 1).to(dq.dtype))
        # GQA: sum the rep replicated heads back into n_kv heads.
        if rep > 1:
            # (n_heads, Lk, D) → (n_kv, rep, Lk, D) → sum over rep dim
            dk_t = dk_t_rep.view(n_kv, rep, Lk, head_dim).sum(dim=1)
            dv_t = dv_t.view(n_kv, rep, Lk, head_dim).sum(dim=1)
        else:
            dk_t = dk_t_rep
        dk[k_lo:k_hi].copy_(dk_t.transpose(0, 1).to(dk.dtype))
        dv[k_lo:k_hi].copy_(dv_t.transpose(0, 1).to(dv.dtype))
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Public dispatchers.
# ---------------------------------------------------------------------------


def flextrain_attention_fwd(
    q, k, v, out, softmax_lse,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    sm_margin=0, softcap=0.0,
):
    # q: (total tokens, n_q_heads, head_dim)
    # k: (total tokens, n_kv_heads, head_dim)
    # v: (total tokens, n_kv_heads, head_dim)
    # out: (total tokens, n_q_heads, head_dim) -- written
    # softmax_lse: (n_q_heads, total tokens) fp32 -- written
    # q_seq_offsets / k_seq_offsets: (num_seqs + 1,) int32 cumsum
    # q_seq_lens / k_seq_lens: (num_seqs,) int32
    # max_seqlen_q / max_seqlen_k: int
    # causal: bool
    # window_size: (left, right)  -- (-1, -1) = no windowing
    pinned = _SELECTED_BACKEND
    if (pinned is None and FLASH_ATTN_4_AVAILABLE) or pinned == "fa4":
        return _flash4_fwd(
            q, k, v, out, softmax_lse,
            q_seq_offsets, k_seq_offsets,
            q_seq_lens, k_seq_lens,
            max_seqlen_q, max_seqlen_k,
            causal=causal, window_size=window_size,
            sm_margin=sm_margin, softcap=softcap,
        )
    if (pinned is None and FLASH_ATTN_3_AVAILABLE) or pinned == "fa3":
        return _flash3_fwd(
            q, k, v, out, softmax_lse,
            q_seq_offsets, k_seq_offsets,
            q_seq_lens, k_seq_lens,
            max_seqlen_q, max_seqlen_k,
            causal=causal, window_size=window_size,
            sm_margin=sm_margin, softcap=softcap,
        )
    if (pinned is None and FLASH_ATTN_2_AVAILABLE) or pinned == "fa2":
        return _flash2_fwd(
            q, k, v, out, softmax_lse,
            q_seq_offsets, k_seq_offsets,
            q_seq_lens, k_seq_lens,
            max_seqlen_q, max_seqlen_k,
            causal=causal, window_size=window_size,
            softcap=softcap,
        )
    if (pinned is None and EAGER_AVAILABLE) or pinned == "eager":
        return _eager_attention_fwd(
            q, k, v, out, softmax_lse,
            q_seq_offsets, k_seq_offsets,
            q_seq_lens, k_seq_lens,
            max_seqlen_q, max_seqlen_k,
            causal=causal, window_size=window_size,
            sm_margin=sm_margin, softcap=softcap,
        )
    raise FlashAttentionNotAvailableError(
        "No attention backend available (probed fa4 / fa3 / fa2 / eager). "
        "This shouldn't be reachable since EAGER_AVAILABLE is always True; "
        "please report."
    )


def flextrain_attention_bwd(
    dout, q, k, v, out, softmax_lse,
    dq, dk, dv,
    q_seq_offsets, k_seq_offsets,
    q_seq_lens, k_seq_lens,
    max_seqlen_q, max_seqlen_k,
    causal=True, window_size=(-1, -1),
    deterministic=True, sm_margin=0, softcap=0.0,
):
    # IMPORTANT — accumulation semantics:
    # ``dq``/``dk``/``dv`` are caller-supplied output buffers. The
    # underlying flash_attn varlen_bwd OVERWRITES these tensors — it
    # does NOT accumulate. Pre-existing values are clobbered.
    #
    # For multi-chunk training where a prior reverse iteration has
    # written cross-chunk dK/dV contributions into a global window
    # at this chunk's positions, callers MUST pass scratch buffers
    # to dk/dv and accumulate the result back into the window
    # themselves. See ``GQAAttentionBlock.bwd`` /
    # ``GQAAttentionGatedBlock.bwd`` for the pattern.
    pinned = _SELECTED_BACKEND
    if (pinned is None and FLASH_ATTN_4_AVAILABLE) or pinned == "fa4":
        return _flash4_bwd(
            dout, q, k, v, out, softmax_lse,
            dq, dk, dv,
            q_seq_offsets, k_seq_offsets,
            q_seq_lens, k_seq_lens,
            max_seqlen_q, max_seqlen_k,
            causal=causal, window_size=window_size,
            deterministic=deterministic,
            sm_margin=sm_margin, softcap=softcap,
        )
    if (pinned is None and FLASH_ATTN_3_AVAILABLE) or pinned == "fa3":
        return _flash3_bwd(
            dout, q, k, v, out, softmax_lse,
            dq, dk, dv,
            q_seq_offsets, k_seq_offsets,
            q_seq_lens, k_seq_lens,
            max_seqlen_q, max_seqlen_k,
            causal=causal, window_size=window_size,
            deterministic=deterministic,
            sm_margin=sm_margin, softcap=softcap,
        )
    if (pinned is None and FLASH_ATTN_2_AVAILABLE) or pinned == "fa2":
        return _flash2_bwd(
            dout, q, k, v, out, softmax_lse,
            dq, dk, dv,
            q_seq_offsets, k_seq_offsets,
            q_seq_lens, k_seq_lens,
            max_seqlen_q, max_seqlen_k,
            causal=causal, window_size=window_size,
            deterministic=deterministic, softcap=softcap,
        )
    if (pinned is None and EAGER_AVAILABLE) or pinned == "eager":
        return _eager_attention_bwd(
            dout, q, k, v, out, softmax_lse,
            dq, dk, dv,
            q_seq_offsets, k_seq_offsets,
            q_seq_lens, k_seq_lens,
            max_seqlen_q, max_seqlen_k,
            causal=causal, window_size=window_size,
            deterministic=deterministic,
            sm_margin=sm_margin, softcap=softcap,
        )
    raise FlashAttentionNotAvailableError(
        "No attention backend available (probed fa4 / fa3 / fa2 / eager). "
        "This shouldn't be reachable since EAGER_AVAILABLE is always True; "
        "please report."
    )
