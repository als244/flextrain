"""Partial-rotary RoPE kernel.

Distinct from :mod:`flextrain.ops._kernels.rope` (full-rotary). Used by
Qwen3-Next / Qwen3.5 / Qwen3.6 where only the first
``partial_rotary_factor * head_dim`` channels are rotated; the remaining
channels pass through unchanged.

The kernel operates on the first ``ROT_D`` channels of each head's
``HEAD_D``-wide head_dim. Channels ``[ROT_D : HEAD_D]`` are untouched.

Layout: pair-interleave on the FIRST ``ROT_D`` channels (so even=cos,
odd=sin within those). Pass-through channels keep their canonical
order; the loader's halved→pair permutation must therefore only permute
the first ``ROT_D`` channels per head.

Variable-length sequences: the kernel grid is ``(T,)`` and operates
per-token; T can be any value per call (chunks vary).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# Fused Forward Kernel — partial rotary
# ----------------------------------------------------------------------------
@triton.jit
def rope_partial_fwd_kernel(
    # Pointers
    Q_ptr,          # (T, Hq, HEAD_D)
    K_ptr,          # (T, Hk, HEAD_D) (Optional)
    POS_ptr,        # (T, 1) int32
    INV_FREQ_ptr,   # fp32 (ROT_D/2,)

    # Strides
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_p_t, stride_p_k,
    stride_if,

    # Shapes
    n_head_q,
    n_head_k,
    HEAD_D: tl.constexpr,   # Full head_dim (e.g. 256 for Qwen3-Next)
    ROT_D: tl.constexpr,    # Rotary sub-dim (e.g. 64 for partial_rotary_factor=0.25)
    HAS_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = tl.load(POS_ptr + pid * stride_p_t)

    # inv_freq has ROT_D/2 entries — one per (even, odd) pair within the
    # rotary portion.
    offs_d_pairs = tl.arange(0, ROT_D // 2)
    inv_freq = tl.load(INV_FREQ_ptr + offs_d_pairs * stride_if)
    freqs = pos * inv_freq
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    offs_d_even = offs_d_pairs * 2
    offs_d_odd = offs_d_pairs * 2 + 1

    # Q heads — rotate only first ROT_D channels per head.
    q_token_ptr = Q_ptr + pid * stride_q_t
    for h in range(n_head_q):
        head_offset = h * stride_q_h
        q_ptrs_even = q_token_ptr + head_offset + offs_d_even * stride_q_d
        q_ptrs_odd  = q_token_ptr + head_offset + offs_d_odd  * stride_q_d
        q_even = tl.load(q_ptrs_even)
        q_odd  = tl.load(q_ptrs_odd)
        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd  = q_even * sin + q_odd * cos
        tl.store(q_ptrs_even, q_rot_even)
        tl.store(q_ptrs_odd, q_rot_odd)
        # Channels [ROT_D : HEAD_D] are intentionally untouched.

    # K heads — same.
    if HAS_K:
        k_token_ptr = K_ptr + pid * stride_k_t
        for h in range(n_head_k):
            head_offset = h * stride_k_h
            k_ptrs_even = k_token_ptr + head_offset + offs_d_even * stride_k_d
            k_ptrs_odd  = k_token_ptr + head_offset + offs_d_odd  * stride_k_d
            k_even = tl.load(k_ptrs_even)
            k_odd  = tl.load(k_ptrs_odd)
            k_rot_even = k_even * cos - k_odd * sin
            k_rot_odd  = k_even * sin + k_odd * cos
            tl.store(k_ptrs_even, k_rot_even)
            tl.store(k_ptrs_odd, k_rot_odd)


# ----------------------------------------------------------------------------
# Fused Backward Kernel — partial rotary
# ----------------------------------------------------------------------------
@triton.jit
def rope_partial_bwd_kernel(
    DQ_ptr, DK_ptr,
    POS_ptr, INV_FREQ_ptr,
    stride_dq_t, stride_dq_h, stride_dq_d,
    stride_dk_t, stride_dk_h, stride_dk_d,
    stride_p_t, stride_p_k,
    stride_if,
    n_head_q, n_head_k,
    HEAD_D: tl.constexpr,
    ROT_D: tl.constexpr,
    HAS_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pos = tl.load(POS_ptr + pid * stride_p_t)

    offs_d_pairs = tl.arange(0, ROT_D // 2)
    inv_freq = tl.load(INV_FREQ_ptr + offs_d_pairs * stride_if)
    freqs = pos * inv_freq
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    offs_d_even = offs_d_pairs * 2
    offs_d_odd = offs_d_pairs * 2 + 1

    # dQ — inverse rotate first ROT_D channels.
    dq_token_ptr = DQ_ptr + pid * stride_dq_t
    for h in range(n_head_q):
        head_offset = h * stride_dq_h
        dq_ptrs_even = dq_token_ptr + head_offset + offs_d_even * stride_dq_d
        dq_ptrs_odd  = dq_token_ptr + head_offset + offs_d_odd  * stride_dq_d
        dx_even = tl.load(dq_ptrs_even)
        dx_odd  = tl.load(dq_ptrs_odd)
        # Inverse RoPE (rotation by -angle): cos(-x)=cos(x), sin(-x)=-sin(x).
        dx_rot_even = dx_even * cos + dx_odd * sin
        dx_rot_odd  = -dx_even * sin + dx_odd * cos
        tl.store(dq_ptrs_even, dx_rot_even)
        tl.store(dq_ptrs_odd, dx_rot_odd)

    # dK — same.
    if HAS_K:
        dk_token_ptr = DK_ptr + pid * stride_dk_t
        for h in range(n_head_k):
            head_offset = h * stride_dk_h
            dk_ptrs_even = dk_token_ptr + head_offset + offs_d_even * stride_dk_d
            dk_ptrs_odd  = dk_token_ptr + head_offset + offs_d_odd  * stride_dk_d
            dk_even = tl.load(dk_ptrs_even)
            dk_odd  = tl.load(dk_ptrs_odd)
            dk_rot_even = dk_even * cos + dk_odd * sin
            dk_rot_odd  = -dk_even * sin + dk_odd * cos
            tl.store(dk_ptrs_even, dk_rot_even)
            tl.store(dk_ptrs_odd, dk_rot_odd)


# ----------------------------------------------------------------------------
# Python wrappers
# ----------------------------------------------------------------------------

_PARTIAL_INV_FREQ_CACHE: dict[tuple[int, float, torch.device, torch.dtype], torch.Tensor] = {}


def _vanilla_partial_inv_freq(
    rot_dim: int, base: float, device: torch.device, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``inv_freq[i] = base ** (-2i / rot_dim)`` for i in [0, rot_dim/2).

    Note: divides by ``rot_dim``, not full ``head_dim`` — this matches HF's
    Qwen3-Next text RoPE which computes ``inv_freq`` over the partial sub-dim.
    """
    key = (rot_dim, float(base), device, dtype)
    cached = _PARTIAL_INV_FREQ_CACHE.get(key)
    if cached is not None:
        return cached
    half = rot_dim // 2
    exponents = torch.arange(0, half, dtype=torch.float32, device=device) * 2.0 / rot_dim
    inv_freq = (1.0 / (float(base) ** exponents)).to(dtype)
    _PARTIAL_INV_FREQ_CACHE[key] = inv_freq
    return inv_freq


def _resolve_partial_freqs(
    freqs_arg: torch.Tensor, rot_dim: int, device: torch.device,
) -> torch.Tensor:
    """Accept a precomputed ``inv_freq`` of length rot_dim/2, OR a 1-element
    scalar base θ tensor (vanilla inv_freq is built+cached). Returns a
    contiguous fp32 CUDA tensor."""
    half = rot_dim // 2
    if freqs_arg.numel() == half:
        f = freqs_arg
        if f.dtype != torch.float32:
            f = f.to(torch.float32)
        if f.device != device:
            f = f.to(device)
        return f.contiguous()
    if freqs_arg.numel() == 1:
        return _vanilla_partial_inv_freq(rot_dim, float(freqs_arg.item()), device)
    raise ValueError(
        f"flextrain_rope_partial: expected freqs of length {half} or 1, "
        f"got numel={freqs_arg.numel()}"
    )


def flextrain_rope_partial_fwd(
    tensors_list: list[torch.Tensor],
    pos_ids: torch.Tensor,
    freqs: torch.Tensor,
    rot_dim: int,
):
    """Partial-rotary RoPE in-place on Q (and optionally K).

    ``rot_dim`` is the rotary sub-dimension (e.g. ``head_dim * 0.25``
    for Qwen3-Next). Channels ``[rot_dim:head_dim]`` of each head are
    passed through untouched.

    ``tensors_list`` shapes: ``[Q]`` or ``[Q, K]`` where each is
    ``(T, n_heads, head_dim)``. T is the variable-length token count
    of the chunk; can differ per call.
    """
    if not tensors_list:
        return tensors_list

    q = tensors_list[0]
    T, Hq, head_d = q.shape
    assert q.is_cuda and pos_ids.is_cuda
    assert rot_dim <= head_d, f"rot_dim={rot_dim} > head_dim={head_d}"
    assert rot_dim % 2 == 0, f"rot_dim must be even, got {rot_dim}"
    inv_freq = _resolve_partial_freqs(freqs, rot_dim, q.device)

    has_k = False
    k = None
    Hk = 0
    stride_k_t, stride_k_h, stride_k_d = 0, 0, 0

    if len(tensors_list) == 2:
        k = tensors_list[1]
        assert k.shape[0] == T
        assert k.shape[2] == head_d
        has_k = True
        Hk = k.shape[1]
        stride_k_t, stride_k_h, stride_k_d = k.stride()

    grid = (T,)
    rope_partial_fwd_kernel[grid](
        q, k, pos_ids, inv_freq,
        q.stride(0), q.stride(1), q.stride(2),
        stride_k_t, stride_k_h, stride_k_d,
        pos_ids.stride(0), pos_ids.stride(1),
        inv_freq.stride(0),
        n_head_q=Hq,
        n_head_k=Hk,
        HEAD_D=head_d,
        ROT_D=rot_dim,
        HAS_K=has_k,
        num_warps=4,
        num_stages=2,
    )

    if len(tensors_list) > 2:
        flextrain_rope_partial_fwd(tensors_list[2:], pos_ids, freqs, rot_dim)

    return tensors_list


def flextrain_rope_partial_bwd(
    grad_tensors_list: list[torch.Tensor],
    pos_ids: torch.Tensor,
    freqs: torch.Tensor,
    rot_dim: int,
):
    """Inverse partial-RoPE on dQ/dK in-place. Same semantics as
    :func:`flextrain_rope_partial_fwd`."""
    if not grad_tensors_list:
        return grad_tensors_list

    dq = grad_tensors_list[0]
    T, Hq, head_d = dq.shape
    assert rot_dim <= head_d
    assert rot_dim % 2 == 0
    inv_freq = _resolve_partial_freqs(freqs, rot_dim, dq.device)

    has_k = False
    dk = None
    Hk = 0
    stride_dk_t, stride_dk_h, stride_dk_d = 0, 0, 0

    if len(grad_tensors_list) == 2:
        dk = grad_tensors_list[1]
        has_k = True
        Hk = dk.shape[1]
        stride_dk_t, stride_dk_h, stride_dk_d = dk.stride()

    grid = (T,)
    rope_partial_bwd_kernel[grid](
        dq, dk, pos_ids, inv_freq,
        dq.stride(0), dq.stride(1), dq.stride(2),
        stride_dk_t, stride_dk_h, stride_dk_d,
        pos_ids.stride(0), pos_ids.stride(1),
        inv_freq.stride(0),
        n_head_q=Hq,
        n_head_k=Hk,
        HEAD_D=head_d,
        ROT_D=rot_dim,
        HAS_K=has_k,
        num_warps=4,
        num_stages=2,
    )

    if len(grad_tensors_list) > 2:
        flextrain_rope_partial_bwd(grad_tensors_list[2:], pos_ids, freqs, rot_dim)

    return grad_tensors_list
