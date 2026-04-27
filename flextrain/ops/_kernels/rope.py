import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------------
# Fused Forward Kernel
# ----------------------------------------------------------------------------
@triton.jit
def rope_fused_fwd_kernel(
    # Pointers
    Q_ptr,          # Pointer to Q tensor (T, Hq, D)
    K_ptr,          # Pointer to K tensor (T, Hk, D) (Optional, can be null)
    POS_ptr,        # Pointer to Position tensor (T, 1) or (T, K_rope)
    INV_FREQ_ptr,   # Pointer to fp32 inv_freq array of length D/2

    # Strides
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_p_t, stride_p_k,
    stride_if,

    # Shapes
    n_head_q,       # Number of heads in Q
    n_head_k,       # Number of heads in K
    BLOCK_SIZE_D: tl.constexpr, # Head Dimension
    HAS_K: tl.constexpr         # Boolean flag to enable K processing
):
    # One program instance per Token (Sequence elements)
    pid = tl.program_id(0)

    # 1. Load Position
    pos = tl.load(POS_ptr + pid * stride_p_t)

    # 2. Load inv_freq[d_pair] from a caller-supplied array (length D/2).
    # This lets callers pass arbitrary frequency curves: vanilla RoPE
    # (inv_freq[i] = theta^(-2i/D)), Llama-3.1 YARN, NTK, etc.
    offs_d_pairs = tl.arange(0, BLOCK_SIZE_D // 2)
    inv_freq = tl.load(INV_FREQ_ptr + offs_d_pairs * stride_if)

    freqs = pos * inv_freq
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    # 3. Process Q Heads
    # We iterate over heads within the kernel to keep cos/sin in registers
    offs_d_even = offs_d_pairs * 2
    offs_d_odd = offs_d_pairs * 2 + 1

    # Pointer to the start of this token's Q data
    q_token_ptr = Q_ptr + pid * stride_q_t

    for h in range(n_head_q):
        # Calculate pointers for this head
        head_offset = h * stride_q_h
        q_ptrs_even = q_token_ptr + head_offset + offs_d_even * stride_q_d
        q_ptrs_odd  = q_token_ptr + head_offset + offs_d_odd  * stride_q_d

        # Load
        q_even = tl.load(q_ptrs_even)
        q_odd  = tl.load(q_ptrs_odd)

        # Apply RoPE
        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd  = q_even * sin + q_odd * cos

        # Store
        tl.store(q_ptrs_even, q_rot_even)
        tl.store(q_ptrs_odd, q_rot_odd)

    # 4. Process K Heads (Fused)
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
# Fused Backward Kernel
# ----------------------------------------------------------------------------
@triton.jit
def rope_fused_bwd_kernel(
    DQ_ptr, DK_ptr,
    POS_ptr, INV_FREQ_ptr,
    stride_dq_t, stride_dq_h, stride_dq_d,
    stride_dk_t, stride_dk_h, stride_dk_d,
    stride_p_t, stride_p_k,
    stride_if,
    n_head_q, n_head_k,
    BLOCK_SIZE_D: tl.constexpr,
    HAS_K: tl.constexpr
):
    pid = tl.program_id(0)

    # 1. Load Pos
    pos = tl.load(POS_ptr + pid * stride_p_t)

    # 2. Load inv_freq from caller-supplied array (matches fwd kernel).
    offs_d_pairs = tl.arange(0, BLOCK_SIZE_D // 2)
    inv_freq = tl.load(INV_FREQ_ptr + offs_d_pairs * stride_if)
    freqs = pos * inv_freq
    
    # Note: Backward pass usually implies conjugate (rotation by -angle)
    # cos(-x) = cos(x), sin(-x) = -sin(x)
    # The logic below matches your reference:
    # rot_even = dx_even * cos + dx_odd * sin
    # rot_odd  = -dx_even * sin + dx_odd * cos
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    # 3. Process dQ
    offs_d_even = offs_d_pairs * 2
    offs_d_odd = offs_d_pairs * 2 + 1
    
    dq_token_ptr = DQ_ptr + pid * stride_dq_t
    
    for h in range(n_head_q):
        head_offset = h * stride_dq_h
        dq_ptrs_even = dq_token_ptr + head_offset + offs_d_even * stride_dq_d
        dq_ptrs_odd  = dq_token_ptr + head_offset + offs_d_odd  * stride_dq_d

        dx_even = tl.load(dq_ptrs_even)
        dx_odd  = tl.load(dq_ptrs_odd)
        
        # Inverse RoPE
        dx_rot_even = dx_even * cos + dx_odd * sin
        dx_rot_odd  = -dx_even * sin + dx_odd * cos
        
        tl.store(dq_ptrs_even, dx_rot_even)
        tl.store(dq_ptrs_odd, dx_rot_odd)

    # 4. Process dK
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
# Python Wrappers
# ----------------------------------------------------------------------------

_INV_FREQ_CACHE: dict[tuple[int, float, torch.device, torch.dtype], torch.Tensor] = {}


def _vanilla_inv_freq(
    head_dim: int, base: float, device: torch.device, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``inv_freq[i] = base ** (-2i / head_dim)`` for i in [0, head_dim/2)."""
    key = (head_dim, float(base), device, dtype)
    cached = _INV_FREQ_CACHE.get(key)
    if cached is not None:
        return cached
    half = head_dim // 2
    exponents = torch.arange(0, half, dtype=torch.float32, device=device) * 2.0 / head_dim
    inv_freq = (1.0 / (float(base) ** exponents)).to(dtype)
    _INV_FREQ_CACHE[key] = inv_freq
    return inv_freq


def _resolve_freqs(
    freqs_arg: torch.Tensor, head_dim: int, device: torch.device,
) -> torch.Tensor:
    """Accept either:
    * a ``(D/2,)`` precomputed ``inv_freq`` array (any fp dtype, on CUDA), or
    * a ``(1,)`` tensor holding a scalar RoPE base θ — we build the
      vanilla ``inv_freq`` array on the fly (cached, so repeated calls
      with the same base are free).
    Returns a ``(D/2,)`` fp32 CUDA tensor."""
    half = head_dim // 2
    if freqs_arg.numel() == half:
        f = freqs_arg
        if f.dtype != torch.float32:
            f = f.to(torch.float32)
        if f.device != device:
            f = f.to(device)
        return f.contiguous()
    if freqs_arg.numel() == 1:
        base = float(freqs_arg.item())
        return _vanilla_inv_freq(head_dim, base, device, torch.float32)
    raise ValueError(
        f"flextrain_rope: expected freqs of length {half} (inv_freq) or 1 "
        f"(scalar base θ), got numel={freqs_arg.numel()}"
    )


def flextrain_rope_fwd(
    tensors_list: list[torch.Tensor],
    pos_ids: torch.Tensor,
    freqs: torch.Tensor,
):
    """
    Applies RoPE in-place.

    ``freqs`` may be either a precomputed ``inv_freq`` array of length
    ``head_dim/2`` (e.g. for Llama-3.1 YARN-scaled RoPE), or a 1-element
    tensor holding a scalar RoPE base θ (vanilla ``inv_freq[i] = θ^(-2i/D)``
    is built and cached).

    Optimized to fuse Q and K when both are passed.
    """
    if not tensors_list:
        return tensors_list

    q = tensors_list[0]
    T, Hq, D = q.shape
    assert q.is_cuda and pos_ids.is_cuda
    inv_freq = _resolve_freqs(freqs, D, q.device)

    has_k = False
    k = None
    Hk = 0
    stride_k_t, stride_k_h, stride_k_d = 0, 0, 0

    if len(tensors_list) == 2:
        k = tensors_list[1]
        assert k.shape[0] == T
        assert k.shape[2] == D
        has_k = True
        Hk = k.shape[1]
        stride_k_t, stride_k_h, stride_k_d = k.stride()

    grid = (T,)
    rope_fused_fwd_kernel[grid](
        q, k, pos_ids, inv_freq,
        q.stride(0), q.stride(1), q.stride(2),
        stride_k_t, stride_k_h, stride_k_d,
        pos_ids.stride(0), pos_ids.stride(1),
        inv_freq.stride(0),
        n_head_q=Hq,
        n_head_k=Hk,
        BLOCK_SIZE_D=D,
        HAS_K=has_k,
        num_warps=4,
        num_stages=2,
    )

    if len(tensors_list) > 2:
        flextrain_rope_fwd(tensors_list[2:], pos_ids, freqs)

    return tensors_list


def flextrain_rope_bwd(
    grad_tensors_list: list[torch.Tensor],
    pos_ids: torch.Tensor,
    freqs: torch.Tensor,
):
    """Inverse RoPE on dQ/dK in-place. ``freqs`` semantics match
    :func:`flextrain_rope_fwd`."""
    if not grad_tensors_list:
        return grad_tensors_list

    dq = grad_tensors_list[0]
    T, Hq, D = dq.shape
    inv_freq = _resolve_freqs(freqs, D, dq.device)

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
    rope_fused_bwd_kernel[grid](
        dq, dk, pos_ids, inv_freq,
        dq.stride(0), dq.stride(1), dq.stride(2),
        stride_dk_t, stride_dk_h, stride_dk_d,
        pos_ids.stride(0), pos_ids.stride(1),
        inv_freq.stride(0),
        n_head_q=Hq,
        n_head_k=Hk,
        BLOCK_SIZE_D=D,
        HAS_K=has_k,
        num_warps=4,
        num_stages=2,
    )

    if len(grad_tensors_list) > 2:
        flextrain_rope_bwd(grad_tensors_list[2:], pos_ids, freqs)

    return grad_tensors_list