import torch
import triton
import triton.language as tl

# ===================================================================
# 1. FORWARD KERNELS
# ===================================================================

@triton.jit
def rms_norm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr, Rstd_ptr,
    stride_x_t, stride_y_t, stride_rstd_t,
    N_COLS,               # Normalization dimension (D or head_dim)
    N_HEADS,              # Number of heads (1 for full_row)
    EPSILON: tl.float32,  # Epsilon for numerical stability
    HAS_WEIGHTS: tl.constexpr,
    IS_BY_HEAD: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for RMSNorm forward pass.
    Grid is (T, N_HEADS).
    """
    pid_t = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # --- Calculate pointers ---
    row_x_ptr = X_ptr + pid_t * stride_x_t
    row_y_ptr = Y_ptr + pid_t * stride_y_t
    
    start_col = pid_h * N_COLS
    group_x_ptr = row_x_ptr + start_col
    group_y_ptr = row_y_ptr + start_col
    rstd_ptr = Rstd_ptr + pid_t * stride_rstd_t + pid_h

    # --- Compute RMS ---
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < N_COLS

    # Load x (fp16/bf16)
    x = tl.load(group_x_ptr + offs_d, mask=mask_d, other=0.0)
    
    # Compute sum of squares in fp32
    x_f32 = x.to(tl.float32)
    var = tl.sum(x_f32 * x_f32, axis=0) / N_COLS
    
    # Compute rstd
    rstd_val = tl.rsqrt(var + EPSILON)
    tl.store(rstd_ptr, rstd_val)

    # --- Normalize ---
    # Match PyTorch: (x.float() * rstd).type_as(x)
    x_norm_fp32 = x_f32 * rstd_val
    x_norm_original_dtype = x_norm_fp32.to(X_ptr.dtype.element_ty)
    
    # --- Apply Weights ---
    if HAS_WEIGHTS:
        if IS_BY_HEAD:
            w_ptr = W_ptr
        else:
            w_ptr = W_ptr + start_col 
        
        w = tl.load(w_ptr + offs_d, mask=mask_d)
        y = x_norm_original_dtype * w
    else:
        y = x_norm_original_dtype

    # Store result
    tl.store(group_y_ptr + offs_d, y, mask=mask_d)


@triton.jit
def rms_norm_fwd_recompute_kernel(
    X_ptr, Y_ptr, W_ptr, Rstd_ptr,
    stride_x_t, stride_y_t, stride_rstd_t,
    N_COLS,
    N_HEADS,
    HAS_WEIGHTS: tl.constexpr,
    IS_BY_HEAD: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for RMSNorm recompute forward pass.
    Uses pre-computed rstd values.
    """
    pid_t = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    row_x_ptr = X_ptr + pid_t * stride_x_t
    row_y_ptr = Y_ptr + pid_t * stride_y_t
    
    start_col = pid_h * N_COLS
    group_x_ptr = row_x_ptr + start_col
    group_y_ptr = row_y_ptr + start_col
    rstd_ptr = Rstd_ptr + pid_t * stride_rstd_t + pid_h

    # Load pre-computed rstd
    rstd_val = tl.load(rstd_ptr).to(tl.float32)

    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < N_COLS

    x = tl.load(group_x_ptr + offs_d, mask=mask_d, other=0.0)
    x_f32 = x.to(tl.float32)
    
    # Normalize
    x_norm_fp32 = x_f32 * rstd_val
    x_norm_original_dtype = x_norm_fp32.to(X_ptr.dtype.element_ty)

    # Apply weights
    if HAS_WEIGHTS:
        if IS_BY_HEAD:
            w_ptr = W_ptr
        else:
            w_ptr = W_ptr + start_col
        w = tl.load(w_ptr + offs_d, mask=mask_d)
        y = x_norm_original_dtype * w
    else:
        y = x_norm_original_dtype

    tl.store(group_y_ptr + offs_d, y, mask=mask_d)


# ===================================================================
# 2. BACKWARD KERNELS (OPTIMIZED)
# ===================================================================

@triton.jit
def rms_norm_bwd_dx_kernel(
    dY_ptr, X_ptr, W_ptr, Rstd_ptr,
    dX_ptr, Y_ptr,
    stride_dy_t, stride_x_t, stride_rstd_t, stride_dx_t, stride_y_t,
    N_COLS,       
    N_HEADS,      
    HAS_WEIGHTS: tl.constexpr,
    IS_BY_HEAD: tl.constexpr,
    ACCUMULATE_DX: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Calculates dX and optionally recomputes Y. 
    This kernel is fully parallel over rows (T) and heads (N_HEADS).
    It does NOT compute dW to avoid atomic contention.
    """
    pid_t = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # --- Pointers ---
    row_dy_ptr = dY_ptr + pid_t * stride_dy_t
    row_x_ptr = X_ptr + pid_t * stride_x_t
    row_dx_ptr = dX_ptr + pid_t * stride_dx_t
    rstd_ptr = Rstd_ptr + pid_t * stride_rstd_t + pid_h

    # Offsets for D dimension
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_d = offs_d < N_COLS
    
    # Handle Heads
    start_col = pid_h * N_COLS
    group_dy_ptr = row_dy_ptr + start_col
    group_x_ptr = row_x_ptr + start_col
    group_dx_ptr = row_dx_ptr + start_col

    # --- Load Data ---
    dy = tl.load(group_dy_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    x = tl.load(group_x_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    rstd_val = tl.load(rstd_ptr).to(tl.float32)

    # --- Recompute Y (Standardized) ---
    y_norm_f32 = x * rstd_val
    
    # --- Apply Weights to dY ---
    if HAS_WEIGHTS:
        if IS_BY_HEAD:
            w_ptr = W_ptr
        else:
            w_ptr = W_ptr + start_col
        w = tl.load(w_ptr + offs_d, mask=mask_d).to(tl.float32)
        
        # If we need to store Y, we multiply by W here
        if RECOMPUTE_OUTPUT:
            y_final = y_norm_f32 * w
            row_y_ptr = Y_ptr + pid_t * stride_y_t 
            tl.store(row_y_ptr + start_col + offs_d, y_final.to(Y_ptr.dtype.element_ty), mask=mask_d)

        # Pre-multiply dy by w for the dX calculation
        dy_scaled = dy * w
    else:
        dy_scaled = dy
        if RECOMPUTE_OUTPUT:
            row_y_ptr = Y_ptr + pid_t * stride_y_t 
            tl.store(row_y_ptr + start_col + offs_d, y_norm_f32.to(Y_ptr.dtype.element_ty), mask=mask_d)

    # --- Compute dX ---
    # Math: dX = rstd * ( dy_scaled - y_norm * sum(dy_scaled * y_norm) / N )
    term_dot = tl.sum(y_norm_f32 * dy_scaled, axis=0)
    k = term_dot / N_COLS
    
    dx = rstd_val * (dy_scaled - y_norm_f32 * k)

    # --- Store ---
    if ACCUMULATE_DX:
        dx_old = tl.load(group_dx_ptr + offs_d, mask=mask_d, other=0.0)
        dx_final = dx_old + dx
        tl.store(group_dx_ptr + offs_d, dx_final.to(dX_ptr.dtype.element_ty), mask=mask_d)
    else:
        tl.store(group_dx_ptr + offs_d, dx.to(dX_ptr.dtype.element_ty), mask=mask_d)


# Legacy single-stage kernel — kept for benchmarking / regression
# reference. Not used by the wrapper after the two-stage rewrite.
@triton.jit
def rms_norm_bwd_dw_kernel_legacy(
    dY_ptr, X_ptr, Rstd_ptr, dW_ptr,
    stride_dy_row, stride_x_row, stride_rstd_row,
    TOTAL_ROWS, N_COLS,
    ACCUMULATE_DW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    num_split_k = tl.num_programs(axis=1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N_COLS
    rows_per_split = tl.cdiv(TOTAL_ROWS, num_split_k)
    start_row = pid_k * rows_per_split
    end_row = tl.minimum(start_row + rows_per_split, TOTAL_ROWS)
    dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row_idx in range(start_row, end_row, BLOCK_M):
        offs_m = row_idx + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end_row
        dy_ptrs = dY_ptr + (offs_m[:, None] * stride_dy_row) + offs_n[None, :]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_x_row) + offs_n[None, :]
        rstd_ptrs = Rstd_ptr + (offs_m * stride_rstd_row)
        dy = tl.load(dy_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        x_norm = x * rstd[:, None]
        prod = dy * x_norm
        dw_acc += tl.sum(prod, axis=0)
    dw_out_ptr = dW_ptr + offs_n
    tl.atomic_add(dw_out_ptr, dw_acc, mask=mask_n)


@triton.jit
def rms_norm_bwd_dw_partial_kernel(
    dY_ptr, X_ptr, Rstd_ptr, dW_partial_ptr,
    stride_dy_row, stride_x_row, stride_rstd_row,
    stride_partial_split,
    TOTAL_ROWS,
    N_COLS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Stage 1 of the two-stage Split-K dW reduction. Each (pid_n, pid_k)
    program processes a contiguous chunk of rows and writes its partial
    sum into ``dW_partial[pid_k, offs_n]`` — no atomics, no contention.

    Stage 2 (``rms_norm_bwd_dw_reduce_kernel``) sums along the SPLIT_K
    axis to produce the final dW.

    Replaces the old single-stage kernel that had every split-K program
    ``atomic_add`` into the same final dW vector. With small N_COLS
    (e.g. head_dim=128) and large SPLIT_K, that pattern serializes on
    the L2 atomics and severely under-utilizes the GPU at large T.
    """
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    num_split_k = tl.num_programs(axis=1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N_COLS

    rows_per_split = tl.cdiv(TOTAL_ROWS, num_split_k)
    start_row = pid_k * rows_per_split
    end_row = tl.minimum(start_row + rows_per_split, TOTAL_ROWS)

    dw_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for row_idx in range(start_row, end_row, BLOCK_M):
        offs_m = row_idx + tl.arange(0, BLOCK_M)
        mask_m = offs_m < end_row

        dy_ptrs = dY_ptr + (offs_m[:, None] * stride_dy_row) + offs_n[None, :]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_x_row) + offs_n[None, :]
        rstd_ptrs = Rstd_ptr + (offs_m * stride_rstd_row)

        dy = tl.load(dy_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        x_norm = x * rstd[:, None]
        prod = dy * x_norm
        dw_acc += tl.sum(prod, axis=0)

    # Direct (non-atomic) store into this program's row of the partial
    # buffer. Layout: dW_partial[pid_k, offs_n], stride = N_COLS.
    out_ptr = dW_partial_ptr + pid_k * stride_partial_split + offs_n
    tl.store(out_ptr, dw_acc, mask=mask_n)


@triton.jit
def rms_norm_bwd_dw_reduce_kernel(
    dW_partial_ptr, dW_ptr,
    stride_partial_split,
    SPLIT_K,
    N_COLS,
    ACCUMULATE_DW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Stage 2: reduce dW_partial[SPLIT_K, N_COLS] along the SPLIT_K axis.
    One program per BLOCK_N column tile; each program loads
    ``(SPLIT_K, BLOCK_N)`` partials in fp32 and sums them down.

    SPLIT_K is small (typ. 32-512) and the partial buffer is tiny
    relative to the main inputs, so this stage is essentially free
    compared to stage 1.
    """
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N_COLS

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in range(0, SPLIT_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < SPLIT_K
        ptrs = (
            dW_partial_ptr
            + offs_k[:, None] * stride_partial_split
            + offs_n[None, :]
        )
        partial = tl.load(
            ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0,
        )
        acc += tl.sum(partial, axis=0)

    out_ptr = dW_ptr + offs_n
    if ACCUMULATE_DW:
        old = tl.load(out_ptr, mask=mask_n, other=0.0)
        tl.store(out_ptr, old + acc, mask=mask_n)
    else:
        tl.store(out_ptr, acc, mask=mask_n)


# ===================================================================
# 3. PYTHON WRAPPERS
# ===================================================================

def _get_norm_configs(X, W, head_dim):
    T, D = X.shape
    device = X.device
    
    if head_dim is None:
        N_COLS = D
        N_HEADS = 1
        IS_BY_HEAD = False
    else:
        if D % head_dim != 0:
            raise ValueError(f"Input dimension D ({D}) must be divisible by head_dim ({head_dim})")
        N_COLS = head_dim
        N_HEADS = D // head_dim
        IS_BY_HEAD = True
        
    if W is not None:
        HAS_WEIGHTS = True
        if IS_BY_HEAD:
            expected_w_shape = (N_COLS,)
        else:
            expected_w_shape = (D,)

        if W.shape != expected_w_shape:
            raise ValueError(f"Weight tensor has incorrect shape. Expected {expected_w_shape}, got {W.shape}")
    else:
        HAS_WEIGHTS = False
        
    if not X.is_contiguous():
        raise ValueError("Input tensor X must be contiguous")
    if W is not None and not W.is_contiguous():
        raise ValueError("Weight tensor W must be contiguous")

    BLOCK_SIZE_D = triton.next_power_of_2(N_COLS)
    grid = (T, N_HEADS)
    
    return T, D, N_COLS, N_HEADS, IS_BY_HEAD, HAS_WEIGHTS, BLOCK_SIZE_D, grid, device

def flextrain_rmsnorm_fwd(
    X: torch.Tensor, 
    W: torch.Tensor = None, 
    head_dim: int = None, 
    output: torch.Tensor = None,
    rstd: torch.Tensor = None,
    rms_norm_eps: float = 1e-5
) -> (torch.Tensor, torch.Tensor):
    
    T, D, N_COLS, N_HEADS, IS_BY_HEAD, HAS_WEIGHTS, \
    BLOCK_SIZE_D, grid, device = _get_norm_configs(X, W, head_dim)

    if output is None:
        Y = torch.empty_like(X)
    else:
        if output.shape != X.shape:
            raise ValueError(f"Output tensor shape {output.shape} must match input shape {X.shape}")
        if not output.is_contiguous():
             raise ValueError("Output tensor must be contiguous")
        Y = output

    if rstd is None:
        rstd = torch.empty(T, N_HEADS, dtype=torch.float32, device=device)
    else:
        if rstd.shape != (T, N_HEADS):
            raise ValueError(f"rstd tensor shape {rstd.shape} must match (T, N_HEADS) {T, N_HEADS}")
        if not rstd.is_contiguous():
            raise ValueError("rstd tensor must be contiguous")

    rms_norm_fwd_kernel[grid](
        X, Y, W, rstd,
        X.stride(0), Y.stride(0), rstd.stride(0),
        N_COLS, N_HEADS,
        rms_norm_eps,
        HAS_WEIGHTS=HAS_WEIGHTS,
        IS_BY_HEAD=IS_BY_HEAD,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    return Y, rstd

def flextrain_rmsnorm_fwd_recompute(
    X: torch.Tensor, 
    W: torch.Tensor, 
    rstd: torch.Tensor, 
    head_dim: int = None,
    output: torch.Tensor = None
) -> torch.Tensor:
    
    T, D, N_COLS, N_HEADS, IS_BY_HEAD, HAS_WEIGHTS, \
    BLOCK_SIZE_D, grid, device = _get_norm_configs(X, W, head_dim)
    
    if rstd.shape != (T, N_HEADS):
        raise ValueError(f"rstd tensor shape {rstd.shape} mismatch. Expected {(T, N_HEADS)}")

    if output is None:
        Y = torch.empty_like(X)
    else:
        if output.shape != X.shape:
            raise ValueError(f"Output tensor shape {output.shape} must match input shape {X.shape}")
        if not output.is_contiguous():
             raise ValueError("Output tensor must be contiguous")
        Y = output

    rms_norm_fwd_recompute_kernel[grid](
        X, Y, W, rstd,
        X.stride(0), Y.stride(0), rstd.stride(0),
        N_COLS, N_HEADS,
        HAS_WEIGHTS=HAS_WEIGHTS,
        IS_BY_HEAD=IS_BY_HEAD,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    
    return Y
    
def flextrain_rmsnorm_bwd(
    dY: torch.Tensor, 
    X: torch.Tensor, 
    W: torch.Tensor, 
    rstd: torch.Tensor, 
    head_dim: int = None,
    dX: torch.Tensor = None,
    dW: torch.Tensor = None,
    recompute_output: bool = False,
    recomputed_output_tensor: torch.Tensor = None
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Backward pass for RMSNorm.
    
    Args:
        dY: Gradient of loss w.r.t. output Y
        X: Original input tensor
        W: Weight tensor (optional)
        rstd: Pre-computed reciprocal standard deviation from forward pass
        head_dim: If specified, applies normalization per-head
        dX: Optional pre-allocated tensor for input gradients. 
            If provided, gradients are ACCUMULATED into this tensor.
        dW: Optional pre-allocated tensor for weight gradients.
            If provided, gradients are ACCUMULATED into this tensor (not zeroed).
            Must be float32 for atomic operations.
        recompute_output: If True, recompute the forward output Y
        recomputed_output_tensor: Optional pre-allocated tensor for recomputed output
        
    Returns:
        dX: Gradient of loss w.r.t. input X
        dW: Gradient of loss w.r.t. weights W (or None if W is None)
        recomputed_output: Recomputed Y if recompute_output=True, else None
    """
    T, D, N_COLS, N_HEADS, IS_BY_HEAD, HAS_WEIGHTS, \
    BLOCK_SIZE_D, grid, device = _get_norm_configs(X, W, head_dim)

    # --- Output Recomputation Setup ---
    final_recomputed_output = None
    Y_stride_0 = 0 
    
    if recompute_output:
        RECOMPUTE_OUTPUT = True
        if recomputed_output_tensor is None:
            final_recomputed_output = torch.empty_like(X)
        else:
            if recomputed_output_tensor.shape != X.shape:
                raise ValueError(f"recomputed_output_tensor shape {recomputed_output_tensor.shape} must match X {X.shape}")
            if not recomputed_output_tensor.is_contiguous():
                raise ValueError("recomputed_output_tensor must be contiguous")
            final_recomputed_output = recomputed_output_tensor
            
        Y_ptr = final_recomputed_output
        Y_stride_0 = final_recomputed_output.stride(0)
    else:
        RECOMPUTE_OUTPUT = False
        Y_ptr = X # Dummy pointer

    # --- dX Setup ---
    if dX is None:
        dX = torch.empty_like(X)
        ACCUMULATE_DX = False
    else:
        if dX.shape != X.shape:
             raise ValueError(f"dX shape {dX.shape} must match X shape {X.shape}")
        if not dX.is_contiguous():
            raise ValueError("dX tensor must be contiguous")
        ACCUMULATE_DX = True

    # --- 1. Launch dX Kernel (Row Parallel) ---
    # This kernel handles the math for dX and optional Y recompute.
    # It does NOT handle dW.
    rms_norm_bwd_dx_kernel[grid](
        dY, X, W, rstd,
        dX, Y_ptr,
        dY.stride(0), X.stride(0), rstd.stride(0), dX.stride(0), Y_stride_0,
        N_COLS, N_HEADS,
        HAS_WEIGHTS=HAS_WEIGHTS,
        IS_BY_HEAD=IS_BY_HEAD,
        ACCUMULATE_DX=ACCUMULATE_DX,
        RECOMPUTE_OUTPUT=RECOMPUTE_OUTPUT,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    # --- 2. Launch dW Kernel (Split-K Column Reduction) ---
    if HAS_WEIGHTS:
        # Determine dW shape
        if IS_BY_HEAD:
            dw_shape = (N_COLS,)
        else:
            dw_shape = (D,)

        # Track whether we need to add back to a non-float32 buffer
        user_dW = None
        user_dW_dtype = None

        if dW is None:
            # No dW provided - allocate a new zeroed tensor
            dW_buffer = torch.zeros(dw_shape, dtype=torch.float32, device=device)
            ACCUMULATE_DW = False
        else:
            # dW provided - validate shape and contiguity
            if dW.shape != dw_shape:
                raise ValueError(f"dW shape {dW.shape} mismatch. Expected {dw_shape}")
            if not dW.is_contiguous():
                raise ValueError("dW tensor must be contiguous")
            
            if dW.dtype == torch.float32:
                # float32 - can use directly with atomic_add
                dW_buffer = dW
                ACCUMULATE_DW = True
            else:
                # Non-float32 (e.g., bfloat16) - need temporary float32 buffer
                # We'll accumulate into temp buffer, then add back to user's dW
                user_dW = dW
                user_dW_dtype = dW.dtype
                dW_buffer = torch.zeros(dw_shape, dtype=torch.float32, device=device)
                ACCUMULATE_DW = False  # Fresh temp buffer, no accumulation needed in kernel

        # --- FLATTENING LOGIC FOR REDUCTION ---
        # To handle IS_BY_HEAD efficiently, we flatten (T, H) into a single "Row" dimension.
        if IS_BY_HEAD:
            total_rows_for_reduction = T * N_HEADS
            reduction_cols = N_COLS
            
            # Strides for the flattened view (T*H, N_COLS)
            # Since tensors are contiguous, the stride to move 1 'logical row' (head) is just N_COLS
            stride_dy_row = N_COLS 
            stride_x_row = N_COLS
            stride_rstd_row = 1 # rstd is (T,H), so flattened stride is 1
        else:
            total_rows_for_reduction = T
            reduction_cols = N_COLS
            
            stride_dy_row = dY.stride(0)
            stride_x_row = X.stride(0)
            stride_rstd_row = rstd.stride(0)

        # ---- Adaptive Split-K, hybrid one- vs two-stage reduction ----
        # The dW reduction is a column-wise sum over ``total_rows`` rows
        # of dy * x * rstd. Two regimes:
        #
        # Wide-N (e.g. attn_norm/ffn_norm at D=2048): ``grid_n =
        # ceil(N/BLOCK_N) = 16`` is already enough to fill the SMs with
        # a small split_k. Use the legacy single-stage kernel with
        # ``tl.atomic_add`` directly into dW — atomic contention is low
        # because each (pid_n, pid_k) writes to a unique 128-element
        # column tile and only competes with the SPLIT_K programs
        # sharing that tile (typically 4-16 atomics on a 128-element
        # region, well-served by H100's L2 atomic path).
        #
        # Narrow-N (per-head q/k_norm at head_dim=128): ``grid_n=1`` so
        # we need a high split_k to fill the SMs. Atomic-add into one
        # 128-element vector with 256-512 atomics serializes badly —
        # the user's profile showed 42 ms on Qwen3-30B-A3B q_norm at
        # T=131072 on H100, ~10% of peak HBM BW. Use the two-stage
        # path: stage-1 writes partials to ``(SPLIT_K, N_COLS)`` with
        # no atomics, stage-2 sums them down. Empirically 2.4× faster
        # on the narrow case at T=32768 RTX 3090.
        BLOCK_N = 128
        BLOCK_M = 32
        n_sms = torch.cuda.get_device_properties(device).multi_processor_count
        grid_n = triton.cdiv(reduction_cols, BLOCK_N)
        # Target ~4 programs per SM total.
        target_programs = 4 * n_sms
        split_k_target = max(1, (target_programs + grid_n - 1) // grid_n)
        max_split_k = max(1, total_rows_for_reduction // BLOCK_M)
        SPLIT_K = min(split_k_target, max_split_k)
        SPLIT_K = min(1024, max(1, triton.next_power_of_2(SPLIT_K)))

        # Heuristic: switch to the two-stage path only when grid_n is
        # so small that a large split_k is required to fill SMs — that's
        # the case where atomic contention on a single output tile
        # serializes badly. Empirically (RTX 3090): legacy at split_k=32
        # for D=2048 (grid_n=16) is 0.32 ms; two-stage is 0.79 ms (2.5x
        # slower because of the launch overhead + partial alloc). At
        # head_dim=128 (grid_n=1, split_k=512) legacy is 4.3 ms,
        # two-stage is 1.8 ms (2.4x faster).
        use_two_stage = SPLIT_K > 32

        if use_two_stage:
            partial = torch.empty(
                (SPLIT_K, reduction_cols),
                dtype=torch.float32, device=device,
            )
            grid_partial = (grid_n, SPLIT_K)
            rms_norm_bwd_dw_partial_kernel[grid_partial](
                dY, X, rstd, partial,
                stride_dy_row, stride_x_row, stride_rstd_row,
                partial.stride(0),
                total_rows_for_reduction,
                reduction_cols,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            BLOCK_K = min(SPLIT_K, 64)
            rms_norm_bwd_dw_reduce_kernel[(grid_n,)](
                partial, dW_buffer,
                partial.stride(0),
                SPLIT_K, reduction_cols,
                ACCUMULATE_DW=ACCUMULATE_DW,
                BLOCK_N=BLOCK_N,
                BLOCK_K=triton.next_power_of_2(BLOCK_K),
            )
        else:
            # Legacy single-stage: each program ``atomic_add`` into dW.
            grid_dw = (grid_n, SPLIT_K)
            rms_norm_bwd_dw_kernel_legacy[grid_dw](
                dY, X, rstd, dW_buffer,
                stride_dy_row, stride_x_row, stride_rstd_row,
                total_rows_for_reduction,
                reduction_cols,
                ACCUMULATE_DW=ACCUMULATE_DW,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        
        # Handle non-float32 accumulation
        if user_dW is not None:
            # Add the float32 results back to the user's buffer (with cast)
            user_dW.add_(dW_buffer.to(user_dW_dtype))
            final_dW = user_dW
        else:
            final_dW = dW_buffer
    else:
        final_dW = None
    
    return dX, final_dW, final_recomputed_output