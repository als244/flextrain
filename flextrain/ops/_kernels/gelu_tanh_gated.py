"""Gated GELU-tanh activation kernel — Gemma 2 / Gemma 3 FFN.

Computes the fused ``gelu_tanh(x1) * x3`` (forward) and the matching
backward producing ``(dx1, dx3)``. Drop-in counterpart to
``flextrain_swiglu_{fwd,bwd}`` for architectures whose FFN is gated by
the tanh-approximated GELU instead of SiLU (Gemma family,
``hidden_activation="gelu_pytorch_tanh"`` in the HF config).

Formula
-------
GELU (tanh approximation), matching ``torch.nn.functional.gelu(...,
approximate="tanh")`` and HF's ``GeluPytorchTanh``::

    c = sqrt(2 / pi) ≈ 0.7978845608
    a = 0.044715
    u(x) = c * (x + a * x^3)
    gelu(x) = 0.5 * x * (1 + tanh(u(x)))

Gradient::

    du/dx = c * (1 + 3 * a * x^2)
    sech^2(u) = 1 - tanh^2(u)
    d/dx gelu(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech^2(u) * du/dx

For the gated form ``out = gelu(x1) * x3``::

    dx1 = dout * x3 * d/dx gelu(x1)
    dx3 = dout * gelu(x1)

Implementation notes
--------------------
* Operates in fp32 internally; inputs/outputs follow the SwiGLU
  kernel's dtype convention (bf16 in, bf16 out by default).
* Uses the ``tanh(x) = 2*sigmoid(2x) - 1`` identity so we don't need
  Triton's ``tl.math.tanh`` (some toolchains miss it; sigmoid is always
  available).
* 1-D flat kernel — same block-size / num_warps tuning as
  ``swiglu_fwd_kernel`` to keep behavior comparable on the same
  shapes.
"""
import torch
import triton
import triton.language as tl


_GELU_TANH_C = 0.7978845608028654  # sqrt(2 / pi)
_GELU_TANH_A = 0.044715


# -----------------
# Forward Kernel
# -----------------


@triton.jit
def gelu_tanh_gated_fwd_kernel(
    X1_PTR,
    X3_PTR,
    OUT_PTR,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x1 = tl.load(X1_PTR + offsets, mask=mask)
    x3 = tl.load(X3_PTR + offsets, mask=mask)

    x1_f32 = x1.to(tl.float32)
    x3_f32 = x3.to(tl.float32)

    x1_cu = x1_f32 * x1_f32 * x1_f32
    u = 0.7978845608028654 * (x1_f32 + 0.044715 * x1_cu)
    tanh_u = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    gelu_x1 = 0.5 * x1_f32 * (1.0 + tanh_u)

    out_f32 = gelu_x1 * x3_f32
    tl.store(
        OUT_PTR + offsets,
        out_f32.to(OUT_PTR.dtype.element_ty),
        mask=mask,
    )


# -----------------
# Backward Kernel
# -----------------


@triton.jit
def gelu_tanh_gated_bwd_kernel(
    X1_PTR,
    X3_PTR,
    DOUT_PTR,
    DX1_PTR,
    DX3_PTR,
    ACT_PTR,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
    STORE_ACTIVATIONS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x1 = tl.load(X1_PTR + offsets, mask=mask)
    x3 = tl.load(X3_PTR + offsets, mask=mask)
    dout = tl.load(DOUT_PTR + offsets, mask=mask)

    x1_f32 = x1.to(tl.float32)
    x3_f32 = x3.to(tl.float32)
    dout_f32 = dout.to(tl.float32)

    x1_sq = x1_f32 * x1_f32
    x1_cu = x1_sq * x1_f32
    u = 0.7978845608028654 * (x1_f32 + 0.044715 * x1_cu)
    tanh_u = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    one_plus_tanh = 1.0 + tanh_u
    gelu_x1 = 0.5 * x1_f32 * one_plus_tanh

    if STORE_ACTIVATIONS:
        act_f32 = gelu_x1 * x3_f32
        tl.store(
            ACT_PTR + offsets,
            act_f32.to(ACT_PTR.dtype.element_ty),
            mask=mask,
        )

    # dx3 = dout * gelu(x1)
    dx3 = dout_f32 * gelu_x1

    # dx1 = dout * x3 * d/dx gelu(x1)
    du_dx = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x1_sq)
    sech_sq = 1.0 - tanh_u * tanh_u
    dgelu_dx1 = 0.5 * one_plus_tanh + 0.5 * x1_f32 * sech_sq * du_dx
    dx1 = dout_f32 * x3_f32 * dgelu_dx1

    tl.store(DX1_PTR + offsets, dx1.to(DX1_PTR.dtype.element_ty), mask=mask)
    tl.store(DX3_PTR + offsets, dx3.to(DX3_PTR.dtype.element_ty), mask=mask)


# -----------------
# Wrappers
# -----------------


def flextrain_gelu_tanh_gated_fwd(
    x1: torch.Tensor, x3: torch.Tensor, out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``gelu_tanh(x1) * x3`` elementwise. Output dtype = ``x1.dtype``
    unless ``out`` is supplied (then ``out.dtype``)."""
    assert x1.shape == x3.shape
    assert x1.is_cuda and x3.is_cuda
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x3.is_contiguous():
        x3 = x3.contiguous()
    if out is None:
        out = torch.empty_like(x1)

    n_elements = x1.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_tanh_gated_fwd_kernel[grid](
        x1, x3, out, n_elements,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
    )
    return out


def flextrain_gelu_tanh_gated_bwd(
    x1: torch.Tensor,
    x3: torch.Tensor,
    dout: torch.Tensor,
    dx1: torch.Tensor | None = None,
    dx3: torch.Tensor | None = None,
    store_activations: bool = False,
    activations: torch.Tensor | None = None,
):
    """Backward for ``out = gelu_tanh(x1) * x3``. Returns ``(dx1, dx3)``
    or ``(dx1, dx3, activations)`` when ``store_activations=True``
    (the recomputed forward output, needed as the left operand for
    ``w_2``'s wgrad — analogous to swiglu_bwd)."""
    assert x1.shape == x3.shape == dout.shape
    assert x1.is_cuda and x3.is_cuda and dout.is_cuda
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x3.is_contiguous():
        x3 = x3.contiguous()
    if not dout.is_contiguous():
        dout = dout.contiguous()
    if dx1 is None:
        dx1 = torch.empty_like(x1)
    if dx3 is None:
        dx3 = torch.empty_like(x3)
    act_ptr = None
    if store_activations:
        if activations is None:
            activations = torch.empty_like(x1)
        act_ptr = activations

    n_elements = x1.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_tanh_gated_bwd_kernel[grid](
        x1, x3, dout, dx1, dx3, act_ptr, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        STORE_ACTIVATIONS=store_activations,
        num_warps=8,
    )
    if store_activations:
        return dx1, dx3, activations
    return dx1, dx3
