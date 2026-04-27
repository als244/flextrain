"""3-way numerical parity for :class:`flextrain.nn.embed.TokenEmbedLayer`.

Runs forward + backward on:

* **naive**:     pure PyTorch reference. Forward is a row-gather
                 (table[token_ids, :]). Backward is an ``index_add_``
                 scatter, done in fp32 to eliminate bf16 round-off on
                 the ground-truth side.
* **orig**:      ``orig.awsm_transformer.embed.TransformerEmbed``
                 unmodified.
* **flextrain**: :class:`flextrain.nn.embed.TokenEmbedLayer`.

Forward is just a fancy-index copy; it should be bit-identical across
all three implementations. Backward goes through
``flextrain_embedding_bwd`` (a CUDA kernel that does a scatter-add into the
embedding's gradient table) and should agree with the fp32 naive
reference within bf16 tolerance.

Why 3-way even for a pure-Python wrapper
----------------------------------------
We don't trust orig blindly. The naive reference establishes the
contract; orig and flextrain are compared against it independently.
If orig were wrong, we would see orig-vs-naive diverge while
flextrain-vs-naive agreed, which is a bug in orig rather than our
port. Lesson from the RoPE convention bug (see NOTES.md
[FINDING — from 3-way parity]).
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ORIG = os.path.join(ROOT, "orig")
if ORIG not in sys.path:
    sys.path.insert(0, ORIG)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Naive reference
# ---------------------------------------------------------------------------


def _embed_forward_naive(
    token_ids: torch.Tensor,
    w_tok_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Fancy-indexed gather, identical to orig/flextrain forward."""
    return w_tok_embeddings[token_ids, :]


def _embed_backward_naive(
    dx: torch.Tensor,  # (num_tokens, d_model)
    token_ids: torch.Tensor,  # (num_tokens,)
    grad_table: torch.Tensor,  # (vocab_size, d_model)
    *,
    scale: float = 1.0,
) -> None:
    """Scatter-add fp32 reference. Accumulates into ``grad_table`` in place.

    Does the math in fp32 regardless of input dtypes so the reference is
    immune to bf16 round-off in the accumulator. Caller is responsible
    for casting the result back to compare against bf16 kernels.
    """
    grad_fp32 = grad_table.float()
    dx_fp32 = dx.float() * scale
    grad_fp32.index_add_(0, token_ids, dx_fp32)
    grad_table.copy_(grad_fp32.to(grad_table.dtype))


# ---------------------------------------------------------------------------
# Orig wrapper
# ---------------------------------------------------------------------------


def _build_orig_embed(vocab_size: int, d_model: int):
    from awsm_transformer.embed import TransformerEmbed  # type: ignore[import-not-found]

    model_dims = {"vocab_size": vocab_size, "d_model": d_model}
    model_hyperparams: dict = {}
    return TransformerEmbed(model_dims, model_hyperparams)


# ---------------------------------------------------------------------------
# Flextrain wrapper
# ---------------------------------------------------------------------------


def _build_flextrain_embed(vocab_size: int, d_model: int):
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer

    return TokenEmbedLayer(
        TokenEmbedConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            compute_dtype=DTYPE,
            master_dtype=DTYPE,
            grad_dtype=DTYPE,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _fixed_inputs(vocab_size: int, d_model: int, num_tokens: int):
    torch.manual_seed(1234)
    table = torch.randn(
        vocab_size, d_model, device=DEVICE, dtype=DTYPE
    ) * (1.0 / (d_model**0.5))
    token_ids = torch.randint(
        0, vocab_size, (num_tokens,), device=DEVICE, dtype=torch.int64
    )
    dx = torch.randn(num_tokens, d_model, device=DEVICE, dtype=DTYPE) * 1e-2
    return table, token_ids, dx


def test_embed_forward_bit_identical() -> None:
    """All three forward implementations are pure fancy-indexed copies.
    They must agree bit-for-bit (no arithmetic, no precision loss)."""
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("test_embed_parity requires CUDA.")

    vocab_size, d_model, num_tokens = 4096, 128, 256
    table, token_ids, _dx = _fixed_inputs(vocab_size, d_model, num_tokens)

    y_naive = _embed_forward_naive(token_ids, table)

    orig_embed = _build_orig_embed(vocab_size, d_model)
    y_orig = orig_embed.forward(token_ids, {"w_tok_embeddings": table})

    ft_embed = _build_flextrain_embed(vocab_size, d_model)
    from flextrain.core.layer import LayerContext

    ctx = LayerContext(
        scratch=lambda shape, dtype: torch.empty(shape, dtype=dtype, device=DEVICE),
        kv_cache=None,
        stream=torch.cuda.current_stream(),
    )
    y_ft = ft_embed.forward(
        token_ids,
        chunk=None,  # type: ignore[arg-type]
        weights={"w_tok_embeddings": table},
        ctx=ctx,
    )

    # Fancy indexing is a pure copy; no tolerance needed.
    assert torch.equal(y_naive, y_orig), (
        f"naive and orig forward differ (max abs delta "
        f"{(y_naive - y_orig).abs().max().item()})"
    )
    assert torch.equal(y_naive, y_ft), (
        f"naive and flextrain forward differ (max abs delta "
        f"{(y_naive - y_ft).abs().max().item()})"
    )


def _run_backward_naive_fp32(
    table: torch.Tensor,
    token_ids: torch.Tensor,
    dx: torch.Tensor,
) -> torch.Tensor:
    """The FP32 ground truth we compare bf16 kernels against."""
    grad = torch.zeros_like(table)
    _embed_backward_naive(dx, token_ids, grad, scale=1.0)
    return grad


def _run_backward_orig(
    vocab_size: int,
    d_model: int,
    table: torch.Tensor,
    token_ids: torch.Tensor,
    dx: torch.Tensor,
) -> torch.Tensor:
    orig_embed = _build_orig_embed(vocab_size, d_model)
    grad = torch.zeros_like(table)
    orig_embed.backward(dx, token_ids, {"g_tok_embeddings": grad})
    torch.cuda.synchronize()
    return grad


def _run_backward_ft(
    vocab_size: int,
    d_model: int,
    table: torch.Tensor,
    token_ids: torch.Tensor,
    dx: torch.Tensor,
) -> torch.Tensor:
    from flextrain.core.layer import LayerContext

    ft_embed = _build_flextrain_embed(vocab_size, d_model)
    grad = torch.zeros_like(table)
    ctx = LayerContext(
        scratch=lambda shape, dtype: torch.empty(shape, dtype=dtype, device=DEVICE),
        kv_cache=None,
        stream=torch.cuda.current_stream(),
    )
    ft_embed.backward(
        dx,
        token_ids,
        chunk=None,  # type: ignore[arg-type]
        weights={"w_tok_embeddings": table},
        grads={"g_tok_embeddings": grad},
        ctx=ctx,
    )
    torch.cuda.synchronize()
    return grad


def _three_way_bwd_check(
    vocab_size: int, d_model: int, num_tokens: int, *, tol: float = 5e-2
) -> None:
    table, token_ids, dx = _fixed_inputs(vocab_size, d_model, num_tokens)

    g_naive = _run_backward_naive_fp32(table, token_ids, dx)
    g_orig = _run_backward_orig(vocab_size, d_model, table, token_ids, dx)
    g_ft = _run_backward_ft(vocab_size, d_model, table, token_ids, dx)

    def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.float()
        b = b.float()
        return ((a - b).norm() / (b.norm() + 1e-6)).item()

    e_no = rel_err(g_orig, g_naive)
    e_nf = rel_err(g_ft, g_naive)
    e_of = rel_err(g_ft, g_orig)

    print(
        f"    (V={vocab_size}, d={d_model}, T={num_tokens})  "
        f"orig-vs-naive={e_no:.2e}  ft-vs-naive={e_nf:.2e}  "
        f"orig-vs-ft={e_of:.2e}"
    )

    # orig vs flextrain MUST be bit-identical: same underlying kernel,
    # same inputs, same stream semantics.
    assert torch.equal(g_orig, g_ft), (
        f"orig and flextrain backward differ (max abs delta "
        f"{(g_orig - g_ft).abs().max().item()})"
    )

    # Both kernels vs fp32 scatter-add reference: expect bf16 round-off
    # on per-row accumulation. 5e-2 mirrors test_llama_parity.
    assert e_no < tol, (
        f"orig grad disagrees with naive fp32 reference: {e_no:.4e}"
    )
    assert e_nf < tol, (
        f"flextrain grad disagrees with naive fp32 reference: {e_nf:.4e}"
    )


def test_embed_backward_three_way_parity_sparse() -> None:
    """Many rows, few tokens per row -- tests gather pattern with little
    accumulation pressure. bf16 round-off stays near zero.
    """
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("test_embed_parity requires CUDA.")
    _three_way_bwd_check(vocab_size=4096, d_model=128, num_tokens=256)


def test_embed_backward_three_way_parity_dense() -> None:
    """Many tokens, small vocab -- each row gets ~32 scatter-adds, so bf16
    accumulation actually matters. This is the case where orig's kernel
    could drift from the fp32 reference if it weren't accumulating in
    fp32 internally; we verify it stays within bf16 tolerance.
    """
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("test_embed_parity requires CUDA.")
    _three_way_bwd_check(vocab_size=128, d_model=256, num_tokens=4096)


def test_embed_layer_shape_and_spec() -> None:
    """InputLayer Protocol surface: schema is empty, ParamSpec has one tensor
    of the expected shape and dtype."""
    ft_embed = _build_flextrain_embed(vocab_size=32000, d_model=4096)
    assert ft_embed.schema.max_tier == 0
    assert ft_embed.schema.fields == ()
    assert ft_embed.param_spec.names() == ("w_tok_embeddings",)
    t = ft_embed.param_spec.get("w_tok_embeddings")
    assert t.shape({"vocab_size": 32000, "d_model": 4096}) == (32000, 4096)
    assert t.compute_dtype == DTYPE


def _run_all() -> None:
    tests = [
        ("test_embed_layer_shape_and_spec", test_embed_layer_shape_and_spec),
        ("test_embed_forward_bit_identical", test_embed_forward_bit_identical),
        (
            "test_embed_backward_three_way_parity_sparse",
            test_embed_backward_three_way_parity_sparse,
        ),
        (
            "test_embed_backward_three_way_parity_dense",
            test_embed_backward_three_way_parity_dense,
        ),
    ]
    for name, fn in tests:
        print(f"  - {name}")
        fn()
    print(f"  OK ({len(tests)} tests)")


if __name__ == "__main__":
    _run_all()
