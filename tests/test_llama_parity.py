"""3-way numerical parity on a single Llama-style transformer block.

Runs the SAME forward pass three different ways and diffs outputs:

* **naive**:     pure PyTorch reference. No custom kernels, no fused matmul,
                 no flash-attn. Implements the block algebra from scratch so
                 it is the independent ground truth.
* **orig**:      ``orig.awsm_transformer.TransformerLayer.forward``
                 unmodified. The implementation the paper was written
                 against.
* **flextrain**: :class:`flextrain.nn.layers.LlamaBlock.forward` on the
                 new contract.

Expected result
---------------
All three outputs agree within bf16 tolerance (~1e-2 relative) on the
residual-stream output. If ``naive`` and ``orig`` disagree but ``orig``
and ``flextrain`` agree, we've inherited a bug from orig (flagged for
review). If ``naive`` and ``orig`` agree but ``flextrain`` diverges,
our port has a bug.

Shape choices kept small so it runs quickly on a 3090.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Also put orig/ on the path so we can import orig.awsm_transformer directly.
ORIG = os.path.join(ROOT, "orig")
if ORIG not in sys.path:
    sys.path.insert(0, ORIG)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Naive reference implementation in pure PyTorch.
# ---------------------------------------------------------------------------


def _rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Full-vector RMSNorm, pure PyTorch."""
    x_fp = x.float()
    rstd = torch.rsqrt(x_fp.pow(2).mean(-1, keepdim=True) + eps)
    return (x_fp * rstd).to(x.dtype) * w


def _rope_ref(
    x: torch.Tensor,  # (T, H, D)
    seq_positions: torch.Tensor,  # (T,)
    theta: float,
) -> torch.Tensor:
    """RoPE with the PAIR-INTERLEAVE convention (``x[..., 2i]`` paired with
    ``x[..., 2i+1]``). Matches the convention in
    ``orig/awsm_transformer/ops/rope.py:38-48``.

    NOT the halved-split convention (``x[:D/2]`` with ``x[D/2:]``) that
    HuggingFace's Llama implementation uses. Both are valid RoPE; they
    produce different tensors for the same logical rotation. See
    tests/test_gqk_investigation.py for the investigation that caught
    this.
    """
    T, H, D = x.shape
    assert D % 2 == 0
    half = D // 2
    pos = seq_positions.view(-1, 1, 1).float()  # (T, 1, 1)
    # Exponent = 2i/D for i in 0..D/2 (matches kernel's ``2.0 * offs / D``)
    exponent = 2.0 * torch.arange(0, half, device=x.device).float() / D
    inv_freq = theta ** (-exponent)  # (D/2,)
    angles = pos * inv_freq  # (T, 1, D/2)
    cos = angles.cos()
    sin = angles.sin()
    x_fp = x.float()
    even = x_fp[..., 0::2]  # x[..., 2i]
    odd = x_fp[..., 1::2]   # x[..., 2i+1]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    # Interleave back into the original layout.
    out = torch.empty_like(x_fp)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out.to(x.dtype)


def _sdpa_causal_ref(
    q: torch.Tensor,  # (T, H, D)
    k: torch.Tensor,  # (T, H_kv, D)
    v: torch.Tensor,  # (T, H_kv, D)
) -> torch.Tensor:
    """Single-sequence causal scaled-dot-product attention. Handles GQA by
    repeating K/V to match Q heads. Returns (T, H, D)."""
    T, H, D = q.shape
    _, H_kv, _ = k.shape
    if H_kv != H:
        rep = H // H_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    # (H, T, D) for batched matmul
    q_ = q.transpose(0, 1).float()
    k_ = k.transpose(0, 1).float()
    v_ = v.transpose(0, 1).float()
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale  # (H, T, T)
    mask = torch.triu(
        torch.full((T, T), float("-inf"), device=q.device), diagonal=1
    )
    scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_)  # (H, T, D)
    return out.transpose(0, 1).to(q.dtype).contiguous()


def _swiglu_ref(x1: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x1.float()).to(x1.dtype) * x3


def naive_llama_block_forward(
    x: torch.Tensor,
    weights: dict,
    seq_positions: torch.Tensor,
    *,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    rope_base: float,
) -> torch.Tensor:
    """Naive PyTorch reference forward. Returns the residual-stream output."""
    # --- Attention norm + Q/K/V ---
    h = _rmsnorm_ref(x, weights["w_attn_norm"], rms_norm_eps)

    xq = (h @ weights["w_q"]).view(-1, n_heads, head_dim)
    xk = (h @ weights["w_k"]).view(-1, n_kv_heads, head_dim)
    xv = (h @ weights["w_v"]).view(-1, n_kv_heads, head_dim)

    # --- RoPE ---
    rope_q = _rope_ref(xq, seq_positions, rope_base)
    rope_k = _rope_ref(xk, seq_positions, rope_base)

    # --- Scaled dot-product attention (causal, single sequence) ---
    attn_out = _sdpa_causal_ref(rope_q, rope_k, xv)  # (T, H, D)
    attn_flat = attn_out.reshape(-1, n_heads * head_dim)

    # --- Output projection + residual ---
    attn_output_with_residual = x + attn_flat @ weights["w_o"]

    # --- FFN norm + SwiGLU + down proj + residual ---
    h2 = _rmsnorm_ref(
        attn_output_with_residual, weights["w_ffn_norm"], rms_norm_eps
    )
    x1 = h2 @ weights["w_1"]
    x3 = h2 @ weights["w_3"]
    mlp = _swiglu_ref(x1, x3) @ weights["w_2"]
    return attn_output_with_residual + mlp


# ---------------------------------------------------------------------------
# Orig-layer forward invocation.
# ---------------------------------------------------------------------------


def build_orig_transformer_layer(
    *, d_model, n_heads, n_kv_heads, head_dim, expert_dim,
    rms_norm_eps, rope_base,
):
    from awsm_transformer.dense_layer import TransformerLayer  # noqa: E402

    model_dims = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "expert_dim": expert_dim,
        "is_causal": True,
        "datatypes": {"residual": "bfloat16"},
    }
    model_hyperparams = {
        "rms_norm_eps": rms_norm_eps,
        "position_angles": torch.tensor(
            [rope_base], dtype=torch.float32, device=DEVICE
        ),
        "window_size_left": -1,
        "window_size_right": 0,
    }
    layer = TransformerLayer(
        layer_id=0,
        model_dims=model_dims,
        model_hyperparams=model_hyperparams,
    )
    return layer, model_dims, model_hyperparams


def run_orig_forward(
    layer,
    x: torch.Tensor,
    weights: dict,
    *,
    seq_len: int,
) -> torch.Tensor:
    """Call orig's TransformerLayer.forward directly. Needs fwd_context
    buffers sized for the KV ring and a chunk_metadata dict matching
    orig's schema."""
    # fwd_context needs K/V buffers sized for total_k tokens.
    n_kv = layer.model_dims["n_kv_heads"]
    head_dim = layer.model_dims["head_dim"]
    fwd_context = {
        "k": torch.zeros(seq_len, n_kv, head_dim, device=DEVICE, dtype=DTYPE),
        "v": torch.zeros(seq_len, n_kv, head_dim, device=DEVICE, dtype=DTYPE),
    }

    # chunk_metadata via orig's helper
    chunk_metadata = layer.make_chunk_metadata(
        seq_lens=[seq_len],
        seq_positions=list(range(seq_len)),
        prior_seq_lens=[0],
        prior_seq_offsets=[0],
        device=DEVICE,
    )

    # Allocate base_act_slot at max tier on device.
    act_slot, _ = layer.make_act_slot(
        seq_len, saved_level=3, buffer=None, device=DEVICE, pin_memory=False
    )

    out, _ = layer.forward(
        x.clone(),
        chunk_metadata,
        weights,
        act_slot,
        fwd_context,
    )
    return out.clone()


# ---------------------------------------------------------------------------
# Flextrain LlamaBlock invocation.
# ---------------------------------------------------------------------------


def build_flextrain_llama_block(
    *, d_model, n_heads, n_kv_heads, head_dim, expert_dim,
    rms_norm_eps, rope_base,
):
    from flextrain.nn.layers import LlamaBlock, LlamaBlockConfig

    cfg = LlamaBlockConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        expert_dim=expert_dim,
        rms_norm_eps=rms_norm_eps,
        rope_base=rope_base,
    )
    return LlamaBlock(layer_id=0, cfg=cfg), cfg


def run_flextrain_forward(
    layer,
    cfg,
    x: torch.Tensor,
    weights: dict,
    *,
    seq_len: int,
) -> torch.Tensor:
    from flextrain.core.activation_schema import ActivationSlot
    from flextrain.core.layer import ChunkMeta, LayerContext
    from flextrain.engine.buffers import KVContextWindow, ScratchPool

    dims = cfg.dims()
    chunk = ChunkMeta.build(
        seq_lens=[seq_len],
        seq_positions=list(range(seq_len)),
        prior_seq_lens=[0],
        prior_seq_offsets=[0],
        device=DEVICE,
    )

    # Device slot at max tier.
    dev_nbytes = layer.schema.device_size_bytes(seq_len, dims)
    dev_buf = torch.zeros(dev_nbytes, device=DEVICE, dtype=torch.uint8)
    slot, _ = ActivationSlot.from_buffer(
        layer.schema,
        layer.schema.max_tier,
        seq_len,
        dims,
        dev_buf,
        include_nonpersistent=True,
    )

    # KV context window (seq_len tokens, matching attention total_k).
    kv = KVContextWindow(
        k=torch.zeros(seq_len, cfg.n_kv_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE),
        v=torch.zeros(seq_len, cfg.n_kv_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE),
        dk=torch.zeros(seq_len, cfg.n_kv_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE),
        dv=torch.zeros(seq_len, cfg.n_kv_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE),
    )
    ctx = LayerContext(
        scratch=ScratchPool(torch.device(DEVICE)),
        kv_cache=kv,
        stream=torch.cuda.current_stream(),
    )

    out = layer.forward(x.clone(), chunk, weights, slot, ctx)
    return out.clone()


# ---------------------------------------------------------------------------
# The 3-way test.
# ---------------------------------------------------------------------------


def _make_inputs(seq_len: int, cfg) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Construct a shared (x, weights, seq_positions) triple used by all
    three implementations. Weights are bf16; we deliberately make them
    small-amplitude so bf16 rounding doesn't blow up comparisons."""
    gen = torch.Generator(device=DEVICE).manual_seed(0)

    def rnd(*shape):
        return (
            torch.randn(*shape, generator=gen, device=DEVICE, dtype=torch.float32)
            * 0.02
        ).to(DTYPE)

    d_model = cfg.d_model
    n_heads = cfg.n_heads
    n_kv = cfg.n_kv_heads
    head_dim = cfg.head_dim
    expert_dim = cfg.expert_dim

    x = rnd(seq_len, d_model)

    weights = {
        "w_attn_norm": (
            1.0
            + torch.randn(
                d_model, generator=gen, device=DEVICE, dtype=torch.float32
            )
            * 0.02
        ).to(DTYPE),
        "w_q": rnd(d_model, n_heads * head_dim),
        "w_k": rnd(d_model, n_kv * head_dim),
        "w_v": rnd(d_model, n_kv * head_dim),
        "w_o": rnd(n_heads * head_dim, d_model),
        "w_ffn_norm": (
            1.0
            + torch.randn(
                d_model, generator=gen, device=DEVICE, dtype=torch.float32
            )
            * 0.02
        ).to(DTYPE),
        "w_1": rnd(d_model, expert_dim),
        "w_2": rnd(expert_dim, d_model),
        "w_3": rnd(d_model, expert_dim),
    }
    seq_positions = torch.arange(seq_len, dtype=torch.int32, device=DEVICE)
    return x, weights, seq_positions


def naive_llama_block_with_autograd(
    x: torch.Tensor,
    weights: dict,
    seq_positions: torch.Tensor,
    dy: torch.Tensor,
    *,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    rope_base: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Naive forward + autograd backward. Returns (y, grad_weights, dx).

    ``grad_weights`` dict mirrors the weight names ("g_q", "g_k", ...) so
    we can compare element-wise against orig/flextrain's in-place grad
    accumulators.
    """
    # All weights + input need grad.
    x_g = x.detach().clone().requires_grad_(True)
    w_g = {
        name: w.detach().clone().requires_grad_(True) for name, w in weights.items()
    }

    y = naive_llama_block_forward(
        x_g,
        w_g,
        seq_positions,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        rope_base=rope_base,
    )
    y.backward(dy)

    # Map weight-name -> grad-name (orig uses "g_" prefix; strip "w_").
    grad_weights = {}
    for name, wt in w_g.items():
        grad_name = "g_" + name[2:] if name.startswith("w_") else "g_" + name
        grad_weights[grad_name] = wt.grad.detach().clone()
    return y.detach().clone(), grad_weights, x_g.grad.detach().clone()


def run_orig_backward(
    layer,
    x: torch.Tensor,
    weights: dict,
    dy: torch.Tensor,
    *,
    seq_len: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Run orig forward + backward. Returns (y, grad_weights, dx)."""
    n_kv = layer.model_dims["n_kv_heads"]
    head_dim = layer.model_dims["head_dim"]
    fwd_context = {
        "k": torch.zeros(seq_len, n_kv, head_dim, device=DEVICE, dtype=DTYPE),
        "v": torch.zeros(seq_len, n_kv, head_dim, device=DEVICE, dtype=DTYPE),
    }
    bwd_context = {
        "dk": torch.zeros(seq_len, n_kv, head_dim, device=DEVICE, dtype=DTYPE),
        "dv": torch.zeros(seq_len, n_kv, head_dim, device=DEVICE, dtype=DTYPE),
    }

    chunk_metadata = layer.make_chunk_metadata(
        seq_lens=[seq_len],
        seq_positions=list(range(seq_len)),
        prior_seq_lens=[0],
        prior_seq_offsets=[0],
        device=DEVICE,
    )

    # Zero gradient buffers.
    grad_weights = layer.create(
        buffer=None, device=DEVICE, pin_memory=False, is_grad=True
    )
    for t in grad_weights.values():
        t.zero_()

    # Allocate act slot at max tier so all fields are populated
    # (forward_recompute becomes a no-op).
    act_slot, _ = layer.make_act_slot(
        seq_len, saved_level=3, buffer=None, device=DEVICE, pin_memory=False
    )

    out, _ = layer.forward(
        x.clone(),
        chunk_metadata,
        weights,
        act_slot,
        fwd_context,
    )
    y = out.clone()
    dx = layer.backward(
        dy.clone(),
        chunk_metadata,
        weights,
        grad_weights,
        act_slot,
        fwd_context,
        bwd_context,
    )
    return y, {k: v.clone() for k, v in grad_weights.items()}, dx.clone()


def run_flextrain_backward(
    layer,
    cfg,
    x: torch.Tensor,
    weights: dict,
    dy: torch.Tensor,
    *,
    seq_len: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Run flextrain forward + backward. Returns (y, grad_weights, dx)."""
    from flextrain.core.activation_schema import ActivationSlot
    from flextrain.core.layer import ChunkMeta, LayerContext
    from flextrain.engine.buffers import KVContextWindow, ScratchPool

    dims = cfg.dims()
    chunk = ChunkMeta.build(
        seq_lens=[seq_len],
        seq_positions=list(range(seq_len)),
        prior_seq_lens=[0],
        prior_seq_offsets=[0],
        device=DEVICE,
    )

    dev_nbytes = layer.schema.device_size_bytes(seq_len, dims)
    dev_buf = torch.zeros(dev_nbytes, device=DEVICE, dtype=torch.uint8)
    slot, _ = ActivationSlot.from_buffer(
        layer.schema,
        layer.schema.max_tier,
        seq_len,
        dims,
        dev_buf,
        include_nonpersistent=True,
    )

    kv = KVContextWindow(
        k=torch.zeros(seq_len, cfg.n_kv_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE),
        v=torch.zeros(seq_len, cfg.n_kv_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE),
        dk=torch.zeros(seq_len, cfg.n_kv_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE),
        dv=torch.zeros(seq_len, cfg.n_kv_heads, cfg.head_dim, device=DEVICE, dtype=DTYPE),
    )
    ctx = LayerContext(
        scratch=ScratchPool(torch.device(DEVICE)),
        kv_cache=kv,
        stream=torch.cuda.current_stream(),
    )

    # Zero gradient buffers. RMSNorm grads must be fp32; attn/FFN grads bf16.
    grads = {}
    for t in layer.param_spec.tensors:
        grad_name = "g_" + t.name[2:] if t.name.startswith("w_") else "g_" + t.name
        shape = t.shape(dims)
        grads[grad_name] = torch.zeros(shape, dtype=t.grad_dtype, device=DEVICE)

    y = layer.forward(x.clone(), chunk, weights, slot, ctx)
    y = y.clone()
    dx = layer.backward(dy.clone(), chunk, weights, grads, slot, ctx)
    return y, {k: v.clone() for k, v in grads.items()}, dx.clone()


def test_three_way_llama_parity() -> None:
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError(
            "test_llama_parity requires CUDA; no device available."
        )

    # Small but realistic.
    d_model = 128
    n_heads = 4
    n_kv_heads = 2
    head_dim = 32  # d_model = n_heads * head_dim
    expert_dim = 256
    seq_len = 64
    rms_norm_eps = 1e-5
    rope_base = 500000.0

    # Build three implementations with matching hyperparams.
    orig_layer, orig_dims, orig_hp = build_orig_transformer_layer(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, expert_dim=expert_dim,
        rms_norm_eps=rms_norm_eps, rope_base=rope_base,
    )
    ft_layer, ft_cfg = build_flextrain_llama_block(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, expert_dim=expert_dim,
        rms_norm_eps=rms_norm_eps, rope_base=rope_base,
    )

    x, weights, seq_positions = _make_inputs(seq_len, ft_cfg)

    # Run naive reference on a clone of x (block may mutate in-place).
    y_naive = naive_llama_block_forward(
        x.clone(),
        weights,
        seq_positions,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        rope_base=rope_base,
    )

    # Run orig.
    y_orig = run_orig_forward(orig_layer, x, weights, seq_len=seq_len)

    # Run flextrain.
    y_ft = run_flextrain_forward(ft_layer, ft_cfg, x, weights, seq_len=seq_len)

    # Compare. bf16 is lossy; compute relative error in fp32.
    def rel_err(a, b):
        a = a.float()
        b = b.float()
        return (a - b).norm() / (b.norm() + 1e-6)

    err_naive_vs_orig = rel_err(y_naive, y_orig).item()
    err_naive_vs_ft = rel_err(y_naive, y_ft).item()
    err_orig_vs_ft = rel_err(y_orig, y_ft).item()

    print(f"  naive vs orig: {err_naive_vs_orig:.4e}")
    print(f"  naive vs ft:   {err_naive_vs_ft:.4e}")
    print(f"  orig  vs ft:   {err_orig_vs_ft:.4e}")

    TOL = 5e-2  # bf16 across a full transformer block; loose but detects bugs
    assert err_naive_vs_orig < TOL, (
        f"orig disagrees with naive PyTorch reference: {err_naive_vs_orig:.4e}"
    )
    assert err_naive_vs_ft < TOL, (
        f"flextrain disagrees with naive PyTorch reference: {err_naive_vs_ft:.4e}"
    )
    assert err_orig_vs_ft < TOL, (
        f"flextrain disagrees with orig: {err_orig_vs_ft:.4e}"
    )


def test_three_way_llama_backward_parity() -> None:
    """Forward + backward 3-way parity. Checks dx and per-weight grads."""
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError(
            "test_llama_parity requires CUDA; no device available."
        )

    d_model = 128
    n_heads = 4
    n_kv_heads = 2
    head_dim = 32
    expert_dim = 256
    seq_len = 64
    rms_norm_eps = 1e-5
    rope_base = 500000.0

    orig_layer, _, _ = build_orig_transformer_layer(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, expert_dim=expert_dim,
        rms_norm_eps=rms_norm_eps, rope_base=rope_base,
    )
    ft_layer, ft_cfg = build_flextrain_llama_block(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, expert_dim=expert_dim,
        rms_norm_eps=rms_norm_eps, rope_base=rope_base,
    )

    x, weights, seq_positions = _make_inputs(seq_len, ft_cfg)

    # Fixed dY so backward is deterministic.
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    dy = (
        torch.randn(
            seq_len, d_model, generator=gen, device=DEVICE, dtype=torch.float32
        )
        * 0.02
    ).to(DTYPE)

    y_naive, g_naive, dx_naive = naive_llama_block_with_autograd(
        x, weights, seq_positions, dy,
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, rms_norm_eps=rms_norm_eps, rope_base=rope_base,
    )
    y_orig, g_orig, dx_orig = run_orig_backward(
        orig_layer, x, weights, dy, seq_len=seq_len
    )
    y_ft, g_ft, dx_ft = run_flextrain_backward(
        ft_layer, ft_cfg, x, weights, dy, seq_len=seq_len
    )

    def rel(a, b):
        a = a.float()
        b = b.float()
        return (a - b).norm() / (b.norm() + 1e-6)

    # Forward (should match the forward-only test).
    print(f"  fwd: naive vs orig: {rel(y_naive, y_orig).item():.4e}")
    print(f"  fwd: naive vs ft:   {rel(y_naive, y_ft).item():.4e}")
    print(f"  fwd: orig  vs ft:   {rel(y_orig, y_ft).item():.4e}")

    # dx.
    print(f"  dx: naive vs orig: {rel(dx_naive, dx_orig).item():.4e}")
    print(f"  dx: naive vs ft:   {rel(dx_naive, dx_ft).item():.4e}")
    print(f"  dx: orig  vs ft:   {rel(dx_orig, dx_ft).item():.4e}")

    # Per-weight grads. Compare keys that exist in all three.
    common_keys = set(g_naive) & set(g_orig) & set(g_ft)
    assert common_keys, f"no common grad keys: {set(g_naive)} {set(g_orig)} {set(g_ft)}"

    per_wt_errors = {}
    for k in sorted(common_keys):
        e_naive_orig = rel(g_naive[k], g_orig[k]).item()
        e_naive_ft = rel(g_naive[k], g_ft[k]).item()
        e_orig_ft = rel(g_orig[k], g_ft[k]).item()
        per_wt_errors[k] = (e_naive_orig, e_naive_ft, e_orig_ft)
        print(
            f"  grad {k:14s}: naive/orig {e_naive_orig:.3e}  "
            f"naive/ft {e_naive_ft:.3e}  orig/ft {e_orig_ft:.3e}"
        )

    # Tolerances.
    #
    # Observed values on an RTX 3090 at this config (seed 0/1) after the
    # naive RoPE convention was corrected to pair-interleave (the
    # convention orig's Triton kernel uses -- see
    # tests/test_gqk_investigation.py for the investigation):
    #
    #   fwd (all 3 impls):        ~1.8e-2 naive-vs-{orig,ft};  0.0 orig-vs-ft
    #   dx:                       ~3.2e-2 naive-vs-{orig,ft};  0.0 orig-vs-ft
    #   g_1 / g_2 / g_3:          ~2.6e-2 naive-vs-{orig,ft};  0.0 orig-vs-ft
    #   g_q / g_k / g_v / g_o:    ~3e-2   naive-vs-{orig,ft};  0.0 orig-vs-ft
    #   g_attn_norm / g_ffn_norm: ~6e-2   naive-vs-{orig,ft};  ~1e-3 orig-vs-ft
    #
    # The g_attn_norm / g_ffn_norm ~1e-3 orig-vs-ft non-zero is atomic-add
    # nondeterminism in flextrain_rmsnorm_bwd's dW accumulator. Not a
    # correctness issue; tolerance reflects observed noise.
    TOL_FWD = 5e-2
    TOL_DX = 1e-1
    TOL_GRAD_NAIVE = 1e-1  # bf16 cascade through a full block
    TOL_GRAD_ORIG_FT = 5e-3  # accommodates atomic-add nondeterminism in RMSNorm bwd

    assert rel(y_naive, y_orig).item() < TOL_FWD
    assert rel(y_naive, y_ft).item() < TOL_FWD
    assert rel(y_orig, y_ft).item() < TOL_FWD

    assert rel(dx_naive, dx_orig).item() < TOL_DX, (
        f"orig dx disagrees with naive autograd: "
        f"{rel(dx_naive, dx_orig).item():.3e}"
    )
    assert rel(dx_naive, dx_ft).item() < TOL_DX, (
        f"flextrain dx disagrees with naive autograd: "
        f"{rel(dx_naive, dx_ft).item():.3e}"
    )
    assert rel(dx_orig, dx_ft).item() < TOL_GRAD_ORIG_FT, (
        f"flextrain dx disagrees with orig: {rel(dx_orig, dx_ft).item():.3e}"
    )

    for k, (e_no, e_nf, e_of) in per_wt_errors.items():
        assert e_no < TOL_GRAD_NAIVE, (
            f"orig g_{k} disagrees with naive autograd: {e_no:.3e}"
        )
        assert e_nf < TOL_GRAD_NAIVE, (
            f"flextrain g_{k} disagrees with naive autograd: {e_nf:.3e}"
        )
        assert e_of < TOL_GRAD_ORIG_FT, (
            f"flextrain g_{k} disagrees with orig: {e_of:.3e}"
        )


def _run_all() -> None:
    tests = [
        test_three_way_llama_parity,
        test_three_way_llama_backward_parity,
    ]
    for fn in tests:
        print(f"... {fn.__name__}", flush=True)
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    print(f"\nAll {len(tests)} Llama-parity tests passed.")


if __name__ == "__main__":
    _run_all()
