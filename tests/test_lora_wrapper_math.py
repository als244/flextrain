"""LoRAWrapperLayer math parity vs autograd reference.

Builds a tiny LlamaBlock, wraps it with LoRA on **all** linear
projections (Q, K, V, O, FFN gate w_1, FFN up w_3, FFN down w_2),
runs fwd+bwd, and compares every gradient against a fully-autograd
PyTorch reference.

The reference path constructs the same arithmetic — RMSNorm + GQA
with effective weights ``W' = W + A @ B * scale`` for every targeted
projection, RoPE, naive softmax attention, SwiGLU FFN, and the down
projection — using ``requires_grad=True`` on A and B and calling
``torch.autograd.backward``. All other base weights stay leaf with
``requires_grad=False``.

Asserts:
* Forward output matches autograd reference within bf16 noise.
* dL/dA, dL/dB for every LoRA target match within bf16 noise.
* dL/dx (residual stream gradient) matches.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.core.activation_schema import ActivationSchema, ActivationSlot
from flextrain.core.layer import ChunkMeta, LayerContext
from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
from flextrain.bench.parity import _rmsnorm, _rope_pair_interleave, DTYPE


DEVICE = "cuda:0"


_LORA_TARGETS = ("w_q", "w_k", "w_v", "w_o", "w_1", "w_2", "w_3")


def _autograd_reference(
    cfg: LlamaBlockConfig,
    base_weights: dict[str, torch.Tensor],
    lora_weights: dict[str, torch.Tensor],
    scale: float,
    x: torch.Tensor,
    seq_positions: torch.Tensor,
    targets_dx: torch.Tensor,
):
    """Autograd Llama block with W' = W + A @ B * scale on ALL targets."""
    # Build leaves for A, B (require grad).
    leaves = {}
    for tgt in _LORA_TARGETS:
        leaves[f"{tgt}_lora_a"] = (
            lora_weights[f"{tgt}_lora_a"].detach().clone().float().requires_grad_(True)
        )
        leaves[f"{tgt}_lora_b"] = (
            lora_weights[f"{tgt}_lora_b"].detach().clone().float().requires_grad_(True)
        )
    x_leaf = x.detach().clone().requires_grad_(True)

    # Reference uses the SAME kernels FT uses for everything except
    # the matmuls (which need autograd to flow LoRA grads). We wrap the
    # kernels in a custom autograd Function so the reference path can
    # call .backward() normally.
    bf = x_leaf.dtype

    def W_eff(name: str) -> torch.Tensor:
        W = base_weights[name].to(torch.float32)
        A = leaves[f"{name}_lora_a"]
        B = leaves[f"{name}_lora_b"]
        return (W + (A @ B) * scale).to(bf)

    W_q_eff = W_eff("w_q")
    W_k_eff = W_eff("w_k")
    W_v_eff = W_eff("w_v")
    W_o_eff = W_eff("w_o")
    W_1_eff = W_eff("w_1")
    W_2_eff = W_eff("w_2")
    W_3_eff = W_eff("w_3")

    # Use FT's RMSNorm kernel via a custom Function for autograd compat.
    # Skipping that complexity — the RMSNorm kernel was shown to be
    # bit-identical to ``_rmsnorm``, so use the latter directly. Same for
    # RoPE: pair-interleave python matches the kernel.
    h = _rmsnorm(x_leaf, base_weights["w_attn_norm"], cfg.rms_norm_eps)
    xq = (h @ W_q_eff).view(-1, cfg.n_heads, cfg.head_dim)
    xk = (h @ W_k_eff).view(-1, cfg.n_kv_heads, cfg.head_dim)
    xv = (h @ W_v_eff).view(-1, cfg.n_kv_heads, cfg.head_dim)
    rope_q = _rope_pair_interleave(xq, seq_positions, cfg.rope_base)
    rope_k = _rope_pair_interleave(xk, seq_positions, cfg.rope_base)
    T = rope_q.shape[0]
    q_b = rope_q.transpose(0, 1).unsqueeze(0)
    k_b = rope_k.transpose(0, 1).unsqueeze(0)
    v_b = xv.transpose(0, 1).unsqueeze(0)
    attn_out = F.scaled_dot_product_attention(
        q_b, k_b, v_b, is_causal=True,
        enable_gqa=(cfg.n_kv_heads != cfg.n_heads),
    )
    attn_flat = attn_out.squeeze(0).transpose(0, 1).contiguous().reshape(T, -1)
    x_after_attn = x_leaf + attn_flat @ W_o_eff

    h2 = _rmsnorm(x_after_attn, base_weights["w_ffn_norm"], cfg.rms_norm_eps)
    x1 = h2 @ W_1_eff
    x3 = h2 @ W_3_eff
    swiglu = (F.silu(x1.float()) * x3.float()).to(bf)
    layer_out = x_after_attn + swiglu @ W_2_eff

    layer_out.backward(targets_dx.detach())
    grads = {name: leaf.grad for name, leaf in leaves.items()}
    return layer_out, grads, x_leaf.grad


def main():
    torch.manual_seed(7)
    cfg = LlamaBlockConfig(
        d_model=128, n_heads=4, n_kv_heads=2, head_dim=32,
        expert_dim=256, rms_norm_eps=1e-5, rope_base=10_000.0,
        is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    T = 32
    base = LlamaBlock(layer_id=0, cfg=cfg)
    dims = {
        "d_model": cfg.d_model, "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads, "head_dim": cfg.head_dim,
        "attn_dim": cfg.n_heads * cfg.head_dim,
        "kv_dim": cfg.n_kv_heads * cfg.head_dim,
        "expert_dim": cfg.expert_dim,
    }
    rank = 8
    alpha = 16.0
    scale = alpha / rank
    layer = LoRAWrapperLayer(
        base, lora_targets="all",
        rank=rank, alpha=alpha, dims=dims,
    )

    # Build random base + LoRA weights.
    weights: dict[str, torch.Tensor] = {}
    for t in layer.param_spec.tensors:
        shape = t.shape(dims)
        w = torch.randn(*shape, dtype=DTYPE, device=DEVICE) * 0.02
        if "norm" in t.name:
            w = (torch.ones(*shape, dtype=DTYPE, device=DEVICE)
                 + 0.01 * torch.randn(*shape, dtype=DTYPE, device=DEVICE))
        if t.name.endswith("_lora_b"):
            w = torch.zeros(*shape, dtype=DTYPE, device=DEVICE)  # PEFT init
        weights[t.name] = w

    # Random input + upstream grad for parity.
    x = torch.randn(T, cfg.d_model, dtype=DTYPE, device=DEVICE)
    upstream = torch.randn(T, cfg.d_model, dtype=DTYPE, device=DEVICE) * 0.01
    seq_positions = torch.arange(T, dtype=torch.int32, device=DEVICE)

    # ---- Make A AND B nonzero for every LoRA target so gradients flow. ----
    for tgt in _LORA_TARGETS:
        weights[f"{tgt}_lora_a"] = (
            torch.randn_like(weights[f"{tgt}_lora_a"]) * 0.02
        )
        weights[f"{tgt}_lora_b"] = (
            torch.randn_like(weights[f"{tgt}_lora_b"]) * 0.02
        )

    # ---- FlexTrain wrapper fwd + bwd ----
    schema = layer.schema
    slot_tensors = {}
    for f in schema.fields:
        shape = f.shape_fn(T, dims)
        slot_tensors[f.name] = torch.empty(shape, dtype=f.dtype, device=DEVICE)
    slot = ActivationSlot(schema=schema, level=schema.max_tier, tensors=slot_tensors)

    # Build a minimal ChunkMeta + LayerContext. The Llama block needs the
    # KV cache, sequence offsets, and a scratch pool.
    from flextrain.engine.buffers import ScratchPool
    scratch = ScratchPool(device=DEVICE)

    class _MiniKV:
        def __init__(self, T, n_kv, head_dim, dtype, device):
            self.k = torch.zeros(T, n_kv, head_dim, dtype=dtype, device=device)
            self.v = torch.zeros(T, n_kv, head_dim, dtype=dtype, device=device)
            self.dk = torch.zeros(T, n_kv, head_dim, dtype=dtype, device=device)
            self.dv = torch.zeros(T, n_kv, head_dim, dtype=dtype, device=device)

    kv = _MiniKV(T, cfg.n_kv_heads, cfg.head_dim, DTYPE, DEVICE)
    chunk = ChunkMeta.build(
        seq_lens=[T], seq_positions=list(range(T)),
        prior_seq_lens=[0], prior_seq_offsets=[0],
        device=DEVICE,
    )
    ctx = LayerContext(
        scratch=scratch, kv_cache=kv,
        stream=torch.cuda.current_stream(),
        secondary_stream=None,
        total_tokens_per_step=T,
    )

    # IMPORTANT: LlamaBlock.forward mutates ``x`` in place (the FFN's
    # final addmm uses ``out_tensor=x``). Snapshot before calling so the
    # autograd reference sees the same input.
    x_orig = x.clone()
    # FT fwd.
    y_ft = layer.forward(x, chunk, weights, slot, ctx)

    # Allocate grad accumulators for the trainable tensors.
    grads: dict[str, torch.Tensor] = {}
    for t in layer.param_spec.tensors:
        if t.frozen:
            continue
        gkey = "g_" + t.name[2:] if t.name.startswith("w_") else "g_" + t.name
        grads[gkey] = torch.zeros(
            *t.shape(dims), dtype=torch.float32, device=DEVICE,
        )

    dx_ft = layer.backward(upstream, chunk, weights, grads, slot, ctx)

    # ---- Autograd reference ----
    base_weights = {
        n: weights[n] for n in (
            "w_q", "w_k", "w_v", "w_o", "w_1", "w_2", "w_3",
            "w_attn_norm", "w_ffn_norm",
        )
    }
    lora_weights = {
        f"{t}_lora_{x}": weights[f"{t}_lora_{x}"]
        for t in _LORA_TARGETS for x in ("a", "b")
    }
    y_ref, ref_grads, dx_ref = _autograd_reference(
        cfg, base_weights, lora_weights, scale,
        x_orig, seq_positions, upstream,
    )

    # ---- Compare ----
    print("=== forward parity ===")
    fwd_delta = (y_ft.float() - y_ref.float()).abs().max().item()
    fwd_max = max(y_ft.abs().max().item(), y_ref.abs().max().item())
    rel = fwd_delta / (fwd_max + 1e-12)
    print(
        f"  fwd max |Δ| = {fwd_delta:.4e}  fwd |y|_max = {fwd_max:.4e}  "
        f"rel = {rel:.4f}"
    )
    # bf16-vs-bf16 with flash-attn on both sides: should be tight (<1% rel).
    assert rel < 0.02, f"forward diverges: rel={rel:.4f}"

    print("\n=== gradient parity (LoRA A/B for all 7 targets) ===")
    max_rel = 0.0
    for tgt in _LORA_TARGETS:
        for ab in ("a", "b"):
            gkey = f"g_{tgt[2:]}_lora_{ab}"
            ft = grads[gkey].float()
            ref = ref_grads[f"{tgt}_lora_{ab}"].float()
            d = (ft - ref).abs().max().item()
            m = ref.abs().max().item()
            rel = d / (m + 1e-12)
            max_rel = max(max_rel, rel)
            print(
                f"  {gkey:<16s}: max |Δ|={d:.4e}  |ref|max={m:.4e}  rel={rel:.4f}"
            )
            assert d < 0.01 or rel < 0.10, (
                f"{gkey} diverges: |Δ|={d:.4e} rel={rel:.4f}"
            )

    print("\n=== dL/dx parity ===")
    dx_delta = (dx_ft.float() - dx_ref.float()).abs().max().item()
    print(f"  dL/dx max |Δ| = {dx_delta:.4e}")
    assert dx_delta < 0.05, f"dx diverges: {dx_delta}"

    print("\n✓ LoRAWrapperLayer math parity PASSED")


if __name__ == "__main__":
    main()
