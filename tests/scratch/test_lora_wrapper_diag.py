"""Per-tensor divergence probe between FT LlamaBlock and naive PyTorch ref.

Walks through the layer step-by-step and reports max|Δ| at each
intermediate (after attn_norm, after Q proj, after RoPE, after attention,
after o proj + residual, after ffn_norm, after FFN, final). Identifies
where the bf16 noise is dominating.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import _rmsnorm, _rope_pair_interleave, DTYPE
from flextrain.core.activation_schema import ActivationSlot
from flextrain.core.layer import ChunkMeta, LayerContext
from flextrain.engine.buffers import ScratchPool
from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig

DEVICE = "cuda:0"


def _delta(name, a, b):
    a32 = a.float()
    b32 = b.float()
    d = (a32 - b32).abs().max().item()
    m = max(a32.abs().max().item(), b32.abs().max().item())
    rel = d / (m + 1e-12)
    print(f"  {name:<30s}: max|Δ|={d:.4e}  |max|={m:.4e}  rel={rel:.4f}")


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
    weights: dict[str, torch.Tensor] = {}
    for t in base.param_spec.tensors:
        shape = t.shape(dims)
        if "norm" in t.name:
            w = torch.ones(*shape, dtype=DTYPE, device=DEVICE) + 0.01 * torch.randn(*shape, dtype=DTYPE, device=DEVICE)
        else:
            w = torch.randn(*shape, dtype=DTYPE, device=DEVICE) * 0.02
        weights[t.name] = w

    x = torch.randn(T, cfg.d_model, dtype=DTYPE, device=DEVICE)
    seq_positions = torch.arange(T, dtype=torch.int32, device=DEVICE)

    # FT fwd via base block (no LoRA).
    schema = base.schema
    slot_tensors = {f.name: torch.empty(f.shape_fn(T, dims), dtype=f.dtype, device=DEVICE) for f in schema.fields}
    slot = ActivationSlot(schema=schema, level=schema.max_tier, tensors=slot_tensors)

    class _MiniKV:
        def __init__(self):
            self.k = torch.zeros(T, cfg.n_kv_heads, cfg.head_dim, dtype=DTYPE, device=DEVICE)
            self.v = torch.zeros(T, cfg.n_kv_heads, cfg.head_dim, dtype=DTYPE, device=DEVICE)
            self.dk = torch.zeros(T, cfg.n_kv_heads, cfg.head_dim, dtype=DTYPE, device=DEVICE)
            self.dv = torch.zeros(T, cfg.n_kv_heads, cfg.head_dim, dtype=DTYPE, device=DEVICE)
    chunk = ChunkMeta.build(
        seq_lens=[T], seq_positions=list(range(T)),
        prior_seq_lens=[0], prior_seq_offsets=[0], device=DEVICE,
    )
    ctx = LayerContext(
        scratch=ScratchPool(device=DEVICE), kv_cache=_MiniKV(),
        stream=torch.cuda.current_stream(), secondary_stream=None,
        total_tokens_per_step=T,
    )

    y_ft = base.forward(x, chunk, weights, slot, ctx)

    # ---- Reference, intercepting at each step ----
    bf = DTYPE
    h_ref = _rmsnorm(x, weights["w_attn_norm"], cfg.rms_norm_eps)
    print("=== After attn_norm ===")
    # FT slot doesn't directly store attn_norm_output; recompute from rstd.
    h_ft = (x.float() * slot.attn_norm_rstd).to(bf) * weights["w_attn_norm"]
    _delta("h (attn_norm output)", h_ft, h_ref)

    xq_ref = (h_ref @ weights["w_q"]).view(-1, cfg.n_heads, cfg.head_dim)
    xk_ref = (h_ref @ weights["w_k"]).view(-1, cfg.n_kv_heads, cfg.head_dim)
    xv_ref = (h_ref @ weights["w_v"]).view(-1, cfg.n_kv_heads, cfg.head_dim)
    print("=== After QKV proj (pre-RoPE) ===")
    # FT's xq is post-RoPE in slot.xq. We don't have pre-RoPE saved. Use
    # the kv cache's xv (V doesn't go through RoPE).
    _delta("xv (pre-RoPE)", slot.xv, xv_ref)
    # For Q/K, compare post-RoPE.
    rope_q = _rope_pair_interleave(xq_ref, seq_positions, cfg.rope_base)
    rope_k = _rope_pair_interleave(xk_ref, seq_positions, cfg.rope_base)
    print("=== After RoPE ===")
    _delta("xq (post-RoPE)", slot.xq, rope_q)
    _delta("xk (post-RoPE)", slot.xk, rope_k)

    # Attention output (FT uses flash-attn, ref uses SDPA which under
    # eligible conditions also dispatches to flash-attn).
    q_b = rope_q.transpose(0, 1).unsqueeze(0)
    k_b = rope_k.transpose(0, 1).unsqueeze(0)
    v_b = xv_ref.transpose(0, 1).unsqueeze(0)
    attn_ref = F.scaled_dot_product_attention(
        q_b, k_b, v_b, is_causal=True,
        enable_gqa=(cfg.n_kv_heads != cfg.n_heads),
    ).squeeze(0).transpose(0, 1).contiguous()
    print("=== After attention ===")
    _delta("attn_result", slot.attn_result.view(T, cfg.n_heads, cfg.head_dim), attn_ref)

    # O proj + residual.
    attn_flat = attn_ref.reshape(T, -1)
    x_after_attn_ref = x + attn_flat @ weights["w_o"]
    print("=== After O proj + residual ===")
    _delta("x_after_attn", slot.xo.view(T, cfg.d_model), x_after_attn_ref)

    # FFN.
    h2_ref = _rmsnorm(x_after_attn_ref, weights["w_ffn_norm"], cfg.rms_norm_eps)
    print("=== After ffn_norm ===")
    h2_ft = (x_after_attn_ref.float() * slot.ffn_norm_rstd).to(bf) * weights["w_ffn_norm"]
    _delta("h2 (ffn_norm output)", h2_ft, h2_ref)

    x1_ref = h2_ref @ weights["w_1"]
    x3_ref = h2_ref @ weights["w_3"]
    print("=== After FFN gate/up ===")
    _delta("x1", slot.x1, x1_ref)
    _delta("x3", slot.x3, x3_ref)
    swiglu_ref = (F.silu(x1_ref.float()) * x3_ref.float()).to(bf)
    layer_out_ref = x_after_attn_ref + swiglu_ref @ weights["w_2"]
    print("=== Final ===")
    _delta("layer_out", y_ft, layer_out_ref)


if __name__ == "__main__":
    main()
