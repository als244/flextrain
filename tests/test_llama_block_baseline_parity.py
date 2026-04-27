"""Baseline check: plain LlamaBlock vs autograd reference (no LoRA).

Measures the inherent forward divergence between FT's LlamaBlock
(flash-attn + Triton kernels) and a naive PyTorch reference using the
same bf16 weights. Lets us judge whether LoRA-added divergence is
significant or just baseline noise.
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


def _ref(cfg, weights, x, seq_positions):
    bf = x.dtype
    h = _rmsnorm(x, weights["w_attn_norm"], cfg.rms_norm_eps)
    xq = (h @ weights["w_q"]).view(-1, cfg.n_heads, cfg.head_dim)
    xk = (h @ weights["w_k"]).view(-1, cfg.n_kv_heads, cfg.head_dim)
    xv = (h @ weights["w_v"]).view(-1, cfg.n_kv_heads, cfg.head_dim)
    rope_q = _rope_pair_interleave(xq, seq_positions, cfg.rope_base)
    rope_k = _rope_pair_interleave(xk, seq_positions, cfg.rope_base)
    if cfg.n_kv_heads != cfg.n_heads:
        rep = cfg.n_heads // cfg.n_kv_heads
        rope_k = rope_k.repeat_interleave(rep, dim=1)
        xv = xv.repeat_interleave(rep, dim=1)
    T = rope_q.shape[0]
    q_ = rope_q.transpose(0, 1).float()
    k_ = rope_k.transpose(0, 1).float()
    v_ = xv.transpose(0, 1).float()
    scale = 1.0 / (cfg.head_dim ** 0.5)
    scores = q_ @ k_.transpose(-2, -1) * scale
    mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
    probs = torch.softmax(scores + mask, dim=-1)
    attn_out = (probs @ v_).transpose(0, 1).to(bf).contiguous().reshape(T, -1)
    x_after = x + attn_out @ weights["w_o"]
    h2 = _rmsnorm(x_after, weights["w_ffn_norm"], cfg.rms_norm_eps)
    x1 = h2 @ weights["w_1"]
    x3 = h2 @ weights["w_3"]
    swiglu = (F.silu(x1.float()) * x3.float()).to(bf)
    return x_after + swiglu @ weights["w_2"]


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
    y_ref = _ref(cfg, weights, x, seq_positions)
    d = (y_ft.float() - y_ref.float()).abs().max().item()
    m = max(y_ft.abs().max().item(), y_ref.abs().max().item())
    print(f"baseline (no LoRA) fwd: max |Δ| = {d:.4e}  |y|_max = {m:.4e}  rel = {d/m:.4f}")


if __name__ == "__main__":
    main()
