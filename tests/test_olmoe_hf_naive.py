"""Sanity: run a naive-PyTorch OLMoE forward with real HF weights and
compare the step-0 loss to HF transformers'. If this matches, the issue
in test_olmoe_1b7b_training is in the FT engine path, not the naive
reference. If it doesn't match, it's in how we interpret HF's tensors.

Runs only one batch, forward-only, CPU-pin weights on-demand to avoid
fitting the whole model on GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import pickle
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.test_llama32_1b_parity import _pull_step_batches  # noqa: E402


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _rms_norm(x, w, eps):
    x_f = x.float()
    rms = (x_f * x_f).mean(dim=-1, keepdim=True).add_(eps).rsqrt_()
    return (x_f * rms).to(x.dtype) * w


def _rope(x, positions, base=10_000.0):
    # Llama-style halved layout.
    T, H, D = x.shape
    half = D // 2
    inv = (1.0 / (base ** (torch.arange(0, half, device=x.device).float() * 2.0 / D)))
    angles = positions.float().unsqueeze(-1) * inv.unsqueeze(0)
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:]
    out1 = x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1)
    out2 = x2 * cos.unsqueeze(1) + x1 * sin.unsqueeze(1)
    return torch.cat([out1, out2], dim=-1)


def _layer_forward(x, positions, sdict, L, cfg):
    """One OLMoE layer. Reads HF weights directly from state_dict slice."""
    d_model = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = d_model // n_heads
    E = cfg["num_experts"]
    K = cfg["num_experts_per_tok"]
    F = cfg["intermediate_size"]
    eps = cfg.get("rms_norm_eps", 1e-5)
    rope_base = cfg.get("rope_theta", 10000.0)

    # Attn norm.
    w_attn_norm = sdict[f"model.layers.{L}.input_layernorm.weight"]
    h = _rms_norm(x, w_attn_norm, eps)

    # QKV.
    wq = sdict[f"model.layers.{L}.self_attn.q_proj.weight"]  # (D, D)
    wk = sdict[f"model.layers.{L}.self_attn.k_proj.weight"]
    wv = sdict[f"model.layers.{L}.self_attn.v_proj.weight"]
    wo = sdict[f"model.layers.{L}.self_attn.o_proj.weight"]
    xq = h @ wq.T
    xk = h @ wk.T
    xv = h @ wv.T
    wq_norm = sdict[f"model.layers.{L}.self_attn.q_norm.weight"]
    wk_norm = sdict[f"model.layers.{L}.self_attn.k_norm.weight"]
    xq = _rms_norm(xq, wq_norm, eps)
    xk = _rms_norm(xk, wk_norm, eps)
    xq = xq.view(-1, n_heads, head_dim)
    xk = xk.view(-1, n_kv, head_dim)
    xv = xv.view(-1, n_kv, head_dim)
    xq = _rope(xq, positions, rope_base)
    xk = _rope(xk, positions, rope_base)
    if n_kv != n_heads:
        rep = n_heads // n_kv
        xk = xk.repeat_interleave(rep, dim=1)
        xv = xv.repeat_interleave(rep, dim=1)
    # Causal attention.
    T = xq.shape[0]
    q_ = xq.transpose(0, 1).float()
    k_ = xk.transpose(0, 1).float()
    v_ = xv.transpose(0, 1).float()
    scale = 1.0 / (head_dim ** 0.5)
    scores = (q_ @ k_.transpose(-2, -1)) * scale
    mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
    probs = torch.softmax(scores + mask, dim=-1)
    attn = (probs @ v_).transpose(0, 1).to(x.dtype).contiguous().reshape(T, -1)
    x_after = x + attn @ wo.T

    # FFN norm.
    w_ffn_norm = sdict[f"model.layers.{L}.post_attention_layernorm.weight"]
    h2 = _rms_norm(x_after, w_ffn_norm, eps)

    # Router.
    w_router = sdict[f"model.layers.{L}.mlp.gate.weight"]  # (E, D)
    logits = h2 @ w_router.T  # (T, E)
    norm_topk = cfg.get("norm_topk_prob", False)
    probs_e = torch.softmax(logits.float(), dim=-1)
    topk_w, topk_ids = torch.topk(probs_e, K, dim=-1)
    if norm_topk:
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    topk_w = topk_w.to(x.dtype)

    # Experts.
    out = torch.zeros_like(x_after)
    for e in range(E):
        mask_e = (topk_ids == e).any(dim=-1)
        if not mask_e.any():
            continue
        token_ids = mask_e.nonzero(as_tuple=False).squeeze(-1)
        # For each selected token, find its k-index
        selected = (topk_ids[token_ids] == e)  # (N_e, K)
        gate_w = sdict[f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight"]  # (F, D)
        up_w = sdict[f"model.layers.{L}.mlp.experts.{e}.up_proj.weight"]
        down_w = sdict[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"]  # (D, F)
        h_e = h2[token_ids]  # (N_e, D)
        g = h_e @ gate_w.T
        u = h_e @ up_w.T
        act = torch.nn.functional.silu(g.float()).to(x.dtype) * u
        y = act @ down_w.T
        # Apply router weight (sum over k-indices where expert = e; since
        # same token may choose expert e in multiple k slots, sum).
        scale_w = (selected.to(topk_w.dtype) * topk_w[token_ids]).sum(dim=-1, keepdim=True)
        out.index_add_(0, token_ids, y * scale_w)
    return x_after + out


@torch.no_grad()
def naive_step0_loss(hf_path, batch):
    with open(os.path.join(hf_path, "config.json")) as f:
        cfg = json.load(f)
    # Load whole state dict via safetensors.
    from safetensors import safe_open
    idx_path = os.path.join(hf_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard_keys = {}
    for name, shard in weight_map.items():
        shard_keys.setdefault(shard, []).append(name)
    print(f"  Loading {len(weight_map)} tensors from {len(shard_keys)} shards...")
    sdict = {}
    for shard, names in shard_keys.items():
        with safe_open(os.path.join(hf_path, shard), framework="pt", device="cpu") as f:
            for name in names:
                sdict[name] = f.get_tensor(name).to(DTYPE)
    print(f"  loaded {len(sdict)} tensors")
    # Only move active tensors to device on demand? Start by moving all.
    # 7B bf16 = 14 GB, fits 22 GB.
    sdict = {k: v.to(DEVICE) for k, v in sdict.items()}

    total_loss = 0.0
    total_active = 0
    for s in batch:
        tokens = s.tokens.to(DEVICE)
        targets = s.targets.to(DEVICE)
        T = tokens.shape[0]
        # Embedding.
        x = sdict["model.embed_tokens.weight"][tokens].to(DTYPE)
        positions = torch.arange(T, device=DEVICE)
        for L in range(cfg["num_hidden_layers"]):
            x = _layer_forward(x, positions, sdict, L, cfg)
        x = _rms_norm(x, sdict["model.norm.weight"], cfg.get("rms_norm_eps", 1e-5))
        logits = x @ sdict["lm_head.weight"].T
        # HF-style labels: labels[1:] = tokens[1:], labels[0] = -100.
        # Loss is on predictions at positions 0..T-2 vs labels 1..T-1.
        # But our ``targets`` already encodes this (torch.roll(tokens, -1)
        # with prompt masked and final set to -100), so at position t
        # the target is targets[t]. Loss at positions 0..T-2 where
        # targets[t] != -100.
        loss_sum = 0.0
        active = 0
        logits_f32 = logits.float()
        for t in range(T - 1):
            tgt = int(targets[t].item())
            if tgt < 0:
                continue
            loss_sum += float(torch.nn.functional.cross_entropy(
                logits_f32[t].unsqueeze(0),
                torch.tensor([tgt], device=DEVICE),
            ).item())
            active += 1
        total_loss += loss_sum
        total_active += active
    avg = total_loss / max(1, total_active)
    return avg


def main():
    hf_path = os.path.join(ROOT, "models", "OLMoE-1B-7B")
    n_steps = 1
    target_tokens_per_step = 4096  # Match test_olmoe_1b7b_training.py
    print(f"Preparing {n_steps} batch...")
    batches = _pull_step_batches(
        hf_path, n_steps=n_steps,
        target_tokens_per_step=target_tokens_per_step,
    )
    print(f"  {len(batches)} batches, batch[0] has {sum(s.tokens.shape[0] for s in batches[0])} tokens")

    t0 = time.time()
    loss = naive_step0_loss(hf_path, batches[0])
    print(f"naive OLMoE step-0 loss: {loss:.4f}  (in {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
