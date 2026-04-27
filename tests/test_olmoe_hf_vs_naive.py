"""Compare HF OLMoE vs my naive impl on a tiny prompt. Track layer-by-layer
hidden state divergence so we can pin down where the naive implementation
disagrees with HF.

Runs in two phases:
1. HF forward — subprocess, dumps per-layer hidden states to pickle.
2. Naive forward — loads HF weights directly, dumps per-layer hidden states.
Then compares.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _hf_worker(hf_path, tokens_pkl, out_pkl):
    """Runs HF OLMoE, records per-layer hidden states."""
    from transformers import AutoModelForCausalLM
    with open(tokens_pkl, "rb") as f:
        tokens = pickle.load(f).to(DEVICE).unsqueeze(0)
    model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="sdpa", output_hidden_states=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(input_ids=tokens, output_hidden_states=True)
    hs = [h[0].cpu() for h in out.hidden_states]  # list of (T, D), incl. embed
    logits = out.logits[0].cpu()
    with open(out_pkl, "wb") as f:
        pickle.dump({"hidden_states": hs, "logits": logits}, f)


def _rms_norm(x, w, eps):
    x_f = x.float()
    rms = (x_f * x_f).mean(dim=-1, keepdim=True).add_(eps).rsqrt_()
    return (x_f * rms).to(x.dtype) * w


def _rope(x, positions, base=10_000.0):
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


@torch.no_grad()
def _naive_forward(hf_path, tokens, record_layers=True):
    with open(os.path.join(hf_path, "config.json")) as f:
        cfg = json.load(f)
    from safetensors import safe_open
    idx_path = os.path.join(hf_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard_keys = {}
    for name, shard in weight_map.items():
        shard_keys.setdefault(shard, []).append(name)
    sdict = {}
    for shard, names in shard_keys.items():
        with safe_open(os.path.join(hf_path, shard), framework="pt", device="cpu") as f:
            for name in names:
                t = f.get_tensor(name)
                # Keep norm weights in their native dtype (usually fp32),
                # cast everything else to DTYPE. Matches HF's behavior where
                # weights are cast per-layer at construction time.
                if "norm" in name or "layernorm" in name:
                    sdict[name] = t
                else:
                    sdict[name] = t.to(DTYPE)
    sdict = {k: v.to(DEVICE) for k, v in sdict.items()}
    d_model = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = d_model // n_heads
    E = cfg["num_experts"]
    K = cfg["num_experts_per_tok"]
    eps = cfg.get("rms_norm_eps", 1e-5)
    rope_base = cfg.get("rope_theta", 10_000.0)
    norm_topk = cfg.get("norm_topk_prob", False)

    tokens = tokens.to(DEVICE)
    T = tokens.shape[0]
    x = sdict["model.embed_tokens.weight"][tokens].to(DTYPE)
    positions = torch.arange(T, device=DEVICE)
    hs = [x.cpu()]
    for L in range(cfg["num_hidden_layers"]):
        w_attn_norm = sdict[f"model.layers.{L}.input_layernorm.weight"]
        h = _rms_norm(x, w_attn_norm, eps)
        wq = sdict[f"model.layers.{L}.self_attn.q_proj.weight"]
        wk = sdict[f"model.layers.{L}.self_attn.k_proj.weight"]
        wv = sdict[f"model.layers.{L}.self_attn.v_proj.weight"]
        wo = sdict[f"model.layers.{L}.self_attn.o_proj.weight"]
        xq = h @ wq.T  # (T, D)
        xk = h @ wk.T  # (T, n_kv*head_dim)
        xv = h @ wv.T
        # OLMoE: RMSNorm on Q and K (full-dim, not per-head) before RoPE.
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
        # Match HF SDPA: bf16 Q @ K^T with implicit fp32 accum, bf16 softmax.
        q_ = xq.transpose(0, 1).unsqueeze(0)  # (1, H, T, D)
        k_ = xk.transpose(0, 1).unsqueeze(0)
        v_ = xv.transpose(0, 1).unsqueeze(0)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, is_causal=True,
        ).squeeze(0).transpose(0, 1).contiguous().reshape(T, -1)
        x = x + attn @ wo.T

        w_ffn_norm = sdict[f"model.layers.{L}.post_attention_layernorm.weight"]
        h2 = _rms_norm(x, w_ffn_norm, eps)
        w_router = sdict[f"model.layers.{L}.mlp.gate.weight"]
        logits = h2 @ w_router.T
        probs_e = torch.softmax(logits.float(), dim=-1)
        topk_w, topk_ids = torch.topk(probs_e, K, dim=-1)
        if norm_topk:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        topk_w = topk_w.to(x.dtype)
        out = torch.zeros_like(x)
        for e in range(E):
            mask_e = (topk_ids == e).any(dim=-1)
            if not mask_e.any():
                continue
            token_ids = mask_e.nonzero(as_tuple=False).squeeze(-1)
            selected = (topk_ids[token_ids] == e)
            gate_w = sdict[f"model.layers.{L}.mlp.experts.{e}.gate_proj.weight"]
            up_w = sdict[f"model.layers.{L}.mlp.experts.{e}.up_proj.weight"]
            down_w = sdict[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight"]
            h_e = h2[token_ids]
            g = h_e @ gate_w.T
            u = h_e @ up_w.T
            act = torch.nn.functional.silu(g.float()).to(x.dtype) * u
            y = act @ down_w.T
            scale_w = (selected.to(topk_w.dtype) * topk_w[token_ids]).sum(dim=-1, keepdim=True)
            out.index_add_(0, token_ids, y * scale_w)
        x = x + out
        hs.append(x.cpu())

    x_norm = _rms_norm(x, sdict["model.norm.weight"], eps)
    out_logits = x_norm @ sdict["lm_head.weight"].T
    return {"hidden_states": hs, "logits": out_logits.cpu()}


def main():
    hf_path = os.path.join(ROOT, "models", "OLMoE-1B-7B")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_path)
    prompt = "The capital of France is"
    tokens = torch.tensor(tok.encode(prompt, add_special_tokens=False), dtype=torch.int64)
    print(f"prompt: {prompt!r}  tokens: {tokens.tolist()}")

    # HF in subprocess.
    with tempfile.TemporaryDirectory() as td:
        tpkl = os.path.join(td, "tokens.pkl")
        hpkl = os.path.join(td, "hf.pkl")
        with open(tpkl, "wb") as f:
            pickle.dump(tokens, f)
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        script = os.path.abspath(__file__)
        subprocess.run(
            [sys.executable, script, "--hf-worker", hf_path, tpkl, hpkl],
            check=True, env=env,
        )
        with open(hpkl, "rb") as f:
            hf_out = pickle.load(f)
        import gc; gc.collect()
        torch.cuda.empty_cache()
        # Naive.
        print("\nRunning naive...")
        naive_out = _naive_forward(hf_path, tokens)

    hf_hs = hf_out["hidden_states"]
    naive_hs = naive_out["hidden_states"]
    assert len(hf_hs) == len(naive_hs), f"layer count differs: {len(hf_hs)} vs {len(naive_hs)}"
    print(f"\n=== per-layer divergence ({len(hf_hs)} hidden states, layer 0 = embed) ===")
    for L, (a, b) in enumerate(zip(hf_hs, naive_hs)):
        d = (a.float() - b.float()).abs().max().item()
        m = (a.float() - b.float()).abs().mean().item()
        print(f"  layer {L:2d}: max|Δ|={d:.4e}  mean|Δ|={m:.4e}  |hf|_max={a.float().abs().max().item():.2f}")

    dlog = (hf_out["logits"].float() - naive_out["logits"].float()).abs()
    print(f"\nlogits: max|Δ|={dlog.max().item():.4f}  mean|Δ|={dlog.mean().item():.4e}")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--hf-worker":
        _hf_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
