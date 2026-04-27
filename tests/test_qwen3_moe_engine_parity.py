"""Qwen3-MoE architecture — engine-integrated parity test.

Small random-init Qwen3-MoE (d=128, head_dim=32, E=4, K=2, 3 layers) in
both FlexTrain (via the engine + Qwen3MoEBlock + MoESwiGLUFFN) and
naive PyTorch. Same random weights in both. 3 SGD steps. Asserts loss
curves match within bf16 noise.

Exercises: per-head QK-norm (Qwen3 style), topk_then_softmax routing
(Qwen3-MoE default), MoE FFN + engine chunk-scoped MoE scratch.
"""
from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import (  # noqa: E402
    ModelShape, _Seq, _flextrain_step, _naive_step,
    _rmsnorm, _rope_pair_interleave, DTYPE,
)

DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Naive PyTorch Qwen3-MoE reference.
# ---------------------------------------------------------------------------


class NaiveQwen3MoEBlock(torch.nn.Module):
    """Qwen3-MoE-style block in pure PyTorch: GQA + per-head QK-norm + MoE."""

    def __init__(self, d_model, n_heads, n_kv_heads, head_dim,
                 expert_dim, num_experts, top_k,
                 rms_norm_eps, rope_base,
                 load_balance_coef):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.load_balance_coef = load_balance_coef

        self.w_attn_norm = torch.nn.Parameter(torch.ones(d_model, dtype=DTYPE))
        attn_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        self.w_q = torch.nn.Parameter(torch.zeros(d_model, attn_dim, dtype=DTYPE))
        self.w_k = torch.nn.Parameter(torch.zeros(d_model, kv_dim, dtype=DTYPE))
        self.w_v = torch.nn.Parameter(torch.zeros(d_model, kv_dim, dtype=DTYPE))
        self.w_o = torch.nn.Parameter(torch.zeros(attn_dim, d_model, dtype=DTYPE))
        # Qwen3 per-head QK-norm weights: (head_dim,) each, broadcast across heads.
        self.w_q_norm = torch.nn.Parameter(torch.ones(head_dim, dtype=DTYPE))
        self.w_k_norm = torch.nn.Parameter(torch.ones(head_dim, dtype=DTYPE))

        self.w_ffn_norm = torch.nn.Parameter(torch.ones(d_model, dtype=DTYPE))
        self.w_router = torch.nn.Parameter(
            torch.zeros(d_model, num_experts, dtype=DTYPE)
        )
        # Expert weights: matched to FT's stacked layout, packed as [up, gate].
        self.w_up = torch.nn.Parameter(
            torch.zeros(num_experts, d_model, 2 * expert_dim, dtype=DTYPE)
        )
        self.w_down = torch.nn.Parameter(
            torch.zeros(num_experts, expert_dim, d_model, dtype=DTYPE)
        )

    def _rmsnorm_head(self, x, w, eps):
        """Per-head RMSNorm: x shape (T, H, head_dim), w shape (head_dim,)."""
        x_f = x.float()
        rms = (x_f * x_f).mean(dim=-1, keepdim=True).add_(eps).rsqrt_()
        return (x_f * rms).to(x.dtype) * w

    def forward(self, x, seq_positions):
        h = _rmsnorm(x, self.w_attn_norm, self.rms_norm_eps)
        xq = (h @ self.w_q).view(-1, self.n_heads, self.head_dim)
        xk = (h @ self.w_k).view(-1, self.n_kv_heads, self.head_dim)
        xv = (h @ self.w_v).view(-1, self.n_kv_heads, self.head_dim)
        xq = self._rmsnorm_head(xq, self.w_q_norm, self.rms_norm_eps)
        xk = self._rmsnorm_head(xk, self.w_k_norm, self.rms_norm_eps)
        rope_q = _rope_pair_interleave(xq, seq_positions, self.rope_base)
        rope_k = _rope_pair_interleave(xk, seq_positions, self.rope_base)
        T, H, D = rope_q.shape
        H_kv = rope_k.shape[1]
        if H_kv != H:
            rep = H // H_kv
            rope_k = rope_k.repeat_interleave(rep, dim=1)
            xv = xv.repeat_interleave(rep, dim=1)
        q_ = rope_q.transpose(0, 1).float()
        k_ = rope_k.transpose(0, 1).float()
        v_ = xv.transpose(0, 1).float()
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_).transpose(0, 1).to(x.dtype).contiguous()
        attn_flat = attn_out.reshape(T, -1)
        x_after_attn = x + attn_flat @ self.w_o

        # MoE FFN — Qwen3 topk_then_softmax (norm_topk_prob=True): take top-K
        # raw logits, softmax over the K selected, weights sum to 1.
        h2 = _rmsnorm(x_after_attn, self.w_ffn_norm, self.rms_norm_eps)
        router_logits = h2 @ self.w_router
        topk_vals, topk_ids = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_w = torch.softmax(topk_vals.float(), dim=-1).to(DTYPE)

        out = torch.zeros_like(x_after_attn)
        for e in range(self.num_experts):
            mask_e = (topk_ids == e)
            if not mask_e.any():
                continue
            tk_pos = mask_e.nonzero(as_tuple=False)
            t_idx = tk_pos[:, 0]
            k_idx = tk_pos[:, 1]
            h_e = h2[t_idx]
            up_e = h_e @ self.w_up[e]  # (N_e, 2F), packed [up, gate]
            up, gate = up_e.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate.float()).to(DTYPE) * up
            down = act @ self.w_down[e]
            scale_w = topk_w[t_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, t_idx, down * scale_w)
        return x_after_attn + out


class NaiveQwen3MoEModel(torch.nn.Module):
    def __init__(self, shape, num_experts, top_k, load_balance_coef):
        super().__init__()
        self.w_tok_embeddings = torch.nn.Parameter(
            torch.zeros(shape.vocab_size, shape.d_model, dtype=DTYPE)
        )
        self.blocks = torch.nn.ModuleList([
            NaiveQwen3MoEBlock(
                shape.d_model, shape.n_heads, shape.n_kv_heads,
                shape.head_dim, shape.expert_dim,
                num_experts, top_k,
                shape.rms_norm_eps, shape.rope_base,
                load_balance_coef,
            )
            for _ in range(shape.n_layers)
        ])
        self.w_final_norm = torch.nn.Parameter(torch.ones(shape.d_model, dtype=DTYPE))
        self.w_head_proj = torch.nn.Parameter(
            torch.zeros(shape.d_model, shape.vocab_size, dtype=DTYPE)
        )

    def forward(self, token_ids, seq_positions, labels):
        x = self.w_tok_embeddings[token_ids, :]
        for b in self.blocks:
            x = b(x, seq_positions)
        x = _rmsnorm(x, self.w_final_norm, 1e-6)
        logits = x @ self.w_head_proj
        return torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="sum"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(4242)
    shape = ModelShape(
        d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, head_dim=32,
        expert_dim=64, vocab_size=256, rms_norm_eps=1e-6, rope_base=1_000_000.0,
    )
    num_experts = 4
    top_k = 2
    load_balance_coef = 0.001
    lr = 3e-4

    # Data: 2 seqs of len 32 each per step.
    n_steps = 3
    tokens_per_step = 64
    batches = []
    for _ in range(n_steps):
        batch = []
        for _ in range(2):
            tokens = torch.randint(0, shape.vocab_size, (32,), dtype=torch.int64)
            targets = torch.roll(tokens, -1)
            targets[-1] = -100
            s = _Seq(tokens)
            s.targets = targets
            batch.append(s)
        batches.append(batch)

    def init_naive():
        torch.manual_seed(4242)
        m = NaiveQwen3MoEModel(shape, num_experts, top_k, load_balance_coef).to(DEVICE)
        with torch.no_grad():
            for p in m.parameters():
                if p.dim() >= 2:
                    p.normal_(mean=0.0, std=0.02)
        for b in m.blocks:
            b.w_attn_norm.data.copy_(
                torch.ones_like(b.w_attn_norm) + 0.01 * torch.randn_like(b.w_attn_norm)
            )
            b.w_ffn_norm.data.copy_(
                torch.ones_like(b.w_ffn_norm) + 0.01 * torch.randn_like(b.w_ffn_norm)
            )
            b.w_q_norm.data.copy_(
                torch.ones_like(b.w_q_norm) + 0.01 * torch.randn_like(b.w_q_norm)
            )
            b.w_k_norm.data.copy_(
                torch.ones_like(b.w_k_norm) + 0.01 * torch.randn_like(b.w_k_norm)
            )
        m.w_final_norm.data.copy_(
            torch.ones_like(m.w_final_norm) + 0.01 * torch.randn_like(m.w_final_norm)
        )
        return m

    print("Running naive Qwen3-MoE 3 steps...")
    naive = init_naive()
    opt = torch.optim.AdamW(
        naive.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
    )
    naive_losses = []
    for i, b in enumerate(batches):
        loss = _naive_step(naive, opt, b, DEVICE)
        print(f"  naive step {i}: {loss:.4f}")
        naive_losses.append(loss)

    # FlexTrain run.
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.qwen3_moe import Qwen3MoEBlock, Qwen3MoEBlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    cfg = Qwen3MoEBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        num_experts=num_experts, top_k=top_k,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True, load_balance_coef=load_balance_coef,
        routing_mode="topk_then_softmax",
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [Qwen3MoEBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
    embed = TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=shape.vocab_size, d_model=shape.d_model,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    ))
    head = LMHead(LMHeadConfig(
        d_model=shape.d_model, vocab_size=shape.vocab_size,
        rms_norm_eps=shape.rms_norm_eps, head_chunk_size=64,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    ))
    dims = dict(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        attn_dim=shape.n_heads * shape.head_dim,
        kv_dim=shape.n_kv_heads * shape.head_dim,
        expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
        num_experts=num_experts, top_k=top_k,
    )
    ws = WorkingSetConfig(
        target_round_tokens=256, max_chunk_size=256,
        max_training_chunks=4, max_total_round_tokens=256,
        target_num_rounds=1,
        n_gpu_layers=shape.n_layers, n_gpu_grads=shape.n_layers,
        n_gpu_opt_layers=shape.n_layers,
        gpu_act_buffer_size=int(0.5 * (1 << 30)),
        host_act_buffer_size=int(1 * (1 << 30)),
        available_gpu_memory_bytes=int(4 * (1 << 30)),
        available_host_memory_bytes=int(8 * (1 << 30)),
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=256, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    opt_ft = AdamW(AdamWHyperparams(
        lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
    ), state_dtype=torch.bfloat16)
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt_ft,
        working_set=ws, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )

    def copy_naive_to_ft(naive, am):
        with torch.no_grad():
            am.buffers.host_embed_params["w_tok_embeddings"].copy_(
                naive.w_tok_embeddings.detach().cpu()
            )
            am.buffers.host_head_params["w_final_norm"].copy_(
                naive.w_final_norm.detach().cpu()
            )
            am.buffers.host_head_params["w_head_proj"].copy_(
                naive.w_head_proj.detach().cpu()
            )
            for i in range(shape.n_layers):
                b = naive.blocks[i]
                hp = am.buffers.host_params[i]
                hp["w_attn_norm"].copy_(b.w_attn_norm.detach().cpu())
                hp["w_ffn_norm"].copy_(b.w_ffn_norm.detach().cpu())
                hp["w_q"].copy_(b.w_q.detach().cpu())
                hp["w_k"].copy_(b.w_k.detach().cpu())
                hp["w_v"].copy_(b.w_v.detach().cpu())
                hp["w_o"].copy_(b.w_o.detach().cpu())
                hp["w_q_norm"].copy_(b.w_q_norm.detach().cpu())
                hp["w_k_norm"].copy_(b.w_k_norm.detach().cpu())
                hp["w_router"].copy_(b.w_router.detach().cpu())
                hp["w_up"].copy_(b.w_up.detach().cpu())
                hp["w_down"].copy_(b.w_down.detach().cpu())
        am._refresh_gpu_residents()
        for name, dev_t in am.buffers.gpu_head_params.items():
            dev_t.copy_(am.buffers.host_head_params[name])
        torch.cuda.synchronize()

    naive2 = init_naive()
    copy_naive_to_ft(naive2, am)

    print("\nRunning FlexTrain Qwen3-MoE 3 steps...")
    ft_losses = []
    for i, b in enumerate(batches):
        seqs = [_Seq(s.tokens.clone()) for s in b]
        for d, s in zip(seqs, b):
            d.targets = s.targets.clone()
        loss = _flextrain_step(am, seqs)
        print(f"  FT step {i}: {loss:.4f}")
        ft_losses.append(loss)

    print("\n=== compare ===")
    max_delta = 0.0
    for i, (nl, ft) in enumerate(zip(naive_losses, ft_losses)):
        d = abs(nl - ft)
        max_delta = max(max_delta, d)
        print(f"  step {i}: naive={nl:.4f}  FT={ft:.4f}  |Δ|={d:.4f}")
    print(f"\nmax |Δ| = {max_delta:.4f}")
    if max_delta > 0.05:
        raise AssertionError(
            f"Qwen3-MoE engine parity failed: max |Δ| = {max_delta:.4f} > 0.05"
        )
    print("✓ Qwen3-MoE engine parity PASSED (within bf16 noise)")


if __name__ == "__main__":
    main()
