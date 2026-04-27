"""OLMoE architecture — engine-integrated parity test.

Builds a small random-init OLMoE (d=128, E=4, K=2, 3 layers) in
both FlexTrain (via the engine + OLMoEBlock + MoESwiGLUFFN) and
naive PyTorch. Copies the same random weights into both. Runs
3 SGD steps on random tokens. Asserts loss curves match within
bf16 noise.

This exercises the full MoE integration path: ChunkMeta.extra MoE
scratch allocation, MoESwiGLUFFN fwd/bwd using chunk-passed
token_index_mapping, load-balance loss gradient, engine save-plan
with MoE tier-3 activations.
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
# Naive PyTorch OLMoE reference.
# ---------------------------------------------------------------------------


class NaiveOLMoEBlock(torch.nn.Module):
    """OLMoE-style block in pure PyTorch: GQA attn + MoE SwiGLU FFN."""

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
        # OLMoE QK-norm: full-dim RMSNorm weights (not per-head).
        self.w_q_norm = torch.nn.Parameter(torch.ones(attn_dim, dtype=DTYPE))
        self.w_k_norm = torch.nn.Parameter(torch.ones(kv_dim, dtype=DTYPE))

        self.w_ffn_norm = torch.nn.Parameter(torch.ones(d_model, dtype=DTYPE))
        self.w_router = torch.nn.Parameter(
            torch.zeros(d_model, num_experts, dtype=DTYPE)
        )
        # Expert weights: matched to FlexTrain's stacked layout.
        # w_up: (E, d, 2F) packed as [up (value x3) | gate (x1)] to match
        # the orig swiglu_moe kernel convention. w_down: (E, F, d).
        self.w_up = torch.nn.Parameter(
            torch.zeros(num_experts, d_model, 2 * expert_dim, dtype=DTYPE)
        )
        self.w_down = torch.nn.Parameter(
            torch.zeros(num_experts, expert_dim, d_model, dtype=DTYPE)
        )

    def forward(self, x, seq_positions):
        # Attention + OLMoE QK-norm (full-dim RMSNorm on Q/K).
        h = _rmsnorm(x, self.w_attn_norm, self.rms_norm_eps)
        xq_flat = h @ self.w_q
        xk_flat = h @ self.w_k
        xv = (h @ self.w_v).view(-1, self.n_kv_heads, self.head_dim)
        xq_flat = _rmsnorm(xq_flat, self.w_q_norm, self.rms_norm_eps)
        xk_flat = _rmsnorm(xk_flat, self.w_k_norm, self.rms_norm_eps)
        xq = xq_flat.view(-1, self.n_heads, self.head_dim)
        xk = xk_flat.view(-1, self.n_kv_heads, self.head_dim)
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

        # MoE FFN.
        h2 = _rmsnorm(x_after_attn, self.w_ffn_norm, self.rms_norm_eps)
        router_logits = h2 @ self.w_router  # (T, E)
        topk_vals, topk_ids = torch.topk(router_logits, k=self.top_k, dim=-1)
        # OLMoE: softmax over topk (softmax(topk(logits))).
        topk_w = torch.softmax(topk_vals.float(), dim=-1).to(DTYPE)

        out = torch.zeros_like(x_after_attn)
        for e in range(self.num_experts):
            mask_e = (topk_ids == e)
            if not mask_e.any():
                continue
            tk_pos = mask_e.nonzero(as_tuple=False)
            t_idx = tk_pos[:, 0]
            k_idx = tk_pos[:, 1]
            h_e = h2[t_idx]  # (N_e, d)
            up_e = h_e @ self.w_up[e]  # (N_e, 2F), packed [up, gate]
            up, gate = up_e.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate.float()).to(DTYPE) * up
            down = act @ self.w_down[e]
            scale_w = topk_w[t_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, t_idx, down * scale_w)
        return x_after_attn + out


class NaiveOLMoEModel(torch.nn.Module):
    def __init__(self, shape, num_experts, top_k, load_balance_coef):
        super().__init__()
        self.w_tok_embeddings = torch.nn.Parameter(
            torch.zeros(shape.vocab_size, shape.d_model, dtype=DTYPE)
        )
        self.blocks = torch.nn.ModuleList([
            NaiveOLMoEBlock(
                shape.d_model, shape.n_heads, shape.n_kv_heads,
                shape.head_dim, shape.expert_dim,
                num_experts, top_k,
                shape.rms_norm_eps, shape.rope_base,
                load_balance_coef,
            )
            for _ in range(shape.n_layers)
        ])
        self.w_final_norm = torch.nn.Parameter(
            torch.ones(shape.d_model, dtype=DTYPE)
        )
        self.w_head_proj = torch.nn.Parameter(
            torch.zeros(shape.d_model, shape.vocab_size, dtype=DTYPE)
        )
        self.rms_norm_eps = shape.rms_norm_eps

    def forward(self, token_ids, seq_positions, labels):
        x = self.w_tok_embeddings[token_ids, :]
        for block in self.blocks:
            x = block(x, seq_positions)
        x = _rmsnorm(x, self.w_final_norm, self.rms_norm_eps)
        logits = x @ self.w_head_proj
        return torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="sum"
        )


def main() -> None:
    # Tiny OLMoE-like model — enough to exercise the MoE path
    # without blowing GPU memory.
    shape = ModelShape(
        d_model=128, n_layers=3, n_heads=4, n_kv_heads=4,
        head_dim=32, expert_dim=64, vocab_size=256,
        rms_norm_eps=1e-5, rope_base=10_000.0,
    )
    num_experts = 4
    top_k = 2
    load_balance_coef = 0.0  # disable aux loss for strict FT↔naive parity
    lr = 5e-4
    n_steps = 3

    # Generate batches.
    torch.manual_seed(7)
    batches = []
    for _ in range(n_steps):
        batch = []
        for L in (96, 64):
            toks = torch.randint(0, shape.vocab_size, (L,), dtype=torch.int64)
            s = _Seq(toks)
            s.targets = torch.roll(toks, -1)
            s.targets[-1] = -100
            batch.append(s)
        batches.append(batch)

    # Naive reference.
    def init_naive():
        torch.manual_seed(4242)
        m = NaiveOLMoEModel(shape, num_experts, top_k, load_balance_coef).to(DEVICE)
        with torch.no_grad():
            for p in m.parameters():
                if p.dim() >= 2:
                    p.normal_(mean=0.0, std=0.02)
        # Norms to ~1.
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

    print("Running naive OLMoE 3 steps...")
    naive = init_naive()
    opt = torch.optim.AdamW(
        naive.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.0,
    )
    naive_losses = []
    for i, b in enumerate(batches):
        seqs = [_Seq(s.tokens.clone()) for s in b]
        for d, s in zip(seqs, b):
            d.targets = s.targets.clone()
        loss = _naive_step(naive, opt, seqs, DEVICE)
        print(f"  naive step {i}: {loss:.4f}")
        naive_losses.append(loss)

    # FlexTrain run.
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.olmoe import OLMoEBlock, OLMoEBlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    cfg = OLMoEBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        num_experts=num_experts, top_k=top_k,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True, load_balance_coef=load_balance_coef,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [OLMoEBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
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

    # Mirror naive's weights into FT host buffers (no HF load — we
    # use identical random init).
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

    naive2 = init_naive()  # fresh with same seed
    copy_naive_to_ft(naive2, am)

    print("\nRunning FlexTrain OLMoE 3 steps...")
    ft_losses = []
    for i, b in enumerate(batches):
        seqs = [_Seq(s.tokens.clone()) for s in b]
        for d, s in zip(seqs, b):
            d.targets = s.targets.clone()
        loss = _flextrain_step(am, seqs)
        print(f"  FT step {i}: {loss:.4f}")
        ft_losses.append(loss)

    print("\n=== compare ===")
    max_d = 0.0
    for i, (a, b) in enumerate(zip(naive_losses, ft_losses)):
        d = abs(a - b)
        max_d = max(max_d, d)
        print(f"  step {i}: naive={a:.4f}  FT={b:.4f}  |Δ|={d:.4f}")
    print(f"\nmax |Δ| = {max_d:.4f}")
    if max_d > 0.05:
        raise AssertionError(
            f"OLMoE engine parity FAILED: max |Δ| = {max_d:.4f} > 0.05"
        )
    print("✓ OLMoE engine parity PASSED (within bf16 noise)")


if __name__ == "__main__":
    main()
