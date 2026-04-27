"""Qwen2 Q/K/V-bias smoke test.

Builds a tiny Qwen2Block with random weights + random biases,
constructs a naive PyTorch equivalent with the same weights, and
checks loss curves match over 3 SGD steps.

This is the algorithmic test for the ``cfg.qkv_bias=True`` path on
:class:`GQAAttentionBlock`. No HF weights needed.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import (  # noqa: E402
    DTYPE, ModelShape, NaiveLlamaModel, NaiveLlamaBlock, _Seq,
    _naive_step, _flextrain_step, _rmsnorm, _rope_pair_interleave,
)

DEVICE = "cuda:0"


class NaiveQwen2Block(NaiveLlamaBlock):
    """LlamaBlock + Q/K/V biases."""

    def __init__(self, d_model, n_heads, n_kv_heads, head_dim,
                 expert_dim, rms_norm_eps, rope_base):
        super().__init__(d_model, n_heads, n_kv_heads, head_dim,
                         expert_dim, rms_norm_eps, rope_base)
        attn_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        self.b_q = torch.nn.Parameter(torch.zeros(attn_dim, dtype=DTYPE))
        self.b_k = torch.nn.Parameter(torch.zeros(kv_dim, dtype=DTYPE))
        self.b_v = torch.nn.Parameter(torch.zeros(kv_dim, dtype=DTYPE))

    def forward(self, x, seq_positions):
        h = _rmsnorm(x, self.w_attn_norm, self.rms_norm_eps)
        xq = (h @ self.w_q + self.b_q).view(-1, self.n_heads, self.head_dim)
        xk = (h @ self.w_k + self.b_k).view(-1, self.n_kv_heads, self.head_dim)
        xv = (h @ self.w_v + self.b_v).view(-1, self.n_kv_heads, self.head_dim)
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
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_).transpose(0, 1).to(x.dtype).contiguous()
        attn_flat = attn_out.reshape(T, -1)
        x_after_attn = x + attn_flat @ self.w_o
        h2 = _rmsnorm(x_after_attn, self.w_ffn_norm, self.rms_norm_eps)
        x1 = h2 @ self.w_1
        x3 = h2 @ self.w_3
        mlp = (torch.nn.functional.silu(x1.float()).to(x1.dtype) * x3) @ self.w_2
        return x_after_attn + mlp


class NaiveQwen2Model(torch.nn.Module):
    """Pure-PyTorch reference for a Qwen2-style model."""

    def __init__(self, shape):
        super().__init__()
        self.w_tok_embeddings = torch.nn.Parameter(
            torch.zeros(shape.vocab_size, shape.d_model, dtype=DTYPE)
        )
        self.blocks = torch.nn.ModuleList([
            NaiveQwen2Block(
                shape.d_model, shape.n_heads, shape.n_kv_heads,
                shape.head_dim, shape.expert_dim,
                shape.rms_norm_eps, shape.rope_base,
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


def _build_ft(shape, lr, n_gpu_layers):
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.qwen2 import Qwen2Block, Qwen2BlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    cfg = Qwen2BlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [Qwen2Block(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
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
        expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
    )
    ws = WorkingSetConfig(
        target_round_tokens=512, max_chunk_size=512,
        max_training_chunks=4, max_total_round_tokens=512,
        target_num_rounds=1,
        n_gpu_layers=n_gpu_layers, n_gpu_grads=n_gpu_layers,
        n_gpu_opt_layers=n_gpu_layers,
        gpu_act_buffer_size=int(1 * (1 << 30)),
        host_act_buffer_size=int(2 * (1 << 30)),
        available_gpu_memory_bytes=int(24 * (1 << 30)),
        available_host_memory_bytes=int(32 * (1 << 30)),
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=512, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    opt = AdamW(
        AdamWHyperparams(lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
        state_dtype=torch.bfloat16,
    )
    return ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=ws, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )


def _copy_naive_to_ft(naive, am, n_layers):
    am.buffers.host_embed_params["w_tok_embeddings"].copy_(
        naive.w_tok_embeddings.detach().cpu()
    )
    am.buffers.host_head_params["w_final_norm"].copy_(
        naive.w_final_norm.detach().cpu()
    )
    am.buffers.host_head_params["w_head_proj"].copy_(
        naive.w_head_proj.detach().cpu()
    )
    for i in range(n_layers):
        b = naive.blocks[i]
        hp = am.buffers.host_params[i]
        hp["w_attn_norm"].copy_(b.w_attn_norm.detach().cpu())
        hp["w_ffn_norm"].copy_(b.w_ffn_norm.detach().cpu())
        hp["w_q"].copy_(b.w_q.detach().cpu())
        hp["w_k"].copy_(b.w_k.detach().cpu())
        hp["w_v"].copy_(b.w_v.detach().cpu())
        hp["w_o"].copy_(b.w_o.detach().cpu())
        hp["b_q"].copy_(b.b_q.detach().cpu())
        hp["b_k"].copy_(b.b_k.detach().cpu())
        hp["b_v"].copy_(b.b_v.detach().cpu())
        hp["w_1"].copy_(b.w_1.detach().cpu())
        hp["w_2"].copy_(b.w_2.detach().cpu())
        hp["w_3"].copy_(b.w_3.detach().cpu())
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


def _batches(n_steps, shape, seq_lens=(96, 128)):
    torch.manual_seed(7)
    out = []
    for _ in range(n_steps):
        batch = []
        for L in seq_lens:
            toks = torch.randint(0, shape.vocab_size, (L,), dtype=torch.int64)
            s = _Seq(toks)
            s.targets = torch.roll(toks, -1)
            s.targets[: L // 4] = -100
            s.targets[-1] = -100
            batch.append(s)
        out.append(batch)
    return out


def test_qwen2_bias_path() -> None:
    shape = ModelShape(
        d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, head_dim=32,
        expert_dim=384, vocab_size=256,
        rms_norm_eps=1e-6, rope_base=1_000_000.0,
    )
    n_steps = 3
    lr = 5e-4
    batches = _batches(n_steps, shape)

    def _init_naive() -> NaiveQwen2Model:
        torch.manual_seed(4242)
        m = NaiveQwen2Model(shape).to(DEVICE)
        with torch.no_grad():
            for p in m.parameters():
                if p.dim() == 2:
                    p.normal_(mean=0.0, std=0.02)
                elif p.dim() == 1:
                    # 1-D: could be either RMSNorm weight or QKV bias.
                    # Initialize both to small random values (bias in
                    # Qwen2 is non-zero in pretrained checkpoints).
                    p.normal_(mean=0.0, std=0.02)
        # Re-init RMSNorm weights to ~1 (they're shape d_model, small).
        for b in m.blocks:
            b.w_attn_norm.data.copy_(
                torch.ones_like(b.w_attn_norm) + 0.01 * torch.randn_like(b.w_attn_norm)
            )
            b.w_ffn_norm.data.copy_(
                torch.ones_like(b.w_ffn_norm) + 0.01 * torch.randn_like(b.w_ffn_norm)
            )
        m.w_final_norm.data.copy_(
            torch.ones_like(m.w_final_norm) + 0.01 * torch.randn_like(m.w_final_norm)
        )
        return m

    print("running naive 3 steps...")
    naive = _init_naive()
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

    print("running FT all-resident 3 steps...")
    naive2 = _init_naive()
    am = _build_ft(shape, lr, n_gpu_layers=shape.n_layers)
    _copy_naive_to_ft(naive2, am, shape.n_layers)
    ft_losses = []
    for i, b in enumerate(batches):
        seqs = [_Seq(s.tokens.clone()) for s in b]
        for d, s in zip(seqs, b):
            d.targets = s.targets.clone()
        loss = _flextrain_step(am, seqs)
        print(f"  FT step {i}: {loss:.4f}")
        ft_losses.append(loss)

    print("\n== compare ==")
    for i, (a, b) in enumerate(zip(naive_losses, ft_losses)):
        print(f"  step {i}: naive={a:.4f}  FT={b:.4f}  |Δ|={abs(a-b):.4f}")
    max_delta = max(abs(a - b) for a, b in zip(naive_losses, ft_losses))
    print(f"\nmax |Δ| = {max_delta:.4f}")
    if max_delta > 0.05:
        raise AssertionError(f"FT diverges from naive: max |Δ|={max_delta:.4f}")
    print("✓ Qwen2 bias path: FT matches naive within bf16 noise")


if __name__ == "__main__":
    test_qwen2_bias_path()
