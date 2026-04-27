"""Random-init pretraining correctness test.

Models pretraining-from-scratch correctness: builds a small (~100M)
Llama-3.2-style model with **standard pretraining init** (normal(0, 0.02)
for all weights, one for norms), and trains it for a few dozen steps
on real Llama-3 tokenizer tokens from MathInstruct. Asserts:

* FT and naive loss curves match within bf16 noise (parity)
* Loss decreases over the run (convergence)

This is complementary to ``test_llama31_8b_training.py``, which starts
from pretrained HF weights. Here we exercise the cold-start regime
where weights are small-magnitude and gradients have different scale
characteristics — catches init-schema / precision bugs that don't
trigger on pretrained weights.
"""
from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import (
    ModelShape, _Seq, _flextrain_step, _naive_step, DTYPE,
)
from tests.test_llama32_1b_parity import (
    NaiveLlamaModel, _pull_step_batches,
)

DEVICE = "cuda:0"


def _llama_init(module: torch.nn.Module, seed: int = 4242) -> None:
    """Standard Llama init: normal(0, 0.02) for matrices, ones() for norms,
    zeros() for embeddings (to be re-initialized below).

    Matches HF's ``Qwen3PreTrainedModel._init_weights`` semantics
    (also used for Llama): ``module.weight.data.normal_(mean=0.0, std=0.02)``
    for Linear / Embedding; RMSNorm weights initialized to 1.0.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.dim() >= 2:
                p.normal_(mean=0.0, std=0.02)
            elif "norm" in name.lower() or "ln" in name.lower():
                p.fill_(1.0)
            else:
                p.normal_(mean=0.0, std=0.02)


def main():
    shape = ModelShape(
        d_model=512, n_layers=6, n_heads=8, n_kv_heads=2, head_dim=64,
        expert_dim=1024, vocab_size=128256,  # Llama-3 tokenizer vocab
        rms_norm_eps=1e-5, rope_base=500_000.0,
    )
    lr = 3e-4
    n_steps = 20
    target_tokens = 2048

    hf_tokenizer_path = os.path.join(ROOT, "models", "Llama-3.2-1B")
    print(f"Tokenizer path: {hf_tokenizer_path}")
    step_batches = _pull_step_batches(
        hf_tokenizer_path, n_steps=n_steps,
        target_tokens_per_step=target_tokens,
    )
    print(f"  {len(step_batches)} batches ready")

    # --- Naive reference ---
    print("\n=== naive PyTorch run ===")
    naive = NaiveLlamaModel(shape).to(DEVICE)
    _llama_init(naive)
    opt = torch.optim.AdamW(
        naive.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.0,
    )
    t0 = time.time()
    naive_losses = []
    for i, b in enumerate(step_batches):
        loss = _naive_step(naive, opt, b, DEVICE)
        naive_losses.append(loss)
        if i < 5 or i % 5 == 0 or i == n_steps - 1:
            print(f"  naive step {i}: loss={loss:.4f}  elapsed={time.time()-t0:.1f}s")
    del naive, opt
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # --- FlexTrain run ---
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    print("\n=== FlexTrain run ===")
    cfg = LlamaBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [LlamaBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
    embed = TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=shape.vocab_size, d_model=shape.d_model,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    ))
    head = LMHead(LMHeadConfig(
        d_model=shape.d_model, vocab_size=shape.vocab_size,
        rms_norm_eps=shape.rms_norm_eps, head_chunk_size=512,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    ))
    dims = dict(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        attn_dim=shape.n_heads * shape.head_dim,
        kv_dim=shape.n_kv_heads * shape.head_dim,
        expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
    )
    ws = WorkingSetConfig(
        target_round_tokens=2048, max_chunk_size=2048,
        max_training_chunks=4, max_total_round_tokens=2048,
        target_num_rounds=1,
        n_gpu_layers=shape.n_layers, n_gpu_grads=shape.n_layers,
        n_gpu_opt_layers=shape.n_layers,
        gpu_act_buffer_size=int(1 * (1 << 30)),
        host_act_buffer_size=int(2 * (1 << 30)),
        available_gpu_memory_bytes=int(20 * (1 << 30)),
        available_host_memory_bytes=int(16 * (1 << 30)),
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=512, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    opt_ft = AdamW(AdamWHyperparams(
        lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
    ), state_dtype=torch.float32)
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt_ft,
        working_set=ws, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )

    # Identical init: build a second naive with same seed, then copy.
    reference = NaiveLlamaModel(shape).to(DEVICE)
    _llama_init(reference)
    with torch.no_grad():
        am.buffers.host_embed_params["w_tok_embeddings"].copy_(
            reference.w_tok_embeddings.detach().cpu()
        )
        am.buffers.host_head_params["w_final_norm"].copy_(
            reference.w_final_norm.detach().cpu()
        )
        am.buffers.host_head_params["w_head_proj"].copy_(
            reference.w_head_proj.detach().cpu()
        )
        for i in range(shape.n_layers):
            b = reference.blocks[i]
            hp = am.buffers.host_params[i]
            hp["w_attn_norm"].copy_(b.w_attn_norm.detach().cpu())
            hp["w_ffn_norm"].copy_(b.w_ffn_norm.detach().cpu())
            hp["w_q"].copy_(b.w_q.detach().cpu())
            hp["w_k"].copy_(b.w_k.detach().cpu())
            hp["w_v"].copy_(b.w_v.detach().cpu())
            hp["w_o"].copy_(b.w_o.detach().cpu())
            hp["w_1"].copy_(b.w_1.detach().cpu())
            hp["w_2"].copy_(b.w_2.detach().cpu())
            hp["w_3"].copy_(b.w_3.detach().cpu())
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    t0 = time.time()
    ft_losses = []
    for i, b in enumerate(step_batches):
        seqs = [_Seq(s.tokens.clone()) for s in b]
        for d, s in zip(seqs, b):
            d.targets = s.targets.clone()
        loss = _flextrain_step(am, seqs)
        ft_losses.append(loss)
        if i < 5 or i % 5 == 0 or i == n_steps - 1:
            print(f"  FT step {i}: loss={loss:.4f}  elapsed={time.time()-t0:.1f}s")

    # --- Compare ---
    print("\n=== comparison ===")
    max_delta = 0.0
    for i, (nl, ft) in enumerate(zip(naive_losses, ft_losses)):
        d = abs(nl - ft)
        max_delta = max(max_delta, d)
        print(f"  step {i:3d}: naive={nl:.4f}  FT={ft:.4f}  |Δ|={d:.4f}")

    first5 = sum(naive_losses[:5]) / 5
    last5 = sum(naive_losses[-5:]) / 5
    print(f"\n  naive first-5 avg: {first5:.4f}  last-5 avg: {last5:.4f}  Δ: {last5-first5:+.4f}")
    first5 = sum(ft_losses[:5]) / 5
    last5 = sum(ft_losses[-5:]) / 5
    print(f"  FT    first-5 avg: {first5:.4f}  last-5 avg: {last5:.4f}  Δ: {last5-first5:+.4f}")

    assert max_delta < 0.15, (
        f"Naive/FT drift: max |Δ| = {max_delta:.4f} > 0.15"
    )
    assert last5 < first5, (
        f"FT didn't reduce loss: first5={first5:.4f} last5={last5:.4f}"
    )
    print(f"\n✓ Random-init pretraining parity PASSED (max |Δ| = {max_delta:.4f})")


if __name__ == "__main__":
    main()
