"""End-to-end Muon + offloading pretraining correctness for MoE (OLMoE).

Same pattern as :mod:`test_muon_offloading_pretraining` but with an
OLMoE-style model: per-expert Muon on the stacked expert matrices
(iterated expert dim internally), AdamW on router / norms / embed /
head. Asserts resident vs offloaded curves are bit-identical.
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
    ModelShape, _Seq, _flextrain_step, DTYPE,
)
from tests.test_llama32_1b_parity import _pull_step_batches

DEVICE = "cuda:0"


def _random_init(am, seed: int = 4242) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, t in am.buffers.host_embed_params.items():
            t.normal_(mean=0.0, std=0.02)
        for name, t in am.buffers.host_head_params.items():
            if "norm" in name:
                t.fill_(1.0).add_(0.01 * torch.randn_like(t))
            else:
                t.normal_(mean=0.0, std=0.02)
        for layer in am.buffers.host_params:
            for name, t in layer.items():
                if "norm" in name:
                    t.fill_(1.0).add_(0.01 * torch.randn_like(t))
                else:
                    t.normal_(mean=0.0, std=0.02)
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


def _build_olmoe_engine(
    shape: ModelShape, lr: float, num_experts: int, top_k: int,
    n_gpu_layers: int, target_round_tokens: int,
):
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.olmoe import OLMoEBlock, OLMoEBlockConfig
    from flextrain.optim.hybrid import (
        HybridMuonAdamW, HybridMuonAdamWHyperparams,
    )
    from flextrain.optim.adamw import AdamWHyperparams
    from flextrain.optim.muon import MuonHyperparams

    cfg = OLMoEBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        num_experts=num_experts, top_k=top_k,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True, load_balance_coef=0.01,
        routing_mode="softmax_then_topk",
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
        num_experts=num_experts, top_k=top_k,
    )
    ws = WorkingSetConfig(
        target_round_tokens=target_round_tokens,
        max_chunk_size=target_round_tokens,
        max_training_chunks=4,
        max_total_round_tokens=target_round_tokens,
        target_num_rounds=1,
        n_gpu_layers=n_gpu_layers,
        n_gpu_grads=n_gpu_layers,
        n_gpu_opt_layers=max(1, n_gpu_layers // 2),
        gpu_act_buffer_size=int(1 * (1 << 30)),
        host_act_buffer_size=int(2 * (1 << 30)),
        available_gpu_memory_bytes=int(20 * (1 << 30)),
        available_host_memory_bytes=int(20 * (1 << 30)),
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=512, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    opt = HybridMuonAdamW(HybridMuonAdamWHyperparams(
        lr=lr,
        adamw=AdamWHyperparams(lr=lr, beta1=0.9, beta2=0.95, eps=1e-8),
        muon=MuonHyperparams(lr=lr, beta=0.95, ns_iters=5),
    ))
    return ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=ws, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )


def _cleanup(am):
    am.buffers.destroy()
    del am
    try:
        from flextrain.engine import unregister_all_process_pinned_memory
        unregister_all_process_pinned_memory()
    except Exception:
        pass
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _train(am, step_batches) -> list[float]:
    curve = []
    t0 = time.time()
    for i, batch in enumerate(step_batches):
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        loss = _flextrain_step(am, seqs)
        curve.append(loss)
        if i < 3 or i % 5 == 0 or i == len(step_batches) - 1:
            max_alloc = torch.cuda.max_memory_allocated() / (1 << 30)
            print(
                f"    step {i:3d}: loss={loss:.4f}  "
                f"elapsed={time.time()-t0:.1f}s  max_alloc={max_alloc:.2f}GiB"
            )
    return curve


def main():
    shape = ModelShape(
        d_model=512, n_layers=6, n_heads=8, n_kv_heads=8, head_dim=64,
        expert_dim=512, vocab_size=128256,
        rms_norm_eps=1e-5, rope_base=10_000.0,
    )
    num_experts = 8
    top_k = 2
    lr = 3e-4
    n_steps = 12
    target_tokens = 2048

    hf_tokenizer = os.path.join(ROOT, "models", "Llama-3.2-1B")
    print(f"Preparing {n_steps} batches of ~{target_tokens} tokens...")
    step_batches = _pull_step_batches(
        hf_tokenizer, n_steps=n_steps, target_tokens_per_step=target_tokens,
    )
    print(f"  {len(step_batches)} batches ready")
    total_params = (
        2 * shape.vocab_size * shape.d_model +  # embed + head
        shape.n_layers * (
            4 * shape.d_model * shape.d_model +
            2 * num_experts * shape.d_model * 2 * shape.expert_dim +
            num_experts * shape.expert_dim * shape.d_model +
            shape.d_model * num_experts
        )
    )
    print(
        f"  ~{total_params / 1e6:.0f}M params "
        f"({num_experts}E × top-{top_k}, d={shape.d_model}, "
        f"F={shape.expert_dim})"
    )

    print("\n=== A. all-resident OLMoE (baseline, Hybrid opt) ===")
    am = _build_olmoe_engine(
        shape, lr=lr, num_experts=num_experts, top_k=top_k,
        n_gpu_layers=shape.n_layers, target_round_tokens=target_tokens,
    )
    print(
        f"  n_gpu_layers={am.working_set.n_gpu_layers}/{shape.n_layers}  "
        f"n_gpu_grads={am.working_set.n_gpu_grads}  "
        f"n_gpu_opt_layers={am.working_set.n_gpu_opt_layers}"
    )
    _random_init(am, seed=4242)
    resident_curve = _train(am, step_batches)
    _cleanup(am)

    print("\n=== B. offloaded OLMoE (half layers resident) ===")
    am = _build_olmoe_engine(
        shape, lr=lr, num_experts=num_experts, top_k=top_k,
        n_gpu_layers=max(1, shape.n_layers // 2),
        target_round_tokens=target_tokens,
    )
    print(
        f"  n_gpu_layers={am.working_set.n_gpu_layers}/{shape.n_layers}  "
        f"n_gpu_grads={am.working_set.n_gpu_grads}  "
        f"n_gpu_opt_layers={am.working_set.n_gpu_opt_layers}"
    )
    _random_init(am, seed=4242)
    offload_curve = _train(am, step_batches)
    _cleanup(am)

    print("\n=== comparison ===")
    max_delta = 0.0
    for i, (a, b) in enumerate(zip(resident_curve, offload_curve)):
        d = abs(a - b)
        max_delta = max(max_delta, d)
        print(f"  step {i:3d}: resident={a:.4f}  offload={b:.4f}  |Δ|={d:.4f}")

    for step, L in enumerate(resident_curve + offload_curve):
        assert L == L and abs(L) < 1e6, f"bad loss {L} at step {step}"
    assert resident_curve[-1] < resident_curve[0] - 0.1, (
        f"MoE resident loss didn't decrease: "
        f"first={resident_curve[0]:.4f} last={resident_curve[-1]:.4f}"
    )
    assert offload_curve[-1] < offload_curve[0] - 0.1, (
        f"MoE offload loss didn't decrease"
    )
    assert max_delta < 0.05, (
        f"MoE resident vs offload: max |Δ| = {max_delta:.4f} (expected "
        f"< 0.05). Possible bugs in per-expert Muon step or opt-state "
        f"rotation for 3-D MoE expert tensors."
    )
    print(
        f"\n✓ MoE Muon+offloading pretraining PASSED "
        f"(max |Δ| = {max_delta:.4f}, "
        f"loss {resident_curve[0]:.3f} → {resident_curve[-1]:.3f})"
    )


if __name__ == "__main__":
    main()
