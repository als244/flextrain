"""End-to-end Muon + offloading pretraining correctness on real data.

Trains a small (~100M) Llama-style model from random init on
MathInstruct (real Llama-3 tokenizer tokens) under two configurations:

* **all-resident**: every layer's params/grads/opt fit on GPU. Baseline.
* **offloaded**: reduce ``n_gpu_layers`` so AdaWS ring rotation kicks
  in for params, grads, and opt state.

Both use :class:`HybridMuonAdamW` (Muon on 2-D projections + per-expert
MoE stacks, AdamW on norms/embed/head/router). Asserts:

* Loss curves for all-resident vs offloaded match within bf16 noise
  (engine is deterministic across offload settings for the same opt).
* Loss decreases over the run.
* No NaN/inf at any step.

This is the Muon counterpart to ``test_random_init_pretraining.py``
(which used AdamW) and complements ``test_save_level_parity.py`` by
also varying the offloading working-set instead of only save level.
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


def _random_init(am, shape, seed: int = 4242) -> None:
    """Standard normal(0, 0.02) init, norms at 1.0, deterministic."""
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


def _build_llama_engine(
    shape: ModelShape, lr: float, n_gpu_layers: int, target_round_tokens: int,
):
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
    from flextrain.optim.hybrid import (
        HybridMuonAdamW, HybridMuonAdamWHyperparams,
    )
    from flextrain.optim.adamw import AdamWHyperparams
    from flextrain.optim.muon import MuonHyperparams

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
        d_model=512, n_layers=8, n_heads=8, n_kv_heads=2, head_dim=64,
        expert_dim=2048, vocab_size=128256,
        rms_norm_eps=1e-5, rope_base=500_000.0,
    )
    lr = 3e-4
    n_steps = 15
    target_tokens = 2048

    hf_tokenizer = os.path.join(ROOT, "models", "Llama-3.2-1B")
    assert os.path.isdir(hf_tokenizer), f"need tokenizer at {hf_tokenizer}"
    print(f"Preparing {n_steps} batches of ~{target_tokens} tokens (Llama-3 tokenizer)...")
    step_batches = _pull_step_batches(
        hf_tokenizer, n_steps=n_steps, target_tokens_per_step=target_tokens,
    )
    print(f"  {len(step_batches)} batches ready")
    total_params = (
        shape.vocab_size * shape.d_model +  # embed
        shape.d_model * shape.vocab_size +  # head
        shape.n_layers * (
            4 * shape.d_model * shape.d_model +  # attn Q/K/V/O (approx)
            3 * shape.d_model * shape.expert_dim  # FFN w1/w2/w3
        )
    )
    print(f"  model ~{total_params / 1e6:.0f}M params")

    # --- Scenario A: all-resident ---
    print("\n=== A. all-resident (baseline, Muon+AdamW hybrid) ===")
    am = _build_llama_engine(
        shape, lr=lr, n_gpu_layers=shape.n_layers,
        target_round_tokens=target_tokens,
    )
    print(
        f"  n_gpu_layers={am.working_set.n_gpu_layers}/{shape.n_layers}  "
        f"n_gpu_grads={am.working_set.n_gpu_grads}  "
        f"n_gpu_opt_layers={am.working_set.n_gpu_opt_layers}"
    )
    _random_init(am, shape, seed=4242)
    resident_curve = _train(am, step_batches)
    _cleanup(am)

    # --- Scenario B: offloaded (n_gpu_layers < n_layers) ---
    print("\n=== B. offloaded (half the layers resident) ===")
    am = _build_llama_engine(
        shape, lr=lr, n_gpu_layers=max(1, shape.n_layers // 2),
        target_round_tokens=target_tokens,
    )
    print(
        f"  n_gpu_layers={am.working_set.n_gpu_layers}/{shape.n_layers}  "
        f"n_gpu_grads={am.working_set.n_gpu_grads}  "
        f"n_gpu_opt_layers={am.working_set.n_gpu_opt_layers}"
    )
    _random_init(am, shape, seed=4242)  # same seed → same init
    offload_curve = _train(am, step_batches)
    _cleanup(am)

    # --- Compare ---
    print("\n=== comparison (same init, same data, different offloading) ===")
    max_delta = 0.0
    for i, (a, b) in enumerate(zip(resident_curve, offload_curve)):
        d = abs(a - b)
        max_delta = max(max_delta, d)
        print(f"  step {i:3d}: resident={a:.4f}  offload={b:.4f}  |Δ|={d:.4f}")

    # Basic health.
    for step, L in enumerate(resident_curve + offload_curve):
        assert L == L and abs(L) < 1e6, f"bad loss {L} at step {step}"
    assert resident_curve[-1] < resident_curve[0] - 0.1, (
        f"resident loss didn't decrease: "
        f"first={resident_curve[0]:.4f} last={resident_curve[-1]:.4f}"
    )
    assert offload_curve[-1] < offload_curve[0] - 0.1, (
        f"offload loss didn't decrease: "
        f"first={offload_curve[0]:.4f} last={offload_curve[-1]:.4f}"
    )
    # Engine is deterministic across offloading for the same optimizer
    # — the two curves should match to bf16 noise.
    assert max_delta < 0.05, (
        f"resident vs offload curves diverge: max |Δ| = {max_delta:.4f} "
        f"(expected < 0.05). Offloading or opt-state rotation may be "
        f"miscomputing grads for Muon-eligible tensors."
    )

    print(
        f"\n✓ Muon + offloading pretraining PASSED "
        f"(max resident-vs-offload |Δ| = {max_delta:.4f}, "
        f"loss reduced {resident_curve[0]:.3f} → {resident_curve[-1]:.3f})"
    )


if __name__ == "__main__":
    main()
