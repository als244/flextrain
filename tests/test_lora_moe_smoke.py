"""Phase 4 smoke: LoRAWrapperLayer on OLMoEBlock with default targets="all".

Verifies:
* The wrapper auto-discovers w_q/k/v/o + w_up/w_down (3-D) + w_1/2/3 etc.
* w_router is excluded.
* Frozen invariant after training.
* Loss decreases.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import ModelShape, _Seq, _flextrain_step, DTYPE
from flextrain.core.save_level import HardwareCost
from flextrain.core.working_set import WorkingSetConfig
from flextrain.engine.active_model import ActiveModel
from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
from flextrain.nn.head import LMHead, LMHeadConfig
from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
from flextrain.nn.layers.olmoe import OLMoEBlock, OLMoEBlockConfig
from flextrain.optim.adamw import AdamW, AdamWHyperparams


DEVICE = "cuda:0"


def main():
    shape = ModelShape(
        d_model=128, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=32,
        expert_dim=64, vocab_size=512,
        rms_norm_eps=1e-5, rope_base=10_000.0,
    )
    num_experts = 4
    top_k = 2
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
    dims = dict(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        attn_dim=shape.n_heads * shape.head_dim,
        kv_dim=shape.n_kv_heads * shape.head_dim,
        expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
        num_experts=num_experts, top_k=top_k,
    )
    rank = 8
    alpha = 16.0
    backbone = []
    for i in range(shape.n_layers):
        base = OLMoEBlock(layer_id=i, cfg=cfg)
        wrapped = LoRAWrapperLayer(
            base, lora_targets="all", rank=rank, alpha=alpha, dims=dims,
        )
        backbone.append(wrapped)
        if i == 0:
            # Inspect what was wrapped.
            target_names = [c.target_name for c in wrapped.targets]
            print(f"  layer 0 LoRA targets: {target_names}")
            assert "w_q" in target_names
            assert "w_up" in target_names
            assert "w_down" in target_names
            assert "w_router" not in target_names, "router should be excluded"

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
    ws = WorkingSetConfig(
        target_round_tokens=128, max_chunk_size=128,
        max_training_chunks=4, max_total_round_tokens=128,
        target_num_rounds=1,
        n_gpu_layers=shape.n_layers, n_gpu_grads=shape.n_layers,
        n_gpu_opt_layers=shape.n_layers,
        gpu_act_buffer_size=int(0.5 * (1 << 30)),
        host_act_buffer_size=int(1 * (1 << 30)),
        available_gpu_memory_bytes=int(4 * (1 << 30)),
        available_host_memory_bytes=int(8 * (1 << 30)),
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=128, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    opt = AdamW(AdamWHyperparams(lr=3e-4))
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=ws, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )

    # Frozen invariant: w_up/w_down/etc. have no grads; LoRA A/B do.
    print("\n=== Frozen invariant ===")
    for i in range(shape.n_layers):
        gkeys = set(am.buffers.host_grads[i].keys())
        # Targeted base tensors absent from grads.
        for tgt in ("w_q", "w_up", "w_down", "w_1", "w_2", "w_3"):
            gkey = "g_" + tgt[2:]
            assert gkey not in gkeys, (
                f"layer {i}: frozen {tgt} has unexpected grad allocation"
            )
        # Router still has g_router (not frozen by LoRA wrapper).
        assert "g_router" in gkeys, (
            f"layer {i}: router should still get a grad (excluded from LoRA)"
        )
        # LoRA grads ARE allocated.
        for tgt in ("w_q", "w_up", "w_down"):
            for ab in ("a", "b"):
                gkey = f"g_{tgt[2:]}_lora_{ab}"
                assert gkey in gkeys, f"layer {i}: missing {gkey}"
    print("  frozen base + LoRA grads correctly allocated ✓")

    # Random init.
    torch.manual_seed(7)
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
                elif name.endswith("_lora_b"):
                    t.zero_()
                elif name.endswith("_lora_a"):
                    t.normal_(mean=0.0, std=0.02)
                else:
                    t.normal_(mean=0.0, std=0.02)
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    # Snapshot frozen base weights.
    snap = {}
    for i in range(shape.n_layers):
        snap[i] = {
            tgt: am.buffers.host_params[i][tgt].clone()
            for tgt in ("w_q", "w_k", "w_v", "w_o", "w_up", "w_down")
        }

    # Train.
    torch.manual_seed(1000)
    tokens = torch.randint(0, shape.vocab_size, (64,), dtype=torch.int64)
    targets = torch.roll(tokens, -1)
    targets[-1] = -100

    print("\n=== Training (fixed batch, 15 steps) ===")
    losses = []
    for step in range(15):
        seq = _Seq(tokens.clone())
        seq.targets = targets.clone()
        loss = _flextrain_step(am, [seq])
        losses.append(loss)
        if step < 3 or step == 14:
            print(f"  step {step:2d}: loss={loss:.4f}")
        assert loss == loss and abs(loss) < 1e6

    print(f"  first vs last: {losses[0]:.4f} → {losses[-1]:.4f}")
    assert losses[-1] < losses[0] - 0.3

    # Verify frozen-base invariant.
    print("\n=== Frozen base unchanged ===")
    for i in range(shape.n_layers):
        for tgt, before in snap[i].items():
            now = am.buffers.host_params[i][tgt]
            d = (now.float() - before.float()).abs().max().item()
            assert d == 0.0, f"layer {i} {tgt}: changed by {d}"
    print("  every frozen base weight bit-identical ✓")

    print("\n✓ Phase 4 MoE LoRA smoke PASSED")


if __name__ == "__main__":
    main()
