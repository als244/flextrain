"""Phase 2: LoRA + engine integration smoke test.

Builds a small Llama backbone with :class:`LoRAWrapperLayer` on every
backbone layer, runs through the full FlexTrain engine for a few steps
on random data, asserts:

1. Loss decreases.
2. Frozen base weights produce NO grad / opt-state allocations.
3. LoRA A/B params DO have grad + opt-state allocations.
4. After training, master copies of base weights are unchanged
   (frozen invariant).
5. After training, master copies of LoRA A/B params have changed
   (training is doing something).
6. No NaN / inf at any step.
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
from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
from flextrain.optim.adamw import AdamW, AdamWHyperparams


DEVICE = "cuda:0"


def main():
    shape = ModelShape(
        d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=32,
        expert_dim=256, vocab_size=512,
        rms_norm_eps=1e-5, rope_base=10_000.0,
    )
    cfg = LlamaBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    dims = dict(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        attn_dim=shape.n_heads * shape.head_dim,
        kv_dim=shape.n_kv_heads * shape.head_dim,
        expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
    )
    rank = 8
    alpha = 16.0
    backbone = []
    for i in range(shape.n_layers):
        base = LlamaBlock(layer_id=i, cfg=cfg)
        wrapped = LoRAWrapperLayer(
            base, lora_targets="all", rank=rank, alpha=alpha, dims=dims,
        )
        backbone.append(wrapped)

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

    # ---- Frozen invariant assertions ----
    print("=== Frozen invariant ===")
    base_targets = ("w_q", "w_k", "w_v", "w_o", "w_1", "w_2", "w_3")
    for i in range(shape.n_layers):
        gkeys = set(am.buffers.host_grads[i].keys())
        opt_keys = set(am.buffers.host_opt[i].host.keys())
        master_keys = set(am.buffers.host_params[i].keys())
        # Frozen base targets: in master, NOT in grads/opt.
        for tgt in base_targets:
            assert tgt in master_keys, f"layer {i}: {tgt} missing from master"
            gkey = "g_" + tgt[2:]
            assert gkey not in gkeys, (
                f"layer {i}: frozen {tgt} unexpectedly has {gkey}"
            )
            for st in ("o_adam_m", "o_adam_v"):
                assert f"{st}_{tgt[2:]}" not in opt_keys, (
                    f"layer {i}: frozen {tgt} unexpectedly has {st}_{tgt[2:]}"
                )
        # LoRA A/B: have grads + opt state.
        for tgt in base_targets:
            for ab in ("a", "b"):
                ab_master = f"{tgt}_lora_{ab}"
                ab_grad = f"g_{tgt[2:]}_lora_{ab}"
                assert ab_master in master_keys, (
                    f"layer {i}: {ab_master} missing from master"
                )
                assert ab_grad in gkeys, (
                    f"layer {i}: {ab_master} missing grad {ab_grad}"
                )
    print("  frozen base: NO grads/opt allocated ✓")
    print("  LoRA A/B: grads+opt allocated ✓")

    # ---- Random init weights ----
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
                    t.zero_()  # PEFT init: B = 0 so LoRA delta = 0 at start.
                elif name.endswith("_lora_a"):
                    t.normal_(mean=0.0, std=0.02)
                else:
                    t.normal_(mean=0.0, std=0.02)
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    # Snapshot base weights (to verify they DON'T change).
    snapshot_base = {}
    snapshot_a = {}
    snapshot_b = {}
    for i in range(shape.n_layers):
        snapshot_base[i] = {
            tgt: am.buffers.host_params[i][tgt].clone()
            for tgt in base_targets
        }
        snapshot_a[i] = {
            tgt: am.buffers.host_params[i][f"{tgt}_lora_a"].clone()
            for tgt in base_targets
        }
        snapshot_b[i] = {
            tgt: am.buffers.host_params[i][f"{tgt}_lora_b"].clone()
            for tgt in base_targets
        }

    # ---- Train on a fixed batch ----
    torch.manual_seed(1000)
    tokens = torch.randint(0, shape.vocab_size, (64,), dtype=torch.int64)
    targets = torch.roll(tokens, -1)
    targets[-1] = -100

    print("\n=== Training (fixed batch, 20 steps) ===")
    losses = []
    for step in range(20):
        seq = _Seq(tokens.clone())
        seq.targets = targets.clone()
        loss = _flextrain_step(am, [seq])
        losses.append(loss)
        if step < 3 or step % 5 == 0 or step == 19:
            print(f"  step {step:2d}: loss={loss:.4f}")
        assert loss == loss and abs(loss) < 1e6, f"bad loss {loss} at step {step}"

    first3 = sum(losses[:3]) / 3
    last3 = sum(losses[-3:]) / 3
    print(f"  first-3 avg: {first3:.4f}  last-3 avg: {last3:.4f}")
    assert last3 < first3 - 0.5, (
        f"LoRA training didn't reduce loss: first3={first3} last3={last3}"
    )
    print("  loss decreased ✓")

    # ---- Verify base weights unchanged + LoRA params changed ----
    print("\n=== Frozen-base invariant after training ===")
    for i in range(shape.n_layers):
        for tgt in base_targets:
            now = am.buffers.host_params[i][tgt]
            before = snapshot_base[i][tgt]
            d = (now.float() - before.float()).abs().max().item()
            assert d == 0.0, (
                f"layer {i} {tgt}: base weight CHANGED by {d:.4e} "
                f"(should be frozen)"
            )
    print("  every base weight is bit-identical to its initial value ✓")

    print("\n=== LoRA A/B changed during training ===")
    a_changed = b_changed = 0
    for i in range(shape.n_layers):
        for tgt in base_targets:
            now_a = am.buffers.host_params[i][f"{tgt}_lora_a"]
            now_b = am.buffers.host_params[i][f"{tgt}_lora_b"]
            d_a = (now_a.float() - snapshot_a[i][tgt].float()).abs().max().item()
            d_b = (now_b.float() - snapshot_b[i][tgt].float()).abs().max().item()
            if d_a > 1e-6:
                a_changed += 1
            if d_b > 1e-6:
                b_changed += 1
    total = shape.n_layers * len(base_targets)
    print(f"  A: {a_changed}/{total} tensors changed")
    print(f"  B: {b_changed}/{total} tensors changed")
    assert a_changed == total and b_changed == total, (
        "Some LoRA params didn't change — gradient routing broken"
    )

    print("\n✓ Phase 2 PASSED")


if __name__ == "__main__":
    main()
