"""Save-level parity: force_saved_act_level=0 vs force_saved_act_level=max
must produce identical loss curves given the same random init and data.

This is the TIGHTEST correctness test for the activation ring / offload /
recompute paths. The only difference between the two runs is whether
higher-tier activations are saved (and read directly at bwd time) or
recomputed at bwd time from tier-0 fields. Any mismatch beyond bf16
noise means ``forward_recompute != forward``, i.e. a bug in the
recompute path or the engine's slot management around it.

[FINDING 17] (docs/NOTES.md) was a bug in exactly this area — the
``ActivationSlot.has()`` check didn't respect the saved level, so
recompute was silently skipped for offloaded layers. This test
exercises it directly: at save=0 every layer recomputes, at
save=max nothing does, and they must agree on every step.

This test intentionally uses a small-but-realistic config with:
* 32 layers (so the ring has to rotate — n_gpu_act_slots < n_layers)
* random-init weights (pretraining regime)
* synthetic random tokens
* 10 SGD steps
"""
from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import ModelShape, _Seq

DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _build_engine(shape: ModelShape, lr: float, *, force_save_level: int | None):
    """Build a FlexTrain engine with either force_saved_act_level=0
    (all layers recompute) or =schema.max_tier (all layers full-save)."""
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

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
        rms_norm_eps=shape.rms_norm_eps, head_chunk_size=256,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    ))
    dims = dict(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
    )
    # Deliberately small GPU act buffer so the ring wraps and offloads
    # happen — this is required to actually exercise the recompute path
    # at force_save=0.
    working_set = WorkingSetConfig(
        target_round_tokens=1024, max_chunk_size=1024,
        max_training_chunks=1, max_total_round_tokens=1024,
        target_num_rounds=1,
        n_gpu_layers=shape.n_layers // 2,
        n_gpu_grads=shape.n_layers // 2,
        n_gpu_opt_layers=max(1, shape.n_layers // 4),
        gpu_act_buffer_size=int(1.0 * (1 << 30)),  # 1 GiB
        host_act_buffer_size=int(2.0 * (1 << 30)),  # 2 GiB
        available_gpu_memory_bytes=int(24 * (1 << 30)),
        available_host_memory_bytes=int(32 * (1 << 30)),
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=1024, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    opt = AdamW(
        AdamWHyperparams(lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
        state_dtype=torch.bfloat16,
    )
    return ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=working_set, hw_cost=hw_cost, dims=dims, device=DEVICE,
        force_saved_act_level=force_save_level,
    )


def _seed_random_weights(am, seed: int = 4242) -> None:
    """Fill host buffers with Gaussian random weights (std=0.02) —
    matches standard Llama-ish pretraining init.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        # Embed + head.
        for name, t in am.buffers.host_embed_params.items():
            t.normal_(mean=0.0, std=0.02)
        for name, t in am.buffers.host_head_params.items():
            if name.endswith("norm"):
                t.fill_(1.0)
                t.add_(0.01 * torch.randn_like(t))
            else:
                t.normal_(mean=0.0, std=0.02)
        # Backbone.
        for host_p in am.buffers.host_params:
            for name, t in host_p.items():
                if "norm" in name:
                    t.fill_(1.0)
                    t.add_(0.01 * torch.randn_like(t))
                else:
                    t.normal_(mean=0.0, std=0.02)
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


def _gen_batches(n_steps: int, shape: ModelShape, seed: int = 11) -> list[list[_Seq]]:
    torch.manual_seed(seed)
    batches = []
    for _ in range(n_steps):
        seq_len = 512
        tokens = torch.randint(0, shape.vocab_size, (seq_len,), dtype=torch.int64)
        s = _Seq(tokens)
        targets = torch.roll(tokens, -1)
        targets[-1] = -100
        s.targets = targets
        # Also a second shorter sequence to exercise packing.
        seq_len2 = 384
        tokens2 = torch.randint(0, shape.vocab_size, (seq_len2,), dtype=torch.int64)
        s2 = _Seq(tokens2)
        s2.targets = torch.roll(tokens2, -1)
        s2.targets[-1] = -100
        batches.append([s, s2])
    return batches


def _run(batches, force_save_level, shape, lr) -> list[float]:
    from flextrain.bench.parity import _flextrain_step
    am = _build_engine(shape, lr, force_save_level=force_save_level)
    _seed_random_weights(am, seed=4242)
    curve = []
    print(f"\n=== save_level={force_save_level} ===")
    for step, batch in enumerate(batches):
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        loss = _flextrain_step(am, seqs)
        print(f"  step {step:3d}: loss = {loss:.6f}")
        curve.append(loss)
    am.buffers.destroy()
    del am
    import gc; gc.collect()
    try:
        from flextrain.engine import unregister_all_process_pinned_memory
        unregister_all_process_pinned_memory()
    except Exception:
        pass
    torch.cuda.empty_cache()
    return curve


def test_save_level_parity() -> None:
    """Small 32-layer Llama-ish model, random init, 10 SGD steps.
    Run twice: force_save=0 and force_save=max. Losses must match within
    bf16 noise (per-step |Δ| < 1e-3 typically, and cumulative curves
    indistinguishable)."""
    shape = ModelShape(
        d_model=256, n_layers=32, n_heads=8, n_kv_heads=2,
        head_dim=32, expert_dim=512, vocab_size=512,
        rms_norm_eps=1e-5, rope_base=10000.0,
    )
    n_steps = 10
    lr = 5e-4  # small pretraining lr
    batches = _gen_batches(n_steps, shape)

    # max_tier for LlamaBlock is 3.
    max_tier = 3

    curve_full = _run(batches, force_save_level=max_tier, shape=shape, lr=lr)
    curve_recomp = _run(batches, force_save_level=0, shape=shape, lr=lr)

    print("\n=== Comparison (full-save vs all-recompute) ===")
    deltas = [abs(a - b) for a, b in zip(curve_full, curve_recomp)]
    max_delta = max(deltas)
    for i, (a, b, d) in enumerate(zip(curve_full, curve_recomp, deltas)):
        print(f"  step {i:3d}: full={a:.6f}  recomp={b:.6f}  |Δ|={d:.2e}")
    print(f"\nmax |Δ| across {n_steps} steps: {max_delta:.2e}")

    # bf16 noise tolerance: per-step |Δ| should be under 1e-2.
    # If recompute is bit-identical, it'll be 0 or well under 1e-3.
    # If there's a drift, we want to fail loud.
    tol = 1e-2
    if max_delta > tol:
        raise AssertionError(
            f"Save-level parity FAILED: max |Δ| = {max_delta:.4e} > "
            f"tolerance {tol:.4e}. Recompute diverges from full-save."
        )
    print(f"\n✓ Save-level parity PASSED (max |Δ| = {max_delta:.2e} < {tol:.0e})")


if __name__ == "__main__":
    test_save_level_parity()
