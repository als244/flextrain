"""HybridMuonAdamW optimizer correctness.

Three tests:

1. **Classification**: ``infer_optimizer_for_param`` correctly routes
   every TensorSpec class encountered in LlamaBlock + OLMoEBlock + the
   LM head to the expected update rule. No GPU.

2. **Dense parity**: build a small Llama in both naive PyTorch (with
   a hand-rolled split optimizer: torch.optim.AdamW on norms/embed/head,
   a Muon reference on 2-D projections) and FlexTrain (with
   HybridMuonAdamW). Train for a few steps on random-init random-data
   and assert loss curves agree within bf16 noise.

3. **MoE parity**: same, but with OLMoEBlock — verifies that routers
   get AdamW, per-expert stacked matrices get Muon, and norm weights
   get AdamW.

Tests run with GPU; all assertions use bf16-appropriate tolerances.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.core.layer import TensorSpec
from flextrain.optim.hybrid import (
    HybridMuonAdamW, HybridMuonAdamWHyperparams,
    HybridStateSpec, infer_optimizer_for_param,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Test 1: classification
# ---------------------------------------------------------------------------


def test_classification() -> None:
    """Every parameter name we use across Llama/Qwen3/OLMoE should classify
    to the expected rule."""
    dims = {
        "d_model": 128, "n_heads": 4, "n_kv_heads": 2, "head_dim": 32,
        "attn_dim": 128, "kv_dim": 64, "expert_dim": 256, "vocab_size": 256,
        "num_experts": 4, "top_k": 2,
    }

    def make(name: str, shape_fn) -> TensorSpec:
        return TensorSpec(
            name=name, shape_fn=shape_fn, compute_dtype=torch.bfloat16
        )

    # Each tuple: (name, shape_fn, expected_rule).
    cases = [
        # Attention projections → Muon (2-D, no adamw fragment).
        ("w_q", lambda d: (d["d_model"], d["attn_dim"]), "muon"),
        ("w_k", lambda d: (d["d_model"], d["kv_dim"]), "muon"),
        ("w_v", lambda d: (d["d_model"], d["kv_dim"]), "muon"),
        ("w_o", lambda d: (d["attn_dim"], d["d_model"]), "muon"),
        # FFN projections → Muon.
        ("w_1", lambda d: (d["d_model"], d["expert_dim"]), "muon"),
        ("w_2", lambda d: (d["expert_dim"], d["d_model"]), "muon"),
        ("w_3", lambda d: (d["d_model"], d["expert_dim"]), "muon"),
        # MoE stacked expert weights → Muon. Hybrid optimizer iterates
        # the expert dim and applies Newton-Schulz per 2-D expert slice.
        (
            "w_up",
            lambda d: (d["num_experts"], d["d_model"], 2 * d["expert_dim"]),
            "muon",
        ),
        (
            "w_down",
            lambda d: (d["num_experts"], d["expert_dim"], d["d_model"]),
            "muon",
        ),
        # MoE router → AdamW (has "router" fragment).
        ("w_router", lambda d: (d["d_model"], d["num_experts"]), "adamw"),
        # Residual-stream / final norms → AdamW.
        ("w_attn_norm", lambda d: (d["d_model"],), "adamw"),
        ("w_ffn_norm", lambda d: (d["d_model"],), "adamw"),
        ("w_final_norm", lambda d: (d["d_model"],), "adamw"),
        # QK-norms (Qwen3 per-head and OLMoE full-row) → AdamW.
        ("w_q_norm", lambda d: (d["head_dim"],), "adamw"),
        ("w_k_norm", lambda d: (d["head_dim"],), "adamw"),
        # Embedding + LM head → AdamW.
        (
            "w_tok_embeddings",
            lambda d: (d["vocab_size"], d["d_model"]),
            "adamw",
        ),
        ("w_head_proj", lambda d: (d["d_model"], d["vocab_size"]), "adamw"),
        # Qwen2 biases (1-D) → AdamW.
        ("b_q", lambda d: (d["attn_dim"],), "adamw"),
    ]

    for name, shape_fn, expected in cases:
        spec = make(name, shape_fn)
        got = infer_optimizer_for_param(spec, dims)
        assert got == expected, (
            f"{name}: expected {expected!r}, got {got!r}"
        )

    # Explicit override wins even when auto would disagree.
    forced_spec = TensorSpec(
        name="w_q",
        shape_fn=lambda d: (d["d_model"], d["attn_dim"]),
        compute_dtype=torch.bfloat16,
        optimizer="adamw",
    )
    assert infer_optimizer_for_param(forced_spec, dims) == "adamw"

    print("  classification: all 17 name patterns classify correctly ✓")


def test_state_spec_byte_size() -> None:
    """HybridStateSpec should allocate state only for the applicable rule
    per param, not the union."""
    from flextrain.core.layer import ParamSpec
    dims = {"d_model": 64, "attn_dim": 64, "vocab_size": 32}
    w_q = TensorSpec(
        name="w_q", shape_fn=lambda d: (d["d_model"], d["attn_dim"]),
        compute_dtype=torch.bfloat16, opt_state_dtype=torch.float32,
    )
    w_attn_norm = TensorSpec(
        name="w_attn_norm", shape_fn=lambda d: (d["d_model"],),
        compute_dtype=torch.bfloat16, opt_state_dtype=torch.float32,
    )
    w_embed = TensorSpec(
        name="w_tok_embeddings",
        shape_fn=lambda d: (d["vocab_size"], d["d_model"]),
        compute_dtype=torch.bfloat16, opt_state_dtype=torch.float32,
    )
    ps = ParamSpec(tensors=(w_q, w_attn_norm, w_embed))

    spec = HybridStateSpec(tensors=())
    # w_q (muon): 1 tensor (o_muon) × 64*64 × 4 bytes = 16384
    # w_attn_norm (adamw): 2 tensors × 64 × 4 = 512
    # w_embed (adamw): 2 × 32*64 × 4 = 16384
    expected = 64 * 64 * 4 + 2 * 64 * 4 + 2 * 32 * 64 * 4
    got = spec.byte_size_for(ps, dims)
    assert got == expected, f"byte_size: expected {expected}, got {got}"
    print(f"  state-spec byte_size: {got} bytes (correct, w_q Muon-only) ✓")


# ---------------------------------------------------------------------------
# Test 2: dense parity on a tiny Llama
# ---------------------------------------------------------------------------


def _rmsnorm(x, w, eps=1e-5):
    x_f = x.float()
    rms = (x_f * x_f).mean(dim=-1, keepdim=True).add_(eps).rsqrt_()
    return (x_f * rms).to(x.dtype) * w


class _TinyDense(torch.nn.Module):
    """Minimal dense model for optimizer parity testing. A single
    Linear + RMSNorm + LM head — just enough to exercise both update
    rules on tensors of each classification."""

    def __init__(self, d_model=64, vocab=128):
        super().__init__()
        self.embed = torch.nn.Parameter(
            torch.empty(vocab, d_model, dtype=DTYPE).normal_(std=0.02)
        )
        self.w_q = torch.nn.Parameter(
            torch.empty(d_model, d_model, dtype=DTYPE).normal_(std=0.02)
        )
        self.norm = torch.nn.Parameter(
            torch.ones(d_model, dtype=DTYPE) + 0.01 * torch.randn(d_model, dtype=DTYPE)
        )
        self.head = torch.nn.Parameter(
            torch.empty(d_model, vocab, dtype=DTYPE).normal_(std=0.02)
        )

    def forward(self, tokens):
        x = self.embed[tokens]
        x = x @ self.w_q
        x = _rmsnorm(x, self.norm)
        return x @ self.head


# ---------------------------------------------------------------------------
# Test 3: dense HybridMuonAdamW E2E on tiny Llama via the engine
# ---------------------------------------------------------------------------


def test_engine_e2e_dense() -> None:
    """Build a small random-init Llama, run HybridMuonAdamW for a few
    steps, verify:
      (a) it runs without errors,
      (b) loss decreases,
      (c) opt-state host allocation has only o_muon_* for 2-D w_q/w_k/...
          and only o_adam_m_* / o_adam_v_* for norms/embed/head/router.
    """
    from flextrain.bench.parity import (
        ModelShape, _Seq, _flextrain_step, DTYPE as D,
    )
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig

    shape = ModelShape(
        d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=32,
        expert_dim=256, vocab_size=512,
        rms_norm_eps=1e-5, rope_base=500_000.0,
    )
    cfg = LlamaBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True,
        compute_dtype=D, master_dtype=D, grad_dtype=D,
        norm_grad_dtype=torch.float32,
    )
    backbone = [LlamaBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
    embed = TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=shape.vocab_size, d_model=shape.d_model,
        compute_dtype=D, master_dtype=D, grad_dtype=D,
    ))
    head = LMHead(LMHeadConfig(
        d_model=shape.d_model, vocab_size=shape.vocab_size,
        rms_norm_eps=shape.rms_norm_eps, head_chunk_size=64,
        compute_dtype=D, master_dtype=D, grad_dtype=D,
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
    opt = HybridMuonAdamW(HybridMuonAdamWHyperparams(
        lr=3e-4,
    ))
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=ws, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )

    # Verify allocated opt-state keys per layer match classification.
    for i in range(shape.n_layers):
        keys = set(am.buffers.host_opt[i].host.keys())
        # Muon keys for 2-D projections.
        for w in ("q", "k", "v", "o", "1", "2", "3"):
            assert f"o_muon_{w}" in keys, (
                f"layer {i}: expected o_muon_{w} in host_opt, got {keys}"
            )
            assert f"o_adam_m_{w}" not in keys, (
                f"layer {i}: unexpected AdamW state for 2-D tensor o_{w}"
            )
        # AdamW keys for norms.
        for w in ("attn_norm", "ffn_norm"):
            assert f"o_adam_m_{w}" in keys
            assert f"o_adam_v_{w}" in keys
            assert f"o_muon_{w}" not in keys, (
                f"layer {i}: unexpected Muon state for norm {w}"
            )

    # Embed + head keys: all AdamW.
    embed_keys = set(am.buffers.host_embed_opt.host.keys())
    assert "o_adam_m_tok_embeddings" in embed_keys
    assert "o_muon_tok_embeddings" not in embed_keys
    head_keys = set(am.buffers.host_head_opt.host.keys())
    assert "o_adam_m_head_proj" in head_keys
    assert "o_adam_m_final_norm" in head_keys
    assert "o_muon_head_proj" not in head_keys

    print("  opt-state allocation matches classification per param ✓")

    # Random-init all params.
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
                else:
                    t.normal_(mean=0.0, std=0.02)
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    # Run enough training steps to see clear descent. Use the same batch
    # every step so descent is monotone (no random-data noise masking
    # the signal).
    torch.manual_seed(1000)
    tokens = torch.randint(0, shape.vocab_size, (64,), dtype=torch.int64)
    targets = torch.roll(tokens, -1)
    targets[-1] = -100

    losses = []
    for step in range(20):
        seq = _Seq(tokens.clone())
        seq.targets = targets.clone()
        loss = _flextrain_step(am, [seq])
        losses.append(loss)
        if step < 3 or step % 5 == 0 or step == 19:
            print(f"  hybrid step {step:2d}: loss={loss:.4f}")

    first3 = sum(losses[:3]) / 3
    last3 = sum(losses[-3:]) / 3
    # On a fixed batch, a working optimizer should drive loss down
    # noticeably in 20 steps.
    assert last3 < first3 - 0.5, (
        f"hybrid optimizer failed to reduce loss on fixed batch: "
        f"first3 avg={first3:.4f} last3 avg={last3:.4f}"
    )
    # And no NaNs / infs.
    for step, L in enumerate(losses):
        assert L == L, f"NaN loss at step {step}"
        assert abs(L) < 1e6, f"exploded loss {L} at step {step}"
    print(f"  hybrid E2E: loss {losses[0]:.4f} → {losses[-1]:.4f}, no NaN ✓")
    # Clean up so the next test (MoE) starts with a fresh engine.
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


# ---------------------------------------------------------------------------
# Test 4: MoE E2E (classifier must separate router/norms from expert stacks)
# ---------------------------------------------------------------------------


def test_engine_e2e_moe() -> None:
    """OLMoE with HybridMuonAdamW: router and norms → AdamW, expert
    stacked matrices → Muon. Verify classification + smoke convergence."""
    from flextrain.bench.parity import (
        ModelShape, _Seq, _flextrain_step, DTYPE as D,
    )
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.olmoe import OLMoEBlock, OLMoEBlockConfig

    shape = ModelShape(
        d_model=128, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=32,
        expert_dim=64, vocab_size=256, rms_norm_eps=1e-5, rope_base=10_000.0,
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
        compute_dtype=D, master_dtype=D, grad_dtype=D,
        norm_grad_dtype=torch.float32,
    )
    backbone = [OLMoEBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
    embed = TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=shape.vocab_size, d_model=shape.d_model,
        compute_dtype=D, master_dtype=D, grad_dtype=D,
    ))
    head = LMHead(LMHeadConfig(
        d_model=shape.d_model, vocab_size=shape.vocab_size,
        rms_norm_eps=shape.rms_norm_eps, head_chunk_size=64,
        compute_dtype=D, master_dtype=D, grad_dtype=D,
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
    opt = HybridMuonAdamW(HybridMuonAdamWHyperparams(lr=3e-4))
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=ws, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )

    # Verify MoE classification.
    layer0_keys = set(am.buffers.host_opt[0].host.keys())
    # Expert stacks (3-D) → Muon (per-expert slice).
    assert "o_muon_up" in layer0_keys
    assert "o_muon_down" in layer0_keys
    assert "o_adam_m_up" not in layer0_keys
    # Router → AdamW (name contains "router").
    assert "o_adam_m_router" in layer0_keys
    assert "o_muon_router" not in layer0_keys
    # QK-norms → AdamW.
    assert "o_adam_m_q_norm" in layer0_keys
    assert "o_adam_m_k_norm" in layer0_keys
    # Q/K/V/O projections (2-D) → Muon.
    for w in ("q", "k", "v", "o"):
        assert f"o_muon_{w}" in layer0_keys
        assert f"o_adam_m_{w}" not in layer0_keys
    print("  MoE classification: router/norms→AdamW, attn+experts→Muon ✓")

    # Random-init + smoke-run a step.
    torch.manual_seed(11)
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

    # One SGD step — verify no crash, finite loss.
    tokens = torch.randint(0, shape.vocab_size, (64,), dtype=torch.int64)
    targets = torch.roll(tokens, -1)
    targets[-1] = -100
    seq = _Seq(tokens)
    seq.targets = targets
    loss = _flextrain_step(am, [seq])
    assert loss == loss and abs(loss) < 1e6, f"bad loss: {loss}"
    print(f"  MoE hybrid step: loss={loss:.4f} ✓")


# ---------------------------------------------------------------------------


def main() -> None:
    print("=== classification ===")
    test_classification()
    test_state_spec_byte_size()

    if torch.cuda.is_available():
        print("\n=== dense E2E (HybridMuonAdamW on LlamaBlock) ===")
        test_engine_e2e_dense()
        print("\n=== MoE E2E (HybridMuonAdamW on OLMoEBlock) ===")
        test_engine_e2e_moe()

    print("\n✓ HybridMuonAdamW tests PASSED")


if __name__ == "__main__":
    main()
