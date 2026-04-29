"""Isolate the recompute bug: single LlamaBlock at 8B dims, compare
fwd outputs vs fwd_recompute outputs.

If recompute is correct, it should produce bit-identical (or bf16-
near-identical) activations to the original forward pass.

We run a full forward at tier-max (all fields saved), record x1, x3,
xq, xo. Then simulate a "saved at tier 0" path: clear xq/xo/x1/x3,
call forward_recompute, and compare against the saved values.
"""
from __future__ import annotations

import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def main():
    from flextrain.core.activation_schema import ActivationSlot
    from flextrain.core.layer import LayerContext, ChunkMeta
    from flextrain.engine.buffers import KVContextWindow
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig

    torch.manual_seed(4242)

    # 8B-ish dims.
    d_model = 4096
    n_heads = 32
    n_kv_heads = 8
    head_dim = 128
    expert_dim = 14336

    cfg = LlamaBlockConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, expert_dim=expert_dim,
        rms_norm_eps=1e-5, rope_base=500000.0, is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    block = LlamaBlock(layer_id=0, cfg=cfg)
    dims = cfg.dims()
    dims["vocab_size"] = 128256

    T = 2048
    # Random weights.
    weights = {
        "w_attn_norm": torch.randn(d_model, dtype=DTYPE, device=DEVICE).abs(),
        "w_q": torch.randn(d_model, n_heads * head_dim, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_k": torch.randn(d_model, n_kv_heads * head_dim, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_v": torch.randn(d_model, n_kv_heads * head_dim, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_o": torch.randn(n_heads * head_dim, d_model, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_ffn_norm": torch.randn(d_model, dtype=DTYPE, device=DEVICE).abs(),
        "w_1": torch.randn(d_model, expert_dim, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_2": torch.randn(expert_dim, d_model, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_3": torch.randn(d_model, expert_dim, dtype=DTYPE, device=DEVICE) * 0.02,
    }
    x = torch.randn(T, d_model, dtype=DTYPE, device=DEVICE) * 0.5

    # Build a ChunkMeta. Minimal: one seq.
    seq_lens = torch.tensor([T], dtype=torch.int32, device=DEVICE)
    chunk_meta = ChunkMeta(
        seq_positions=torch.arange(T, dtype=torch.int32, device=DEVICE).unsqueeze(-1),
        q_seq_offsets=torch.tensor([0, T], dtype=torch.int32, device=DEVICE),
        k_seq_offsets=torch.tensor([0, T], dtype=torch.int32, device=DEVICE),
        q_seq_lens=seq_lens,
        k_seq_lens=seq_lens,
        q_seq_offsets_i64=torch.tensor([0, T], dtype=torch.int64, device=DEVICE),
        fla_chunk_indices_64=torch.tensor(
            [[0, c] for c in range((T + 63) // 64)],
            dtype=torch.int64, device=DEVICE,
        ).reshape(-1, 2),
        seq_lens_host=[T],
        prior_seq_lens_host=[0],
        prior_seq_offsets_host=[0],
        max_seqlen_q=T, max_seqlen_k=T, total_q=T, total_k=T,
    )

    # Build an ActivationSlot at max_tier backed by a fresh buffer.
    nbytes = block.schema.home_size_bytes(T, dims, block.schema.max_tier)
    buffer = torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)
    slot, _ = ActivationSlot.from_buffer(
        block.schema, level=block.schema.max_tier,
        num_tokens=T, dims=dims, buffer=buffer,
        include_nonpersistent=True,
    )

    # Scratch + KV cache + LayerContext.
    scratch_pool = {}
    def _scratch(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=DEVICE)
    kv = KVContextWindow.create(
        max_context_tokens=T, n_kv_heads=n_kv_heads, head_dim=head_dim,
        dtype=DTYPE, device=DEVICE,
    )
    ctx = LayerContext(scratch=_scratch, kv_cache=kv, stream=torch.cuda.current_stream())

    # Forward.
    print("Running original forward (saves all fields at tier max)...")
    y = block.forward(x.clone(), chunk_meta, weights, slot, ctx)

    # Save copies.
    xq_orig = slot.xq.clone()
    xo_orig = slot.xo.clone()
    x1_orig = slot.x1.clone()
    x3_orig = slot.x3.clone()
    attn_result_orig = slot.attn_result.clone()
    rstd_attn_orig = slot.attn_norm_rstd.clone()
    rstd_ffn_orig = slot.ffn_norm_rstd.clone()

    # Now simulate "saved at tier 0": only keep x_inp, xk, xv, rstds.
    # Clear the higher-tier fields AND remove them from _tensors so
    # slot.has(name) returns False for them (forcing recompute).
    slot.xq.zero_()
    slot.xo.zero_()
    slot.x1.zero_()
    slot.x3.zero_()
    slot.attn_result.zero_()
    slot.softmax_lse.zero_()

    # Drop higher-tier fields so has() returns False.
    tier0_tensors = {
        k: v for k, v in slot._tensors.items()
        if {f.name: f.tier for f in slot.schema.fields}[k] == 0
    }
    slot_tier0 = ActivationSlot(slot.schema, 0, tier0_tensors)
    # But recompute writes INTO the same memory (via .set). We need to
    # keep references to the higher-tier tensors so fwd_recompute can
    # write into them. Work around by giving slot_tier0 a ``set`` path
    # by pre-populating the high-tier slots anyway (they'll be
    # overwritten by recompute).
    for name, t in slot._tensors.items():
        if name not in slot_tier0._tensors:
            # Insert zeroed high-tier tensors. has() now returns True
            # which is WRONG for our simulation... Instead just call
            # fwd_recompute manually with the original slot that had
            # level=0. The logic in forward_recompute checks has().
            pass
    # The cleanest way: directly modify slot._tensors to remove the
    # high-tier fields BEFORE calling recompute.
    saved_high_tier = {}
    for name in ("xq", "xo", "x1", "x3", "attn_result", "softmax_lse"):
        if name in slot._tensors:
            saved_high_tier[name] = slot._tensors.pop(name)
    # But recompute will need a place to write. forward_recompute writes
    # to slot.xq etc. via attribute access, which goes through
    # __getattr__ which checks _tensors. If xq is not there, access
    # fails. Put them back but also set slot.level=0.
    for name, t in saved_high_tier.items():
        slot._tensors[name] = t
    # Use a wrapper to override has() behavior.
    class _LevelSlot:
        def __init__(self, inner, level):
            self._inner = inner
            self._level = level
            self._field_tiers = {f.name: f.tier for f in inner.schema.fields}
        def has(self, name):
            return self._field_tiers.get(name, 99) <= self._level
        def __getattr__(self, name):
            return getattr(self._inner, name)
    slot = _LevelSlot(slot, 0)

    print("Running forward_recompute from tier 0...")
    block.forward_recompute(slot, chunk_meta, weights, ctx)

    def _cmp(name, a, b):
        delta = (a.float() - b.float()).abs()
        max_d = float(delta.max())
        mean_d = float(delta.mean())
        max_a = float(a.abs().max())
        rel = max_d / max_a if max_a > 0 else float('inf')
        print(f"  {name:14s}  max|Δ|={max_d:.6g}  mean|Δ|={mean_d:.6g}  max|A|={max_a:.4g}  rel={rel:.4g}")

    print("\n=== Comparison (should be ~bf16 noise) ===")
    _cmp("xq", xq_orig, slot.xq)
    _cmp("xo", xo_orig, slot.xo)
    _cmp("x1", x1_orig, slot.x1)
    _cmp("x3", x3_orig, slot.x3)
    _cmp("attn_result", attn_result_orig, slot.attn_result)

    # Also check residual-stream y: call bwd path?
    # For simplicity, skip bwd parity here — this test is forward-only.


if __name__ == "__main__":
    main()
