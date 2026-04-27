"""Qwen3-Next forward parity vs HF transformers.

Constructs a small **heterogeneous** Qwen3-Next model with HF
transformers (random init), copies the weights into the equivalent FT
backbone, and compares forward output of:

* HF model end-to-end (logits + per-layer hidden states).
* FT engine end-to-end (logits via a logit-capturing CrossEntropy hook).

This is the rigorous correctness test for Qwen3-Next: it exercises
the full heterogeneous backbone (3 linear-attn + 1 full-attn) with
real HF weights, on the same input, and surfaces every divergence
between FT's and HF's chosen math.

Usage::

    python tests/test_qwen3_next_vs_hf.py

It is expected to FAIL initially. Each failure pinpoints a specific
bug in our Qwen3-Next implementation. The expected failures are:

1. ``(1 + weight)`` RMSNorm convention not applied at load time.
2. Full-attention output gate (``q_proj`` outputs ``n_heads*head_dim*2``)
   not implemented in ``GQAAttentionBlock``.
3. Partial-rotary RoPE (``partial_rotary_factor=0.25``) not in our
   RoPE kernel.
4. Shared-expert path (``shared_expert`` + ``shared_expert_gate``) not
   in our MoE FFN.

Each fix lands once this test (or a focused subset of it) shows the
specific divergence reduced.
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


def _build_hf_mini():
    """A 4-layer mini Qwen3-Next: [L, L, L, F]. Random-init, deterministic
    via fixed seed. Returns ``(model, config)``."""
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    cfg = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,        # dense MLP path (only used if mlp_only_layers); here MoE is always on
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_value_heads=8,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=64,
        rope_theta=500_000.0,
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        full_attention_interval=4,
        partial_rotary_factor=0.25,
        attention_bias=False,
        norm_topk_prob=False,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        tie_word_embeddings=False,
        torch_dtype=DTYPE,
        attn_implementation="eager",  # avoid flash-attn API surprises
    )
    torch.manual_seed(20260427)
    model = Qwen3NextForCausalLM(cfg).eval().to(DEVICE, dtype=DTYPE)
    return model, cfg


def main():
    print("Building HF mini-Qwen3-Next ...", flush=True)
    hf, cfg = _build_hf_mini()
    n_params = sum(p.numel() for p in hf.parameters()) / 1e6
    print(f"  params: {n_params:.2f} M, n_layers={cfg.num_hidden_layers}, layer_types={cfg.layer_types}", flush=True)

    # ----- Build inputs -----
    torch.manual_seed(0)
    T = 32
    tokens = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.int64, device=DEVICE)

    # ----- HF forward, capturing per-layer hidden states + logits -----
    print("\n=== HF forward ===")
    with torch.no_grad():
        out = hf(input_ids=tokens, output_hidden_states=True, use_cache=False)
    hf_logits = out.logits[0].detach().to(torch.float32).cpu()
    hf_hidden = [h[0].detach().to(torch.float32).cpu() for h in out.hidden_states]
    print(f"  logits shape: {tuple(hf_logits.shape)}, max|x|={float(hf_logits.abs().max().item()):.3e}")
    print("  per-layer hidden state magnitudes:")
    for i, h in enumerate(hf_hidden):
        label = "embed" if i == 0 else f"after layer {i-1} ({cfg.layer_types[i-1] if i-1 < len(cfg.layer_types) else 'final-norm'})"
        print(f"    {label:>40s}: max|h|={float(h.abs().max().item()):.3e}  mean|h|={float(h.abs().mean().item()):.3e}")

    # We'll add the FT equivalent build below once we've identified the
    # required pieces. For now this script confirms HF runs and prints
    # the reference hidden-state magnitudes.
    print("\n(FT side not yet wired in. Next step: build FT mini-model with the same weights.)")


if __name__ == "__main__":
    main()
