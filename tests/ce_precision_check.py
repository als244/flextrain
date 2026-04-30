"""Isolated precision check: FT CE pipeline vs HF CE pipeline on the
SAME logits + labels. Tells us whether the residual ~0.1 step-1 loss
gap between FT and HF-PEFT in test_arch_lora_e2e is precision (FT
softmaxes in fp32 then casts to bf16 before log; HF upcasts logits
to fp32 and stays in fp32 through log) or a real bug.

Usage:
    PYTHONPATH=. python tests/ce_precision_check.py

What it does:
    1. Synthesize a (T, V) bf16 logits tensor from a Qwen3.5-2B
       embedding-norm-ish distribution (random with realistic scale).
    2. Build labels = random valid token ids.
    3. Compute CE via FT path (flextrain_softmax + flextrain_cross_entropy_loss).
    4. Compute CE via HF path (logits.float(); cross_entropy_loss).
    5. Compare per-position L values.

If FT and HF disagree by ~0.001-0.01 average on ~248k vocab → confirms
the bf16-cast-after-softmax precision hypothesis.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.ops import flextrain_softmax, flextrain_cross_entropy_loss


def _ft_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """FT CE: bf16 logits -> fp32 softmax -> cast back to bf16 ->
    -log(probs[label]). Mirrors flextrain/nn/loss.py:CrossEntropyLoss."""
    probs = torch.empty_like(logits)
    aux_idx = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
    aux_val = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    probs, _, _ = flextrain_softmax(
        logits, out=probs, max_idx_out=aux_idx, max_val_out=aux_val,
    )
    L = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
    _dZ, L = flextrain_cross_entropy_loss(probs, labels, L=L)
    return L


def _hf_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """HF CE: cast logits to fp32, then cross_entropy. Mirrors
    transformers.loss.loss_utils.fixed_cross_entropy."""
    L = F.cross_entropy(logits.float(), labels, reduction="none")
    return L


def main() -> int:
    torch.manual_seed(0)
    device = "cuda:0"
    T = 256
    V = 248320  # Qwen3.5-2B vocab
    # Realistic logit scale: roughly N(0, 5) with a few peaks; resembles
    # what a small model's last-layer head produces.
    logits_fp32 = torch.randn(T, V, device=device) * 5.0
    # Add a peak at a "preferred" token for each row to mimic the
    # confident-prediction regime where bf16 vs fp32 log() differs most.
    peaks = torch.randint(0, V, (T,), device=device)
    boost = torch.randn(T, device=device).abs() * 8.0 + 4.0
    logits_fp32[torch.arange(T), peaks] += boost
    logits_bf16 = logits_fp32.to(torch.bfloat16).contiguous()
    labels = torch.randint(0, V, (T,), device=device, dtype=torch.int64)
    # 30% of labels = the peak (high confidence) so we sample both
    # confident and unconfident regimes.
    confident = torch.rand(T, device=device) < 0.3
    labels = torch.where(confident, peaks, labels)

    L_ft = _ft_ce(logits_bf16, labels)
    L_hf_bf16in = _hf_ce(logits_bf16, labels)
    L_hf_fp32in = _hf_ce(logits_fp32, labels)

    diff_ft_vs_hfbf = (L_ft - L_hf_bf16in).abs()
    diff_ft_vs_hffp = (L_ft - L_hf_fp32in).abs()

    print(f"=== CE precision check ({T=} {V=}) ===")
    print(f"  L_ft         mean={L_ft.mean().item():.6f}  std={L_ft.std().item():.6f}")
    print(f"  L_hf_bf16in  mean={L_hf_bf16in.mean().item():.6f}  std={L_hf_bf16in.std().item():.6f}")
    print(f"  L_hf_fp32in  mean={L_hf_fp32in.mean().item():.6f}  std={L_hf_fp32in.std().item():.6f}")
    print()
    print(f"  |L_ft - L_hf_bf16in|.max  = {diff_ft_vs_hfbf.max().item():.6f}")
    print(f"  |L_ft - L_hf_bf16in|.mean = {diff_ft_vs_hfbf.mean().item():.6f}")
    print(f"  |L_ft - L_hf_fp32in|.max  = {diff_ft_vs_hffp.max().item():.6f}")
    print(f"  |L_ft - L_hf_fp32in|.mean = {diff_ft_vs_hffp.mean().item():.6f}")

    # The HF path used in test_arch_lora_e2e takes BF16 logits from the
    # forward and upcasts inside ForCausalLMLoss. The FT side has bf16
    # logits in the head and runs softmax in fp32 internally but writes
    # back as bf16 before flextrain_cross_entropy_loss. So the relevant
    # comparison is L_ft vs L_hf_bf16in (both starting from bf16 logits;
    # difference comes from where the fp32->bf16 cast happens).
    print()
    print("  Interpretation: if L_ft and L_hf_bf16in agree but L_ft and")
    print("  L_hf_fp32in disagree, the divergence is the bf16-after-softmax")
    print("  cast in FT (vs HF's fp32-throughout). If L_ft and L_hf_bf16in")
    print("  ALSO disagree by a similar margin, the divergence is the")
    print("  FT softmax kernel itself (vs torch's softmax).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
