"""LoRA init must give bit-identical step-0 output to full FT.

At initialization LoRA has ``A ~ N(0, σ)`` and ``B = 0``, so the
adapter delta ``X @ A @ B = 0``. The model output at step 0 should be
identical to the base model — same forward, same loss, same per-
parameter gradients on the base-block parameters. Any divergence
points at a real bug (e.g. an A/B init that's not zero, a LoRA forward
that mis-routes, a kernel that doesn't short-circuit on zero B).

This test is the regression guard for that invariant on the new
Gemma 3 LoRA path (and incidentally covers Gemma 2 / any other arch
with a registered LoRA wrapper). It also documents why the
verified-runs table sometimes shows small step-0 differences between
LoRA and full rows: those come from data-prefetch ordering in
``train.py``, NOT from engine numerics — when both modes see the
exact same batch, losses agree bit-for-bit.
"""
from __future__ import annotations

import gc
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


_SPECS = [
    ("Gemma-3-1B-Instruct", 8),
    ("Gemma-2-2B-Instruct", 10),
]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LoRA equivalence requires CUDA",
)


def _run_mode(model_dir: str, mode: str, ids, targets, active: int) -> float:
    from flextrain.api import from_pretrained
    from flextrain.bench.parity import _Seq
    from flextrain.optim import AdamW, AdamWHyperparams

    opt = AdamW(AdamWHyperparams(
        lr=1e-5, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
    ))
    kw = dict(
        max_seq_len=128, max_global_batch_tokens=512,
        max_gpu_mem_bytes=int(24 * 2**30),
        max_host_mem_bytes=int(60 * 2**30),
    )
    if mode == "lora":
        kw.update(lora_targets="all", lora_rank=16, lora_alpha=16.0)
    am = from_pretrained(model_dir, optimizer=opt, **kw)
    seq = _Seq(ids.clone())
    seq.targets = targets.clone()
    stats = am.fwd_bwd([seq], loss_scale_factor=1.0 / active, verbose=False)
    loss = stats.total_loss / active
    try:
        am.buffers.destroy()
    except Exception:
        pass
    del am
    gc.collect()
    torch.cuda.empty_cache()
    return float(loss)


@pytest.mark.parametrize("model_name,memory_gb", _SPECS)
def test_lora_step0_matches_full_bitwise(model_name: str, memory_gb: int):
    """LoRA init has B=0 → adapter delta is zero → step-0 forward
    must produce the same loss as full FT on the same data."""
    md = os.path.join(ROOT, "models", model_name)
    if not os.path.isdir(md):
        pytest.skip(f"{model_name} not present under models/")
    free_b, _ = torch.cuda.mem_get_info()
    if free_b < memory_gb * 2**30:
        pytest.skip(f"{model_name}: need ~{memory_gb} GB free")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(md)
    prompt = (
        "Instruction:\nWhat is 6 times 7? Show steps.\n\n"
        "Response:\nFirst step: I will multiply 6 by 7. "
        "6 * 7 = 42. The answer is 42."
    )
    ids = torch.tensor(
        tok(prompt, add_special_tokens=False).input_ids
        + [int(tok.eos_token_id)],
        dtype=torch.int64,
    )
    prompt_only = (
        "Instruction:\nWhat is 6 times 7? Show steps.\n\n"
        "Response:\n"
    )
    plen = len(tok(prompt_only, add_special_tokens=False).input_ids)
    targets = torch.roll(ids, -1)
    targets[: plen - 1] = -100
    targets[-1] = -100
    active = int((targets != -100).sum().item())
    assert active > 0, "no active response tokens — test data is broken"

    loss_lora = _run_mode(md, "lora", ids, targets, active)
    loss_full = _run_mode(md, "full", ids, targets, active)

    # bit-identical: LoRA at init = base model exactly.
    assert loss_lora == loss_full, (
        f"{model_name}: LoRA step-0 ({loss_lora:.6f}) != full step-0 "
        f"({loss_full:.6f}); diff={abs(loss_lora - loss_full):.6e}. "
        f"This means the LoRA forward path is applying a non-zero delta "
        f"at initialization, or A/B aren't initialized to the LoRA "
        f"convention (A ~ N(0,σ), B=0). The verified-runs table's "
        f"step-0 LoRA-vs-full gap should be from data-prefetch race, "
        f"not from numerics — if THIS test fails, the gap is real."
    )
