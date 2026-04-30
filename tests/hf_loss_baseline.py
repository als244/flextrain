"""Compute HF reference cross-entropy loss on a mathinstruct chunk for
direct comparison against FT step-1 LoRA loss.

If FT step-1 LoRA loss (1.87 for Qwen3.5-27B) matches the HF reference
loss for the same tokens, then 27B's loss is "correct" — the base model
just performs differently on this dataset. If HF's loss is much lower,
then FT has a bug.

For 27B we do CPU offloading via accelerate's auto-device-map (or just
CPU+GPU split) to fit in 24GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_chunk_tokens(model_path: str, dataset_path: str, target: int) -> list[int]:
    """Tokenize records from JSONL until we hit target tokens. Returns
    a single concatenated id list."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    bos = tok.bos_token_id
    ids: list[int] = []
    if bos is not None:
        ids.append(bos)
    with open(dataset_path) as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text") or rec.get("instruction", "") or ""
            if not text:
                continue
            piece = tok(text, add_special_tokens=False)["input_ids"]
            ids.extend(piece)
            if len(ids) >= target:
                break
    return ids[:target]


def _hf_loss(model_path: str, ids: list[int]) -> float:
    """Run HF eager fwd, compute next-token CE loss."""
    from transformers import AutoModelForCausalLM
    print(f"  loading HF model from {model_path} ...", flush=True)
    # Try GPU; if OOM fall back to CPU.
    try:
        hf = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        device = "cuda:0"
        print(f"  loaded on GPU", flush=True)
    except torch.OutOfMemoryError:
        print(f"  OOM on GPU; falling back to CPU", flush=True)
        hf = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        device = "cpu"
    hf.eval()
    ids_t = torch.tensor([ids], dtype=torch.int64, device=device)
    print(f"  running fwd (T={len(ids)}) ...", flush=True)
    with torch.no_grad():
        out = hf(input_ids=ids_t, output_hidden_states=False, use_cache=False)
    logits = out.logits[0]  # (T, V)
    # Next-token CE: predict ids[1:] from logits[:-1].
    # Match the FT loss path: per-token CE summed then divided by tokens.
    targets = ids_t[0, 1:]  # (T-1,)
    pred = logits[:-1, :]  # (T-1, V)
    loss = torch.nn.functional.cross_entropy(
        pred.float(), targets, reduction="mean",
    )
    return float(loss.item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="datasets/mathinstruct.jsonl")
    ap.add_argument("--tokens", type=int, default=4096)
    args = ap.parse_args()

    print(f"=== HF reference loss for {args.model} ===")
    ids = _load_chunk_tokens(args.model, args.dataset, args.tokens)
    print(f"  tokens: {len(ids)}")
    loss = _hf_loss(args.model, ids)
    print(f"  HF cross-entropy loss: {loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
