"""HF capture: tokenize a long real document, run HF eager/sdpa fwd,
save per-position logits to disk for later FT comparison.

The FT replay script then loads the bundle, runs FT multi-chunk fwd,
and writes a parallel bundle. A separate compare script does the
post-hoc parity analysis. Splitting fwd-then-compare across processes
keeps GPU memory footprints sane on a 24 GB card at 32k tokens.

Usage:
    python tests/multi_chunk_dense_parity/hf_capture.py \\
        --model models/Llama-3.2-1B \\
        --fixture tests/fixtures/long_real_sample.txt \\
        --max-tokens 32000 \\
        --out tests/multi_chunk_logs/Llama-3.2-1B__hf.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _tokenize(model_path: str, fixture_path: str, max_tokens: int) -> tuple[list[int], object]:
    from transformers import AutoTokenizer
    with open(fixture_path) as f:
        text = f.read()
    tok = AutoTokenizer.from_pretrained(model_path)
    bos = tok.bos_token_id
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if bos is not None:
        ids = [bos] + ids
    if max_tokens is not None and max_tokens > 0:
        ids = ids[:max_tokens]
    print(f"  tokenized to {len(ids)} ids (head: {ids[:6]} ...)")
    return ids, tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True,
                    help="Long-text fixture (single document)")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn-impl", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = ap.parse_args()

    print(f"=== HF capture: {args.model} ===")
    ids, _tok = _tokenize(args.model, args.fixture, args.max_tokens)

    from transformers import AutoModelForCausalLM
    print(f"  loading HF model (attn={args.attn_impl}) ...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to(args.device)
    hf.eval()
    ids_t = torch.tensor([ids], dtype=torch.int64, device=args.device)

    print(f"  running HF forward (T={len(ids)}) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = hf(input_ids=ids_t, output_hidden_states=False, use_cache=False)
    logits = out.logits.detach().squeeze(0).contiguous().to(torch.bfloat16).cpu()
    print(f"  HF fwd done in {time.time()-t0:.1f}s, logits {tuple(logits.shape)}")

    bundle = {
        "model": args.model,
        "fixture": args.fixture,
        "input_ids": torch.tensor(ids, dtype=torch.int64),
        "logits": logits,                         # (T, V) bf16 cpu
        "attn_impl": args.attn_impl,
        "num_input_tokens": len(ids),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(bundle, args.out)
    print(f"  saved -> {args.out}  ({os.path.getsize(args.out) / (1<<20):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
