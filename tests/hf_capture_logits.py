"""HF logit capture — run on a machine that fits the model in HF.

Loads HF model, applies chat template to ``--prompt``, runs single
forward pass with ``attn_implementation="eager"``, saves
``input_ids`` and full per-position ``logits`` to a torch ``.pt`` file.

Pair with ``tests/ft_replay_compare_logits.py`` which consumes the
saved file and runs FlexTrain on the same token ids, then compares.

Usage (on Hopper machine):

    PYTHONPATH=. python tests/hf_capture_logits.py \\
        --model /path/to/Qwen3.5-27B \\
        --prompt "What is the capital of France?" \\
        --out hf_capture_27b.pt

Notes
-----
* Uses chat template (matches what training / inference would do).
* Captures full ``logits`` tensor — for a 27B vocab=248320 at
  T=~30 tokens that's ~30 * 248320 * 2 bytes = ~15 MiB. Easy to ship.
* For longer prompts the file size scales linearly with T. Add
  ``--max-tokens`` to cap the prompt length if needed.
* Forces text-only collapse of mRoPE by passing default 2D
  position_ids (HF expands to 3-axis with all axes equal).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model dir (e.g. models/Qwen3.5-27B)")
    ap.add_argument("--prompt", required=True,
                    help="User-message text. Will be chat-templated.")
    ap.add_argument("--out", required=True,
                    help="Output .pt path.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Cap prompt length post-chat-template (truncate). "
                         "If unset, use full chat-templated prompt.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                    default="bfloat16",
                    help="Model dtype (default bf16 — matches FT).")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="Bypass chat template; tokenize prompt as raw text.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"=== HF capture: {args.model} ===")
    print(f"  device={args.device}  dtype={args.dtype}")

    print("  loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.no_chat_template:
        prompt_ids = tok.encode(args.prompt, add_special_tokens=True)
    else:
        msgs = [{"role": "user", "content": args.prompt}]
        enc = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
        )
        prompt_ids = list(enc["input_ids"]) if isinstance(enc, dict) else list(enc)

    if args.max_tokens is not None and len(prompt_ids) > args.max_tokens:
        prompt_ids = prompt_ids[: args.max_tokens]

    T = len(prompt_ids)
    print(f"  prompt: {T} tokens  head={prompt_ids[:8]}...")

    print("  loading model ...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    hf.eval()
    print(f"  model loaded; {sum(p.numel() for p in hf.parameters()) / 1e9:.2f}B params")

    ids_t = torch.tensor([prompt_ids], dtype=torch.int64, device=args.device)

    print(f"  running forward (T={T}) ...", flush=True)
    with torch.no_grad():
        out = hf(input_ids=ids_t, output_hidden_states=False, use_cache=False)
    logits = out.logits.detach().squeeze(0).contiguous()  # (T, V)
    # Always store on CPU bf16 — keeps file size manageable + interop'd.
    logits_cpu = logits.to(dtype=torch.bfloat16, device="cpu").contiguous()

    # Sanity: argmax of last position (gives "what HF would generate next").
    last_arg = int(logits_cpu[-1].argmax().item())
    last_decoded = tok.decode([last_arg])
    print(f"  last-position argmax: {last_arg} ({last_decoded!r})")
    print(f"  logits.shape = {tuple(logits_cpu.shape)}  dtype={logits_cpu.dtype}")
    print(f"  logits.norm = {logits_cpu.float().norm().item():.4f}")

    bundle = {
        "model": os.path.basename(os.path.normpath(args.model)),
        "model_path_arg": args.model,
        "prompt": args.prompt,
        "no_chat_template": args.no_chat_template,
        "input_ids": torch.tensor(prompt_ids, dtype=torch.int64),
        "logits": logits_cpu,
        "vocab_size": int(logits_cpu.shape[-1]),
        "last_argmax": last_arg,
        "dtype_used_for_fwd": str(dtype),
    }
    torch.save(bundle, args.out)
    print(f"  saved -> {args.out}  "
          f"(~{os.path.getsize(args.out) / (1<<20):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
