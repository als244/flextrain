"""HF logit capture with greedy generation — run on a HF-friendly box.

Loads HF model, applies chat template (or raw tokenization) to
``--prompt``, then GREEDY-decodes until EOS or ``--max-new-tokens``.
At each step we capture the next-token logit. After generation we
also run one big forward over the full final sequence to make sure
prompt-position logits are populated. Saves a bundle compatible with
``tests/ft_replay_compare_logits.py``.

Bundle format
-------------
A torch ``.pt`` file with these keys:

* ``input_ids``: 1D LongTensor of shape ``(T,)`` — full sequence
  (prompt + greedy-generated tokens).
* ``prompt_T``: int — prompt length, so ``generated = input_ids[prompt_T:]``.
* ``logits``: bf16 tensor of shape ``(T, V)`` — at index ``i``, the
  logit vector that predicts ``input_ids[i+1]`` (or, for ``i == T-1``,
  what *would* be predicted next). Sources:
    - ``i in [0, prompt_T - 1]`` from the up-front prompt forward.
    - ``i in [prompt_T, T - 1]`` from the per-generation-step forward.
  This way the replay side just runs ONE FT fwd over ``input_ids``
  and compares position-by-position.
* ``last_argmax_per_step``: 1D LongTensor of shape ``(T_gen,)`` — the
  argmax that produced each generated token. Useful sanity check.
* ``model``, ``model_path_arg``, ``prompt``, ``no_chat_template``,
  ``vocab_size``, ``dtype_used_for_fwd``: metadata.

Usage:

    PYTHONPATH=. python tests/hf_capture_logits.py \\
        --model models/Qwen3.5-27B \\
        --prompt "What is the capital of France?" \\
        --max-new-tokens 256 \\
        --out hf_capture_27b.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model dir (e.g. models/Qwen3.5-27B)")
    ap.add_argument("--prompt", required=True,
                    help="User-message text. Will be chat-templated by default.")
    ap.add_argument("--out", required=True,
                    help="Output .pt path.")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="Max greedy-generation length. Stops earlier on EOS.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                    default="bfloat16",
                    help="Model dtype (default bf16 — matches FT).")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="Bypass chat template; tokenize prompt as raw text.")
    ap.add_argument("--max-prompt-tokens", type=int, default=None,
                    help="Cap prompt length post-chat-template (truncate). "
                         "If unset, use full chat-templated prompt.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"=== HF capture: {args.model} ===")
    print(f"  device={args.device}  dtype={args.dtype}  max_new={args.max_new_tokens}")

    print("  loading tokenizer ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    if args.no_chat_template:
        prompt_ids = list(tok.encode(args.prompt, add_special_tokens=True))
    else:
        msgs = [{"role": "user", "content": args.prompt}]
        enc = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
        )
        prompt_ids = list(enc["input_ids"]) if isinstance(enc, dict) else list(enc)

    if args.max_prompt_tokens is not None and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[: args.max_prompt_tokens]

    prompt_T = len(prompt_ids)
    prompt_decoded = tok.decode(prompt_ids)
    print(f"  prompt: {prompt_T} tokens")
    print(f"  prompt input_ids: {prompt_ids}")
    print(f"  prompt decoded:")
    print("    " + prompt_decoded.replace("\n", "\n    "))

    print("  loading model ...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(args.device)
    hf.eval()
    n_params_b = sum(p.numel() for p in hf.parameters()) / 1e9
    print(f"  model loaded; {n_params_b:.2f}B params")

    eos_id = tok.eos_token_id
    if eos_id is None:
        print("  WARN: tokenizer has no eos_token_id; using max-new-tokens only.")

    # ---- Greedy generation, capturing per-step next-token logits. ----
    cur_ids = list(prompt_ids)
    gen_logits_list: list[torch.Tensor] = []  # one (V,) per generated token
    last_argmax_per_step: list[int] = []

    t0 = time.time()
    for step in range(args.max_new_tokens):
        ids_t = torch.tensor([cur_ids], dtype=torch.int64, device=args.device)
        with torch.no_grad():
            out = hf(input_ids=ids_t, output_hidden_states=False, use_cache=False)
        # logits: (1, T, V); last position predicts the next token.
        next_logits = out.logits[0, -1, :].detach().to(
            dtype=torch.bfloat16, device="cpu",
        ).contiguous()
        gen_logits_list.append(next_logits)

        next_id = int(next_logits.argmax().item())
        last_argmax_per_step.append(next_id)
        cur_ids.append(next_id)

        if step < 3 or (step + 1) % 16 == 0:
            print(
                f"  [step {step+1:4d}] next_id={next_id} "
                f"({tok.decode([next_id])!r})  elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

        if eos_id is not None and next_id == eos_id:
            print(f"  EOS hit at step {step + 1}", flush=True)
            break

    n_gen = len(gen_logits_list)
    print(f"  generated {n_gen} tokens in {time.time()-t0:.1f}s")

    # ---- Final pass: full-sequence forward to grab prompt-position logits. ----
    # The autoregressive loop only captured logits at each step's last
    # position. For positions [0, prompt_T - 2] we want logits too, so
    # rerun a single forward over the full sequence and grab those.
    full_ids = cur_ids
    T = len(full_ids)
    print(f"  running final full-sequence forward (T={T}) ...", flush=True)
    ids_t = torch.tensor([full_ids], dtype=torch.int64, device=args.device)
    with torch.no_grad():
        out = hf(input_ids=ids_t, output_hidden_states=False, use_cache=False)
    full_logits = out.logits[0].detach().to(
        dtype=torch.bfloat16, device="cpu",
    ).contiguous()  # (T, V)

    # Sanity: the per-step captured next-token logits should match the
    # full-sequence forward's same positions. (HF eager fwd is
    # deterministic in fp/bf math; identity should hold modulo any
    # internal kv-cache batching paths, which we disabled via
    # use_cache=False.) Compute the deviation as a fact-check.
    if n_gen > 0:
        gen_stack = torch.stack(gen_logits_list, dim=0)   # (n_gen, V)
        # gen_stack[i] is the next-token logit at sequence-context-len
        # = prompt_T + i  (i.e. position prompt_T + i - 1 in full
        # sequence = position whose successor is generated_ids[i]).
        # In full_logits this is full_logits[prompt_T + i - 1] for
        # i = 0..n_gen-1. Special case i=0: position prompt_T - 1.
        ref = full_logits[prompt_T - 1 : prompt_T - 1 + n_gen]  # (n_gen, V)
        if ref.shape == gen_stack.shape:
            diff = (ref.float() - gen_stack.float()).abs()
            rel = float(diff.max().item()) / max(
                float(ref.float().norm().item()), 1e-12,
            )
            print(
                f"  internal sanity: stepwise vs full-fwd logit max|Δ|="
                f"{diff.max().item():.4f}  rel={rel:.3e}",
                flush=True,
            )

    gen_ids = full_ids[prompt_T:]
    gen_text = tok.decode(gen_ids) if n_gen > 0 else ""
    full_text = tok.decode(full_ids)

    bundle = {
        "model": os.path.basename(os.path.normpath(args.model)),
        "model_path_arg": args.model,
        "prompt": args.prompt,
        "prompt_decoded": prompt_decoded,            # str — what HF saw
        "no_chat_template": args.no_chat_template,
        "input_ids": torch.tensor(full_ids, dtype=torch.int64),
        "prompt_T": prompt_T,
        # Convenience: explicit output-only token list. Equivalent to
        # input_ids[prompt_T:] but saved separately so you can grep
        # the bundle without slicing.
        "output_ids": torch.tensor(gen_ids, dtype=torch.int64),
        "output_decoded": gen_text,                  # str
        "full_decoded": full_text,                   # prompt + generated
        "logits": full_logits,                       # (T, V) bf16
        "last_argmax_per_step": torch.tensor(
            last_argmax_per_step, dtype=torch.int64,
        ),
        "vocab_size": int(full_logits.shape[-1]),
        "dtype_used_for_fwd": str(dtype),
        "n_generated": n_gen,
    }
    torch.save(bundle, args.out)
    size_mib = os.path.getsize(args.out) / (1 << 20)
    print(f"  saved -> {args.out}  ({size_mib:.1f} MiB)")
    print(f"  T_total = {T}  (prompt={prompt_T}  generated={n_gen})")
    print()
    print(f"  output_ids ({n_gen}): {gen_ids}")
    print()
    print(f"  === FULL DECODED (prompt + generated) ===")
    print(full_text)
    print(f"  === END ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
