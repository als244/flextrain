"""Greedy generation parity: FlexTrain Qwen3.5-2B vs HF transformers.

Runs the same chat-templated prompt through both stacks under greedy
decoding (``do_sample=False``), prints both completions, and reports
the prefix length where they agree token-for-token. The earliest
divergence localizes the forward-pass bug if one exists.

Why greedy: greedy decode is deterministic, so two correct
implementations on identical weights/inputs should give identical
token streams up to the first floating-point tie that breaks
differently between cuBLAS reduction orders. In practice, divergence
within the first ~10 tokens means a structural bug; divergence after
50+ tokens is acceptable bf16 reorder noise.

Why FT generation is custom: ``ActiveModel`` only exposes
``fwd_bwd``. We drive its internal ``_forward_pass`` directly, then
run the head's RMSNorm + lm_head projection ourselves, argmax, and
append. Re-runs the engine forward each step (no KV cache yet).

Usage:
    PYTHONPATH=. python tests/qwen3_5_generate_compare.py \\
        --prompt "Four score and" --max-new-tokens 64
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


def _ft_greedy_generate(
    am, prompt_ids: list[int], *, max_new_tokens: int,
    eos_token_id: int | None, verbose: bool = False,
) -> list[int]:
    """Greedy decode from FlexTrain. Returns the full token sequence
    (prompt + generated). Stops at EOS or after ``max_new_tokens``."""
    from flextrain.engine.schedule import (
        prepare_training_chunks, ChunkPolicy,
    )
    from flextrain.ops import flextrain_rmsnorm_fwd

    device = am.device
    head_cfg = am.head.cfg
    head_weights = am.buffers.gpu_head_params
    w_final_norm = head_weights["w_final_norm"]
    w_head_proj = head_weights["w_head_proj"]
    rms_eps = float(head_cfg.rms_norm_eps)

    class _Seq:
        """Minimal duck-typed sequence for ``prepare_training_chunks``."""
        def __init__(self, t):
            self.tokens = t
            T = len(t)
            self.targets = torch.zeros(T, dtype=torch.int64)
            self.per_token_loss = torch.zeros(T, dtype=torch.float32)
            self.seq_id = 0

        def __len__(self):
            return len(self.tokens)

    out_ids = list(prompt_ids)
    t0 = time.time()
    n_layers = len(am.backbone)
    N_P = am.working_set.n_gpu_layers
    for step in range(max_new_tokens):
        # The engine assumes the first N_P layers' weights are resident
        # in slots 0..N_P-1 at the start of every fwd pass. ``fwd_bwd``
        # restores this invariant after each step (active_model.py:574-
        # 586); since we drive ``_forward_pass`` directly we must repeat
        # the restore between steps. Without it, step N>0 reads stale
        # weights from whichever layers happened to land in those slots
        # at the end of step N-1.
        if step > 0:
            am.events.weight_inbound.clear()
            with torch.cuda.stream(am.streams.inbound):
                for slot_idx in range(min(N_P, n_layers)):
                    layer_id = am.backbone[slot_idx].layer_id
                    am.buffers.fetch_layer_params(
                        layer_id, slot_idx, non_blocking=True,
                    )
                    am.events.weight_inbound.record_on(
                        layer_id, am.streams.inbound,
                    )
            am.streams.inbound.synchronize()

        tokens_t = torch.tensor(out_ids, dtype=torch.int64)
        seq = _Seq(tokens_t)

        prepared = prepare_training_chunks(
            [seq],
            max_chunk_size=am.working_set.max_chunk_size,
            device=device,
            policy=am.chunk_policy,
        )
        am._allocate_moe_chunk_scratch(prepared)
        am.events.clear_per_round()

        plan = am._plan_save_levels(prepared)
        am.streams.compute.synchronize()
        am._setup_round(prepared, plan)
        am._forward_pass(prepared, plan)
        am.streams.compute.synchronize()

        # Last chunk holds the position we want to predict from.
        last_chunk = prepared.chunks[-1]
        x = am.buffers.transitions[last_chunk.id]
        # x has shape (chunk_tokens, d_model). The token whose next
        # logits we want is the last one in the prompt; locate it
        # within this chunk.
        n_in_chunk = int(last_chunk.meta.total_q)
        last_pos_in_chunk = n_in_chunk - 1
        x_last = x[last_pos_in_chunk : last_pos_in_chunk + 1, :]

        # RMSNorm + lm_head projection (matches head.forward_backward).
        head_proj_in, _rstd = flextrain_rmsnorm_fwd(
            x_last, W=w_final_norm, rms_norm_eps=rms_eps,
        )
        logits = torch.mm(head_proj_in, w_head_proj)  # (1, V)
        next_id = int(logits.argmax(dim=-1).item())
        out_ids.append(next_id)

        if verbose and (step < 5 or step % 10 == 0):
            print(
                f"  [ft] step {step+1}: token {next_id} "
                f"(elapsed {time.time()-t0:.1f}s)", flush=True,
            )
        if eos_token_id is not None and next_id == eos_token_id:
            break
    return out_ids


def _hf_only(args, prompt_ids, eos_id, out_path):
    """Run HF greedy in this process, write tokens to ``out_path``."""
    from transformers import AutoModelForCausalLM
    import json
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    hf.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.int64).cuda()
    with torch.no_grad():
        out = hf.generate(
            ids, max_new_tokens=args.max_new_tokens, do_sample=False,
            eos_token_id=eos_id, pad_token_id=eos_id,
        )
    hf_ids = out[0].tolist()
    with open(out_path, "w") as f:
        json.dump({"tokens": hf_ids, "prompt_len": len(prompt_ids)}, f)
    print(f"  HF: generated {len(hf_ids) - len(prompt_ids)} new tokens")


def _ft_only(args, prompt_ids, eos_id, out_path):
    """Run FT greedy in this process, write tokens to ``out_path``."""
    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    import json
    am = from_pretrained(
        args.model,
        optimizer=AdamW(
            AdamWHyperparams(lr=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
            state_dtype=torch.bfloat16,
        ),
        max_seq_len=args.max_seq_len,
        max_global_batch_tokens=args.max_seq_len,
        max_gpu_mem_bytes=int(args.gpu_gib * (1 << 30)),
        max_host_mem_bytes=int(args.host_gib * (1 << 30)),
        device="cuda:0",
        leeway_gpu_mem_bytes=int(2 * (1 << 30)),
        leeway_host_mem_bytes=int(8 * (1 << 30)),
        strict=False, verbose=False,
        # Allow small chunk sizes (greedy inference uses tiny prompts).
        min_chunk_size=1,
    )
    ft_ids = _ft_greedy_generate(
        am, prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
        verbose=args.ft_verbose,
    )
    with open(out_path, "w") as f:
        json.dump({"tokens": ft_ids, "prompt_len": len(prompt_ids)}, f)
    print(f"  FT: generated {len(ft_ids) - len(prompt_ids)} new tokens")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-2B")
    ap.add_argument("--prompt", default="Four score and")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gpu-gib", type=float, default=20.0)
    ap.add_argument("--host-gib", type=float, default=80.0)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--ft-verbose", action="store_true")
    ap.add_argument(
        "--mode", choices=["both", "hf", "ft"], default="both",
        help="'both' spawns hf+ft as subprocesses (clean GPU between).",
    )
    ap.add_argument("--out", default=None,
                    help="(internal) when mode in {hf,ft}: tokens output path.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    eos_id = int(tok.eos_token_id)

    msgs = [{"role": "user", "content": args.prompt}]
    enc = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
    )
    prompt_ids = list(enc["input_ids"])

    if args.mode == "hf":
        _hf_only(args, prompt_ids, eos_id, args.out)
        return 0
    if args.mode == "ft":
        _ft_only(args, prompt_ids, eos_id, args.out)
        return 0

    # ---- mode == "both": orchestrate two subprocesses ----
    import subprocess, json, tempfile, sys
    print(f"Prompt (chat-templated, {len(prompt_ids)} tokens):")
    print(repr(tok.decode(prompt_ids)))
    print()

    hf_out = tempfile.NamedTemporaryFile(mode="w", suffix=".hf.json", delete=False)
    hf_out.close()
    ft_out = tempfile.NamedTemporaryFile(mode="w", suffix=".ft.json", delete=False)
    ft_out.close()

    base_cmd = [sys.executable, sys.argv[0],
                "--model", args.model,
                "--prompt", args.prompt,
                "--max-new-tokens", str(args.max_new_tokens),
                "--gpu-gib", str(args.gpu_gib),
                "--host-gib", str(args.host_gib),
                "--max-seq-len", str(args.max_seq_len)]
    if args.ft_verbose:
        base_cmd.append("--ft-verbose")

    print("=== HF greedy (subprocess) ===")
    subprocess.run(base_cmd + ["--mode", "hf", "--out", hf_out.name], check=True)
    hf_data = json.load(open(hf_out.name))
    hf_ids = hf_data["tokens"]
    hf_new = hf_ids[len(prompt_ids):]
    print(f"  generated {len(hf_new)} tokens; hit EOS: {hf_new and hf_new[-1] == eos_id}")
    print(repr(tok.decode(hf_ids)))
    print()

    print("=== FT greedy (subprocess) ===")
    subprocess.run(base_cmd + ["--mode", "ft", "--out", ft_out.name], check=True)
    ft_data = json.load(open(ft_out.name))
    ft_ids = ft_data["tokens"]
    ft_new = ft_ids[len(prompt_ids):]
    print(f"  generated {len(ft_new)} tokens; hit EOS: {ft_new and ft_new[-1] == eos_id}")
    print(repr(tok.decode(ft_ids)))
    print()

    print("=== Token-level agreement ===")
    print(f"  HF  new tokens ({len(hf_new)}): {hf_new[:30]}{'...' if len(hf_new)>30 else ''}")
    print(f"  FT  new tokens ({len(ft_new)}): {ft_new[:30]}{'...' if len(ft_new)>30 else ''}")
    agree = 0
    for i in range(min(len(hf_new), len(ft_new))):
        if hf_new[i] == ft_new[i]:
            agree += 1
        else:
            break
    print(f"  Greedy agreement prefix: {agree} tokens")
    if agree < len(hf_new) and agree < len(ft_new):
        print(f"  First divergence at step {agree}: "
                f"HF={hf_new[agree]} ({repr(tok.decode([hf_new[agree]]))})  "
                f"FT={ft_new[agree]} ({repr(tok.decode([ft_new[agree]]))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
