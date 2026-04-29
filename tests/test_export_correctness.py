"""End-to-end correctness test for ``flextrain.export``.

Uses Llama-3.2-1B (small enough to load on CPU + GPU side-by-side) to
validate that the three export paths produce HF checkpoints / adapters
that, when loaded by ``transformers`` (and PEFT), generate the SAME
top-1 token as the in-memory FlexTrain model after a few steps of
LoRA fine-tuning.

What we check
-------------
1. ``save_hf_full`` on a freshly-loaded (untrained) base reproduces
   the original model: HF re-load → top-1 logits match the
   downloaded base.
2. After 5 LoRA steps:
   a. ``save_hf_merged`` → HF re-load top-1 == FT top-1.
   b. ``save_lora_adapter`` → HF + PEFT re-load top-1 == FT top-1.

Tolerance: we compare top-1 token (argmax over vocab). bf16/fp16
arithmetic differences mean strict logit match is too tight.

Run
---
::

    python tests/test_export_correctness.py --model models/Llama-3.2-1B
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ft_top1_next_token(am, prompt_ids: list[int]) -> int:
    """Forward through FlexTrain and return argmax next-token id."""
    import time
    from flextrain.engine.schedule import prepare_training_chunks
    from flextrain.ops import flextrain_rmsnorm_fwd

    device = am.device
    head_cfg = am.head.cfg
    hw = am.buffers.gpu_head_params
    w_final_norm = hw["w_final_norm"]
    w_head_proj = hw["w_head_proj"]
    rms_eps = float(head_cfg.rms_norm_eps)

    class _Seq:
        def __init__(self, t):
            self.tokens = t
            T = len(t)
            self.targets = torch.zeros(T, dtype=torch.int64)
            self.per_token_loss = torch.zeros(T, dtype=torch.float32)
            self.seq_id = 0

        def __len__(self):
            return len(self.tokens)

    n_layers = len(am.backbone)
    N_P = am.working_set.n_gpu_layers
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

    tokens_t = torch.tensor(prompt_ids, dtype=torch.int64)
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

    last_chunk = prepared.chunks[-1]
    x = am.buffers.transitions[last_chunk.id]
    n_in_chunk = int(last_chunk.meta.total_q)
    last_pos_in_chunk = n_in_chunk - 1
    x_last = x[last_pos_in_chunk : last_pos_in_chunk + 1, :]
    head_proj_in, _rstd = flextrain_rmsnorm_fwd(
        x_last, W=w_final_norm, rms_norm_eps=rms_eps,
    )
    logits = torch.mm(head_proj_in, w_head_proj)
    return int(logits.argmax(dim=-1).item())


_HF_TOP1_WORKER = """
import json, sys, torch
from transformers import AutoModelForCausalLM
model_dir, prompt_json, adapter_dir = sys.argv[1], sys.argv[2], sys.argv[3]
prompt_ids = json.loads(prompt_json)
m = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16,
).to("cuda").eval()
if adapter_dir != "":
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, adapter_dir).to("cuda").eval()
inp = torch.tensor([prompt_ids], dtype=torch.int64, device="cuda")
with torch.no_grad():
    out = m(inp)
logits = out.logits[0, -1, :]
print(int(logits.argmax().item()))
"""


def _hf_top1_subprocess(
    model_dir: str, prompt_ids: list[int], adapter_dir: str = "",
) -> int:
    """Run HF (or HF+PEFT) reload in a subprocess so its CUDA context
    and weights aren't fighting FlexTrain's pinned host buffers / GPU
    residents for memory. Returns the argmax next-token id."""
    import json
    import subprocess

    res = subprocess.run(
        [
            sys.executable, "-c", _HF_TOP1_WORKER,
            model_dir, json.dumps(prompt_ids), adapter_dir,
        ],
        capture_output=True, text=True, timeout=600,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"HF reload subprocess failed (rc={res.returncode}):\n"
            f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
        )
    # The model loader prints progress bars; the last line is our answer.
    last_line = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1]
    return int(last_line)


def _hf_top1_next_token(model_dir: str, prompt_ids: list[int]) -> int:
    return _hf_top1_subprocess(model_dir, prompt_ids, adapter_dir="")


def _hf_peft_top1_next_token(
    base_dir: str, adapter_dir: str, prompt_ids: list[int],
) -> int:
    return _hf_top1_subprocess(base_dir, prompt_ids, adapter_dir=adapter_dir)


def _train_a_few_steps(am, n_steps: int = 5) -> list[float]:
    """Run a few synthetic-token training steps so the LoRA adapter
    has a non-trivial delta. Returns the loss trajectory."""
    from flextrain.io.sources import SyntheticTokenSource

    source = SyntheticTokenSource(
        vocab_size=int(am.dims["vocab_size"]),
        seq_lens=512,
        seed=0,
    )
    losses = []
    for step in range(n_steps):
        sequences = source.get_sequences(max_token_count=512)
        stats = am.fwd_bwd(sequences, total_tokens_per_step=512)
        am.step()
        mean_loss = stats.total_loss / max(1, stats.total_tokens)
        losses.append(mean_loss)
        print(f"  step {step+1}: loss={mean_loss:.4f}", flush=True)
    return losses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Llama-3.2-1B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-gpu-gib", type=float, default=22.0)
    ap.add_argument("--max-host-gib", type=float, default=80.0)
    ap.add_argument("--n-steps", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out-dir", default=None,
                    help="Persist exports here for inspection (default: tmp).")
    args = ap.parse_args()

    if args.out_dir is None:
        out_root = tempfile.mkdtemp(prefix="ft_export_test_")
    else:
        out_root = args.out_dir
        os.makedirs(out_root, exist_ok=True)
    print(f"export outputs at: {out_root}")

    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.export import save_hf_full, save_hf_merged, save_lora_adapter

    # Tokenize the prompt.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tok(args.prompt, add_special_tokens=True).input_ids
    print(f"prompt: {args.prompt!r} → {prompt_ids}")

    # ------------------------------------------------------------------
    # Phase 1: load base model with LoRA enabled, untrained.
    # ------------------------------------------------------------------
    print("\n[1/3] loading base + LoRA wrapper...")
    am = from_pretrained(
        args.model,
        optimizer=AdamW(AdamWHyperparams(lr=1e-3)),
        max_seq_len=512,
        max_global_batch_tokens=512,
        max_gpu_mem_bytes=int(args.max_gpu_gib * (1 << 30)),
        max_host_mem_bytes=int(args.max_host_gib * (1 << 30)),
        device="cuda:0",
        lora_targets="all",
        lora_rank=args.rank,
        lora_alpha=float(args.rank),
        verbose=False,
    )
    base_top1 = _ft_top1_next_token(am, prompt_ids)
    print(f"   FT (base, LoRA=0) top-1 next token: {base_top1} "
          f"({tok.decode([base_top1])!r})")

    # Sanity: with B=0, FT and base HF should match.
    base_export = os.path.join(out_root, "base-export")
    save_hf_full(am, base_export)

    # ------------------------------------------------------------------
    # Phase 2: train a few steps so LoRA delta is non-zero.
    # ------------------------------------------------------------------
    print("\n[2/3] training a few LoRA steps (synthetic tokens)...")
    _train_a_few_steps(am, n_steps=args.n_steps)
    trained_top1 = _ft_top1_next_token(am, prompt_ids)
    print(f"   FT (after {args.n_steps} steps) top-1: {trained_top1} "
          f"({tok.decode([trained_top1])!r})")

    # Save merged + adapter while FT still has its in-memory state.
    print("\n[3/3] writing exports...")
    merged_dir = os.path.join(out_root, "merged-export")
    save_hf_merged(am, merged_dir, keep_lora_after_merge=True)
    adapter_dir = os.path.join(out_root, "lora-adapter")
    save_lora_adapter(am, adapter_dir, base_model_name_or_path=args.model)

    # Free FT engine BEFORE loading any HF model — HF needs GPU room.
    print("   freeing FT engine to make room for HF reload...")
    del am
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Now do all three HF-side reloads.
    print("\n   HF reload (base export)...")
    hf_base_top1 = _hf_top1_next_token(base_export, prompt_ids)
    print(f"   HF (base) top-1: {hf_base_top1} ({tok.decode([hf_base_top1])!r})")

    print("   HF reload (merged export)...")
    hf_merged_top1 = _hf_top1_next_token(merged_dir, prompt_ids)
    print(f"   HF (merged) top-1: {hf_merged_top1} "
          f"({tok.decode([hf_merged_top1])!r})")

    print("   HF + PEFT (adapter)...")
    hf_peft_top1 = _hf_peft_top1_next_token(args.model, adapter_dir, prompt_ids)
    print(f"   HF+PEFT (adapter) top-1: {hf_peft_top1} "
          f"({tok.decode([hf_peft_top1])!r})")

    base_ok = hf_base_top1 == base_top1
    merged_ok = hf_merged_top1 == trained_top1
    adapter_ok = hf_peft_top1 == trained_top1

    print("\n" + "=" * 60)
    print(f"base      : FT={base_top1} HF={hf_base_top1} "
          f"{'PASS' if base_ok else 'FAIL'}")
    print(f"merged    : FT={trained_top1} HF={hf_merged_top1} "
          f"{'PASS' if merged_ok else 'FAIL'}")
    print(f"adapter   : FT={trained_top1} HF+PEFT={hf_peft_top1} "
          f"{'PASS' if adapter_ok else 'FAIL'}")
    overall = base_ok and merged_ok and adapter_ok
    print(f"OVERALL   : {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
