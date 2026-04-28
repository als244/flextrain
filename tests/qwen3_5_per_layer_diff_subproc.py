"""Per-layer FT vs HF activation diff with subprocess isolation.

Runs HF and FT in separate Python processes (each gets a fresh CUDA
context, no GPU-residency carryover), each writes its captured layer
outputs + final_norm + logits to a .pt file. The orchestrator loads
both and reports diffs.

Usage:
    PYTHONPATH=. python tests/qwen3_5_per_layer_diff_subproc.py \
        --model models/Qwen3.5-9B --max-seq-len 64
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _hf_capture(model_path, prompt_ids, out_path):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    hf.eval()
    lm = hf.model.language_model if hasattr(hf.model, "language_model") else hf.model
    text_layers = lm.layers
    n_layers = len(text_layers)

    layer_outputs = [None] * n_layers
    handles = []
    for i, layer in enumerate(text_layers):
        def _mk(idx):
            def hook(m, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                layer_outputs[idx] = t.detach().clone()
            return hook
        handles.append(layer.register_forward_hook(_mk(i)))

    embed_out = [None]
    eh = lm.embed_tokens.register_forward_hook(
        lambda m, i, o: embed_out.__setitem__(0, o.detach().clone())
    )
    final_norm_out = [None]
    nh = lm.norm.register_forward_hook(
        lambda m, i, o: final_norm_out.__setitem__(0, o.detach().clone())
    )

    ids = torch.tensor([prompt_ids], dtype=torch.int64).cuda()
    with torch.no_grad():
        out = hf(input_ids=ids, output_hidden_states=False, use_cache=False)
    for h in handles: h.remove()
    eh.remove(); nh.remove()

    captured = {
        "embed": embed_out[0].squeeze(0).contiguous().cpu(),
        "final_norm": final_norm_out[0].squeeze(0).contiguous().cpu(),
        "logits": out.logits.detach().squeeze(0).contiguous().cpu(),
        "n_layers": n_layers,
    }
    for i, t in enumerate(layer_outputs):
        captured[f"layer_{i}"] = t.squeeze(0).contiguous().cpu()
    cfg = hf.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    captured["layer_types"] = list(cfg.layer_types)
    torch.save(captured, out_path)
    print(f"  HF: captured {n_layers} layers, saved to {out_path}")


def _ft_capture(args, prompt_ids, out_path):
    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.engine.schedule import prepare_training_chunks
    from flextrain.ops import flextrain_rmsnorm_fwd

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
        strict=False, verbose=True,
        # Allow tiny chunk sizes for inference-style probes; the
        # arithmetic-intensity heuristic would otherwise reject sizes
        # smaller than ~512 tokens for a 9B model.
        min_chunk_size=1,
    )

    class _Seq:
        def __init__(self, t):
            self.tokens = t
            T = len(t)
            self.targets = torch.zeros(T, dtype=torch.int64)
            self.per_token_loss = torch.zeros(T, dtype=torch.float32)
            self.seq_id = 0
        def __len__(self): return len(self.tokens)

    seq = _Seq(torch.tensor(prompt_ids, dtype=torch.int64))
    prepared = prepare_training_chunks(
        [seq], max_chunk_size=am.working_set.max_chunk_size,
        device=am.device, policy=am.chunk_policy,
    )
    am._allocate_moe_chunk_scratch(prepared)
    am.events.clear_per_round()
    plan = am._plan_save_levels(prepared)
    am.streams.compute.synchronize()
    am._setup_round(prepared, plan)

    n_layers = len(am.backbone)
    captured_layers = [None] * n_layers
    originals = []
    for i, layer in enumerate(am.backbone):
        orig = layer.forward
        originals.append((layer, orig))
        def _mk(idx, fn):
            def w(x, c, w_, s, ctx_):
                y = fn(x, c, w_, s, ctx_)
                captured_layers[idx] = y.detach().clone()
                return y
            return w
        layer.forward = _mk(i, orig)

    am._forward_pass(prepared, plan)
    am.streams.compute.synchronize()
    for layer, orig in originals: layer.forward = orig

    last_chunk = prepared.chunks[-1]
    x = am.buffers.transitions[last_chunk.id]
    head_weights = am.buffers.gpu_head_params
    rms_eps = float(am.head.cfg.rms_norm_eps)
    head_proj_in, _ = flextrain_rmsnorm_fwd(
        x, W=head_weights["w_final_norm"], rms_norm_eps=rms_eps,
    )
    final_norm = head_proj_in.detach().clone()
    logits = torch.mm(head_proj_in, head_weights["w_head_proj"]).detach().clone()
    embed_w = am.buffers.gpu_embed_params.get("w_tok_embeddings")
    if embed_w is None:
        embed_w = am.buffers.host_embed_params["w_tok_embeddings"]
    embed = embed_w[torch.tensor(prompt_ids, dtype=torch.int64, device="cuda:0")].clone()

    captured = {
        "embed": embed.cpu(),
        "final_norm": final_norm.cpu(),
        "logits": logits.cpu(),
        "n_layers": n_layers,
    }
    for i, t in enumerate(captured_layers):
        captured[f"layer_{i}"] = t.cpu()
    torch.save(captured, out_path)
    print(f"  FT: captured {n_layers} layers, saved to {out_path}")


def _stats(name, ref, got, layer_type=""):
    if ref.shape != got.shape:
        print(f"  {name:32s} {layer_type:14s} SHAPE MISMATCH ref={tuple(ref.shape)} got={tuple(got.shape)}")
        return float("inf"), float("inf")
    diff = (ref.float() - got.float()).abs()
    max_abs = float(diff.max().item())
    rel = max_abs / max(float(ref.float().norm().item()), 1e-12)
    flag = "OK" if rel < 5e-3 else ("DIVERGE" if rel < 0.1 else "BAD")
    print(f"  {name:32s} {layer_type:14s} max|Δ|={max_abs:9.4f}  "
            f"rel={rel:.3e}  [{flag}]", flush=True)
    return max_abs, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--prompt", default="Four score and")
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--gpu-gib", type=float, default=22.0)
    ap.add_argument("--host-gib", type=float, default=100.0)
    ap.add_argument("--mode", choices=["both", "hf", "ft"], default="both")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra-tokens", type=int, default=0,
                    help="Append N tokens after prompt (uses HF's known greedy agreement).")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [{"role": "user", "content": args.prompt}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    prompt_ids = list(enc["input_ids"])
    if getattr(args, "extra_tokens", 0):
        # Append HF's first N greedy tokens after the prompt, so we can
        # probe FT at progressive context lengths past the chat template.
        FT_HF_AGREEMENT = [90700]   # both stacks predicted 'Thinking' at T=13
        prompt_ids = prompt_ids + FT_HF_AGREEMENT[:args.extra_tokens]

    if args.mode == "hf":
        _hf_capture(args.model, prompt_ids, args.out)
        return 0
    if args.mode == "ft":
        _ft_capture(args, prompt_ids, args.out)
        return 0

    # Orchestrate
    print(f"Prompt ({len(prompt_ids)} tokens): {repr(tok.decode(prompt_ids))}")
    hf_pt = tempfile.NamedTemporaryFile(suffix=".hf.pt", delete=False).name
    ft_pt = tempfile.NamedTemporaryFile(suffix=".ft.pt", delete=False).name

    base_cmd = [sys.executable, sys.argv[0],
                "--model", args.model, "--prompt", args.prompt,
                "--max-seq-len", str(args.max_seq_len),
                "--gpu-gib", str(args.gpu_gib),
                "--host-gib", str(args.host_gib),
                "--extra-tokens", str(args.extra_tokens)]
    print("=== Capturing HF (subprocess) ===")
    subprocess.run(base_cmd + ["--mode", "hf", "--out", hf_pt], check=True)
    print("=== Capturing FT (subprocess) ===")
    subprocess.run(base_cmd + ["--mode", "ft", "--out", ft_pt], check=True)

    hf_cap = torch.load(hf_pt, weights_only=False)
    ft_cap = torch.load(ft_pt, weights_only=False)
    n_layers = hf_cap["n_layers"]
    layer_types = hf_cap["layer_types"]

    print()
    print("=== Per-layer diff (FT vs HF) ===")
    _stats("embed", hf_cap["embed"], ft_cap["embed"])
    rels = []
    earliest = None
    for i in range(n_layers):
        _, rel = _stats(f"layer_{i}", hf_cap[f"layer_{i}"], ft_cap[f"layer_{i}"], layer_type=layer_types[i])
        rels.append(rel)
        if earliest is None and rel > 5e-3:
            earliest = i
    _stats("final_norm", hf_cap["final_norm"], ft_cap["final_norm"])
    _stats("logits", hf_cap["logits"], ft_cap["logits"])

    hf_arg = hf_cap["logits"][-1].argmax().item()
    ft_arg = ft_cap["logits"][-1].argmax().item()
    print(f"\n  HF argmax of last position: {hf_arg} ({repr(tok.decode([hf_arg]))})")
    print(f"  FT argmax of last position: {ft_arg} ({repr(tok.decode([ft_arg]))})")
    if earliest is not None:
        print(f"\n  EARLIEST DIVERGENCE: layer {earliest} ({layer_types[earliest]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
