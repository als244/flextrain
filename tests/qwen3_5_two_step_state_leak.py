"""Run two consecutive forward passes through the same FT engine instance,
mimicking the greedy decode loop. Capture the layer outputs of BOTH passes
and compare against HF (computed standalone for the same inputs).

If pass 0 matches HF and pass 1 does NOT match HF, that confirms FT
state leak between steps. We can then bisect which buffer/state
persisted.

Inputs:
  pass 0: prompt (T=13)
  pass 1: prompt + 'Thinking' (T=14)
"""
from __future__ import annotations
import argparse, os, sys, subprocess, tempfile
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _hf_capture_logits(model_path, prompt_ids, out_path):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    hf.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.int64).cuda()
    with torch.no_grad():
        out = hf(input_ids=ids, use_cache=False)
    last_logits = out.logits[0, -1, :].detach().cpu()
    torch.save(last_logits, out_path)


def _ft_two_step(args, prompt_ids, hf_token, out_path):
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
        strict=False, verbose=False, min_chunk_size=1,
    )

    class _Seq:
        def __init__(self, t):
            self.tokens = t
            T = len(t)
            self.targets = torch.zeros(T, dtype=torch.int64)
            self.per_token_loss = torch.zeros(T, dtype=torch.float32)
            self.seq_id = 0
        def __len__(self): return len(self.tokens)

    head_weights = am.buffers.gpu_head_params
    rms_eps = float(am.head.cfg.rms_norm_eps)

    def _step(tokens):
        seq = _Seq(torch.tensor(tokens, dtype=torch.int64))
        prepared = prepare_training_chunks(
            [seq], max_chunk_size=am.working_set.max_chunk_size,
            device=am.device, policy=am.chunk_policy,
        )
        am._allocate_moe_chunk_scratch(prepared)
        am.events.clear_per_round()
        plan = am._plan_save_levels(prepared)
        am.streams.compute.synchronize()
        am._setup_round(prepared, plan)

        # Capture layer outputs (residual stream after each layer).
        n_layers = len(am.backbone)
        outs = [None] * n_layers
        originals = []
        for i, layer in enumerate(am.backbone):
            orig = layer.forward
            originals.append((layer, orig))
            def _mk(idx, fn):
                def w(x, c, w_, s, ctx_):
                    y = fn(x, c, w_, s, ctx_)
                    outs[idx] = y.detach().clone().cpu()
                    return y
                return w
            layer.forward = _mk(i, orig)
        am._forward_pass(prepared, plan)
        am.streams.compute.synchronize()
        for layer, orig in originals: layer.forward = orig

        last_chunk = prepared.chunks[-1]
        x = am.buffers.transitions[last_chunk.id]
        head_proj_in, _ = flextrain_rmsnorm_fwd(
            x, W=head_weights["w_final_norm"], rms_norm_eps=rms_eps,
        )
        last_logits = torch.mm(head_proj_in, head_weights["w_head_proj"])[-1].cpu()
        return last_logits, outs

    # Pass 0: T=13 prompt
    logits0, layers0 = _step(prompt_ids)
    next0 = int(logits0.argmax().item())
    print(f"FT step 0 (T={len(prompt_ids)}): predicted token {next0}")

    # Pass 1: T=14 (prompt + step-0 prediction)
    tokens1 = list(prompt_ids) + [next0]
    logits1, layers1 = _step(tokens1)
    next1 = int(logits1.argmax().item())
    print(f"FT step 1 (T={len(tokens1)}): predicted token {next1}")

    torch.save({
        "logits0": logits0,
        "logits1": logits1,
        "next0": next0,
        "next1": next1,
        "layers0": layers0,
        "layers1": layers1,
        "tokens0": prompt_ids,
        "tokens1": tokens1,
    }, out_path)


def _ft_one_pass_t14(args, tokens, out_path):
    """Run a fresh FT engine with ONE forward pass at the given tokens.
    Captures all 32 layer outputs. Compare against the second pass of
    a two-step FT run to detect state leak between steps."""
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
        strict=False, verbose=False, min_chunk_size=1,
    )

    class _Seq:
        def __init__(self, t):
            self.tokens = t
            T = len(t)
            self.targets = torch.zeros(T, dtype=torch.int64)
            self.per_token_loss = torch.zeros(T, dtype=torch.float32)
            self.seq_id = 0
        def __len__(self): return len(self.tokens)

    seq = _Seq(torch.tensor(tokens, dtype=torch.int64))
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
    outs = [None] * n_layers
    originals = []
    for i, layer in enumerate(am.backbone):
        orig = layer.forward
        originals.append((layer, orig))
        def _mk(idx, fn):
            def w(x, c, w_, s, ctx_):
                y = fn(x, c, w_, s, ctx_)
                outs[idx] = y.detach().clone().cpu()
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
    last_logits = torch.mm(head_proj_in, head_weights["w_head_proj"])[-1].cpu()
    next_id = int(last_logits.argmax().item())
    print(f"FT fresh T={len(tokens)}: predicted token {next_id}")
    torch.save({"logits": last_logits, "layers": outs, "next": next_id}, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--prompt", default="Four score and")
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--gpu-gib", type=float, default=22.0)
    ap.add_argument("--host-gib", type=float, default=100.0)
    ap.add_argument("--mode", choices=["both", "ft_two", "ft_fresh"], default="both")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [{"role": "user", "content": args.prompt}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    prompt_ids = list(enc["input_ids"])

    if args.mode == "ft_two":
        _ft_two_step(args, prompt_ids, None, args.out)
        return 0
    if args.mode == "ft_fresh":
        # tokens passed via env
        tokens = [int(t) for t in os.environ["TOKENS_LIST"].split(",")]
        _ft_one_pass_t14(args, tokens, args.out)
        return 0

    # Orchestrate: two ways of running T=14 forward in FT, compare layer outputs.
    print(f"Prompt ({len(prompt_ids)} tokens): {repr(tok.decode(prompt_ids))}")
    two_pt = tempfile.NamedTemporaryFile(suffix=".two.pt", delete=False).name
    fresh_pt = tempfile.NamedTemporaryFile(suffix=".fresh.pt", delete=False).name

    base_cmd = [sys.executable, sys.argv[0],
                "--model", args.model, "--prompt", args.prompt,
                "--max-seq-len", str(args.max_seq_len),
                "--gpu-gib", str(args.gpu_gib), "--host-gib", str(args.host_gib)]

    print("=== Path A: two-step FT (T=13 then T=14, same engine) ===")
    subprocess.run(base_cmd + ["--mode", "ft_two", "--out", two_pt], check=True)
    two = torch.load(two_pt, weights_only=False)
    print(f"  step 0 next={two['next0']} ({repr(tok.decode([two['next0']]))})")
    print(f"  step 1 next={two['next1']} ({repr(tok.decode([two['next1']]))})")

    print("=== Path B: fresh-engine FT at T=14 (prompt + step-0 token) ===")
    tokens14 = two["tokens1"]
    env = os.environ.copy()
    env["TOKENS_LIST"] = ",".join(str(t) for t in tokens14)
    subprocess.run(base_cmd + ["--mode", "ft_fresh", "--out", fresh_pt], check=True, env=env)
    fresh = torch.load(fresh_pt, weights_only=False)
    print(f"  fresh next={fresh['next']} ({repr(tok.decode([fresh['next']]))})")

    print()
    print("=== Compare two-step.layers1 vs fresh.layers (both T=14 forward) ===")
    if two["next1"] == fresh["next"]:
        print("  Same prediction — no state leak.")
    else:
        print(f"  DIFFERENT: two-step={two['next1']}, fresh={fresh['next']} -- state leak!")
    n = min(len(two["layers1"]), len(fresh["layers"]))
    earliest = None
    for i in range(n):
        a = two["layers1"][i].float()
        b = fresh["layers"][i].float()
        if a.shape != b.shape:
            print(f"  layer_{i}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        d = (a - b).abs()
        rel = d.max().item() / max(a.norm().item(), 1e-12)
        flag = "OK" if rel < 1e-5 else ("DIVERGE" if rel < 1e-2 else "BAD")
        print(f"  layer_{i:2d}  max|Δ|={d.max().item():.4e}  rel={rel:.3e}  [{flag}]")
        if earliest is None and rel >= 1e-5:
            earliest = i
    diff_l = (two["logits1"] - fresh["logits"]).abs()
    print(f"  logits(last)  max|Δ|={diff_l.max().item():.4e}  rel={diff_l.max().item()/max(two['logits1'].norm().item(),1e-9):.3e}")
    if earliest is not None:
        print(f"\n  EARLIEST DIFFERENCE: layer {earliest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
