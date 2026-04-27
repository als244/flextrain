r"""Generic FT-vs-HF arch parity diagnostic on REAL DATA.

For each registered architecture with locally-available HF weights,
runs both **LoRA fine-tuning** and **full-parameter fine-tuning** on
real MathInstruct prompts (tokenized by the target model's own
tokenizer) and compares loss curves between FlexTrain and a
reference HF + (PEFT or torch.optim) stack.

What's compared per arch
------------------------
* Loss curve over N training steps (HF vs FT).
* Step 0 logit max\|Δ\| and per-token-CE max\|Δ\| (forward parity floor).
* Step N final loss agreement (both stacks should descend similarly).

Two modes per arch
------------------
* ``lora`` — both stacks use LoRA on every linear projection
  (target_modules = q/k/v/o/gate/up/down). FT auto-inits A/B; we
  copy the same A values into HF PEFT and B=0 in both.
* ``full`` — both stacks update every parameter. No LoRA. Test that
  full fine-tuning agrees too.

Usage::

    python tests/test_arch_parity.py                            # all available
    python tests/test_arch_parity.py Qwen2.5-0.5B               # one model
    python tests/test_arch_parity.py Qwen2.5-0.5B --mode lora   # one model, lora only
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16
LORA_R = 16
LORA_ALPHA = 16.0
N_STEPS = 5
TOKENS_PER_STEP = 512   # small per-step token budget so this runs quickly
LR_LORA = 1e-4
LR_FULL = 5e-6
SEED = 0

# HF arch IDs we know how to do PEFT for.
_HF_TARGETS = {
    "LlamaForCausalLM": ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"),
    "MistralForCausalLM": ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"),
    "Qwen2ForCausalLM": ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"),
    "Qwen3ForCausalLM": ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"),
    "Gemma2ForCausalLM": ("q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"),
    "Gemma3ForCausalLM": ("q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"),
}
_FT2HF = {
    "w_q": "q_proj", "w_k": "k_proj", "w_v": "v_proj", "w_o": "o_proj",
    "w_1": "gate_proj", "w_3": "up_proj", "w_2": "down_proj",
}


# ===========================================================================
# HF subprocess
# ===========================================================================


def _hf_worker(hf_path, mode, init_pkl, batches_pkl, out_pkl):
    """HF training-loop reference. Mode: 'lora' or 'full'."""
    import torch as _t
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    with open(os.path.join(hf_path, "config.json")) as f:
        cfg = json.load(f)
    arch_id = cfg["architectures"][0]
    if arch_id == "Gemma3ForConditionalGeneration":
        arch_id = "Gemma3ForCausalLM"
    targets_list = list(_HF_TARGETS[arch_id])

    with open(batches_pkl, "rb") as f:
        batches = pickle.load(f)

    model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="sdpa",
    )

    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
            target_modules=targets_list, init_lora_weights=False,
        )
        model = get_peft_model(model, lora_cfg)

        # Replay FT's auto-init A values + zero B.
        with open(init_pkl, "rb") as f:
            init = pickle.load(f)
        with _t.no_grad():
            for (L, ft_name), A in init.items():
                hf_name = _FT2HF[ft_name]
                if hf_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    parent = model.model.model.layers[L].self_attn
                else:
                    parent = model.model.model.layers[L].mlp
                lora = getattr(parent, hf_name, None)
                if lora is None:
                    continue
                lora.lora_A["default"].weight.data.copy_(
                    A.t().to(lora.lora_A["default"].weight.dtype).to(DEVICE)
                )
                lora.lora_B["default"].weight.data.zero_()
        trainable = [p for p in model.parameters() if p.requires_grad]
        lr = LR_LORA
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = list(model.parameters())
        lr = LR_FULL

    model.train()
    opt = _t.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    losses = []
    step0_logits_list = []
    step0_per_token_list = []

    for step, batch in enumerate(batches):
        opt.zero_grad(set_to_none=False)
        step_loss_sum = 0.0
        step_active_total = 0
        # Process one seq at a time, concatenating logits/per-token-CE
        # across the whole batch on step 0 so the shape matches FT's
        # captured (sum_T, V) tensor.
        for tokens_cpu, targets_cpu in batch:
            tokens = tokens_cpu.to(DEVICE).unsqueeze(0)
            our_targets = targets_cpu.to(DEVICE)
            T_seq = int(our_targets.shape[0])
            hf_labels = _t.full((T_seq,), -100, dtype=_t.int64, device=DEVICE)
            hf_labels[1:] = our_targets[:-1]
            active = int((hf_labels != -100).sum().item())
            out = model(input_ids=tokens, labels=hf_labels.unsqueeze(0))
            (out.loss * active).backward()
            step_loss_sum += float(out.loss.item()) * active
            step_active_total += active

            if step == 0:
                step0_logits_list.append(out.logits[0].detach().to(_t.float32).cpu())
                pt = F.cross_entropy(
                    out.logits[0].detach().float(),
                    our_targets, reduction="none", ignore_index=-100,
                )
                step0_per_token_list.append(
                    (pt * (our_targets != -100).float()).cpu()
                )

        for p in trainable:
            if p.grad is not None:
                p.grad.div_(step_active_total)
        opt.step()
        losses.append(step_loss_sum / max(1, step_active_total))

    payload = {
        "losses": losses,
        "step0_logits": (
            torch.cat(step0_logits_list, dim=0) if step0_logits_list else None
        ),
        "step0_per_token": (
            torch.cat(step0_per_token_list, dim=0) if step0_per_token_list else None
        ),
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)


def _run_hf(hf_path, mode, init, batches):
    with tempfile.TemporaryDirectory() as td:
        init_pkl = os.path.join(td, "i.pkl")
        bat_pkl = os.path.join(td, "b.pkl")
        out_pkl = os.path.join(td, "o.pkl")
        if init is not None:
            with open(init_pkl, "wb") as f:
                pickle.dump(init, f)
        with open(bat_pkl, "wb") as f:
            pickle.dump(
                [[(s.tokens.cpu(), s.targets.cpu()) for s in b] for b in batches],
                f,
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--hf-worker", hf_path, mode, init_pkl, bat_pkl, out_pkl],
            check=True, env=env,
        )
        with open(out_pkl, "rb") as f:
            return pickle.load(f)


# ===========================================================================
# FT subprocess
# ===========================================================================


def _ft_worker(hf_path, mode, batches_pkl, out_pkl):
    from flextrain import from_pretrained
    from flextrain.bench.parity import _Seq, _flextrain_step
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.nn.loss import CrossEntropyLoss

    class _LogitsCapture(CrossEntropyLoss):
        def __init__(self):
            super().__init__()
            self.first_step = []
            self.captured_first = False
        def compute(self, logits, token_slice, *, loss_scale, per_token_loss_out):
            if not self.captured_first:
                self.first_step.append(logits.detach().to(torch.float32).cpu().clone())
            return super().compute(
                logits, token_slice,
                loss_scale=loss_scale, per_token_loss_out=per_token_loss_out,
            )

    with open(batches_pkl, "rb") as f:
        batches_raw = pickle.load(f)

    # Reconstruct seqs.
    batches = []
    for raw in batches_raw:
        b = []
        for tokens_cpu, targets_cpu in raw:
            seq = _Seq(tokens_cpu)
            seq.targets = targets_cpu
            b.append(seq)
        batches.append(b)

    if mode == "lora":
        opt = AdamW(
            AdamWHyperparams(lr=LR_LORA, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
            state_dtype=torch.float32,
        )
        am = from_pretrained(
            hf_path, optimizer=opt,
            max_seq_len=512, max_global_batch_tokens=TOKENS_PER_STEP,
            max_gpu_mem_bytes=int(20 * (1 << 30)),
            max_host_mem_bytes=int(110 * (1 << 30)),
            leeway_gpu_mem_bytes=int(2 * (1 << 30)),
            leeway_host_mem_bytes=int(4 * (1 << 30)),
            device=DEVICE,
            lora_targets="all", lora_rank=LORA_R, lora_alpha=LORA_ALPHA,
            lora_adapter_compute_dtype=torch.bfloat16,
            lora_adapter_master_dtype=torch.float32,
            lora_adapter_grad_dtype=torch.float32,
            lora_adapter_opt_state_dtype=torch.float32,
        )
    else:
        opt = AdamW(
            AdamWHyperparams(lr=LR_FULL, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
            state_dtype=torch.float32,
        )
        am = from_pretrained(
            hf_path, optimizer=opt,
            max_seq_len=512, max_global_batch_tokens=TOKENS_PER_STEP,
            max_gpu_mem_bytes=int(20 * (1 << 30)),
            max_host_mem_bytes=int(110 * (1 << 30)),
            leeway_gpu_mem_bytes=int(2 * (1 << 30)),
            leeway_host_mem_bytes=int(4 * (1 << 30)),
            device=DEVICE,
        )
    n_layers = len(am.backbone)

    # If LoRA, capture init A values for HF replay.
    init = {}
    if mode == "lora":
        for L in range(n_layers):
            host = am.buffers.host_params[L]
            for ft_name in _FT2HF:
                a_key = f"{ft_name}_lora_a"
                if a_key in host:
                    init[(L, ft_name)] = host[a_key].detach().float().cpu().clone()

    capture = _LogitsCapture()
    losses = []
    step0_logits = None
    step0_per_token = None

    for step, batch in enumerate(batches):
        # Capture step-0 logits via the loss fn.
        if step == 0:
            capture.captured_first = False
            capture.first_step.clear()

        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        # Use plain _flextrain_step on step >= 1 to advance.  On step 0
        # we want the captured logits, so call fwd_bwd + step manually
        # with our capturing loss_fn.
        if step == 0:
            active = sum(int((s.targets != -100).sum().item()) for s in seqs)
            stats = am.fwd_bwd(
                seqs, loss_scale_factor=1.0 / max(1, active), verbose=False,
                loss_fn=capture,
            )
            am.step()
            loss = stats.total_loss / max(1, active)
            # Gather per-token loss across all seqs in batch.
            per_token_concat = torch.cat([s.per_token_loss.detach().clone() for s in seqs])
            step0_per_token = per_token_concat
            if capture.first_step:
                step0_logits = torch.cat(capture.first_step, dim=0)
            capture.captured_first = True
        else:
            loss = _flextrain_step(am, seqs)
        losses.append(loss)
        torch.cuda.synchronize()

    payload = {
        "init": init if mode == "lora" else None,
        "losses": losses,
        "step0_logits": step0_logits,
        "step0_per_token": step0_per_token,
        "n_layers": n_layers,
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)


def _run_ft(hf_path, mode, batches):
    with tempfile.TemporaryDirectory() as td:
        bat_pkl = os.path.join(td, "b.pkl")
        out_pkl = os.path.join(td, "o.pkl")
        with open(bat_pkl, "wb") as f:
            pickle.dump(
                [[(s.tokens.cpu(), s.targets.cpu()) for s in b] for b in batches],
                f,
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--ft-worker", hf_path, mode, bat_pkl, out_pkl],
            check=True, env=env,
        )
        with open(out_pkl, "rb") as f:
            return pickle.load(f)


# ===========================================================================
# Per-model run
# ===========================================================================


def _diff_summary(name, a, b):
    if a is None or b is None:
        return None
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
        return None
    delta = (a - b).abs()
    out = {
        "max": float(delta.max().item()),
        "mean": float(delta.mean().item()),
        "ref_max": float(b.abs().max().item()),
    }
    print(f"  {name:20s} max|Δ|={out['max']:.3e}  mean|Δ|={out['mean']:.3e}  ref|max|={out['ref_max']:.3e}")
    return out


def run_model(model_dir: str, modes=("lora", "full")) -> dict:
    from tests.test_llama32_1b_parity import _pull_step_batches

    hf_path = os.path.join(ROOT, "models", model_dir)
    if not os.path.isdir(hf_path):
        print(f"\n=== SKIP {model_dir} (not at {hf_path}) ===")
        return {"status": "skip"}

    with open(os.path.join(hf_path, "config.json")) as f:
        cfg = json.load(f)
    arch_id = cfg["architectures"][0]
    if arch_id == "Gemma3ForConditionalGeneration":
        arch_id = "Gemma3ForCausalLM"
    if arch_id not in _HF_TARGETS:
        print(f"\n=== SKIP {model_dir}: unknown arch {arch_id} ===")
        return {"status": "skip"}

    # Gemma 2 / Gemma 3 currently have ``forward`` but a stubbed
    # ``backward`` (the dual-residual-norm bwd is in flight). Skip
    # training-mode parity until the bwd lands.
    if arch_id in ("Gemma2ForCausalLM", "Gemma3ForCausalLM"):
        print(f"\n=== SKIP {model_dir} ({arch_id}) — bwd is stubbed; needs handwritten bwd. See SESSION_NOTES. ===")
        return {"status": "skip", "reason": "Gemma 2/3 bwd not yet implemented"}

    # Skip full-FT mode for very large models that would OOM HF's
    # plain torch.optim.AdamW path (no offloading on the HF side).
    n_params = (cfg.get("text_config") or cfg).get("num_hidden_layers", 0) * (cfg.get("text_config") or cfg).get("hidden_size", 0)
    if n_params > 4096 * 28 and "full" in modes:  # >~ 1B-equivalent
        print(f"  (skipping full-FT mode for {model_dir}: too large for HF on a single 24 GiB GPU)")
        modes = tuple(m for m in modes if m != "full")

    print(f"\n{'='*78}")
    print(f"=== {model_dir} ({arch_id})")
    print(f"{'='*78}")

    # Build batches once (deterministic via _pull_step_batches' file order).
    batches = _pull_step_batches(
        hf_path, n_steps=N_STEPS, target_tokens_per_step=TOKENS_PER_STEP,
    )
    print(f"  built {len(batches)} batches × ~{TOKENS_PER_STEP} tokens/step")

    results = {"status": "ok", "arch": arch_id}

    for mode in modes:
        print(f"\n--- mode = {mode} ---")
        ft = _run_ft(hf_path, mode, batches)
        hf = _run_hf(hf_path, mode, ft.get("init"), batches)

        # Side-by-side loss curve.
        print(f"  step  HF       FT       Δ")
        for i, (h, f) in enumerate(zip(hf["losses"], ft["losses"])):
            print(f"  {i:>3d}   {h:8.4f} {f:8.4f}  {f-h:+.4f}")

        logit_diff = _diff_summary("step0 logits", ft["step0_logits"], hf["step0_logits"])
        per_tok_diff = _diff_summary("step0 per-token CE", ft["step0_per_token"], hf["step0_per_token"])

        max_loss_delta = max(abs(f - h) for h, f in zip(hf["losses"], ft["losses"]))
        results[mode] = {
            "hf_losses": hf["losses"],
            "ft_losses": ft["losses"],
            "max_loss_delta": max_loss_delta,
            "logit_diff": logit_diff,
            "per_tok_diff": per_tok_diff,
        }
        print(f"  ⇒ max |Δ_loss| over {N_STEPS} steps = {max_loss_delta:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=None)
    parser.add_argument("--mode", choices=("lora", "full", "both"), default="both")
    args = parser.parse_args()

    models = args.models or [
        "Llama-3.2-1B",
        "Qwen2.5-0.5B",
        "Qwen3-1.7B",
        "Mistral-7B-v0.3",
        "gemma-2-2b",
        "gemma-3-1b-pt",
    ]
    modes = ("lora",) if args.mode == "lora" else ("full",) if args.mode == "full" else ("lora", "full")

    summary = []
    for m in models:
        try:
            r = run_model(m, modes=modes)
            summary.append((m, r))
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary.append((m, {"status": "fail", "error": str(e)}))

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"{'model':25s} {'mode':6s} {'max|Δ_loss|':>12s} {'step0 max|Δlogit|':>20s}")
    for m, r in summary:
        if r.get("status") != "ok":
            print(f"{m:25s} {r.get('status', '?'):6s} {r.get('error', '')}")
            continue
        for mode in modes:
            md = r.get(mode)
            if md is None:
                continue
            ld = md["logit_diff"]["max"] if md["logit_diff"] else float("nan")
            print(f"{m:25s} {mode:6s} {md['max_loss_delta']:>12.4f} {ld:>20.3e}")


if __name__ == "__main__":
    if len(sys.argv) >= 7 and sys.argv[1] == "--hf-worker":
        _hf_worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    elif len(sys.argv) >= 6 and sys.argv[1] == "--ft-worker":
        _ft_worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main()
