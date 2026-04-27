"""Llama-3.1-8B end-to-end LoRA E2E vs HF PEFT.

Same pattern as ``test_lora_e2e_llama_dense.py`` but with the 8B
model. The 8B base weights (~16 GiB bf16) plus activations don't fit
in 24 GiB GPU at full precision, so the solver picks offloading
automatically.

Two runs with identical LoRA inits:

* **HF PEFT reference** (subprocess) — `LlamaForCausalLM.from_pretrained` +
  `peft.LoraConfig` on every linear projection.
* **FT** — solver picks the working-set config based on actual machine
  memory (24 GiB GPU + ~120 GiB host on this box). LoRA's frozen-base
  trick eliminates per-base-param grad+opt allocations, so memory is
  dominated by the (~16 GiB) base weights and activation buffer.

Steps: 50, tokens/step: 1024.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import _Seq, _flextrain_step

from tests.test_llama32_1b_parity import (  # noqa: E402
    _halved_to_pair_perm,
    _permute_qk_for_pair_interleave,
    _pull_step_batches,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16

FT_TO_HF = {
    "w_q": "q_proj", "w_k": "k_proj", "w_v": "v_proj", "w_o": "o_proj",
    "w_1": "gate_proj", "w_3": "up_proj", "w_2": "down_proj",
}
LORA_TARGET_NAMES = tuple(FT_TO_HF.keys())

LORA_R = 16
LORA_ALPHA = 16.0
N_STEPS = 50
TARGET_TOKENS_PER_STEP = 1024
LR = 1e-4


def _gen_lora_init_values(hf_path: str) -> dict:
    """Same recipe as the 1B test: A ~ N(0, 0.02), B = 0."""
    with open(os.path.join(hf_path, "config.json")) as f:
        cfg = json.load(f)
    d_model = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim") or (d_model // n_heads)
    inter = cfg["intermediate_size"]
    n_layers = cfg["num_hidden_layers"]
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    in_dim = {"w_q": d_model, "w_k": d_model, "w_v": d_model, "w_o": attn_dim,
              "w_1": d_model, "w_2": inter, "w_3": d_model}
    out_dim = {"w_q": attn_dim, "w_k": kv_dim, "w_v": kv_dim, "w_o": d_model,
               "w_1": inter, "w_2": d_model, "w_3": inter}

    torch.manual_seed(20260424)
    init = {}
    for L in range(n_layers):
        for tgt in LORA_TARGET_NAMES:
            A = torch.empty(in_dim[tgt], LORA_R, dtype=torch.float32).normal_(std=0.02)
            B = torch.zeros(LORA_R, out_dim[tgt], dtype=torch.float32)
            init[(L, tgt)] = (A, B)
    return init


# ===========================================================================
# HF PEFT subprocess
# ===========================================================================


def _hf_peft_worker(hf_path, init_pkl, batch_pkl, out_pkl):
    import torch as _t
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    with open(init_pkl, "rb") as f:
        init = pickle.load(f)
    with open(batch_pkl, "rb") as f:
        batches = pickle.load(f)

    model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="sdpa",
    )

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        target_modules=list(FT_TO_HF.values()),
        init_lora_weights=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    n_layers = len(set(L for (L, _) in init))
    overwrite = 0
    with _t.no_grad():
        for L in range(n_layers):
            for tgt, hf_name in FT_TO_HF.items():
                A, B = init[(L, tgt)]
                if hf_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    parent = model.model.model.layers[L].self_attn
                else:
                    parent = model.model.model.layers[L].mlp
                lora_layer = getattr(parent, hf_name)
                lora_A = lora_layer.lora_A["default"].weight
                lora_B = lora_layer.lora_B["default"].weight
                lora_A.data.copy_(A.t().to(lora_A.dtype).to(lora_A.device))
                lora_B.data.copy_(B.t().to(lora_B.dtype).to(lora_B.device))
                overwrite += 1
    print(f"  [hf-worker] overwrote {overwrite} LoRA pairs", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = _t.optim.AdamW(
        trainable, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
    )

    losses = []
    t0 = time.time()
    for step, batch in enumerate(batches):
        opt.zero_grad(set_to_none=False)
        batch_loss = 0.0
        active_total = 0
        for tokens_cpu, targets_cpu in batch:
            tokens = tokens_cpu.to(DEVICE).unsqueeze(0)
            our_targets = targets_cpu.to(DEVICE)
            T = int(our_targets.shape[0])
            hf_labels = _t.full((T,), -100, dtype=_t.int64, device=DEVICE)
            hf_labels[1:] = our_targets[:-1]
            active = int((hf_labels != -100).sum().item())
            out = model(input_ids=tokens, labels=hf_labels.unsqueeze(0))
            (out.loss * active).backward()
            batch_loss += float(out.loss.item()) * active
            active_total += active
        for p in trainable:
            if p.grad is not None:
                p.grad.div_(active_total)
        opt.step()
        losses.append(batch_loss / max(1, active_total))
        if step < 3 or step % 5 == 0 or step == len(batches) - 1:
            print(
                f"  [hf-worker] step {step:3d}: loss={losses[-1]:.4f}  "
                f"elapsed={time.time()-t0:.1f}s", flush=True,
            )

    with open(out_pkl, "wb") as f:
        pickle.dump(losses, f)


def _run_hf_peft(hf_path, init, step_batches):
    print("\n=== HF PEFT (subprocess) ===")
    with tempfile.TemporaryDirectory() as td:
        init_pkl = os.path.join(td, "init.pkl")
        batch_pkl = os.path.join(td, "batches.pkl")
        out_pkl = os.path.join(td, "losses.pkl")
        with open(init_pkl, "wb") as f:
            pickle.dump(init, f)
        with open(batch_pkl, "wb") as f:
            pickle.dump(
                [[(s.tokens.cpu(), s.targets.cpu()) for s in b]
                 for b in step_batches], f,
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--hf-worker", hf_path, init_pkl, batch_pkl, out_pkl],
            check=True, env=env,
        )
        with open(out_pkl, "rb") as f:
            return pickle.load(f)


# ===========================================================================
# FT subprocess
# ===========================================================================


def _build_ft_model(hf_path, hf_cfg, init, *, label):
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import determine_working_set_config
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
    from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    d_model = hf_cfg["hidden_size"]
    n_heads = hf_cfg["num_attention_heads"]
    n_kv = hf_cfg["num_key_value_heads"]
    head_dim = hf_cfg.get("head_dim") or (d_model // n_heads)
    inter = hf_cfg["intermediate_size"]
    n_layers = hf_cfg["num_hidden_layers"]
    vocab = hf_cfg["vocab_size"]
    rope_theta = hf_cfg.get("rope_theta", 500_000.0)
    rms_eps = hf_cfg.get("rms_norm_eps", 1e-5)

    cfg = LlamaBlockConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv,
        head_dim=head_dim, expert_dim=inter,
        rms_norm_eps=rms_eps, rope_base=rope_theta,
        rope_scaling=hf_cfg.get("rope_scaling"),
        is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    dims = dict(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv,
        head_dim=head_dim, attn_dim=n_heads*head_dim,
        kv_dim=n_kv*head_dim, expert_dim=inter, vocab_size=vocab,
    )
    # Match HF PEFT dtypes: base bf16 (frozen), LoRA A/B fp32 master +
    # fp32 grad + fp32 AdamW state. PEFT keeps lora_A/lora_B in fp32 by
    # default while the base linear layer stays bf16, so we mirror that.
    backbone = []
    for i in range(n_layers):
        base = LlamaBlock(layer_id=i, cfg=cfg)
        wrapped = LoRAWrapperLayer(
            base, lora_targets="all",
            rank=LORA_R, alpha=LORA_ALPHA, dims=dims,
            adapter_compute_dtype=torch.bfloat16,
            adapter_master_dtype=torch.float32,
            adapter_grad_dtype=torch.float32,
            adapter_opt_state_dtype=torch.float32,
        )
        backbone.append(wrapped)

    embed = TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=vocab, d_model=d_model,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    ))
    head = LMHead(LMHeadConfig(
        d_model=d_model, vocab_size=vocab,
        rms_norm_eps=rms_eps, head_chunk_size=512,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    ))

    print(f"  [{label}] solving working set (auto: solver picks based on machine)...", flush=True)
    # No manual budget cap — solver inspects available GPU + host memory
    # and chooses the working-set config (n_gpu_layers, n_gpu_grads,
    # save level, chunk size) automatically. The big input is the
    # leeway, which gives the solver headroom.
    working_set = determine_working_set_config(
        model_dims=dict(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv,
            head_dim=head_dim, expert_dim=inter, vocab_size=vocab,
            n_layers=n_layers, num_shared_experts=1, num_routed_experts=0,
            top_k=0, is_causal=True,
            datatypes={"embed": "bfloat16", "head_proj": "bfloat16",
                       "attn_proj": "bfloat16", "expert_proj": "bfloat16",
                       "router": "bfloat16", "norm": "bfloat16",
                       "residual": "bfloat16"},
        ),
        max_seq_len=1024, max_global_batch_tokens=TARGET_TOKENS_PER_STEP,
        training_config={"master_weight_dtype": "bfloat16",
                         "grad_dtype": "bfloat16",
                         "opt_choice": "AdamW", "opt_dtype": "float32"},
        has_embed=True, has_head=True, num_local_layers=n_layers,
        max_gpu_mem_bytes=int(24 * (1 << 30)),
        max_host_mem_bytes=int(110 * (1 << 30)),
        leeway_gpu_mem_bytes=int(2 * (1 << 30)),
        leeway_host_mem_bytes=int(4 * (1 << 30)),
        verbose=True, fixed_seq_len=False,
    )
    print(
        f"  [{label}] solver: n_gpu_layers={working_set.n_gpu_layers}/{n_layers}, "
        f"target_round_tokens={working_set.target_round_tokens}, "
        f"act_buffer={working_set.gpu_act_buffer_size/(1<<30):.2f} GiB",
        flush=True,
    )

    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    # Match HF: fp32 AdamW state. Base is frozen so this only allocates
    # state for the LoRA A/B params; cost is small.
    opt = AdamW(
        AdamWHyperparams(lr=LR, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
        state_dtype=torch.float32,
    )
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=working_set, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )
    am.load_hf(hf_path, strict=False)

    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    q_perm = torch.tensor(_halved_to_pair_perm(attn_dim, head_dim), dtype=torch.int64)
    k_perm = torch.tensor(_halved_to_pair_perm(kv_dim, head_dim), dtype=torch.int64)
    for i in range(n_layers):
        for name in ("w_q", "w_k"):
            w = am.buffers.host_params[i][name]
            am.buffers.host_params[i][name].copy_(
                _permute_qk_for_pair_interleave(w, head_dim)
            )

    head_w = am.buffers.host_head_params.get("w_head_proj")
    embed_w = am.buffers.host_embed_params.get("w_tok_embeddings")
    if head_w is not None and embed_w is not None:
        if head_w.abs().sum().item() == 0.0:
            head_w.copy_(embed_w.t())

    for L in range(n_layers):
        for tgt in LORA_TARGET_NAMES:
            A_fp32, B_fp32 = init[(L, tgt)]
            host_A = am.buffers.host_params[L][f"{tgt}_lora_a"]
            host_B = am.buffers.host_params[L][f"{tgt}_lora_b"]
            host_A.copy_(A_fp32.to(host_A.dtype))
            host_B.copy_(B_fp32.to(host_B.dtype))
            if tgt == "w_q":
                am.buffers.host_params[L][f"{tgt}_lora_b"].copy_(host_B[:, q_perm])
            elif tgt == "w_k":
                am.buffers.host_params[L][f"{tgt}_lora_b"].copy_(host_B[:, k_perm])

    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()
    return am, n_layers


def _ft_worker(hf_path, init_pkl, batch_pkl, out_pkl, label):
    with open(init_pkl, "rb") as f:
        init = pickle.load(f)
    with open(batch_pkl, "rb") as f:
        batches_raw = pickle.load(f)
    step_batches = []
    for batch_raw in batches_raw:
        batch = []
        for tokens_cpu, targets_cpu in batch_raw:
            seq = _Seq(tokens_cpu)
            seq.targets = targets_cpu
            batch.append(seq)
        step_batches.append(batch)

    with open(os.path.join(hf_path, "config.json")) as f:
        hf_cfg = json.load(f)

    am, _ = _build_ft_model(hf_path, hf_cfg, init, label=label)
    losses = []
    t0 = time.time()
    for step, batch in enumerate(step_batches):
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        loss = _flextrain_step(am, seqs)
        losses.append(loss)
        if step < 3 or step % 5 == 0 or step == len(step_batches) - 1:
            max_alloc = torch.cuda.max_memory_allocated() / (1 << 30)
            print(
                f"  [{label}] step {step:3d}: loss={loss:.4f}  "
                f"elapsed={time.time()-t0:.1f}s  "
                f"max_alloc={max_alloc:.2f}GiB", flush=True,
            )

    with open(out_pkl, "wb") as f:
        pickle.dump(losses, f)


def _run_ft_subprocess(hf_path, init, step_batches, *, label):
    with tempfile.TemporaryDirectory() as td:
        init_pkl = os.path.join(td, "init.pkl")
        batch_pkl = os.path.join(td, "batches.pkl")
        out_pkl = os.path.join(td, "losses.pkl")
        with open(init_pkl, "wb") as f:
            pickle.dump(init, f)
        with open(batch_pkl, "wb") as f:
            pickle.dump(
                [[(s.tokens.cpu(), s.targets.cpu()) for s in b]
                 for b in step_batches], f,
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--ft-worker", hf_path, init_pkl, batch_pkl, out_pkl, label],
            check=True, env=env,
        )
        with open(out_pkl, "rb") as f:
            return pickle.load(f)


def main():
    hf_path = os.path.join(ROOT, "models", "Llama-3.1-8B")
    if not os.path.isdir(hf_path):
        raise FileNotFoundError(f"Llama-3.1-8B weights not found at {hf_path}")

    with open(os.path.join(hf_path, "config.json")) as f:
        hf_cfg = json.load(f)

    init = _gen_lora_init_values(hf_path)
    n_layers = hf_cfg["num_hidden_layers"]
    print(f"Llama-3.1-8B LoRA: r={LORA_R}, alpha={LORA_ALPHA}, "
          f"{n_layers} layers × {len(LORA_TARGET_NAMES)} targets = "
          f"{n_layers * len(LORA_TARGET_NAMES)} (A,B) pairs")

    print(f"\nPreparing {N_STEPS} batches × {TARGET_TOKENS_PER_STEP} tokens...")
    step_batches = _pull_step_batches(
        hf_path, n_steps=N_STEPS,
        target_tokens_per_step=TARGET_TOKENS_PER_STEP,
    )
    print(f"  {len(step_batches)} batches ready")

    hf_losses = _run_hf_peft(hf_path, init, step_batches)
    print(f"  HF PEFT first/last: {hf_losses[0]:.4f} / {hf_losses[-1]:.4f}")

    print("\n=== FT (auto working set) ===")
    ft_losses = _run_ft_subprocess(hf_path, init, step_batches, label="ft")
    print(f"  FT first/last: {ft_losses[0]:.4f} / {ft_losses[-1]:.4f}")

    out_dir = os.path.join(ROOT, "parity_results", "lora_llama_8b")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "loss_curves.csv")
    with open(csv_path, "w") as f:
        f.write("step,hf_peft,ft\n")
        for i in range(N_STEPS):
            f.write(f"{i},{hf_losses[i]:.6f},{ft_losses[i]:.6f}\n")

    max_hf_ft = max(abs(h - f) for h, f in zip(hf_losses, ft_losses))

    print(f"\n=== Comparison ===")
    print(f"  HF PEFT vs FT  max |Δ| = {max_hf_ft:.4f}")

    assert max_hf_ft < 0.15, (
        f"HF PEFT vs FT: max |Δ| = {max_hf_ft:.4f} > 0.15"
    )

    summary = os.path.join(out_dir, "summary.md")
    with open(summary, "w") as f:
        f.write(
            f"# Llama-3.1-8B LoRA E2E parity\n\n"
            f"{N_STEPS} steps on MathInstruct, identical LoRA inits.\n\n"
            f"| pair | max \\|Δ\\| |\n|---|---|\n"
            f"| HF PEFT vs FT | {max_hf_ft:.4f} |\n\n"
            f"| run | first | last |\n|---|---|---|\n"
            f"| HF PEFT | {hf_losses[0]:.4f} | {hf_losses[-1]:.4f} |\n"
            f"| FT | {ft_losses[0]:.4f} | {ft_losses[-1]:.4f} |\n"
        )
    print(f"\nCSV: {csv_path}\nSummary: {summary}")
    print("\n✓ Llama-3.1-8B LoRA E2E PASSED")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--hf-worker":
        _hf_peft_worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) >= 7 and sys.argv[1] == "--ft-worker":
        _ft_worker(
            sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6],
        )
    else:
        main()
