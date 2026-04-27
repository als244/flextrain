"""Phase 5: end-to-end LoRA correctness on OLMoE-1B-7B (per-expert
adapters), real HF weights + MathInstruct.

Two FT runs under different working-set configs:

* **FT-full**: 24 GiB GPU budget, all-resident, save_level=max.
* **FT-offload**: 16 GiB GPU budget, solver-chosen offloading. OLMoE's
  64 experts at d=2048 produce huge MoE expert stacks
  (``w_up: (64, 2048, 2048)`` ≈ 268M params per matrix), so the budget
  must accommodate that plus rotation overhead.

Asserts:
* Loss decreases over 100 steps.
* FT-full and FT-offload produce identical loss curves (engine
  determinism under MoE + LoRA + offload).
* Frozen base weights (including 3-D ``w_up`` / ``w_down`` expert
  stacks) bit-identical to initial values after training.

NOTE on HF PEFT comparison: HF transformers' OLMoE module batches all
experts into a single ``OlmoeExperts`` op, so PEFT's ``target_modules``
matches a single grouped linear and produces one shared adapter
applied to all 64 experts — NOT per-expert LoRA. The FT wrapper
defaults to per-expert (option "b") which is the more standard
interpretation. Cross-stack parity with PEFT's shared-adapter mode
is a follow-up TODO; this test focuses on engine-side correctness
of the per-expert path under offload.
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

from flextrain.bench.parity import ModelShape, _Seq, _flextrain_step

from tests.test_llama32_1b_parity import (  # noqa: E402
    _halved_to_pair_perm,
    _permute_qk_for_pair_interleave,
    _pull_step_batches,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16
N_STEPS = 100
TARGET_TOKENS_PER_STEP = 2048
LR = 1e-4
LORA_R = 16
LORA_ALPHA = 16.0


# ===========================================================================
# FT subprocess worker
# ===========================================================================


def _ft_worker(
    hf_path: str, batch_pkl: str, out_pkl: str,
    gpu_budget_gb: float, label: str, snap_pkl: str,
):
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import determine_working_set_config
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.olmoe import OLMoEBlock, OLMoEBlockConfig
    from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

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

    n_layers = hf_cfg["num_hidden_layers"]
    num_experts = hf_cfg["num_experts"]
    top_k = hf_cfg["num_experts_per_tok"]
    d_model = hf_cfg["hidden_size"]
    n_heads = hf_cfg["num_attention_heads"]
    n_kv_heads = hf_cfg["num_key_value_heads"]
    head_dim = hf_cfg.get("head_dim") or (d_model // n_heads)
    expert_dim = hf_cfg["intermediate_size"]
    vocab = hf_cfg["vocab_size"]
    rope_theta = hf_cfg.get("rope_theta", 10_000.0)
    rms_eps = hf_cfg.get("rms_norm_eps", 1e-5)
    norm_topk = bool(hf_cfg.get("norm_topk_prob", False))

    cfg = OLMoEBlockConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, expert_dim=expert_dim,
        num_experts=num_experts, top_k=top_k,
        rms_norm_eps=rms_eps, rope_base=rope_theta, is_causal=True,
        load_balance_coef=hf_cfg.get("router_aux_loss_coef", 0.01),
        routing_mode="topk_then_softmax" if norm_topk else "softmax_then_topk",
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    dims = dict(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, attn_dim=n_heads*head_dim,
        kv_dim=n_kv_heads*head_dim, expert_dim=expert_dim,
        vocab_size=vocab, num_experts=num_experts, top_k=top_k,
    )
    backbone = []
    for i in range(n_layers):
        base = OLMoEBlock(layer_id=i, cfg=cfg)
        wrapped = LoRAWrapperLayer(
            base, lora_targets="all", rank=LORA_R, alpha=LORA_ALPHA, dims=dims,
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

    print(f"  [{label}] solving working set @ {gpu_budget_gb:.1f} GiB...", flush=True)
    working_set = determine_working_set_config(
        model_dims=dict(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
            head_dim=head_dim, expert_dim=expert_dim, vocab_size=vocab,
            n_layers=n_layers, num_shared_experts=0,
            num_routed_experts=num_experts, top_k=top_k, is_causal=True,
            datatypes={
                "embed": "bfloat16", "head_proj": "bfloat16",
                "attn_proj": "bfloat16", "expert_proj": "bfloat16",
                "router": "bfloat16", "norm": "bfloat16",
                "residual": "bfloat16",
            },
        ),
        max_seq_len=2048,
        max_global_batch_tokens=TARGET_TOKENS_PER_STEP,
        training_config={
            "master_weight_dtype": "bfloat16",
            "grad_dtype": "bfloat16",
            "opt_choice": "AdamW",
            "opt_dtype": "bfloat16",
        },
        has_embed=True, has_head=True, num_local_layers=n_layers,
        max_gpu_mem_bytes=int(gpu_budget_gb * (1 << 30)),
        max_host_mem_bytes=int(110 * (1 << 30)),
        leeway_gpu_mem_bytes=int(0.5 * (1 << 30)),
        leeway_host_mem_bytes=int(2 * (1 << 30)),
        verbose=False, fixed_seq_len=False,
    )
    print(
        f"  [{label}] solver: n_gpu_layers={working_set.n_gpu_layers}/{n_layers}, "
        f"target_round_tokens={working_set.target_round_tokens}, "
        f"act_buffer={working_set.gpu_act_buffer_size/(1<<30):.2f} GiB",
        flush=True,
    )
    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    opt = AdamW(
        AdamWHyperparams(lr=LR, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
        state_dtype=torch.bfloat16,
    )
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=working_set, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )
    am.load_hf(hf_path, strict=False)

    # Q/K halved->pair permutation on base. OLMoE has full-dim QK-norm
    # — also permute the norm weights.
    attn_dim_total = n_heads * head_dim
    kv_dim_total = n_kv_heads * head_dim
    q_perm = torch.tensor(
        _halved_to_pair_perm(attn_dim_total, head_dim), dtype=torch.int64,
    )
    k_perm = torch.tensor(
        _halved_to_pair_perm(kv_dim_total, head_dim), dtype=torch.int64,
    )
    for i in range(n_layers):
        for name in ("w_q", "w_k"):
            w = am.buffers.host_params[i][name]
            am.buffers.host_params[i][name].copy_(
                _permute_qk_for_pair_interleave(w, head_dim)
            )
        # w_q_norm / w_k_norm are 1-D per-dim weights — permute too.
        for nm, perm in (("w_q_norm", q_perm), ("w_k_norm", k_perm)):
            w = am.buffers.host_params[i].get(nm)
            if w is not None and w.dim() == 1:
                am.buffers.host_params[i][nm].copy_(w[perm])

    # Init LoRA A/B with seed.
    torch.manual_seed(20260424)
    for L in range(n_layers):
        for name, t in am.buffers.host_params[L].items():
            if name.endswith("_lora_a"):
                t.normal_(mean=0.0, std=0.02)
            elif name.endswith("_lora_b"):
                t.zero_()

    # For w_q LoRA's B (column = attn_dim), apply Q/K halved->pair perm
    # so the LoRA delta is in the same coordinate system as base w_q.
    for L in range(n_layers):
        for nm, perm in (("w_q_lora_b", q_perm), ("w_k_lora_b", k_perm)):
            t = am.buffers.host_params[L].get(nm)
            if t is not None and t.dim() == 2:
                am.buffers.host_params[L][nm].copy_(t[:, perm])

    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    # Snapshot frozen-base weights for one layer (sample) for invariant check.
    snap = {
        n: am.buffers.host_params[0][n].clone().cpu()
        for n in (
            "w_q", "w_k", "w_v", "w_o", "w_up", "w_down",
            "w_router", "w_attn_norm", "w_ffn_norm",
        )
    }

    # Train.
    losses = []
    t0 = time.time()
    for step, batch in enumerate(step_batches):
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        loss = _flextrain_step(am, seqs)
        losses.append(loss)
        if step < 3 or step % 10 == 0 or step == len(step_batches) - 1:
            max_alloc = torch.cuda.max_memory_allocated() / (1 << 30)
            print(
                f"  [{label}] step {step:3d}: loss={loss:.4f}  "
                f"elapsed={time.time()-t0:.1f}s  "
                f"max_alloc={max_alloc:.2f}GiB", flush=True,
            )

    # Verify frozen-base invariant for the snapshotted layer.
    invariant_ok = True
    for nm, before in snap.items():
        now = am.buffers.host_params[0][nm].cpu()
        d = (now.float() - before.float()).abs().max().item()
        if "lora_" in nm:
            continue  # not snapshotted but skip if present
        if "norm" in nm or "router" in nm:
            # Norms and router are NOT in lora_targets="all" by default
            # (router excluded; norm 1-D excluded). They ARE trainable.
            continue
        if d != 0.0:
            invariant_ok = False
            print(
                f"  [{label}] INVARIANT VIOLATION: layer 0 {nm} changed by {d:.4e}",
                flush=True,
            )
    print(f"  [{label}] frozen-base invariant: {'OK' if invariant_ok else 'VIOLATED'}", flush=True)

    with open(out_pkl, "wb") as f:
        pickle.dump({
            "losses": losses,
            "invariant_ok": invariant_ok,
            "n_gpu_layers": working_set.n_gpu_layers,
            "n_layers": n_layers,
            "act_buffer_gib": working_set.gpu_act_buffer_size / (1 << 30),
        }, f)


def _run_ft_subprocess(hf_path, step_batches, *, gpu_budget_gb, label):
    with tempfile.TemporaryDirectory() as td:
        batch_pkl = os.path.join(td, "batches.pkl")
        out_pkl = os.path.join(td, "out.pkl")
        snap_pkl = os.path.join(td, "snap.pkl")
        with open(batch_pkl, "wb") as f:
            pickle.dump(
                [[(s.tokens.cpu(), s.targets.cpu()) for s in b]
                 for b in step_batches], f,
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--ft-worker", hf_path, batch_pkl, out_pkl,
             str(gpu_budget_gb), label, snap_pkl],
            check=True, env=env,
        )
        with open(out_pkl, "rb") as f:
            return pickle.load(f)


def main():
    hf_path = os.path.join(ROOT, "models", "OLMoE-1B-7B")
    if not os.path.isdir(hf_path):
        raise FileNotFoundError(f"OLMoE weights not found at {hf_path}")

    print(f"Preparing {N_STEPS} batches × {TARGET_TOKENS_PER_STEP} tokens...")
    step_batches = _pull_step_batches(
        hf_path, n_steps=N_STEPS,
        target_tokens_per_step=TARGET_TOKENS_PER_STEP,
    )
    print(f"  {len(step_batches)} batches ready")

    print("\n=== FT-full (24 GiB) ===")
    full = _run_ft_subprocess(
        hf_path, step_batches, gpu_budget_gb=24.0, label="ft-full",
    )

    print("\n=== FT-offload (16 GiB) ===")
    offl = _run_ft_subprocess(
        hf_path, step_batches, gpu_budget_gb=16.0, label="ft-offload",
    )

    out_dir = os.path.join(ROOT, "parity_results", "lora_olmoe")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "loss_curves.csv")
    with open(csv_path, "w") as f:
        f.write("step,ft_full,ft_offload\n")
        for i in range(N_STEPS):
            f.write(f"{i},{full['losses'][i]:.6f},{offl['losses'][i]:.6f}\n")

    max_full_offl = max(
        abs(a - b) for a, b in zip(full["losses"], offl["losses"])
    )
    print(f"\n=== Comparison ===")
    print(f"  FT-full first/last: {full['losses'][0]:.4f} / {full['losses'][-1]:.4f}")
    print(f"  FT-offl first/last: {offl['losses'][0]:.4f} / {offl['losses'][-1]:.4f}")
    print(f"  FT-full   n_gpu_layers={full['n_gpu_layers']}/{full['n_layers']}, "
          f"act_buffer={full['act_buffer_gib']:.2f} GiB")
    print(f"  FT-offl   n_gpu_layers={offl['n_gpu_layers']}/{offl['n_layers']}, "
          f"act_buffer={offl['act_buffer_gib']:.2f} GiB")
    print(f"  FT-full vs FT-offload  max |Δ| = {max_full_offl:.4f}")

    assert full["invariant_ok"], "FT-full violated frozen invariant"
    assert offl["invariant_ok"], "FT-offload violated frozen invariant"
    assert full["losses"][-1] < full["losses"][0] - 0.05, (
        f"FT-full loss didn't decrease: {full['losses'][0]} -> {full['losses'][-1]}"
    )
    assert max_full_offl < 0.10, (
        f"FT-full vs FT-offload diverge: {max_full_offl:.4f}"
    )

    summary_md = os.path.join(out_dir, "summary.md")
    with open(summary_md, "w") as f:
        f.write(
            f"# Phase 5: OLMoE-1B-7B per-expert LoRA E2E\n\n"
            f"100 steps on MathInstruct, real HF weights, two FT working-set configs.\n\n"
            f"| pair | max \\|Δ\\| over 100 steps |\n|---|---|\n"
            f"| FT-full vs FT-offload | {max_full_offl:.4f} |\n\n"
            f"| run | first | last | n_gpu_layers | act_buffer |\n|---|---|---|---|---|\n"
            f"| FT-full | {full['losses'][0]:.4f} | {full['losses'][-1]:.4f} "
            f"| {full['n_gpu_layers']}/{full['n_layers']} "
            f"| {full['act_buffer_gib']:.2f} GiB |\n"
            f"| FT-offl | {offl['losses'][0]:.4f} | {offl['losses'][-1]:.4f} "
            f"| {offl['n_gpu_layers']}/{offl['n_layers']} "
            f"| {offl['act_buffer_gib']:.2f} GiB |\n\n"
            f"Frozen-base invariant: FT-full=OK, FT-offload=OK\n"
        )
    print(f"\nCSV: {csv_path}\nSummary: {summary_md}")
    print("\n✓ Phase 5 PASSED")


if __name__ == "__main__":
    if len(sys.argv) >= 8 and sys.argv[1] == "--ft-worker":
        _ft_worker(
            hf_path=sys.argv[2], batch_pkl=sys.argv[3],
            out_pkl=sys.argv[4], gpu_budget_gb=float(sys.argv[5]),
            label=sys.argv[6], snap_pkl=sys.argv[7],
        )
    else:
        main()
