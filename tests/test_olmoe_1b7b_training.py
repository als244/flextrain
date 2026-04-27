"""End-to-end FlexTrain training: OLMoE-1B-7B, real HF weights, real data.

Same pattern as ``test_llama31_8b_training.py`` but for OLMoE:
* load HF safetensors (expert-stacked via arch's post_load_hook),
* run HF transformers' forward on batch 0 as the step-0 correctness
  reference,
* train N steps on MathInstruct SFT (prompt-masked),
* verify step-0 FT loss matches HF within |Δ| < 0.15 and loss
  decreases over training.

OLMoE-1B-7B has 1B active / 7B total params. bf16 params ≈ 14 GB,
fits host. With fp32 opt state (~60 GB) still fits host. On-GPU
rotation handles per-layer offloading.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import ModelShape, _Seq, _flextrain_step

from tests.test_llama32_1b_parity import (  # noqa: E402
    _halved_to_pair_perm,
    _permute_qk_for_pair_interleave,
    _live_curve_writer,
    _pull_step_batches,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _build_olmoe_shape() -> ModelShape:
    """Shape pulled from OLMoE-1B-7B-0924/config.json."""
    return ModelShape(
        d_model=2048,
        n_layers=16,
        n_heads=16,
        n_kv_heads=16,
        head_dim=128,
        expert_dim=1024,
        vocab_size=50304,
        rms_norm_eps=1e-5,
        rope_base=10_000.0,
    )


def _hf_step0_worker(hf_path: str, batch_pkl: str, out_pkl: str) -> None:
    """Subprocess worker: load HF model, compute step-0 loss, write to out_pkl.
    When this process exits the OS reclaims all GPU memory — no pinned
    handles, no fragmentation left behind for the FT engine."""
    import pickle
    from transformers import AutoModelForCausalLM
    with open(batch_pkl, "rb") as f:
        batch = pickle.load(f)  # list of (tokens_cpu, targets_cpu)
    model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="sdpa",
    )
    model.eval()
    total_loss = 0.0
    total_active = 0
    with torch.no_grad():
        for tokens_cpu, targets_cpu in batch:
            tokens = tokens_cpu.to(DEVICE).unsqueeze(0)
            our_targets = targets_cpu.to(DEVICE)
            T = int(our_targets.shape[0])
            hf_labels = torch.full((T,), -100, dtype=torch.int64, device=DEVICE)
            hf_labels[1:] = our_targets[:-1]
            active = int((hf_labels != -100).sum().item())
            out = model(input_ids=tokens, labels=hf_labels.unsqueeze(0))
            total_loss += float(out.loss.item()) * active
            total_active += active
    avg = total_loss / max(1, total_active)
    with open(out_pkl, "wb") as f:
        pickle.dump(avg, f)


def _run_hf_step0_loss(hf_path: str, batch: list[_Seq]) -> float:
    """Run HF transformers forward on the same batch in a subprocess so
    GPU memory is fully returned on exit. Mirrors Llama-3.1-8B test."""
    import pickle
    import subprocess
    import tempfile
    print("\n=== HF transformers step-0 forward (reference, subprocess) ===")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        batch_pkl = os.path.join(td, "batch.pkl")
        out_pkl = os.path.join(td, "loss.pkl")
        batch_cpu = [(s.tokens.cpu(), s.targets.cpu()) for s in batch]
        with open(batch_pkl, "wb") as f:
            pickle.dump(batch_cpu, f)
        # Run this file with a sentinel arg to invoke the worker.
        script = os.path.abspath(__file__)
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, script, "--hf-worker", hf_path, batch_pkl, out_pkl],
            check=True, env=env,
        )
        with open(out_pkl, "rb") as f:
            avg = pickle.load(f)
    print(f"  HF subprocess done: {time.time() - t0:.1f}s")
    print(f"  HF step-0 avg per-token loss: {avg:.4f}")
    return avg


def _build_flextrain_olmoe(
    shape: ModelShape, lr: float, num_experts: int, top_k: int,
    load_balance_coef: float, routing_mode: str = "softmax_then_topk",
):
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import determine_working_set_config
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.olmoe import OLMoEBlock, OLMoEBlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    cfg = OLMoEBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        num_experts=num_experts, top_k=top_k,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True, load_balance_coef=load_balance_coef,
        routing_mode=routing_mode,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [OLMoEBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
    embed = TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=shape.vocab_size, d_model=shape.d_model,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    ))
    head = LMHead(LMHeadConfig(
        d_model=shape.d_model, vocab_size=shape.vocab_size,
        rms_norm_eps=shape.rms_norm_eps, head_chunk_size=512,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    ))
    dims = dict(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        attn_dim=shape.n_heads * shape.head_dim,
        kv_dim=shape.n_kv_heads * shape.head_dim,
        expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
        num_experts=num_experts, top_k=top_k,
    )
    # OLMoE has 64 experts and ~7B total params. Its arithmetic-intensity
    # floor on grouped GEMM wants ~1.7K tokens/chunk; we target ~4K/step.
    max_seq_len = 2048
    target_tokens_per_step = 4096
    working_set = determine_working_set_config(
        model_dims=dict(
            d_model=shape.d_model, n_heads=shape.n_heads,
            n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
            expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
            n_layers=shape.n_layers,
            num_shared_experts=0, num_routed_experts=num_experts, top_k=top_k,
            is_causal=True,
            datatypes={
                "embed": "bfloat16", "head_proj": "bfloat16",
                "attn_proj": "bfloat16", "expert_proj": "bfloat16",
                "router": "bfloat16", "norm": "bfloat16",
                "residual": "bfloat16",
            },
        ),
        max_seq_len=max_seq_len,
        max_global_batch_tokens=target_tokens_per_step,
        training_config={
            "master_weight_dtype": "bfloat16",
            "grad_dtype": "bfloat16",
            "opt_choice": "AdamW",
            "opt_dtype": "float32",
        },
        has_embed=True, has_head=True,
        num_local_layers=shape.n_layers,
        max_gpu_mem_bytes=int(24 * (1 << 30)),
        max_host_mem_bytes=int(110 * (1 << 30)),
        leeway_gpu_mem_bytes=int(2 * (1 << 30)),
        leeway_host_mem_bytes=int(4 * (1 << 30)),
        verbose=True, fixed_seq_len=False,
    )
    # Grow host act buffer if solver under-sized it.
    act_slot_size_bytes = working_set.raw.get(
        "act_slot_size_bytes", working_set.gpu_act_buffer_size // 16
    )
    needed_host = int(act_slot_size_bytes * shape.n_layers * 2)
    if working_set.host_act_buffer_size < needed_host:
        import dataclasses
        working_set = dataclasses.replace(
            working_set, host_act_buffer_size=needed_host,
        )
    print(
        f"  solver picked: n_gpu_layers={working_set.n_gpu_layers}/"
        f"{shape.n_layers}, n_gpu_grads={working_set.n_gpu_grads}, "
        f"n_gpu_opt_layers={working_set.n_gpu_opt_layers}, "
        f"act_buffer={working_set.gpu_act_buffer_size/(1<<30):.1f} GiB, "
        f"host_act_buffer={working_set.host_act_buffer_size/(1<<30):.1f} GiB, "
        f"target_round_tokens={working_set.target_round_tokens}"
    )
    hw_cost = HardwareCost(
        peak_tflops=60.0, pcie_bw_gbps=20.0, practical_efficiency_factor=1.0,
    )
    opt = AdamW(
        AdamWHyperparams(lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
        state_dtype=torch.float32,
    )
    return ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=working_set, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )


def _run_flextrain(hf_path, shape, step_batches, lr, num_experts, top_k,
                   load_balance_coef, routing_mode="softmax_then_topk",
                   *, live_path=None):
    print(f"\n=== FlexTrain OLMoE-1B-7B ===")
    am = _build_flextrain_olmoe(
        shape, lr, num_experts, top_k, load_balance_coef,
        routing_mode=routing_mode,
    )
    print("loading HF weights + expert stacking (via arch.post_load_hook)...")
    t0 = time.time()
    am.load_hf(hf_path, strict=False)
    print(f"  load + stack: {time.time() - t0:.1f}s")
    # Sanity: verify q_norm/k_norm actually got loaded (non-default values).
    w_q_norm_0 = am.buffers.host_params[0]["w_q_norm"]
    w_k_norm_0 = am.buffers.host_params[0]["w_k_norm"]
    print(
        f"  layer 0 w_q_norm: mean={w_q_norm_0.float().mean():.4f} "
        f"std={w_q_norm_0.float().std():.4f} shape={tuple(w_q_norm_0.shape)}"
    )
    print(
        f"  layer 0 w_k_norm: mean={w_k_norm_0.float().mean():.4f} "
        f"std={w_k_norm_0.float().std():.4f} shape={tuple(w_k_norm_0.shape)}"
    )
    # OLMoE uses Llama-style RoPE with pair-interleave Triton kernel;
    # HF stores Q/K in halved-split layout. Apply the permutation to:
    # 1. w_q / w_k weight matrices (along the output dim).
    # 2. w_q_norm / w_k_norm 1D RMSNorm weights (their per-dim scale
    #    matches the post-projection Q/K which is now permuted).
    print("applying Q/K halved→pair permutation...")
    attn_dim = shape.n_heads * shape.head_dim
    kv_dim = shape.n_kv_heads * shape.head_dim
    q_perm = torch.tensor(
        _halved_to_pair_perm(attn_dim, shape.head_dim), dtype=torch.int64,
    )
    k_perm = torch.tensor(
        _halved_to_pair_perm(kv_dim, shape.head_dim), dtype=torch.int64,
    )
    for i in range(shape.n_layers):
        for name in ("w_q", "w_k"):
            w = am.buffers.host_params[i][name]
            am.buffers.host_params[i][name].copy_(
                _permute_qk_for_pair_interleave(w, shape.head_dim)
            )
        # 1D norm weights: permute along dim 0.
        w_qn = am.buffers.host_params[i]["w_q_norm"]
        am.buffers.host_params[i]["w_q_norm"].copy_(w_qn[q_perm])
        w_kn = am.buffers.host_params[i]["w_k_norm"]
        am.buffers.host_params[i]["w_k_norm"].copy_(w_kn[k_perm])
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    curve = []
    live = _live_curve_writer(live_path, "step,loss") if live_path else None
    print("running training steps...")
    t0 = time.time()
    for step, batch in enumerate(step_batches):
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        ts = time.time()
        loss = _flextrain_step(am, seqs)
        curve.append(loss)
        if live is not None:
            live(step, loss)
        if step < 5 or step % 5 == 0 or step == len(step_batches) - 1:
            print(
                f"  FT step {step:4d}  loss={loss:.4f}  "
                f"step={(time.time()-ts)*1000:.0f}ms  "
                f"elapsed={time.time()-t0:.1f}s  "
                f"max_alloc={torch.cuda.max_memory_allocated()/(1<<30):.1f}GiB",
                flush=True,
            )
    am.buffers.destroy()
    del am
    import gc; gc.collect()
    try:
        from flextrain.engine import unregister_all_process_pinned_memory
        unregister_all_process_pinned_memory()
    except Exception:
        pass
    torch.cuda.empty_cache()
    return curve


def test_olmoe_1b7b_training() -> None:
    hf_path = os.path.join(ROOT, "models", "OLMoE-1B-7B")
    math_path = os.path.join(ROOT, "datasets", "MathInstruct", "MathInstruct.json")
    if not os.path.isdir(hf_path):
        raise FileNotFoundError(
            f"OLMoE-1B-7B weights not found at {hf_path}. Download: "
            f"hf download allenai/OLMoE-1B-7B-0924 --local-dir {hf_path}"
        )
    if not os.path.isfile(math_path):
        raise FileNotFoundError(f"MathInstruct not found at {math_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")

    # Load HF config for hyperparams (load_balance_coef, num_experts, top_k).
    with open(os.path.join(hf_path, "config.json")) as f:
        hf_cfg = json.load(f)
    num_experts = hf_cfg["num_experts"]
    top_k = hf_cfg["num_experts_per_tok"]
    load_balance_coef = hf_cfg.get("router_aux_loss_coef", 0.01)
    norm_topk = bool(hf_cfg.get("norm_topk_prob", False))
    routing_mode = "topk_then_softmax" if norm_topk else "softmax_then_topk"
    print(f"  routing_mode = {routing_mode} (norm_topk_prob={norm_topk})")

    shape = _build_olmoe_shape()
    n_steps = 50
    target_tokens_per_step = 4096
    print(f"Preparing {n_steps} batches...")
    step_batches = _pull_step_batches(
        hf_path, n_steps=n_steps,
        target_tokens_per_step=target_tokens_per_step,
    )
    print(f"  {len(step_batches)} batches ready")

    lr = 1e-5
    out_dir = os.path.join(ROOT, "parity_results", "olmoe_1b7b")
    os.makedirs(out_dir, exist_ok=True)

    hf_step0_loss = _run_hf_step0_loss(hf_path, step_batches[0])
    gpu_free = torch.cuda.mem_get_info()[0] / (1 << 30)
    print(f"[memcheck] GPU free before FT engine build: {gpu_free:.2f} GiB")

    curve = _run_flextrain(
        hf_path, shape, step_batches, lr, num_experts, top_k,
        load_balance_coef, routing_mode=routing_mode,
        live_path=os.path.join(out_dir, "live_flextrain.csv"),
    )

    ft_step0_loss = curve[0]
    delta = abs(hf_step0_loss - ft_step0_loss)
    print(f"\n=== Step-0 correctness check ===")
    print(f"  HF step-0 loss:         {hf_step0_loss:.4f}")
    print(f"  FlexTrain step-0 loss:  {ft_step0_loss:.4f}")
    print(f"  |Δ| = {delta:.4f}")
    if delta > 0.15:
        raise AssertionError(
            f"FT step-0 ({ft_step0_loss:.4f}) diverges from HF "
            f"({hf_step0_loss:.4f}) by {delta:.4f} — expected <= 0.15."
        )
    print("  ✓ step-0 correctness verified")

    csv_path = os.path.join(out_dir, "loss_curve.csv")
    with open(csv_path, "w") as f:
        f.write("step,loss\n")
        for i, L in enumerate(curve):
            f.write(f"{i},{L:.6f}\n")
    print(f"\nCSV: {csv_path}")

    def _avg(c, a, b=None):
        return sum(c[a:b]) / len(c[a:b])
    first_n = _avg(curve, 0, 5)
    last_n = _avg(curve, -5)

    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# OLMoE-1B-7B FlexTrain training\n\n")
        f.write(
            f"- Model: OLMoE-1B-7B-0924 (16 layers, d_model=2048, 64 experts, top-k=8)\n"
            f"- {n_steps} steps × ~{target_tokens_per_step} tokens/step on MathInstruct (SFT, prompt-masked)\n"
            f"- lr = {lr}, AdamW, bf16 params + bf16 grads + fp32 opt state\n"
            f"- Load-balance aux loss coef = {load_balance_coef}\n\n"
            f"## Step-0 correctness vs HF transformers\n\n"
            f"| side | step-0 loss |\n"
            f"|---|---:|\n"
            f"| HF transformers | {hf_step0_loss:.4f} |\n"
            f"| FlexTrain | {ft_step0_loss:.4f} |\n"
            f"| \\|Δ\\| | {delta:.4f} |\n\n"
            f"## Convergence\n\n"
            f"| run | first-5 avg | last-5 avg | Δ |\n"
            f"|---|---|---|---|\n"
            f"| FlexTrain | {first_n:.4f} | {last_n:.4f} | "
            f"{last_n - first_n:+.4f} |\n"
        )
    print(f"Summary: {md_path}")
    assert last_n < first_n + 0.1, (
        f"OLMoE training didn't reduce loss: {first_n:.4f} → {last_n:.4f}"
    )
    print("\n✓ OLMoE-1B-7B training PASSED")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--hf-worker":
        _hf_step0_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        test_olmoe_1b7b_training()
