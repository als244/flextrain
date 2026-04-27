"""Qwen3-Next end-to-end training test on a larger machine.

Loads HF Qwen3-Next weights, applies the Q/K halved->pair RoPE
permutation (full-attention layers only — linear-attention layers don't
use RoPE), runs N steps on MathInstruct, compares step-0 loss to HF
transformers as a correctness reference.

Requires:
* Qwen3-Next checkpoint at ``models/Qwen3-Next/``
  (e.g. ``hf download Qwen/Qwen3-Next-80B-A3B-Instruct --local-dir models/Qwen3-Next``)
* ``flash-linear-attention`` installed (``pip install flash-linear-attention``).
* Enough host RAM for ~80B-A3B params in bf16 (~160 GB).
* Enough GPU memory for the working set (>=24 GB recommended).
* HF transformers >= 4.56 (Qwen3-Next was added in that release).

The "small machine" path is not supported — Qwen3-Next-80B doesn't fit
in 16 GB GPU + 128 GB host. Run on a beefier box.

How to run:

    cd /path/to/FlexTrain
    PYTHONPATH=. python tests/test_qwen3_next_training.py

Output appears in ``parity_results/qwen3_next/loss_curve.csv`` and
``summary.md``.
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
    _live_curve_writer,
    _permute_qk_for_pair_interleave,
    _pull_step_batches,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _hf_step0_worker(hf_path, batch_pkl, out_pkl):
    """Subprocess HF step-0 loss reference. Releases all GPU memory on exit."""
    from transformers import AutoModelForCausalLM
    with open(batch_pkl, "rb") as f:
        batch = pickle.load(f)
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


def _run_hf_step0_loss(hf_path, batch):
    print("\n=== HF transformers step-0 forward (reference, subprocess) ===")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        batch_pkl = os.path.join(td, "batch.pkl")
        out_pkl = os.path.join(td, "loss.pkl")
        with open(batch_pkl, "wb") as f:
            pickle.dump([(s.tokens.cpu(), s.targets.cpu()) for s in batch], f)
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--hf-worker", hf_path, batch_pkl, out_pkl],
            check=True, env=env,
        )
        with open(out_pkl, "rb") as f:
            avg = pickle.load(f)
    print(f"  HF subprocess done: {time.time() - t0:.1f}s")
    print(f"  HF step-0 loss: {avg:.4f}")
    return avg


def _build_flextrain_qwen3_next(hf_cfg, n_layers, lr):
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import determine_working_set_config
    from flextrain.engine.active_model import ActiveModel
    from flextrain.io.arch.qwen3_next import (
        hf_config_to_flextrain, hf_config_to_hyperparams,
    )
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.qwen3_next import (
        Qwen3NextLayerConfig, build_qwen3_next_backbone,
    )
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    dims_cfg = hf_config_to_flextrain(hf_cfg)
    hp = hf_config_to_hyperparams(hf_cfg)
    layer_types = hp["layer_types"]
    if layer_types is None:
        # Fallback: use the decoder_sparse_step pattern (1-of-N is full).
        sparse_step = hp.get("decoder_sparse_step", 1) or 1
        layer_types = [
            "full_attention" if (i + 1) % sparse_step == 0 else "linear_attention"
            for i in range(n_layers)
        ]

    cfg = Qwen3NextLayerConfig(
        d_model=dims_cfg["d_model"],
        n_heads=dims_cfg["n_heads"],
        n_kv_heads=dims_cfg["n_kv_heads"],
        head_dim=dims_cfg["head_dim"],
        expert_dim=dims_cfg["expert_dim"],
        num_experts=dims_cfg["num_routed_experts"],
        top_k=dims_cfg["top_k"],
        linear_num_v_heads=dims_cfg["linear_num_v_heads"],
        linear_num_k_heads=dims_cfg["linear_num_k_heads"],
        linear_head_k_dim=dims_cfg["linear_head_k_dim"],
        linear_head_v_dim=dims_cfg["linear_head_v_dim"],
        linear_conv_kernel=dims_cfg["linear_conv_kernel"],
        rms_norm_eps=hp["rms_norm_eps"],
        rope_base=hp["rope_theta"],
        is_causal=True,
        load_balance_coef=hp["load_balance_coef"],
        routing_mode=hp["routing_mode"],
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = build_qwen3_next_backbone(cfg, layer_types[:n_layers])
    embed = TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=dims_cfg["vocab_size"], d_model=dims_cfg["d_model"],
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    ))
    head = LMHead(LMHeadConfig(
        d_model=dims_cfg["d_model"], vocab_size=dims_cfg["vocab_size"],
        rms_norm_eps=hp["rms_norm_eps"], head_chunk_size=512,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    ))
    dims = dict(
        d_model=dims_cfg["d_model"],
        n_heads=dims_cfg["n_heads"],
        n_kv_heads=dims_cfg["n_kv_heads"],
        head_dim=dims_cfg["head_dim"],
        attn_dim=dims_cfg["n_heads"] * dims_cfg["head_dim"],
        kv_dim=dims_cfg["n_kv_heads"] * dims_cfg["head_dim"],
        expert_dim=dims_cfg["expert_dim"],
        vocab_size=dims_cfg["vocab_size"],
        num_experts=dims_cfg["num_routed_experts"],
        top_k=dims_cfg["top_k"],
        # Linear-attention dims for layer paramspecs.
        num_v_heads=dims_cfg["linear_num_v_heads"],
        num_k_heads=dims_cfg["linear_num_k_heads"],
        head_k_dim=dims_cfg["linear_head_k_dim"],
        head_v_dim=dims_cfg["linear_head_v_dim"],
        key_dim=dims_cfg["linear_num_k_heads"] * dims_cfg["linear_head_k_dim"],
        value_dim=dims_cfg["linear_num_v_heads"] * dims_cfg["linear_head_v_dim"],
        conv_dim=(
            2 * dims_cfg["linear_num_k_heads"] * dims_cfg["linear_head_k_dim"]
            + dims_cfg["linear_num_v_heads"] * dims_cfg["linear_head_v_dim"]
        ),
        proj_qkvz_dim=(
            2 * dims_cfg["linear_num_k_heads"] * dims_cfg["linear_head_k_dim"]
            + 2 * dims_cfg["linear_num_v_heads"] * dims_cfg["linear_head_v_dim"]
        ),
        proj_ba_dim=2 * dims_cfg["linear_num_v_heads"],
        conv_kernel_size=dims_cfg["linear_conv_kernel"],
    )

    max_seq_len = 2048
    target_tokens_per_step = 4096
    working_set = determine_working_set_config(
        model_dims=dict(
            d_model=dims_cfg["d_model"], n_heads=dims_cfg["n_heads"],
            n_kv_heads=dims_cfg["n_kv_heads"], head_dim=dims_cfg["head_dim"],
            expert_dim=dims_cfg["expert_dim"],
            vocab_size=dims_cfg["vocab_size"],
            n_layers=n_layers,
            num_shared_experts=0,
            num_routed_experts=dims_cfg["num_routed_experts"],
            top_k=dims_cfg["top_k"],
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
            "master_weight_dtype": "bfloat16", "grad_dtype": "bfloat16",
            "opt_choice": "AdamW", "opt_dtype": "bfloat16",
        },
        has_embed=True, has_head=True, num_local_layers=n_layers,
        max_gpu_mem_bytes=int(80 * (1 << 30)),
        max_host_mem_bytes=int(220 * (1 << 30)),
        leeway_gpu_mem_bytes=int(2 * (1 << 30)),
        leeway_host_mem_bytes=int(4 * (1 << 30)),
        verbose=True, fixed_seq_len=False,
    )
    print(
        f"  solver: n_gpu_layers={working_set.n_gpu_layers}/{n_layers}  "
        f"target_round_tokens={working_set.target_round_tokens}"
    )
    hw_cost = HardwareCost(
        peak_tflops=300.0, pcie_bw_gbps=20.0,
        practical_efficiency_factor=1.0,
    )
    opt = AdamW(
        AdamWHyperparams(lr=lr, beta1=0.9, beta2=0.95, eps=1e-8),
        state_dtype=torch.bfloat16,
    )
    return ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=working_set, hw_cost=hw_cost, dims=dims, device=DEVICE,
    ), layer_types


def main():
    hf_path = os.path.join(ROOT, "models", "Qwen3-Next")
    if not os.path.isdir(hf_path):
        raise FileNotFoundError(
            f"Qwen3-Next weights not found at {hf_path}. Download with: "
            f"hf download Qwen/Qwen3-Next-80B-A3B-Instruct --local-dir {hf_path}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")

    with open(os.path.join(hf_path, "config.json")) as f:
        hf_cfg = json.load(f)
    n_layers = hf_cfg["num_hidden_layers"]

    n_steps = 30
    target_tokens_per_step = 4096
    print(f"Preparing {n_steps} batches...")
    step_batches = _pull_step_batches(
        hf_path, n_steps=n_steps,
        target_tokens_per_step=target_tokens_per_step,
    )
    print(f"  {len(step_batches)} batches")

    hf_step0_loss = _run_hf_step0_loss(hf_path, step_batches[0])

    print("\n=== FlexTrain Qwen3-Next ===")
    am, layer_types = _build_flextrain_qwen3_next(
        hf_cfg, n_layers=n_layers, lr=1e-5,
    )
    full_attn_layers = [
        i for i, lt in enumerate(layer_types) if lt == "full_attention"
    ]
    print(f"  layer_types: {len(full_attn_layers)} full / "
          f"{n_layers - len(full_attn_layers)} linear")

    print("loading HF weights...")
    t0 = time.time()
    am.load_hf(hf_path, strict=False)
    print(f"  load: {time.time()-t0:.1f}s")

    # Q/K halved->pair permutation only on full-attention layers.
    head_dim = hf_cfg.get("head_dim") or (
        hf_cfg["hidden_size"] // hf_cfg["num_attention_heads"]
    )
    n_heads = hf_cfg["num_attention_heads"]
    n_kv = hf_cfg["num_key_value_heads"]
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    q_perm = torch.tensor(_halved_to_pair_perm(attn_dim, head_dim), dtype=torch.int64)
    k_perm = torch.tensor(_halved_to_pair_perm(kv_dim, head_dim), dtype=torch.int64)
    print("applying Q/K halved->pair permutation to full-attention layers...")
    for i in full_attn_layers:
        for name in ("w_q", "w_k"):
            w = am.buffers.host_params[i].get(name)
            if w is not None:
                am.buffers.host_params[i][name].copy_(
                    _permute_qk_for_pair_interleave(w, head_dim)
                )
        # 1-D q_norm / k_norm (per-head): no permutation on per-head
        # norm weights — they're (head_dim,) and apply across ALL heads
        # uniformly. The pair-interleave swaps positions WITHIN a head's
        # head_dim, so q_norm/k_norm — which are (head_dim,) — DO need
        # a permutation by `head_dim` slots. Apply.
        for name in ("w_q_norm", "w_k_norm"):
            w = am.buffers.host_params[i].get(name)
            if w is None or w.dim() != 1:
                continue
            if w.numel() == head_dim:
                # Per-head, single (head_dim,) — permute by head_dim slots.
                hperm = torch.tensor(
                    _halved_to_pair_perm(head_dim, head_dim), dtype=torch.int64,
                )
                am.buffers.host_params[i][name].copy_(w[hperm])
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    out_dir = os.path.join(ROOT, "parity_results", "qwen3_next")
    os.makedirs(out_dir, exist_ok=True)
    csv = _live_curve_writer(os.path.join(out_dir, "live_curve.csv"), "step,loss")

    print("running training steps...")
    curve = []
    t0 = time.time()
    for step, batch in enumerate(step_batches):
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        ts = time.time()
        loss = _flextrain_step(am, seqs)
        curve.append(loss)
        csv(step, loss)
        if step < 5 or step % 5 == 0 or step == n_steps - 1:
            print(
                f"  FT step {step:3d}: loss={loss:.4f}  "
                f"step={(time.time()-ts)*1000:.0f}ms  "
                f"max_alloc={torch.cuda.max_memory_allocated()/(1<<30):.1f}GiB"
            )

    delta = abs(hf_step0_loss - curve[0])
    print(f"\n=== Step-0 correctness ===")
    print(f"  HF: {hf_step0_loss:.4f}  FT: {curve[0]:.4f}  |Δ| = {delta:.4f}")
    csv_path = os.path.join(out_dir, "loss_curve.csv")
    with open(csv_path, "w") as f:
        f.write("step,loss\n")
        for i, L in enumerate(curve):
            f.write(f"{i},{L:.6f}\n")
    print(f"\nCSV: {csv_path}")
    if delta > 0.20:
        raise AssertionError(
            f"step-0 |Δ| = {delta:.4f} > 0.20 — FT diverges from HF"
        )
    print("✓ Qwen3-Next training PASSED")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--hf-worker":
        _hf_step0_worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
