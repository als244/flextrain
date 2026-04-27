"""Phase 3: end-to-end LoRA correctness on dense Llama-3.2-1B vs HF PEFT.

Three runs of the same model on the same batches with identical
LoRA A/B initializations:

* **HF PEFT reference** (subprocess) — `transformers.LlamaForCausalLM`
  loaded from `models/Llama-3.2-1B`, wrapped with `peft.LoraConfig` on
  every linear projection (Q/K/V/O + gate/up/down), AdamW on trainable
  params for 100 steps on MathInstruct.
* **FlexTrain LoRA, full-save** — `LoRAWrapperLayer(LlamaBlock,
  lora_targets="all")`, all-resident working set, save_level=max.
* **FlexTrain LoRA, offloaded** — same model, working set capped at
  4 GiB GPU budget so the solver picks aggressive offloading.

LoRA A/B values are written into all three from the same fp32 source
tensors, so loss curves should match within bf16 noise across stacks
and be bit-identical between the two FT working-set configs.

Outputs: ``parity_results/lora_dense_llama/`` with per-run loss curves
and a summary.md table.
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
    _live_curve_writer,
    _pull_step_batches,
)


DEVICE = "cuda:0"
# Compute / param dtype for the BASE block.
# Override with FT_LORA_DTYPE=float32 for the tight-parity run (everything fp32).
_DTYPE_NAME = os.environ.get("FT_LORA_DTYPE", "bfloat16")
DTYPE = {"bfloat16": torch.bfloat16, "float32": torch.float32}[_DTYPE_NAME]
TIGHT_PARITY = (_DTYPE_NAME == "float32")

# Mapping FT name -> HF Llama linear-proj name.
FT_TO_HF = {
    "w_q": "q_proj",
    "w_k": "k_proj",
    "w_v": "v_proj",
    "w_o": "o_proj",
    "w_1": "gate_proj",   # SwiGLU gate
    "w_3": "up_proj",     # SwiGLU up
    "w_2": "down_proj",   # SwiGLU down
}
LORA_TARGET_NAMES = tuple(FT_TO_HF.keys())

# LoRA hyperparams.
LORA_R = 16
LORA_ALPHA = 16.0
N_STEPS = 100
TARGET_TOKENS_PER_STEP = 1024
LR = 1e-4


def _gen_lora_init_values(hf_path: str) -> dict:
    """Generate LoRA A and B initial values once. Returns a dict
    mapping (layer_idx, ft_name) -> (A_fp32, B_fp32) where:

      * A: (d_in, r) — random normal(0, 0.02), kept fp32.
      * B: (r, d_out) — zeros (PEFT default; LoRA delta starts at zero).
    """
    with open(os.path.join(hf_path, "config.json")) as f:
        cfg = json.load(f)
    d_model = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim") or (d_model // n_heads)
    intermediate = cfg["intermediate_size"]
    n_layers = cfg["num_hidden_layers"]
    attn_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    in_dim = {
        "w_q": d_model, "w_k": d_model, "w_v": d_model,
        "w_o": attn_dim,
        "w_1": d_model, "w_2": intermediate, "w_3": d_model,
    }
    out_dim = {
        "w_q": attn_dim, "w_k": kv_dim, "w_v": kv_dim,
        "w_o": d_model,
        "w_1": intermediate, "w_2": d_model, "w_3": intermediate,
    }

    torch.manual_seed(20260424)
    init = {}
    for L in range(n_layers):
        for tgt in LORA_TARGET_NAMES:
            d_in, d_out = in_dim[tgt], out_dim[tgt]
            A = torch.empty(d_in, LORA_R, dtype=torch.float32).normal_(std=0.02)
            B = torch.zeros(LORA_R, d_out, dtype=torch.float32)
            init[(L, tgt)] = (A, B)
    return init


# ===========================================================================
# HF PEFT subprocess worker
# ===========================================================================


def _hf_peft_worker(hf_path: str, init_pkl: str, batch_pkl: str, out_pkl: str) -> None:
    """Subprocess: load HF model + apply PEFT LoRA + train N steps."""
    import torch as _t
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    with open(init_pkl, "rb") as f:
        init = pickle.load(f)
    with open(batch_pkl, "rb") as f:
        batches = pickle.load(f)  # list of list of (tokens_cpu, targets_cpu)

    model = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="sdpa",
    )

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, bias="none",
        target_modules=list(FT_TO_HF.values()),
        init_lora_weights=False,   # we'll overwrite with our values
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    # Overwrite PEFT's auto-generated A/B with our fixed values.
    # PEFT layout: lora_A.weight: (r, d_in)  — applied as x @ lora_A.T -> (T, r)
    #              lora_B.weight: (d_out, r) — applied as xa @ lora_B.T -> (T, d_out)
    # Our init: A_ft (d_in, r), B_ft (r, d_out). So PEFT's lora_A = A_ft.T,
    # PEFT's lora_B = B_ft.T.
    n_layers = len(set(L for (L, _) in init))
    overwrite_count = 0
    with _t.no_grad():
        for L in range(n_layers):
            for tgt, hf_name in FT_TO_HF.items():
                A, B = init[(L, tgt)]
                # Find the PEFT-wrapped Linear at this position.
                # HF Llama path: model.model.model.layers[L].self_attn.<hf_name>
                #                model.model.model.layers[L].mlp.<hf_name>
                if hf_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    parent = model.model.model.layers[L].self_attn
                else:
                    parent = model.model.model.layers[L].mlp
                lora_layer = getattr(parent, hf_name)
                lora_A = lora_layer.lora_A["default"].weight   # (r, d_in)
                lora_B = lora_layer.lora_B["default"].weight   # (d_out, r)
                lora_A.data.copy_(A.t().to(lora_A.dtype).to(lora_A.device))
                lora_B.data.copy_(B.t().to(lora_B.dtype).to(lora_B.device))
                overwrite_count += 1
    print(f"  [hf-worker] overwrote {overwrite_count} LoRA pairs", flush=True)

    # Optimizer — only trainable params (PEFT marks base as
    # ``requires_grad=False`` automatically).
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = _t.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

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
            loss = out.loss * active   # un-mean to get sum
            loss.backward()
            batch_loss += float(out.loss.item()) * active
            active_total += active
        # Match FT's per-token loss-scale: divide grads by total active tokens.
        for p in trainable:
            if p.grad is not None:
                p.grad.div_(active_total)
        opt.step()
        losses.append(batch_loss / max(1, active_total))
        if step < 3 or step % 10 == 0 or step == len(batches) - 1:
            print(
                f"  [hf-worker] step {step:3d}: loss={losses[-1]:.4f}  "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    with open(out_pkl, "wb") as f:
        pickle.dump(losses, f)


def _run_hf_peft(hf_path, init, step_batches):
    print(f"\n=== HF PEFT (subprocess) ===")
    with tempfile.TemporaryDirectory() as td:
        init_pkl = os.path.join(td, "init.pkl")
        batch_pkl = os.path.join(td, "batches.pkl")
        out_pkl = os.path.join(td, "losses.pkl")
        with open(init_pkl, "wb") as f:
            pickle.dump(init, f)
        with open(batch_pkl, "wb") as f:
            pickle.dump(
                [[(s.tokens.cpu(), s.targets.cpu()) for s in b] for b in step_batches],
                f,
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
# FlexTrain run
# ===========================================================================


def _build_ft_model(hf_path, hf_cfg, init, *, gpu_budget_gb: float, label: str):
    """Build a Llama backbone wrapped with LoRA, load HF base weights,
    overwrite LoRA A/B with the fixed init, return ActiveModel + metadata."""
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
    n_kv_heads = hf_cfg["num_key_value_heads"]
    head_dim = hf_cfg.get("head_dim") or (d_model // n_heads)
    intermediate = hf_cfg["intermediate_size"]
    n_layers = hf_cfg["num_hidden_layers"]
    vocab = hf_cfg["vocab_size"]
    rope_theta = hf_cfg.get("rope_theta", 500_000.0)
    rms_eps = hf_cfg.get("rms_norm_eps", 1e-5)

    cfg = LlamaBlockConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, expert_dim=intermediate,
        rms_norm_eps=rms_eps, rope_base=rope_theta, is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    dims = dict(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, attn_dim=n_heads*head_dim,
        kv_dim=n_kv_heads*head_dim, expert_dim=intermediate,
        vocab_size=vocab,
    )
    backbone = []
    for i in range(n_layers):
        base = LlamaBlock(layer_id=i, cfg=cfg)
        wrapped = LoRAWrapperLayer(
            base, lora_targets="all",
            rank=LORA_R, alpha=LORA_ALPHA, dims=dims,
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

    # Working-set solver.
    max_seq_len = 1024
    dtype_name = "float32" if TIGHT_PARITY else "bfloat16"
    print(f"  [{label}] solving working set with GPU cap = {gpu_budget_gb:.1f} GiB...")
    working_set = determine_working_set_config(
        model_dims=dict(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
            head_dim=head_dim, expert_dim=intermediate, vocab_size=vocab,
            n_layers=n_layers, num_shared_experts=1, num_routed_experts=0,
            top_k=0, is_causal=True,
            datatypes={
                "embed": dtype_name, "head_proj": dtype_name,
                "attn_proj": dtype_name, "expert_proj": dtype_name,
                "router": dtype_name, "norm": dtype_name,
                "residual": dtype_name,
            },
        ),
        max_seq_len=max_seq_len,
        max_global_batch_tokens=TARGET_TOKENS_PER_STEP,
        training_config={
            "master_weight_dtype": dtype_name,
            "grad_dtype": dtype_name,
            "opt_choice": "AdamW",
            "opt_dtype": dtype_name,
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
        f"act_buffer={working_set.gpu_act_buffer_size/(1<<30):.2f} GiB"
    )

    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    opt_state_dtype = torch.float32 if TIGHT_PARITY else torch.bfloat16
    opt = AdamW(
        AdamWHyperparams(lr=LR, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
        state_dtype=opt_state_dtype,
    )
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=working_set, hw_cost=hw_cost, dims=dims, device=DEVICE,
    )

    # Load base HF weights.
    am.load_hf(hf_path, strict=False)

    # Q/K halved->pair permutation on base weights.
    attn_dim_total = n_heads * head_dim
    kv_dim_total = n_kv_heads * head_dim
    for i in range(n_layers):
        for name in ("w_q", "w_k"):
            w = am.buffers.host_params[i][name]
            am.buffers.host_params[i][name].copy_(
                _permute_qk_for_pair_interleave(w, head_dim)
            )

    # Tied LM head: copy embed -> head if head_proj missing or zero.
    # Embed is (vocab, d_model); head_proj is (d_model, vocab) — transpose.
    head_w = am.buffers.host_head_params.get("w_head_proj")
    embed_w = am.buffers.host_embed_params.get("w_tok_embeddings")
    if head_w is not None and embed_w is not None:
        if head_w.abs().sum().item() == 0.0:
            head_w.copy_(embed_w.t())

    # Overwrite LoRA A/B with fixed init values.
    for L in range(n_layers):
        for tgt, (A_fp32, B_fp32) in [
            ((L, t), init[(L, t)]) for t in LORA_TARGET_NAMES
        ]:
            ft_layer_idx, ft_name = tgt
            host_A = am.buffers.host_params[ft_layer_idx][f"{ft_name}_lora_a"]
            host_B = am.buffers.host_params[ft_layer_idx][f"{ft_name}_lora_b"]
            host_A.copy_(A_fp32.to(host_A.dtype))
            host_B.copy_(B_fp32.to(host_B.dtype))
            # A/B are shape (d_in, r) and (r, d_out) — but we ALSO need to
            # account for the Q/K halved->pair permutation on A for w_q
            # and w_k (since A is post-projection-equivalent). Wait:
            # A is (d_in, r) = (d_model, r), output of `x @ A` is (T, r)
            # which then feeds B to give (T, attn_dim). The base Q matmul
            # output gets permuted... hmm. Actually the LoRA delta
            # ``(x @ A) @ B`` produces (T, attn_dim) which is added to the
            # base Q output. Both base and LoRA outputs need to be in the
            # same "pair-interleave" coordinate system.
            #
            # FT permutes BASE w_q's columns (out dim). To keep the LoRA
            # delta in the same coordinate system, B's columns (out dim)
            # must be permuted the same way for w_q and w_k. A's columns
            # (the rank dim) and rows (input dim) are unaffected.
            if ft_name in ("w_q", "w_k"):
                if ft_name == "w_q":
                    perm = torch.tensor(
                        _halved_to_pair_perm(attn_dim_total, head_dim),
                        dtype=torch.int64,
                    )
                else:
                    perm = torch.tensor(
                        _halved_to_pair_perm(kv_dim_total, head_dim),
                        dtype=torch.int64,
                    )
                # B shape (r, d_out): permute columns.
                host_B.copy_(host_B[:, perm])

    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()
    return am, n_layers


def _run_ft(am, step_batches, *, label):
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
                f"max_alloc={max_alloc:.2f}GiB"
            )
    return losses


def _ft_worker(hf_path, init_pkl, batch_pkl, out_pkl, gpu_budget_gb, label):
    """Subprocess worker for FT runs: build model, run training, dump losses."""
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

    am, _ = _build_ft_model(
        hf_path, hf_cfg, init,
        gpu_budget_gb=gpu_budget_gb, label=label,
    )
    losses = _run_ft(am, step_batches, label=label)
    with open(out_pkl, "wb") as f:
        pickle.dump(losses, f)


def _run_ft_subprocess(hf_path, init, step_batches, *, gpu_budget_gb, label):
    """Run an FT training in a subprocess so GPU memory is fully reclaimed."""
    with tempfile.TemporaryDirectory() as td:
        init_pkl = os.path.join(td, "init.pkl")
        batch_pkl = os.path.join(td, "batches.pkl")
        out_pkl = os.path.join(td, "losses.pkl")
        with open(init_pkl, "wb") as f:
            pickle.dump(init, f)
        with open(batch_pkl, "wb") as f:
            pickle.dump(
                [[(s.tokens.cpu(), s.targets.cpu()) for s in b]
                 for b in step_batches],
                f,
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--ft-worker", hf_path, init_pkl, batch_pkl, out_pkl,
             str(gpu_budget_gb), label],
            check=True, env=env,
        )
        with open(out_pkl, "rb") as f:
            return pickle.load(f)


def _cleanup(am):
    am.buffers.destroy()
    del am
    try:
        from flextrain.engine import unregister_all_process_pinned_memory
        unregister_all_process_pinned_memory()
    except Exception:
        pass
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    hf_path = os.path.join(ROOT, "models", "Llama-3.2-1B")
    if not os.path.isdir(hf_path):
        raise FileNotFoundError(
            f"Llama-3.2-1B weights not found at {hf_path}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")

    with open(os.path.join(hf_path, "config.json")) as f:
        hf_cfg = json.load(f)

    # Generate fixed LoRA inits.
    init = _gen_lora_init_values(hf_path)
    n_layers = hf_cfg["num_hidden_layers"]
    print(f"LoRA init: r={LORA_R}, alpha={LORA_ALPHA}, "
          f"{n_layers} layers × {len(LORA_TARGET_NAMES)} targets = "
          f"{n_layers * len(LORA_TARGET_NAMES)} (A,B) pairs")

    # Pull batches.
    print(f"\nPreparing {N_STEPS} batches × {TARGET_TOKENS_PER_STEP} tokens (Llama-3 tokenizer)...")
    step_batches = _pull_step_batches(
        hf_path, n_steps=N_STEPS,
        target_tokens_per_step=TARGET_TOKENS_PER_STEP,
    )
    print(f"  {len(step_batches)} batches ready")

    # ---- Run HF PEFT ----
    hf_losses = _run_hf_peft(hf_path, init, step_batches)
    print(f"\n  HF PEFT final loss: {hf_losses[-1]:.4f} (first: {hf_losses[0]:.4f})")
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU free after HF subprocess: "
          f"{torch.cuda.mem_get_info()[0]/(1<<30):.2f} GiB")

    # ---- Run each FT config in its own subprocess for clean memory ----
    print("\n=== FT LoRA, full-save (24 GiB GPU budget, subprocess) ===")
    ft_full_losses = _run_ft_subprocess(
        hf_path, init, step_batches,
        gpu_budget_gb=24.0, label="ft-full",
    )
    print(f"  FT full-save final loss: {ft_full_losses[-1]:.4f}")
    print(f"  GPU free: {torch.cuda.mem_get_info()[0]/(1<<30):.2f} GiB")

    # 8 GiB on a 1B-param Llama-3.2 forces:
    #  - n_gpu_layers < n_layers (param ring rotation H2D)
    #  - small act buffer (most activations on host)
    print("\n=== FT LoRA, offloaded (8 GiB GPU budget, subprocess) ===")
    ft_offl_losses = _run_ft_subprocess(
        hf_path, init, step_batches,
        gpu_budget_gb=8.0, label="ft-offload",
    )
    print(f"  FT offloaded final loss: {ft_offl_losses[-1]:.4f}")

    # ---- Compare ----
    out_dir = os.path.join(ROOT, "parity_results", "lora_dense_llama")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "loss_curves.csv")
    with open(csv_path, "w") as f:
        f.write("step,hf_peft,ft_full,ft_offload\n")
        for i in range(N_STEPS):
            f.write(f"{i},{hf_losses[i]:.6f},{ft_full_losses[i]:.6f},{ft_offl_losses[i]:.6f}\n")

    # Compute per-step deltas.
    max_hf_full = max(abs(h - f) for h, f in zip(hf_losses, ft_full_losses))
    max_hf_offl = max(abs(h - o) for h, o in zip(hf_losses, ft_offl_losses))
    max_full_offl = max(abs(f - o) for f, o in zip(ft_full_losses, ft_offl_losses))

    print(f"\n=== Comparison ===")
    print(f"  HF PEFT  vs FT-full     max |Δ| = {max_hf_full:.4f}")
    print(f"  HF PEFT  vs FT-offload  max |Δ| = {max_hf_offl:.4f}")
    print(f"  FT-full  vs FT-offload  max |Δ| = {max_full_offl:.4f}")

    # Acceptance gates.
    assert max_full_offl < 0.05, (
        f"FT determinism violated: full vs offload max |Δ| = {max_full_offl:.4f} "
        f"(expected < 0.05)"
    )
    assert max_hf_full < 0.10, (
        f"HF PEFT vs FT-full diverge: max |Δ| = {max_hf_full:.4f}"
    )

    summary = os.path.join(out_dir, "summary.md")
    with open(summary, "w") as f:
        f.write(
            f"# Phase 3: Llama-3.2-1B LoRA E2E parity\n\n"
            f"100-step training on MathInstruct, identical LoRA inits "
            f"copied across HF PEFT, FT full-save, and FT offloaded.\n\n"
            f"## Results\n\n"
            f"| pair | max \\|Δ\\| over 100 steps |\n"
            f"|---|---|\n"
            f"| HF PEFT vs FT-full | {max_hf_full:.4f} |\n"
            f"| HF PEFT vs FT-offload | {max_hf_offl:.4f} |\n"
            f"| FT-full vs FT-offload | {max_full_offl:.4f} |\n\n"
            f"| run | first loss | last loss |\n"
            f"|---|---|---|\n"
            f"| HF PEFT | {hf_losses[0]:.4f} | {hf_losses[-1]:.4f} |\n"
            f"| FT-full | {ft_full_losses[0]:.4f} | {ft_full_losses[-1]:.4f} |\n"
            f"| FT-offload | {ft_offl_losses[0]:.4f} | {ft_offl_losses[-1]:.4f} |\n"
        )
    print(f"\nCSV: {csv_path}\nSummary: {summary}")
    print("\n✓ Phase 3 PASSED")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--hf-worker":
        _hf_peft_worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) >= 7 and sys.argv[1] == "--ft-worker":
        _ft_worker(
            hf_path=sys.argv[2], init_pkl=sys.argv[3],
            batch_pkl=sys.argv[4], out_pkl=sys.argv[5],
            gpu_budget_gb=float(sys.argv[6]), label=sys.argv[7],
        )
    else:
        main()
