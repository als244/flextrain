"""8B LoRA correctness diagnostics — step-0 only, multi-config.

Goal: localize the FT-vs-HF systematic loss bias on 8B Llama. Step 0
has B=0 so the LoRA branch contributes nothing; any difference is
purely the *forward-pass* of the bf16 base model.

Three checks, in order of cost:

1. **FT-vs-FT bit-identity across two working set configs** (auto
   solver vs forced full-residency cap). If both FT runs match
   bit-for-bit on per-token loss, the FT engine is deterministic and
   the FT-vs-HF gap is purely cross-stack kernel choice.

2. **Per-token loss diff: FT vs HF** at step 0 on the same input.
   Localizes the gap from a scalar to a ``(T,)`` vector — we can see
   *which positions* diverge most and by how much.

3. **LoRA A/B grad diff** (post-bwd, pre-step) FT vs HF, again at
   step 0. Tests bwd parity through the LoRA wrapper (B=0 means
   ``dL/dA = 0`` exactly in both stacks; ``dL/dB`` is the interesting
   signal).

All three checks share one input batch (1 sequence, 256 tokens) for
speed, so the whole diagnostic runs in well under a minute per FT
config.
"""
from __future__ import annotations

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

from flextrain.bench.parity import _Seq

from tests.test_llama32_1b_parity import (  # noqa: E402
    _halved_to_pair_perm,
    _permute_qk_for_pair_interleave,
    _pull_step_batches,
)


DEVICE = "cuda:0"
# Override with FT_DIAG_DTYPE=float32 to run everything fp32 (rules out
# bf16 noise as the cause of cross-stack disagreement).
_DTYPE_NAME = os.environ.get("FT_DIAG_DTYPE", "bfloat16")
DTYPE = {"bfloat16": torch.bfloat16, "float32": torch.float32}[_DTYPE_NAME]
print(f"[diag] using DTYPE = {_DTYPE_NAME}")

FT_TO_HF = {
    "w_q": "q_proj", "w_k": "k_proj", "w_v": "v_proj", "w_o": "o_proj",
    "w_1": "gate_proj", "w_3": "up_proj", "w_2": "down_proj",
}
LORA_TARGET_NAMES = tuple(FT_TO_HF.keys())

LORA_R = 16
LORA_ALPHA = 16.0
SEQ_LEN = 256
LR = 1e-4


def _gen_lora_init_values(hf_path: str) -> dict:
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
# HF PEFT subprocess: runs step 0, returns per-token CE + LoRA B grads.
# ===========================================================================


def _hf_peft_worker(hf_path, init_pkl, batch_pkl, out_pkl):
    import torch as _t
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    with open(init_pkl, "rb") as f:
        init = pickle.load(f)
    with open(batch_pkl, "rb") as f:
        tokens, targets = pickle.load(f)

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

    # Forward + backward at step 0.
    tokens = tokens.to(DEVICE).unsqueeze(0)         # (1, T)
    targets = targets.to(DEVICE)                    # (T,)
    T = int(targets.shape[0])

    # HF expects shifted labels: labels[i+1] = our_targets[i].
    hf_labels = _t.full((T,), -100, dtype=_t.int64, device=DEVICE)
    hf_labels[1:] = targets[:-1]

    # Forward through the model with output_hidden_states so we can
    # compare per-layer activations across stacks.
    out = model(
        input_ids=tokens,
        labels=None,                 # we'll compute CE explicitly
        output_hidden_states=True,
    )
    # logits: (1, T, V); compute per-position CE explicitly.
    # FT computes loss at position i using target ``our_targets[i]``;
    # ignored positions (target == -100) are zeroed.
    logits_2d = out.logits[0]  # (T, V), bf16
    # CE in fp32 to match FT's internal reduction precision.
    per_token_xent_full = F.cross_entropy(
        logits_2d.float(), targets,
        reduction="none", ignore_index=-100,
    )
    valid_mask = targets != -100
    per_token_xent = per_token_xent_full * valid_mask.float()
    active = int(valid_mask.sum().item())

    # Backward: sum(per-token-CE).backward() — FT's fwd_bwd at
    # loss_scale_factor=1.0 produces grads = sum of per-token grads,
    # so this matches.
    per_token_xent.sum().backward()

    # Collect LoRA B grads (B is fp32, A grad is identically zero
    # because B=0 at step 0 → dL/dA = dL/dW @ B^T * scale = 0).
    lora_b_grads = {}
    lora_a_grads = {}
    for L in range(n_layers):
        for tgt, hf_name in FT_TO_HF.items():
            if hf_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parent = model.model.model.layers[L].self_attn
            else:
                parent = model.model.model.layers[L].mlp
            lora_layer = getattr(parent, hf_name)
            # PEFT layout: B is (d_out, r); transpose to FT's (r, d_out).
            lora_b_grads[(L, tgt)] = lora_layer.lora_B["default"].weight.grad.detach().t().contiguous().cpu()
            lora_a_grads[(L, tgt)] = lora_layer.lora_A["default"].weight.grad.detach().t().contiguous().cpu()

    # Hidden states: tuple of (n_layers + 1) tensors, each (1, T, d_model).
    # hidden_states[0] = embeddings, hidden_states[i] = output of layer i-1.
    hidden = [h[0].detach().to(torch.float32).cpu() for h in out.hidden_states]

    summed = float(per_token_xent.sum().item())
    payload = {
        "per_token_xent": per_token_xent.detach().cpu(),
        "active": active,
        "loss_mean": summed / max(1, active),
        "lora_b_grads": lora_b_grads,
        "lora_a_grads": lora_a_grads,
        "hidden_states": hidden,
        "logits": logits_2d.detach().to(torch.float32).cpu(),
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)


def _run_hf(hf_path, init, tokens, targets):
    print("\n=== HF PEFT step-0 (subprocess) ===")
    with tempfile.TemporaryDirectory() as td:
        init_pkl = os.path.join(td, "init.pkl")
        batch_pkl = os.path.join(td, "batch.pkl")
        out_pkl = os.path.join(td, "out.pkl")
        with open(init_pkl, "wb") as f:
            pickle.dump(init, f)
        with open(batch_pkl, "wb") as f:
            pickle.dump((tokens.cpu(), targets.cpu()), f)
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
# FT subprocess: runs step 0, returns per-token CE + LoRA grads.
# ===========================================================================


def _build_ft_model(hf_path, hf_cfg, init, *, gpu_budget_gb, label):
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
    backbone = []
    for i in range(n_layers):
        base = LlamaBlock(layer_id=i, cfg=cfg)
        wrapped = LoRAWrapperLayer(
            base, lora_targets="all",
            rank=LORA_R, alpha=LORA_ALPHA, dims=dims,
            adapter_compute_dtype=DTYPE,
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

    print(f"  [{label}] solving working set with GPU cap = {gpu_budget_gb:.1f} GiB...", flush=True)
    working_set = determine_working_set_config(
        model_dims=dict(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv,
            head_dim=head_dim, expert_dim=inter, vocab_size=vocab,
            n_layers=n_layers, num_shared_experts=1, num_routed_experts=0,
            top_k=0, is_causal=True,
            datatypes={"embed": _DTYPE_NAME, "head_proj": _DTYPE_NAME,
                       "attn_proj": _DTYPE_NAME, "expert_proj": _DTYPE_NAME,
                       "router": _DTYPE_NAME, "norm": _DTYPE_NAME,
                       "residual": _DTYPE_NAME},
        ),
        max_seq_len=SEQ_LEN, max_global_batch_tokens=SEQ_LEN,
        training_config={"master_weight_dtype": _DTYPE_NAME,
                         "grad_dtype": _DTYPE_NAME,
                         "opt_choice": "AdamW", "opt_dtype": "float32"},
        has_embed=True, has_head=True, num_local_layers=n_layers,
        max_gpu_mem_bytes=int(gpu_budget_gb * (1 << 30)),
        max_host_mem_bytes=int(110 * (1 << 30)),
        leeway_gpu_mem_bytes=int(2 * (1 << 30)),
        leeway_host_mem_bytes=int(4 * (1 << 30)),
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


def _ft_worker(hf_path, init_pkl, batch_pkl, out_pkl, label, gpu_budget_str):
    from flextrain.bench.parity import _Seq
    from flextrain.nn.loss import CrossEntropyLoss

    class _LogitsCapturingCE(CrossEntropyLoss):
        """Wrap CE so we capture per-chunk logits at step 0."""
        def __init__(self):
            super().__init__()
            self.captured: list[torch.Tensor] = []

        def compute(self, logits, token_slice, *, loss_scale, per_token_loss_out):
            self.captured.append(logits.detach().to(torch.float32).cpu().clone())
            return super().compute(
                logits, token_slice,
                loss_scale=loss_scale,
                per_token_loss_out=per_token_loss_out,
            )

    gpu_budget_gb = float(gpu_budget_str)

    with open(init_pkl, "rb") as f:
        init = pickle.load(f)
    with open(batch_pkl, "rb") as f:
        tokens_cpu, targets_cpu = pickle.load(f)

    with open(os.path.join(hf_path, "config.json")) as f:
        hf_cfg = json.load(f)

    am, n_layers = _build_ft_model(
        hf_path, hf_cfg, init,
        gpu_budget_gb=gpu_budget_gb, label=label,
    )

    # Step 0: do fwd+bwd ONLY (no opt.step), capture per-token loss +
    # LoRA grads in one pass. Grads are summed (no /active scaling) so
    # they match HF's hand-rolled "summed_loss.backward()" convention.
    seq = _Seq(tokens_cpu)
    seq.targets = targets_cpu
    active = int((targets_cpu != -100).sum().item())
    capturing_ce = _LogitsCapturingCE()
    stats = am.fwd_bwd(
        [seq], loss_scale_factor=1.0, verbose=False,
        loss_fn=capturing_ce,
    )
    per_token = seq.per_token_loss.detach().clone()
    loss_mean = stats.total_loss / max(1, active)
    # Concat logits across head sub-chunks into one (T, V) tensor.
    logits_full = torch.cat(capturing_ce.captured, dim=0) if capturing_ce.captured else None

    # After fwd_bwd, the engine has offloaded grads for *every* layer
    # to host buffers via ``offload_layer_grads`` (the last N_G also
    # stay GPU-resident, but the host copy is always written). Wait for
    # those DMAs before reading.
    torch.cuda.synchronize()
    lora_a_grads = {}
    lora_b_grads = {}
    for L in range(n_layers):
        for tgt in LORA_TARGET_NAMES:
            ga_key = f"g_{tgt[2:]}_lora_a"
            gb_key = f"g_{tgt[2:]}_lora_b"
            ga = am.buffers.host_grads[L].get(ga_key)
            gb = am.buffers.host_grads[L].get(gb_key)
            lora_a_grads[(L, tgt)] = ga.detach().cpu() if ga is not None else None
            lora_b_grads[(L, tgt)] = gb.detach().cpu() if gb is not None else None

    payload = {
        "per_token_loss": per_token,
        "active": active,
        "loss_mean": loss_mean,
        "lora_a_grads": lora_a_grads,
        "lora_b_grads": lora_b_grads,
        "n_gpu_layers": am.working_set.n_gpu_layers,
        "n_layers": n_layers,
        "logits": logits_full,
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)


def _run_ft(hf_path, init, tokens, targets, *, label, gpu_budget_gb):
    print(f"\n=== FT {label} (subprocess, GPU cap {gpu_budget_gb} GiB) ===")
    with tempfile.TemporaryDirectory() as td:
        init_pkl = os.path.join(td, "init.pkl")
        batch_pkl = os.path.join(td, "batch.pkl")
        out_pkl = os.path.join(td, "out.pkl")
        with open(init_pkl, "wb") as f:
            pickle.dump(init, f)
        with open(batch_pkl, "wb") as f:
            pickle.dump((tokens.cpu(), targets.cpu()), f)
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--ft-worker", hf_path, init_pkl, batch_pkl, out_pkl,
             label, f"{gpu_budget_gb}"],
            check=True, env=env,
        )
        with open(out_pkl, "rb") as f:
            return pickle.load(f)


# ===========================================================================
# Main: orchestrate, compare.
# ===========================================================================


def _stat(name, x):
    """Print summary stats for a tensor."""
    if x is None:
        print(f"  {name}: <missing>")
        return
    mx = float(x.abs().max().item())
    mn = float(x.abs().mean().item())
    print(f"  {name:30s} shape={tuple(x.shape)} max|x|={mx:.4e} mean|x|={mn:.4e}")


def _diff_stats(name, a, b):
    if a is None or b is None:
        print(f"  {name}: missing in one stack")
        return None, None
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH a={tuple(a.shape)} b={tuple(b.shape)}")
        return None, None
    a, b = a.float(), b.float()
    delta = (a - b).abs()
    ref = b.abs().clamp_min(1e-8)
    mx = float(delta.max().item())
    rel = float((delta / ref).max().item())
    print(f"  {name:30s} max|Δ|={mx:.4e}  max-rel={rel:.4f}  ref-max={float(b.abs().max().item()):.4e}")
    return mx, rel


def _permute_hf_grad_to_ft_layout(hf, hf_cfg):
    """FT permutes w_q / w_k base AND the LoRA B's column dim from
    halved to pair-interleave layout. HF keeps the halved layout. To
    compare grads, apply the same permutation to HF's q/k LoRA B grads."""
    d_model = hf_cfg["hidden_size"]
    n_heads = hf_cfg["num_attention_heads"]
    n_kv = hf_cfg["num_key_value_heads"]
    head_dim = hf_cfg.get("head_dim") or (d_model // n_heads)
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    q_perm = torch.tensor(_halved_to_pair_perm(attn_dim, head_dim), dtype=torch.int64)
    k_perm = torch.tensor(_halved_to_pair_perm(kv_dim, head_dim), dtype=torch.int64)
    out = {}
    for key, g in hf["lora_b_grads"].items():
        L, tgt = key
        if tgt == "w_q":
            out[key] = g[:, q_perm].contiguous()
        elif tgt == "w_k":
            out[key] = g[:, k_perm].contiguous()
        else:
            out[key] = g
    hf["lora_b_grads"] = out


def main():
    # Override with FT_DIAG_MODEL=Llama-3.2-1B (or any subdir of models/)
    # to run on a smaller model. Useful with FT_DIAG_DTYPE=float32 since
    # the 8B fp32 weights don't fit in 24 GiB GPU.
    model_name = os.environ.get("FT_DIAG_MODEL", "Llama-3.1-8B")
    hf_path = os.path.join(ROOT, "models", model_name)
    if not os.path.isdir(hf_path):
        raise FileNotFoundError(f"{model_name} weights not found at {hf_path}")
    with open(os.path.join(hf_path, "config.json")) as f:
        hf_cfg = json.load(f)
    n_layers = hf_cfg["num_hidden_layers"]

    # Build a single fixed batch (1 seq × SEQ_LEN tokens) for all checks.
    print(f"Preparing 1 fixed batch × {SEQ_LEN} tokens for diagnostics...")
    step_batches = _pull_step_batches(
        hf_path, n_steps=1, target_tokens_per_step=SEQ_LEN,
    )
    seq = step_batches[0][0]
    tokens = seq.tokens
    targets = seq.targets
    active = int((targets != -100).sum().item())
    print(f"  T={tokens.shape[0]}, active positions={active}")

    init = _gen_lora_init_values(hf_path)

    # ----- Run all three configs. -----
    hf = _run_hf(hf_path, init, tokens, targets)
    print(f"  HF mean loss = {hf['loss_mean']:.6f}, active = {hf['active']}")
    _permute_hf_grad_to_ft_layout(hf, hf_cfg)

    # FT config A: large GPU budget (auto solver picks more residency).
    ft_a = _run_ft(
        hf_path, init, tokens, targets,
        label="ft-A", gpu_budget_gb=24,
    )
    print(f"  FT-A mean loss = {ft_a['loss_mean']:.6f}  n_gpu_layers={ft_a['n_gpu_layers']}/{n_layers}")

    # FT config B: tighter GPU budget → more offloading.
    ft_b = _run_ft(
        hf_path, init, tokens, targets,
        label="ft-B", gpu_budget_gb=18,
    )
    print(f"  FT-B mean loss = {ft_b['loss_mean']:.6f}  n_gpu_layers={ft_b['n_gpu_layers']}/{n_layers}")

    # =========================================================
    # 1. FT-vs-FT determinism.
    # =========================================================
    print("\n" + "="*78)
    print("1. FT-vs-FT bit-identity (cfg-A vs cfg-B)")
    print("="*78)
    _diff_stats(
        "per_token_loss",
        ft_a["per_token_loss"], ft_b["per_token_loss"],
    )
    pa = torch.cat([ft_a["lora_b_grads"][(L, t)].flatten()
                    for L in range(n_layers) for t in LORA_TARGET_NAMES])
    pb = torch.cat([ft_b["lora_b_grads"][(L, t)].flatten()
                    for L in range(n_layers) for t in LORA_TARGET_NAMES])
    _diff_stats("all_lora_B_grads (concat)", pa, pb)

    # =========================================================
    # 2. FT vs HF: forward parity.
    # =========================================================
    print("\n" + "="*78)
    print("2. Forward parity: per-token CE  FT vs HF")
    print("="*78)
    print(f"  HF  mean loss = {hf['loss_mean']:.6f}")
    print(f"  FT-A mean loss = {ft_a['loss_mean']:.6f}  Δ = {ft_a['loss_mean']-hf['loss_mean']:+.6f}")
    print(f"  FT-B mean loss = {ft_b['loss_mean']:.6f}  Δ = {ft_b['loss_mean']-hf['loss_mean']:+.6f}")
    _diff_stats(
        "per_token CE  (FT-A vs HF)",
        ft_a["per_token_loss"], hf["per_token_xent"],
    )
    # Top-K largest per-token disagreements with the position context.
    a = ft_a["per_token_loss"].float()
    b = hf["per_token_xent"].float()
    delta = (a - b).abs()
    topk = torch.topk(delta, k=min(10, delta.numel())).indices.tolist()
    print("  Top-10 disagreeing positions:")
    print(f"  {'pos':>4}  {'target':>7}  {'FT':>10}  {'HF':>10}  {'|Δ|':>10}")
    for p in topk:
        tgt = int(targets[p].item())
        print(f"  {p:>4d}  {tgt:>7d}  {a[p].item():>10.4f}  {b[p].item():>10.4f}  {delta[p].item():>10.4f}")

    # =========================================================
    # 3. FT vs HF: LoRA gradient parity.
    # =========================================================
    print("\n" + "="*78)
    print("3. Backward parity: LoRA grads  FT vs HF")
    print("   (B=0 at step 0 ⇒ dL/dA must be 0 in both stacks; dL/dB is the signal)")
    print("="*78)
    # Show a couple of representative layers + targets, then concat
    # across all for a global stat.
    for L in (0, n_layers // 2, n_layers - 1):
        for tgt in ("w_v", "w_o", "w_3"):
            _diff_stats(
                f"L{L:02d} g_{tgt}_lora_b",
                ft_a["lora_b_grads"][(L, tgt)],
                hf["lora_b_grads"][(L, tgt)],
            )
    # A grads must be zero — confirm.
    a_max_ft = max(
        float(ft_a["lora_a_grads"][(L, t)].abs().max().item()) if ft_a["lora_a_grads"][(L, t)] is not None else 0.0
        for L in range(n_layers) for t in LORA_TARGET_NAMES
    )
    a_max_hf = max(
        float(hf["lora_a_grads"][(L, t)].abs().max().item())
        for L in range(n_layers) for t in LORA_TARGET_NAMES
    )
    print(f"  max|g_A| FT={a_max_ft:.4e}  HF={a_max_hf:.4e}  (both should be 0)")

    # Global concat over all layers / targets.
    ft_b_flat = torch.cat([
        ft_a["lora_b_grads"][(L, t)].flatten()
        for L in range(n_layers) for t in LORA_TARGET_NAMES
    ])
    hf_b_flat = torch.cat([
        hf["lora_b_grads"][(L, t)].flatten()
        for L in range(n_layers) for t in LORA_TARGET_NAMES
    ])
    _diff_stats("ALL g_lora_B (concat, FT-A vs HF)", ft_b_flat, hf_b_flat)

    # Per-layer summary: does the bias drift with depth?
    print("\n  Per-layer max|Δ g_lora_B| (FT-A vs HF), all targets:")
    print(f"  {'L':>3s}  {'max|Δ|':>10s}  {'max|ref|':>10s}  {'rel':>8s}")
    for L in range(n_layers):
        delta_max = 0.0
        ref_max = 0.0
        for t in LORA_TARGET_NAMES:
            a = ft_a["lora_b_grads"][(L, t)].float()
            b = hf["lora_b_grads"][(L, t)].float()
            delta_max = max(delta_max, float((a - b).abs().max().item()))
            ref_max = max(ref_max, float(b.abs().max().item()))
        rel = delta_max / max(ref_max, 1e-8)
        print(f"  {L:>3d}  {delta_max:>10.4e}  {ref_max:>10.4e}  {rel:>8.4f}")

    # =========================================================
    # 4. Logit-level diff (rules out CE-kernel-numerics from the
    #    per-token CE divergence — if logits already disagree at
    #    the head output, the gap is upstream of CE).
    # =========================================================
    print("\n" + "="*78)
    print("4. Logit parity (FT vs HF) — pre-CE")
    print("="*78)
    if ft_a.get("logits") is not None:
        L_ft = ft_a["logits"].float()
        L_hf = hf["logits"].float()
        if L_ft.shape == L_hf.shape:
            print(f"  shape: {tuple(L_ft.shape)} (T, V)")
            delta = (L_ft - L_hf).abs()
            print(f"  max|Δ_logit|     = {float(delta.max().item()):.4e}")
            print(f"  mean|Δ_logit|    = {float(delta.mean().item()):.4e}")
            ref = L_hf.abs().clamp_min(1e-6)
            rel = (delta / ref)
            print(f"  max relative err = {float(rel.max().item()):.4f}")
            print(f"  HF logits  max|x|={float(L_hf.abs().max().item()):.3e}  mean|x|={float(L_hf.abs().mean().item()):.3e}")
            print(f"  FT logits  max|x|={float(L_ft.abs().max().item()):.3e}  mean|x|={float(L_ft.abs().mean().item()):.3e}")

            # Per-position max|Δ| and the position with the worst diff.
            per_pos_max = delta.max(dim=-1).values  # (T,)
            print()
            print("  Top-5 positions by max|Δ_logit|:")
            top5 = torch.topk(per_pos_max, k=min(5, per_pos_max.numel())).indices.tolist()
            print(f"  {'pos':>4} {'tgt':>7} {'max|Δ|':>11} {'argmax_FT':>10} {'argmax_HF':>10} {'pred_match':>11}")
            argmax_ft = L_ft.argmax(dim=-1)
            argmax_hf = L_hf.argmax(dim=-1)
            for p in top5:
                tgt = int(targets[p].item()) if targets[p] != -100 else -1
                af, ah = int(argmax_ft[p].item()), int(argmax_hf[p].item())
                print(f"  {p:>4d} {tgt:>7d} {per_pos_max[p].item():>11.4e} {af:>10d} {ah:>10d} {str(af==ah):>11}")

            # Logit-mean and logit-norm by position — does FT's residual
            # stream have systematically different scale than HF's?
            print()
            print("  Logit norm at first/middle/last positions (||logits[t]||):")
            T = L_ft.shape[0]
            for p in (0, T // 2, T - 1):
                print(f"  pos {p:3d}: ||FT||={L_ft[p].norm().item():.4e}  ||HF||={L_hf[p].norm().item():.4e}  ||Δ||={(L_ft[p]-L_hf[p]).norm().item():.4e}")
        else:
            print(f"  shape mismatch: FT {tuple(L_ft.shape)} vs HF {tuple(L_hf.shape)}")

    # =========================================================
    # 5. HF hidden-state magnitudes by depth (sanity check)
    # =========================================================
    print("\n" + "="*78)
    print("5. HF hidden-state magnitudes by depth (sanity check)")
    print("="*78)
    for i, h in enumerate(hf["hidden_states"]):
        if i % max(1, n_layers // 8) == 0 or i == n_layers:
            label = "embed" if i == 0 else f"after layer {i-1}"
            print(f"  {label:>22s}: max|h|={float(h.abs().max().item()):.3e}  mean|h|={float(h.abs().mean().item()):.3e}")

    print("\n✓ Diagnostics complete.")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--hf-worker":
        _hf_peft_worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) >= 8 and sys.argv[1] == "--ft-worker":
        _ft_worker(
            sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
            sys.argv[6], sys.argv[7],
        )
    else:
        main()
