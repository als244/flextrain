"""End-to-end parity: Qwen3-1.7B — FlexTrain vs pure PyTorch, on
MathInstruct.

Same structure as ``test_llama32_1b_parity.py`` but for Qwen3.
Qwen3-specific bits:
* QK-norm (``self_attn.q_norm.weight`` / ``k_norm.weight``).
* No attention biases.
* Default rope base 1e6.
* Tied embedding (small variants; lm_head absent from checkpoint).

Produces ``parity_results/qwen3_1b7/`` with live CSVs + summary.md.
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

from flextrain.bench.parity import (  # noqa: E402
    ModelShape,
    NaiveQwen3Model,
    _Seq,
    _flextrain_step,
    _naive_step,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


# Reuse the Q/K permutation + tokenizer helpers from the Llama test
# — the RoPE-convention fixup is identical (both Llama and Qwen3
# Triton RoPE use pair-interleave; HF uses halved-split).
from tests.test_llama32_1b_parity import (  # noqa: E402
    _halved_to_pair_perm,
    _permute_qk_for_pair_interleave,
    _live_curve_writer,
)


def _build_qwen3_17b_shape() -> ModelShape:
    # Config:
    #   d_model=2048, n_layers=28, n_heads=16, n_kv_heads=8,
    #   head_dim=128, expert_dim=6144, vocab=151936
    #   rms_norm_eps=1e-6, rope_theta=1e6, tied embed
    return ModelShape(
        d_model=2048,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        expert_dim=6144,
        vocab_size=151936,
        rms_norm_eps=1e-6,
        rope_base=1_000_000.0,
    )


# ---------------------------------------------------------------------------
# Load HF safetensors into NaiveQwen3Model.
# ---------------------------------------------------------------------------


def _load_hf_weights_into_qwen3_naive(
    hf_path: str, naive: NaiveQwen3Model, n_layers: int,
) -> None:
    """Populate NaiveQwen3Model from Qwen3-1.7B safetensors."""
    from safetensors.torch import safe_open

    needed = {
        "model.embed_tokens.weight": ("embed",),
        "model.norm.weight": ("final_norm",),
        "lm_head.weight": ("head",),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        for hf_name, attr in [
            (p + "input_layernorm.weight", ("block", i, "w_attn_norm")),
            (p + "self_attn.q_proj.weight", ("block", i, "w_q")),
            (p + "self_attn.k_proj.weight", ("block", i, "w_k")),
            (p + "self_attn.v_proj.weight", ("block", i, "w_v")),
            (p + "self_attn.o_proj.weight", ("block", i, "w_o")),
            (p + "self_attn.q_norm.weight", ("block", i, "w_q_norm")),
            (p + "self_attn.k_norm.weight", ("block", i, "w_k_norm")),
            (p + "post_attention_layernorm.weight", ("block", i, "w_ffn_norm")),
            (p + "mlp.gate_proj.weight", ("block", i, "w_1")),
            (p + "mlp.up_proj.weight", ("block", i, "w_3")),
            (p + "mlp.down_proj.weight", ("block", i, "w_2")),
        ]:
            needed[hf_name] = attr

    idx_path = os.path.join(hf_path, "model.safetensors.index.json")
    tensor_cache: dict[str, torch.Tensor] = {}
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)["weight_map"]
        by_file: dict[str, list[str]] = {}
        for name in needed:
            if name in idx:
                by_file.setdefault(idx[name], []).append(name)
        for shard, names in by_file.items():
            with safe_open(os.path.join(hf_path, shard),
                           framework="pt", device="cpu") as f:
                for n in names:
                    tensor_cache[n] = f.get_tensor(n)
    else:
        with safe_open(os.path.join(hf_path, "model.safetensors"),
                       framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for name in needed:
                if name in keys:
                    tensor_cache[name] = f.get_tensor(name)

    with torch.no_grad():
        naive.w_tok_embeddings.copy_(
            tensor_cache["model.embed_tokens.weight"].to(DTYPE).to(DEVICE)
        )
        naive.w_final_norm.copy_(
            tensor_cache["model.norm.weight"].to(DTYPE).to(DEVICE)
        )
        # Tied embed fallback.
        head_hf = tensor_cache.get(
            "lm_head.weight", tensor_cache["model.embed_tokens.weight"]
        )
        naive.w_head_proj.copy_(head_hf.T.contiguous().to(DTYPE).to(DEVICE))

        head_dim = naive.blocks[0].head_dim
        for i, block in enumerate(naive.blocks):
            p = f"model.layers.{i}."
            block.w_attn_norm.copy_(
                tensor_cache[p + "input_layernorm.weight"].to(DTYPE).to(DEVICE)
            )
            block.w_ffn_norm.copy_(
                tensor_cache[p + "post_attention_layernorm.weight"].to(DTYPE).to(DEVICE)
            )
            # Qwen3-specific: QK norm — 1-D per-head_dim vector.
            # HF applies it in halved-split layout; our tensors are in
            # pair-interleave layout (post Q/K permute). So the norm
            # weight vector itself needs the halved→pair permutation
            # over its head_dim axis.
            def _perm_norm_vec(w: torch.Tensor) -> torch.Tensor:
                half = head_dim // 2
                perm = []
                for i in range(half):
                    perm.append(i)
                    perm.append(half + i)
                return w[perm].contiguous()

            block.w_q_norm.copy_(_perm_norm_vec(
                tensor_cache[p + "self_attn.q_norm.weight"].to(DTYPE).to(DEVICE)
            ))
            block.w_k_norm.copy_(_perm_norm_vec(
                tensor_cache[p + "self_attn.k_norm.weight"].to(DTYPE).to(DEVICE)
            ))
            for hf_name, attr in [
                ("self_attn.q_proj.weight", "w_q"),
                ("self_attn.k_proj.weight", "w_k"),
                ("self_attn.v_proj.weight", "w_v"),
                ("self_attn.o_proj.weight", "w_o"),
                ("mlp.gate_proj.weight", "w_1"),
                ("mlp.up_proj.weight", "w_3"),
                ("mlp.down_proj.weight", "w_2"),
            ]:
                hf_t = tensor_cache[p + hf_name]
                w = hf_t.T.contiguous().to(DTYPE).to(DEVICE)
                if attr in ("w_q", "w_k"):
                    w = _permute_qk_for_pair_interleave(w, head_dim)
                getattr(block, attr).copy_(w)


def _pull_step_batches(
    tokenizer_path: str, n_steps: int, target_tokens_per_step: int,
    min_len: int = 128, max_len: int = 512,
) -> list[list[_Seq]]:
    """MathInstruct batches tokenized with Qwen3 tokenizer, loss mask
    on the prompt."""
    import json
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    local_path = os.path.join(
        ROOT, "datasets", "MathInstruct", "MathInstruct.json"
    )
    print(f"loading MathInstruct from {local_path}...")
    with open(local_path) as f:
        records = json.load(f)
    print(f"  {len(records)} records")
    it = iter(records)

    def _build_seq() -> _Seq | None:
        while True:
            try:
                rec = next(it)
            except StopIteration:
                return None
            q = rec.get("instruction", "") or ""
            a = rec.get("output", "") or ""
            prompt_text = f"Problem: {q}\nSolution: "
            response_text = a
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            response_ids = tok.encode(response_text, add_special_tokens=False)
            total_ids = prompt_ids + response_ids
            if len(total_ids) < min_len:
                continue
            if len(total_ids) > max_len:
                if len(prompt_ids) >= max_len:
                    continue
                response_ids = response_ids[: max_len - len(prompt_ids)]
                total_ids = prompt_ids + response_ids
            tokens = torch.tensor(total_ids, dtype=torch.int64)
            targets = torch.roll(tokens, -1)
            targets[: len(prompt_ids)] = -100
            targets[-1] = -100
            seq = _Seq(tokens)
            seq.targets = targets
            return seq

    step_batches: list[list[_Seq]] = []
    for _ in range(n_steps):
        batch, total = [], 0
        while total < target_tokens_per_step:
            seq = _build_seq()
            if seq is None:
                raise RuntimeError("MathInstruct exhausted")
            batch.append(seq)
            total += len(seq)
        step_batches.append(batch)
    return step_batches


def _run_naive(hf_path: str, shape: ModelShape, step_batches, lr: float,
               *, live_path: str | None = None):
    print("\n=== naive Qwen3 PyTorch reference ===")
    print("building model...")
    naive = NaiveQwen3Model(shape).to(DEVICE)
    print("loading HF weights...")
    _load_hf_weights_into_qwen3_naive(hf_path, naive, shape.n_layers)
    opt = torch.optim.AdamW(
        naive.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.0,
    )
    curve = []
    live = _live_curve_writer(live_path, "step,loss") if live_path else None
    t0 = time.time()
    for step, batch in enumerate(step_batches):
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        # Preserve -100 targets
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        ts = time.time()
        loss = _naive_step(naive, opt, seqs, DEVICE)
        curve.append(loss)
        if live is not None:
            live(step, loss)
        if step < 5 or step % 10 == 0 or step == len(step_batches) - 1:
            print(
                f"  naive step {step:4d}  loss={loss:.4f}  "
                f"step={(time.time()-ts)*1000:.0f}ms  "
                f"elapsed={time.time()-t0:.1f}s", flush=True,
            )
    del naive, opt
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return curve


def _build_flextrain_qwen3(shape: ModelShape, lr: float, device: str,
                            n_gpu_layers: int, act_buffer_gib: float):
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.qwen3 import (
        Qwen3DenseBlock, Qwen3DenseBlockConfig,
    )
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    cfg = Qwen3DenseBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [Qwen3DenseBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
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
        expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
    )
    working_set = WorkingSetConfig(
        target_round_tokens=2048, max_chunk_size=2048,
        max_training_chunks=4, max_total_round_tokens=2048,
        target_num_rounds=1,
        n_gpu_layers=n_gpu_layers,
        n_gpu_grads=n_gpu_layers,
        n_gpu_opt_layers=n_gpu_layers,
        gpu_act_buffer_size=int(act_buffer_gib * (1 << 30)),
        host_act_buffer_size=int(4 * (1 << 30)),
        available_gpu_memory_bytes=int(24 * (1 << 30)),
        available_host_memory_bytes=int(96 * (1 << 30)),
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=2048, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    opt = AdamW(
        AdamWHyperparams(lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
        state_dtype=torch.bfloat16,
    )
    return ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=opt,
        working_set=working_set, hw_cost=hw_cost, dims=dims, device=device,
    )


def _run_flextrain(hf_path: str, shape: ModelShape, step_batches, lr: float,
                   *, label: str, n_gpu_layers: int, act_buffer_gib: float,
                   live_path: str | None = None):
    print(f"\n=== FlexTrain Qwen3 ({label}) ===")
    am = _build_flextrain_qwen3(shape, lr, DEVICE, n_gpu_layers, act_buffer_gib)
    print("loading HF weights into FT host buffers...")
    am.load_hf(hf_path, strict=False)
    # Tied embed: copy embed → head.T.
    hh = am.buffers.host_head_params["w_head_proj"]
    if hh.abs().sum().item() == 0:
        hh.copy_(am.buffers.host_embed_params["w_tok_embeddings"].T)
    # Q/K RoPE convention fixup (weights + the QK-norm vectors).
    half = shape.head_dim // 2
    norm_perm = []
    for i in range(half):
        norm_perm.append(i)
        norm_perm.append(half + i)
    for i in range(shape.n_layers):
        for name in ("w_q", "w_k"):
            w = am.buffers.host_params[i][name]
            am.buffers.host_params[i][name].copy_(
                _permute_qk_for_pair_interleave(w, shape.head_dim)
            )
        # And the QK-norm weight vectors (halved → pair layout).
        for name in ("w_q_norm", "w_k_norm"):
            w = am.buffers.host_params[i][name]
            am.buffers.host_params[i][name].copy_(w[norm_perm].contiguous())
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    curve = []
    live = _live_curve_writer(live_path, "step,loss") if live_path else None
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
        if step < 5 or step % 10 == 0 or step == len(step_batches) - 1:
            print(
                f"  FT({label}) step {step:4d}  loss={loss:.4f}  "
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


def test_qwen3_17b_e2e_parity() -> None:
    hf_path = os.path.join(ROOT, "models", "Qwen3-1.7B")
    math_path = os.path.join(ROOT, "datasets", "MathInstruct", "MathInstruct.json")
    if not os.path.isdir(hf_path):
        raise FileNotFoundError(
            f"Qwen3-1.7B weights not found. "
            f"Download: huggingface-cli download Qwen/Qwen3-1.7B "
            f"--local-dir {hf_path}"
        )
    if not os.path.isfile(math_path):
        raise FileNotFoundError(f"MathInstruct not found at {math_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")

    shape = _build_qwen3_17b_shape()
    n_steps = 200
    target_tokens_per_step = 2048
    step_batches = _pull_step_batches(
        hf_path, n_steps=n_steps,
        target_tokens_per_step=target_tokens_per_step,
    )
    print(f"Prepared {len(step_batches)} steps")

    lr = 5e-5
    out_dir = os.path.join(ROOT, "parity_results", "qwen3_1b7")
    os.makedirs(out_dir, exist_ok=True)

    naive_curve = _run_naive(
        hf_path, shape, step_batches, lr,
        live_path=os.path.join(out_dir, "live_naive.csv"),
    )
    ft_all_curve = _run_flextrain(
        hf_path, shape, step_batches, lr,
        label="all-resident", n_gpu_layers=shape.n_layers,
        act_buffer_gib=8.0,  # 28 layers × ~280MB = 7.8 GB for opt ring
        live_path=os.path.join(out_dir, "live_flextrain_all_resident.csv"),
    )
    ft_offload_curve = _run_flextrain(
        hf_path, shape, step_batches, lr,
        label="offload-half", n_gpu_layers=max(1, shape.n_layers // 2),
        act_buffer_gib=8.0,
        live_path=os.path.join(out_dir, "live_flextrain_offload_half.csv"),
    )

    # Summary files
    csv_path = os.path.join(out_dir, "loss_curves.csv")
    with open(csv_path, "w") as f:
        f.write("step,naive_pytorch,flextrain_all_resident,flextrain_offload_half\n")
        for i, (ln, la, lo) in enumerate(
            zip(naive_curve, ft_all_curve, ft_offload_curve)
        ):
            f.write(f"{i},{ln:.6f},{la:.6f},{lo:.6f}\n")
    print(f"\nCSV: {csv_path}")

    def _avg(c, a, b=None):
        return sum(c[a:b]) / len(c[a:b])

    avg_first_n = _avg(naive_curve, 0, 10)
    avg_last_n = _avg(naive_curve, -10)
    avg_first_a = _avg(ft_all_curve, 0, 10)
    avg_last_a = _avg(ft_all_curve, -10)
    avg_first_o = _avg(ft_offload_curve, 0, 10)
    avg_last_o = _avg(ft_offload_curve, -10)
    max_abs_a = max(abs(a - b) for a, b in zip(naive_curve, ft_all_curve))
    max_abs_o = max(abs(a - b) for a, b in zip(naive_curve, ft_offload_curve))

    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Qwen3-1.7B E2E parity — FlexTrain vs pure PyTorch\n\n")
        f.write(
            f"- Model: Qwen3-1.7B (28 layers, d_model=2048, QK-norm, tied embed)\n"
            f"- {n_steps} steps × ~{target_tokens_per_step} tokens/step on MathInstruct (SFT, prompt-masked)\n"
            f"- lr = {lr}, AdamW, bf16 params + bf16 grads + bf16 opt state\n\n"
            f"## Convergence\n\n"
            f"| run | first-10 avg | last-10 avg | Δ (train down) |\n"
            f"|---|---|---|---|\n"
            f"| naive PyTorch | {avg_first_n:.4f} | {avg_last_n:.4f} | "
            f"{avg_last_n - avg_first_n:+.4f} |\n"
            f"| FlexTrain all-resident | {avg_first_a:.4f} | {avg_last_a:.4f} | "
            f"{avg_last_a - avg_first_a:+.4f} |\n"
            f"| FlexTrain offload-half | {avg_first_o:.4f} | {avg_last_o:.4f} | "
            f"{avg_last_o - avg_first_o:+.4f} |\n\n"
            f"## Parity\n\n"
            f"| FT config | max per-step \\|Δ\\| | \\|Δ last-10\\| |\n"
            f"|---|---|---|\n"
            f"| all-resident | {max_abs_a:.4f} | "
            f"{abs(avg_last_n - avg_last_a):.4f} |\n"
            f"| offload-half | {max_abs_o:.4f} | "
            f"{abs(avg_last_n - avg_last_o):.4f} |\n"
        )
    print(f"Summary: {md_path}")
    assert avg_last_n < avg_first_n, "naive didn't learn"
    assert avg_last_a < avg_first_a, "FT all-resident didn't learn"
    assert avg_last_o < avg_first_o, "FT offload didn't learn"
    assert max_abs_a < 2.0, f"FT all-resident diverged: {max_abs_a:.4f}"
    assert max_abs_o < 2.0, f"FT offload diverged: {max_abs_o:.4f}"
    print("\n✓ Qwen3-1.7B E2E parity PASSED")


def _run_all() -> None:
    test_qwen3_17b_e2e_parity()


if __name__ == "__main__":
    _run_all()
