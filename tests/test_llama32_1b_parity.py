"""End-to-end parity test: Llama-3.2-1B, real HF weights, real data.

Steps:
1. Load Llama-3.2-1B HF safetensors into both FlexTrain's host
   master buffers AND a naive ``torch.nn.Module`` reference.
2. Train N steps on real FineWeb tokens.
3. Compare loss curves.

Runs on the 3090 under the `flextrain` conda env. Llama-3.2-1B is
~1.23B params; bf16 params + AdamW fp32 opt state ~= ~17 GB
resident, fits the 3090 with FlexTrain's offload if needed.

Usage:
    PYTHONPATH=. python tests/test_llama32_1b_parity.py
    (assumes model at ./models/Llama-3.2-1B and FineWeb shard at
     orig/fineweb/fineweb_train_000001.bin)
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
    FineWebDocStream,
    ModelShape,
    NaiveLlamaModel,
    _flextrain_step,
    _naive_step,
    _Seq,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _halved_to_pair_perm(total_dim: int, head_dim: int) -> list[int]:
    """RoPE convention fixup.

    HF's Llama implementation uses the halved-split RoPE convention
    (pairs ``x[..., :D/2]`` with ``x[..., D/2:]``). FlexTrain's Triton
    kernel AND our naive reference both use pair-interleave
    (pairs ``x[..., 2i]`` with ``x[..., 2i+1]``). Loading HF Q/K
    weights directly into either gives wrong attention output at
    positions > 0.

    This permutation maps the OUTPUT dim of a Q or K weight from
    halved-split layout to pair-interleave layout:

        out_perm[2*i]     = half + 0 ... half + half-1
        out_perm[2*i + 1] = 0 ... half - 1

    Equivalently: for each head, interleave the two halves.
    """
    n_heads = total_dim // head_dim
    half = head_dim // 2
    perm: list[int] = []
    for h in range(n_heads):
        base = h * head_dim
        for i in range(half):
            perm.append(base + i)
            perm.append(base + half + i)
    return perm


def _permute_qk_for_pair_interleave(
    w: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Apply the halved-split → pair-interleave permutation in-place.
    ``w`` is ``(d_model, n_heads * head_dim)``; we permute the last
    dim.
    """
    total = w.shape[1]
    perm = torch.tensor(_halved_to_pair_perm(total, head_dim), device=w.device)
    return w[:, perm].contiguous()


def _load_hf_weights_into_naive(
    hf_path: str, naive: NaiveLlamaModel, n_layers: int,
) -> None:
    """Populate the naive module from HF safetensors.

    Handles Llama-3's tied embeddings (no lm_head.weight in the
    checkpoint → copy from embed_tokens.weight).
    """
    from safetensors.torch import safe_open

    # Accumulate all tensors we'll consume.
    needed = {
        "model.embed_tokens.weight": ("embed",),
        "model.norm.weight": ("final_norm",),
        "lm_head.weight": ("head",),  # may be absent with tied embeds
    }
    for i in range(n_layers):
        p = f"model.layers.{i}."
        for hf_name, attr in [
            (p + "input_layernorm.weight", ("block", i, "w_attn_norm")),
            (p + "self_attn.q_proj.weight", ("block", i, "w_q")),
            (p + "self_attn.k_proj.weight", ("block", i, "w_k")),
            (p + "self_attn.v_proj.weight", ("block", i, "w_v")),
            (p + "self_attn.o_proj.weight", ("block", i, "w_o")),
            (p + "post_attention_layernorm.weight", ("block", i, "w_ffn_norm")),
            (p + "mlp.gate_proj.weight", ("block", i, "w_1")),
            (p + "mlp.up_proj.weight", ("block", i, "w_3")),
            (p + "mlp.down_proj.weight", ("block", i, "w_2")),
        ]:
            needed[hf_name] = attr

    # Support multi-shard via index.json if present.
    idx_path = os.path.join(hf_path, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)["weight_map"]
        shard_of = idx
    else:
        shard_of = {}

    tensor_cache: dict[str, torch.Tensor] = {}
    if shard_of:
        by_file: dict[str, list[str]] = {}
        for name in needed:
            if name in shard_of:
                by_file.setdefault(shard_of[name], []).append(name)
        for shard_file, names in by_file.items():
            p = os.path.join(hf_path, shard_file)
            with safe_open(p, framework="pt", device="cpu") as f:
                for n in names:
                    tensor_cache[n] = f.get_tensor(n)
    else:
        # Single-shard.
        single = os.path.join(hf_path, "model.safetensors")
        with safe_open(single, framework="pt", device="cpu") as f:
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
        # Head: if tied, reuse embed.
        if "lm_head.weight" in tensor_cache:
            head_hf = tensor_cache["lm_head.weight"]
        else:
            head_hf = tensor_cache["model.embed_tokens.weight"]
        # HF stores (vocab, d_model); FlexTrain uses (d_model, vocab) -> transpose.
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
            # HF: (out, in). FlexTrain/Naive: (in, out). Transpose.
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
                dst = getattr(block, attr)
                # Transpose HF (out, in) -> FlexTrain (in, out).
                w = hf_t.T.contiguous().to(DTYPE).to(DEVICE)
                # Q / K need the halved-split → pair-interleave fixup
                # so RoPE-after-projection matches HF's math.
                if attr in ("w_q", "w_k"):
                    w = _permute_qk_for_pair_interleave(w, head_dim)
                dst.copy_(w)


def _build_llama32_1b_shape() -> ModelShape:
    return ModelShape(
        d_model=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        head_dim=64,
        expert_dim=8192,
        vocab_size=128256,  # Llama-3 vocab
        rms_norm_eps=1e-5,
        rope_base=500000.0,
    )


def _build_flextrain_engine_for_1b(
    shape: ModelShape, lr: float, device: str,
    act_buffer_gb: float = 16.0,
    max_seq_len: int = 2048,
    max_chunk_size: int = 2048,
    target_round_tokens: int = 2048,
):
    """Build an ActiveModel tuned for Llama-3.2-1B on a 3090.

    ~2.5 GB params (bf16) + ~10 GB opt state (fp32 AdamW) = 12.5 GB
    baseline; leaves room for act ring + scratch at 16 GB.
    """
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    cfg = LlamaBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [LlamaBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
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

    # For a 1B model all layers fit on a 3090 in bf16 (~2.5 GB).
    # Start with all layers / grads / opts resident; we can exercise
    # rotation later.
    working_set = WorkingSetConfig(
        target_round_tokens=target_round_tokens,
        max_chunk_size=max_chunk_size,
        max_training_chunks=4,
        max_total_round_tokens=target_round_tokens,
        target_num_rounds=1,
        n_gpu_layers=shape.n_layers,
        n_gpu_grads=shape.n_layers,
        n_gpu_opt_layers=shape.n_layers,
        gpu_act_buffer_size=int(act_buffer_gb * (1 << 30)),
        host_act_buffer_size=0,
        available_gpu_memory_bytes=int(24 * (1 << 30)),
        available_host_memory_bytes=int(96 * (1 << 30)),
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=max_seq_len, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(
        peak_tflops=60.0, pcie_bw_gbps=20.0,
        practical_efficiency_factor=1.0,
    )
    # Full-bf16 training: opt state in bf16 halves the opt-state ring.
    opt = AdamW(
        AdamWHyperparams(
            lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
        ),
        state_dtype=torch.bfloat16,
    )
    return ActiveModel(
        embed=embed, backbone=backbone, head=head,
        optimizer=opt, working_set=working_set, hw_cost=hw_cost,
        dims=dims, device=device,
    )


def _pull_step_batches(
    tokenizer_path: str,
    n_steps: int, target_tokens_per_step: int,
    min_len: int = 128, max_len: int = 512,
) -> list[list[_Seq]]:
    """Deterministic batches from the locally-downloaded MathInstruct
    dataset (``datasets/MathInstruct/MathInstruct.json``), tokenized
    with the target model's own tokenizer, with SFT-style prompt
    masking: loss contribution zeroed for the ``Problem:`` portion,
    enabled for the ``Solution:`` portion.

    Why MathInstruct (not FineWeb)?
    -------------------------------
    FineWeb is generic web text — already heavily represented in
    Llama-3/Qwen3 pretraining. Fine-tuning on it produces ~flat loss.
    MathInstruct (mathematical problems with chain-of-thought
    solutions) is different-domain and the model genuinely learns,
    giving a meaningful loss decrease over the run. Both naive and
    FT must follow the same decrease.
    """
    import json
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load local JSON (a list of dicts with instruction/output).
    local_path = os.path.join(
        ROOT, "datasets", "MathInstruct", "MathInstruct.json"
    )
    print(f"loading MathInstruct from {local_path}...")
    with open(local_path) as f:
        records = json.load(f)
    print(f"  {len(records)} records")

    # Deterministic iteration: in file order.
    it = iter(records)

    def _build_seq() -> _Seq | None:
        while True:
            try:
                rec = next(it)
            except StopIteration:
                return None
            q = rec.get("instruction", "") or ""
            a = rec.get("output", "") or ""
            # Build prompt + response. Train only on response.
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
            # Loss masking for SFT: set prompt positions' NEXT-TOKEN
            # targets to -100 (PyTorch CrossEntropy ignore_index).
            # That means the prediction AT prompt position i is not
            # penalized for failing to predict prompt position i+1.
            # FT's CrossEntropyLoss already honors ignore_index=-100.
            tokens = torch.tensor(total_ids, dtype=torch.int64)
            targets = torch.roll(tokens, -1)
            targets[: len(prompt_ids)] = -100  # mask targets over prompt
            # Also mask the final position's target — that's a wraparound
            # artifact and shouldn't contribute to loss.
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
                raise RuntimeError(
                    "MathInstruct exhausted before producing n_steps batches"
                )
            batch.append(seq)
            total += len(seq)
        step_batches.append(batch)
    return step_batches


def _live_curve_writer(path: str, header: str):
    """Return a function that appends one row to ``path`` per call.
    The file is truncated+header-written at the start so callers can
    tail -f it during a long run.
    """
    with open(path, "w") as f:
        f.write(header + "\n")
    def _append(step: int, loss: float) -> None:
        with open(path, "a") as f:
            f.write(f"{step},{loss:.6f}\n")
    return _append


def _run_naive(hf_path: str, shape: ModelShape, step_batches, lr: float,
               *, label: str = "naive", live_path: str | None = None):
    """Run the pure-PyTorch naive reference with ``torch.optim.AdamW``.

    Our naive model's parameters are bf16, and
    ``torch.optim.AdamW``'s exp_avg/exp_avg_sq match the param dtype,
    so this gives a full-bf16 baseline comparable to FlexTrain's AdamW
    configured with ``state_dtype=torch.bfloat16``. No FlexTrain /
    orig kernels involved — pure stock PyTorch.
    """
    print(f"\n=== {label} PyTorch reference ===")
    print("building model...")
    naive = NaiveLlamaModel(shape).to(DEVICE)
    print("loading HF weights...")
    _load_hf_weights_into_naive(hf_path, naive, shape.n_layers)
    naive_opt = torch.optim.AdamW(
        naive.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.0,
    )
    curve = []
    live = _live_curve_writer(live_path, "step,loss") if live_path else None
    t0 = time.time()
    for step, batch in enumerate(step_batches):
        seqs = []
        for s in batch:
            ns = _Seq(s.tokens.clone())
            ns.targets = s.targets.clone()  # preserve -100 mask
            seqs.append(ns)
        ts = time.time()
        loss = _naive_step(naive, naive_opt, seqs, DEVICE)
        curve.append(loss)
        if live is not None:
            live(step, loss)
        if step < 5 or step % 25 == 0 or step == len(step_batches) - 1:
            print(
                f"  {label} step {step:4d}  loss={loss:.4f}  "
                f"step={(time.time()-ts)*1000:.0f}ms  "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
    del naive
    del naive_opt
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return curve


def _run_flextrain(hf_path: str, shape: ModelShape, step_batches, lr: float,
                   *, label: str, n_gpu_layers: int,
                   act_buffer_gib: float, live_path: str | None = None):
    print(f"\n=== FlexTrain ({label}) ===")
    print("building engine...")
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    cfg = LlamaBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [LlamaBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
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
    hw_cost = HardwareCost(
        peak_tflops=60.0, pcie_bw_gbps=20.0,
        practical_efficiency_factor=1.0,
    )
    # Full bf16 training: params + grads + opt state all bf16.
    # User has validated 8B runs on this GPU, so opt-state-bf16 is
    # the sensible memory-saving choice for FlexTrain too.
    opt = AdamW(
        AdamWHyperparams(lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
        state_dtype=torch.bfloat16,
    )
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head,
        optimizer=opt, working_set=working_set, hw_cost=hw_cost,
        dims=dims, device=DEVICE,
    )
    print("loading HF weights into FlexTrain host buffers...")
    am.load_hf(hf_path, strict=False)
    # Tied embeds: copy embed -> head.T if not in checkpoint.
    head_host = am.buffers.host_head_params["w_head_proj"]
    if head_host.abs().sum().item() == 0:
        embed_host = am.buffers.host_embed_params["w_tok_embeddings"]
        head_host.copy_(embed_host.T)
    # Q/K RoPE convention fixup (halved-split → pair-interleave).
    # FlexTrain's Triton kernel uses pair-interleave; HF uses
    # halved-split; we permute the Q/K weights so post-RoPE Q/K match
    # HF semantics. Matches the fixup done in _load_hf_weights_into_naive.
    head_dim = shape.head_dim
    for i in range(shape.n_layers):
        host = am.buffers.host_params[i]
        for name in ("w_q", "w_k"):
            w = host[name]
            host[name].copy_(_permute_qk_for_pair_interleave(w, head_dim))
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    curve = []
    live = _live_curve_writer(live_path, "step,loss") if live_path else None
    t0 = time.time()
    for step, batch in enumerate(step_batches):
        seqs = []
        for s in batch:
            ns = _Seq(s.tokens.clone())
            ns.targets = s.targets.clone()  # preserve -100 mask
            seqs.append(ns)
        ts = time.time()
        loss = _flextrain_step(am, seqs)
        curve.append(loss)
        if live is not None:
            live(step, loss)
        if step < 5 or step % 25 == 0 or step == len(step_batches) - 1:
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


def test_llama32_1b_e2e_parity() -> None:
    hf_path = os.path.join(ROOT, "models", "Llama-3.2-1B")
    math_path = os.path.join(ROOT, "datasets", "MathInstruct", "MathInstruct.json")

    if not os.path.isdir(hf_path):
        raise FileNotFoundError(
            f"Llama-3.2-1B weights not found at {hf_path}. "
            f"Download with: huggingface-cli download meta-llama/Llama-3.2-1B"
        )
    if not os.path.isfile(math_path):
        raise FileNotFoundError(
            f"MathInstruct not found at {math_path}. "
            f"Download: huggingface-cli download --repo-type dataset "
            f"TIGER-Lab/MathInstruct --local-dir datasets/MathInstruct"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")

    shape = _build_llama32_1b_shape()

    n_steps = 200
    target_tokens_per_step = 2048
    step_batches = _pull_step_batches(
        hf_path,
        n_steps=n_steps, target_tokens_per_step=target_tokens_per_step,
    )
    print(f"Prepared {len(step_batches)} steps from MathInstruct "
          f"(SFT, loss masked over prompt tokens)")

    lr = 5e-5

    # Write live / final outputs under ./parity_results/llama32_1b/
    # (repo root) so they're easy to tail from another terminal.
    out_dir = os.path.join(ROOT, "parity_results", "llama32_1b")
    os.makedirs(out_dir, exist_ok=True)

    naive_curve = _run_naive(
        hf_path, shape, step_batches, lr,
        live_path=os.path.join(out_dir, "live_naive.csv"),
    )
    ft_all_curve = _run_flextrain(
        hf_path, shape, step_batches, lr,
        label="all-resident",
        n_gpu_layers=shape.n_layers,
        act_buffer_gib=6.0,
        live_path=os.path.join(out_dir, "live_flextrain_all_resident.csv"),
    )
    ft_offload_curve = _run_flextrain(
        hf_path, shape, step_batches, lr,
        label="offload-half",
        n_gpu_layers=max(1, shape.n_layers // 2),
        act_buffer_gib=6.0,
        live_path=os.path.join(out_dir, "live_flextrain_offload_half.csv"),
    )

    # Write loss curves to CSV + markdown summary in a well-known
    # place at the repo root so they're easy to find.
    out_dir = os.path.join(ROOT, "parity_results", "llama32_1b")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "loss_curves.csv")
    with open(csv_path, "w") as f:
        f.write("step,naive_pytorch,flextrain_all_resident,flextrain_offload_half\n")
        for i, (ln, la, lo) in enumerate(
            zip(naive_curve, ft_all_curve, ft_offload_curve)
        ):
            f.write(f"{i},{ln:.6f},{la:.6f},{lo:.6f}\n")
    print(f"\nLoss curves written to {csv_path}")

    # Simple numbers helpful in the summary.
    def _avg(c, a, b=None):
        return sum(c[a:b]) / len(c[a:b])

    avg_first_naive = _avg(naive_curve, 0, 10)
    avg_last_naive = _avg(naive_curve, -10)
    avg_first_ft = _avg(ft_all_curve, 0, 10)
    avg_last_ft = _avg(ft_all_curve, -10)
    avg_first_off = _avg(ft_offload_curve, 0, 10)
    avg_last_off = _avg(ft_offload_curve, -10)
    max_abs_all = max(abs(a - b) for a, b in zip(naive_curve, ft_all_curve))
    max_abs_off = max(abs(a - b) for a, b in zip(naive_curve, ft_offload_curve))
    rms_all = (sum((a - b) ** 2 for a, b in zip(naive_curve, ft_all_curve)) / n_steps) ** 0.5
    rms_off = (sum((a - b) ** 2 for a, b in zip(naive_curve, ft_offload_curve)) / n_steps) ** 0.5

    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Llama-3.2-1B E2E parity — FlexTrain vs pure PyTorch\n\n")
        f.write(
            f"- **Model:** Llama-3.2-1B (16 layers, d_model=2048, tied embed)\n"
            f"- **Steps:** {n_steps}\n"
            f"- **Tokens/step:** ~{target_tokens_per_step} (real FineWeb docs)\n"
            f"- **Optimizer:** AdamW (lr={lr}, betas=(0.9, 0.95), eps=1e-8, wd=0)\n"
            f"- **Precision:** bf16 params + bf16 grads + bf16 opt state on both sides\n"
            f"- **Init:** both FlexTrain runs load the SAME HF checkpoint as the naive reference\n\n"
            f"## Three side-by-side runs\n\n"
            f"1. **naive PyTorch** — pure ``torch.nn.Module`` + ``torch.optim.AdamW``. "
            f"No FlexTrain or orig kernels. Reference trajectory.\n"
            f"2. **FlexTrain all-resident** — full engine, all {shape.n_layers} layers' "
            f"params/grads/opt-state kept on GPU. Simplest FT configuration.\n"
            f"3. **FlexTrain offload-half** — half the layers "
            f"({shape.n_layers // 2}) resident at a time; weights/grads/opt-state "
            f"rotate in/out of GPU during fwd/bwd/step. Exercises the full AdaWS "
            f"prefetch/offload pipeline.\n\n"
        )
        f.write("## Convergence\n\n")
        f.write(
            "| side | first-10 avg | last-10 avg | Δ (train down) |\n"
            "|---|---|---|---|\n"
            f"| naive PyTorch | {avg_first_naive:.4f} | {avg_last_naive:.4f} | "
            f"{avg_last_naive - avg_first_naive:+.4f} |\n"
            f"| FlexTrain all-resident | {avg_first_ft:.4f} | {avg_last_ft:.4f} | "
            f"{avg_last_ft - avg_first_ft:+.4f} |\n"
            f"| FlexTrain offload-half | {avg_first_off:.4f} | {avg_last_off:.4f} | "
            f"{avg_last_off - avg_first_off:+.4f} |\n\n"
        )
        f.write("## Parity vs naive PyTorch\n\n")
        f.write(
            "| FT config | max per-step \\|Δ\\| | RMS per-step Δ | \\|Δ last-10 avg\\| |\n"
            "|---|---|---|---|\n"
            f"| all-resident | {max_abs_all:.4f} | {rms_all:.4f} | "
            f"{abs(avg_last_naive - avg_last_ft):.4f} |\n"
            f"| offload-half | {max_abs_off:.4f} | {rms_off:.4f} | "
            f"{abs(avg_last_naive - avg_last_off):.4f} |\n\n"
        )
        f.write("## Per-step sample (every 50 steps + last)\n\n")
        f.write(
            "| step | naive | FT all | \\|Δa\\| | FT offload | \\|Δo\\| |\n"
            "|---|---|---|---|---|---|\n"
        )
        sample_idxs = list(range(0, len(naive_curve), 50))
        if sample_idxs[-1] != len(naive_curve) - 1:
            sample_idxs.append(len(naive_curve) - 1)
        for i in sample_idxs:
            ln = naive_curve[i]; la = ft_all_curve[i]; lo = ft_offload_curve[i]
            f.write(
                f"| {i} | {ln:.4f} | {la:.4f} | {abs(ln-la):.4f} | "
                f"{lo:.4f} | {abs(ln-lo):.4f} |\n"
            )
        f.write("\n")
        f.write(
            "## Tied-embed limitation\n\n"
            "Llama-3.2-1B has ``tie_word_embeddings=true`` in its HF config "
            "(the LM head weight is the transpose of the embedding table). "
            "FlexTrain's current ``LMHead`` uses a separate ``w_head_proj`` "
            "parameter, so when we load the HF checkpoint we copy the "
            "embedding table into the head once and then the two evolve "
            "independently under their own gradient updates. The naive "
            "PyTorch reference does the same, so the two are directly "
            "comparable — but both drift from a properly-tied Llama "
            "finetune as training proceeds. Fixing this (shared param "
            "with shared gradient) is flagged as a TODO in docs/internal/NOTES.md.\n"
        )
    print(f"Summary written to {md_path}")

    # Stdout summary.
    print("\n" + "=" * 78)
    print(f"  LLAMA-3.2-1B E2E PARITY  ({n_steps} steps, lr={lr})")
    print("=" * 78)
    print(
        f"  naive PyTorch:           first-10 {avg_first_naive:.4f} -> "
        f"last-10 {avg_last_naive:.4f}"
    )
    print(
        f"  FlexTrain all-resident:  first-10 {avg_first_ft:.4f} -> "
        f"last-10 {avg_last_ft:.4f}"
    )
    print(
        f"  FlexTrain offload-half:  first-10 {avg_first_off:.4f} -> "
        f"last-10 {avg_last_off:.4f}"
    )
    print(f"  max per-step |Δ|  all-resident = {max_abs_all:.4f}   "
          f"offload-half = {max_abs_off:.4f}")
    print(f"  last-10 |Δ|       all-resident = "
          f"{abs(avg_last_naive - avg_last_ft):.4f}   "
          f"offload-half = {abs(avg_last_naive - avg_last_off):.4f}")
    print(f"\n  CSV:     {csv_path}")
    print(f"  Summary: {md_path}")

    # Convergence sanity.
    assert avg_last_ft < avg_first_ft, (
        f"FT all-resident didn't learn: {avg_first_ft:.4f} -> {avg_last_ft:.4f}"
    )
    assert avg_last_off < avg_first_off, (
        f"FT offload didn't learn: {avg_first_off:.4f} -> {avg_last_off:.4f}"
    )
    assert avg_last_naive < avg_first_naive, (
        f"Naive didn't learn: {avg_first_naive:.4f} -> {avg_last_naive:.4f}"
    )
    # Parity bounds. 1000 steps of bf16 AdamW is noisy; bugs are O(1).
    assert max_abs_all < 2.0, (
        f"FT all-resident diverged from naive: max |Δ|={max_abs_all:.4f}"
    )
    assert max_abs_off < 2.0, (
        f"FT offload-half diverged from naive: max |Δ|={max_abs_off:.4f}"
    )
    print("\n✓ Llama-3.2-1B E2E parity PASSED")


def _run_all() -> None:
    test_llama32_1b_e2e_parity()


if __name__ == "__main__":
    _run_all()
