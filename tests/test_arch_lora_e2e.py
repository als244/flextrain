"""End-to-end LoRA correctness sweep across architectures, vs HF PEFT.

For each configured arch:

1. **FT side** -- a subprocess that mimics ``train.py``'s setup
   (``from_pretrained`` + ``JsonSFTTokenSource`` + AdamW + the standard
   training loop), then dumps the LoRA A/B init values so the parity
   side can use them. This is the same code path as ``train.py``; we
   inline a copy here rather than modifying ``train.py`` to keep the
   user-facing CLI clean.
2. **HF PEFT side** -- a sibling subprocess that builds the same model
   via ``transformers``, applies ``peft.LoraConfig``, overwrites
   ``lora_A`` / ``lora_B`` with the values FT dumped, then trains on
   the same JSONL with the same LR / batch / step count.
3. Compares the two loss curves and asserts max |Δ| within tolerance.

Modes (``--mode``):
  ``lora``  -- FT + HF PEFT parity, asserts max|Δ| < per-arch tolerance.
  ``full``  -- FT only (full param finetune). Asserts loss decreases +
              no NaNs. No HF PEFT counterpart in this mode.
  ``smoke`` -- 5 steps in lora mode, NaN check + sanity. Fast.

Usage:

    bash scripts/download_test_models.sh                  # populate models/ + datasets/
    python tests/test_arch_lora_e2e.py --list             # see configured archs
    python tests/test_arch_lora_e2e.py --arch llama-3.2-1b --mode lora
    python tests/test_arch_lora_e2e.py --all  --mode lora
    python tests/test_arch_lora_e2e.py --all  --mode full --keep-going
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Per-arch config registry.
# ---------------------------------------------------------------------------


LLAMA_DENSE_TARGETS = {
    "w_q": "q_proj", "w_k": "k_proj", "w_v": "v_proj", "w_o": "o_proj",
    "w_1": "gate_proj", "w_3": "up_proj", "w_2": "down_proj",
}
ATTN_ONLY_TARGETS = {
    "w_q": "q_proj", "w_k": "k_proj", "w_v": "v_proj", "w_o": "o_proj",
}


@dataclass(frozen=True)
class ArchSpec:
    name: str
    hf_dir: str  # under models/
    seq_len: int
    global_batch_tokens: int
    steps: int
    gpu_gib: float
    host_gib: float
    ft_to_hf: dict = field(default_factory=lambda: dict(LLAMA_DENSE_TARGETS))
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lr_lora: float = 1e-4
    lr_full: float = 3e-5
    tolerance: float = 0.15
    notes: str = ""


ARCHES: dict[str, ArchSpec] = {
    "llama-3.2-1b": ArchSpec(
        name="llama-3.2-1b", hf_dir="Llama-3.2-1B",
        seq_len=2048, global_batch_tokens=8192, steps=50,
        gpu_gib=20.0, host_gib=80.0,
        notes="Reference. Fits on a 24 GB box.",
    ),
    "llama-3.1-8b": ArchSpec(
        name="llama-3.1-8b", hf_dir="Llama-3.1-8B",
        seq_len=2048, global_batch_tokens=16384, steps=30,
        gpu_gib=70.0, host_gib=180.0,
        notes="32 layers @ 4096; HF PEFT bwd needs ~40 GiB peak.",
    ),
    "qwen3-1.7b": ArchSpec(
        name="qwen3-1.7b", hf_dir="Qwen3-1.7B",
        seq_len=2048, global_batch_tokens=8192, steps=50,
        gpu_gib=24.0, host_gib=80.0,
        notes="QK-norm + tied embeddings.",
    ),
    "qwen3-8b": ArchSpec(
        name="qwen3-8b", hf_dir="Qwen3-8B",
        seq_len=2048, global_batch_tokens=16384, steps=30,
        gpu_gib=70.0, host_gib=180.0,
    ),
    "qwen3.5-2b": ArchSpec(
        name="qwen3.5-2b", hf_dir="Qwen3.5-2B",
        ft_to_hf=dict(ATTN_ONLY_TARGETS),
        # Smaller batch than the typical 8192: working_set's baseline
        # accounting still allocates full FT optimizer state for embed
        # + head + every backbone layer even in LoRA mode (the
        # solver doesn't yet know which params are frozen). The 2B's
        # 248k vocab makes embed+head ~7.6 GiB of "required" GPU
        # baseline, which compresses the host activation pool. A 4096
        # global batch keeps the per-round activation tier under the
        # available host slot count.
        seq_len=2048, global_batch_tokens=4096, steps=50,
        gpu_gib=24.0, host_gib=80.0,
        notes=(
            "Qwen3.5 hybrid linear-attn + full-attn. ATTN_ONLY_TARGETS "
            "for parity: HF stores split in_proj_qkv/z/b/a in linear-"
            "attn layers but FT bundles them into w_lin_qkvz/w_lin_ba; "
            "rank-r LoRA on bundled != rank-r on each split, so "
            "linear-attn projections are not parity-comparable. We "
            "LoRA only the full-attn q/k/v/o (which exist on full-"
            "attention layers, every 4th layer)."
        ),
    ),
    "qwen3.5-9b": ArchSpec(
        name="qwen3.5-9b", hf_dir="Qwen3.5-9B",
        ft_to_hf=dict(ATTN_ONLY_TARGETS),
        seq_len=2048, global_batch_tokens=8192, steps=20,
        gpu_gib=22.0, host_gib=100.0,
        notes=(
            "Qwen3.5-9B (grp=2 GVA on linear-attn layers). Same "
            "ATTN_ONLY_TARGETS reasoning as 2B."
        ),
    ),
    "qwen3.6-27b": ArchSpec(
        name="qwen3.6-27b", hf_dir="Qwen3.6-27B",
        ft_to_hf=dict(ATTN_ONLY_TARGETS),
        seq_len=2048, global_batch_tokens=8192, steps=5,
        gpu_gib=22.5, host_gib=110.0, tolerance=0.20,
        notes=(
            "Qwen3.6-27B dense; reuses Qwen3_5 arch_id "
            "(Qwen3_5ForConditionalGeneration). 64 layers, "
            "linear-attn grp=3 GVA (n_v=48, n_k=16). "
            "Multimodal wrapper, tie_word_embeddings=False."
        ),
    ),
    "qwen2.5-1.5b": ArchSpec(
        name="qwen2.5-1.5b", hf_dir="Qwen2.5-1.5B",
        seq_len=2048, global_batch_tokens=8192, steps=50,
        gpu_gib=20.0, host_gib=80.0,
        notes="qkv_bias path.",
    ),
    "mistral-7b": ArchSpec(
        name="mistral-7b", hf_dir="Mistral-7B-v0.1",
        seq_len=2048, global_batch_tokens=16384, steps=30,
        gpu_gib=70.0, host_gib=180.0,
    ),
    "olmoe-1b-7b": ArchSpec(
        name="olmoe-1b-7b", hf_dir="OLMoE-1B-7B-0924",
        ft_to_hf=dict(ATTN_ONLY_TARGETS),
        seq_len=2048, global_batch_tokens=4096, steps=30,
        gpu_gib=40.0, host_gib=120.0, tolerance=0.20,
        notes=(
            "Attention-only LoRA (HF PEFT shares an adapter across the "
            "batched OlmoeExperts op; FT uses per-expert adapters)."
        ),
    ),
    "qwen3-moe-30b": ArchSpec(
        name="qwen3-moe-30b", hf_dir="Qwen3-30B-A3B",
        ft_to_hf=dict(ATTN_ONLY_TARGETS),
        seq_len=2048, global_batch_tokens=65536, steps=5,
        gpu_gib=22.5, host_gib=110.0, tolerance=0.20,
        notes=(
            "Qwen3-MoE 30B-A3B (128 experts, top-K=8, 48 layers). "
            "Attn-only LoRA at batch=65536 fits 24 GiB GPU + 117 GiB host. "
            "Same shape as Qwen3.5-MoE-35B-A3B run."
        ),
    ),
    "qwen3.5-moe-35b": ArchSpec(
        name="qwen3.5-moe-35b", hf_dir="Qwen3.5-35B-A3B",
        ft_to_hf=dict(ATTN_ONLY_TARGETS),
        seq_len=512, global_batch_tokens=512, steps=5,
        gpu_gib=22.5, host_gib=110.0, tolerance=0.20,
        notes=(
            "Hybrid linear-attn + full-attn MoE; 256 routed experts, "
            "top-K=8, shared expert with sigmoid gate. Attn-only LoRA "
            "on full-attention layers (every 4th layer). Working set "
            "fits within 24 GiB GPU + 117 GiB host."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Subprocess plumbing.
# ---------------------------------------------------------------------------


def _stream(cmd: list[str], log_path: str | None = None) -> tuple[int, str]:
    """Stream subprocess output to stdout while capturing for parsing."""
    print("\n$ " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd, cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    captured: list[str] = []
    log_fh = open(log_path, "w") if log_path else None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)
            if log_fh is not None:
                log_fh.write(line)
                log_fh.flush()
    finally:
        proc.wait()
        if log_fh is not None:
            log_fh.close()
    return proc.returncode, "".join(captured)


# ---------------------------------------------------------------------------
# FT worker -- mimics train.py's setup + loop. Inlined so we can dump
# LoRA inits for parity without bloating train.py.
# ---------------------------------------------------------------------------


def _ft_worker_main():
    """Sibling subprocess: replicates the ``train.py`` skeleton (same
    ``from_pretrained``, same ``JsonSFTTokenSource``, same AdamW +
    LR schedule + step loop), and dumps LoRA A/B inits to a pickle so
    the HF PEFT side can mirror them.

    Why inline instead of importing _run_training_loop from train.py:
    train.py uses ``argparse.Namespace`` for its args + a couple of
    LR-schedule helpers; we reproduce the relevant minimum here so the
    harness owns the parity-specific glue without train.py needing
    to expose hooks.
    """
    import torch as _t
    from flextrain import from_pretrained
    from flextrain.io.sources import JsonSFTTokenSource
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    p = argparse.ArgumentParser(prog="ft-worker", add_help=False)
    p.add_argument("--ft-worker", action="store_true")
    p.add_argument("--hf-path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", choices=("lora", "full"), required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--global-batch-tokens", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--gpu-gib", type=float, required=True)
    p.add_argument("--host-gib", type=float, required=True)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--init-dump-pkl", default=None)
    p.add_argument("--losses-out-pkl", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(sys.argv[1:])

    optimizer = AdamW(
        AdamWHyperparams(
            lr=args.lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
        ),
        state_dtype=(_t.float32 if args.mode == "lora" else _t.bfloat16),
    )
    lora_targets = "all" if args.mode == "lora" else None

    am = from_pretrained(
        args.hf_path,
        optimizer=optimizer,
        max_seq_len=args.seq_len,
        max_global_batch_tokens=args.global_batch_tokens,
        max_gpu_mem_bytes=int(args.gpu_gib * (1 << 30)),
        max_host_mem_bytes=int(args.host_gib * (1 << 30)),
        device=args.device,
        leeway_gpu_mem_bytes=int(5 * (1 << 30)),
        leeway_host_mem_bytes=int(8 * (1 << 30)),
        lora_targets=lora_targets,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        strict=False, verbose=True,
    )

    # Dump LoRA inits before training. The HF PEFT worker replays these.
    if args.mode == "lora" and args.init_dump_pkl:
        dump: dict[tuple[int, str, str], "_t.Tensor"] = {}
        for L, host in enumerate(am.buffers.host_params):
            for nm, t in host.items():
                if nm.endswith("_lora_a"):
                    dump[(L, nm[: -len("_lora_a")], "lora_a")] = (
                        t.detach().cpu().clone()
                    )
                elif nm.endswith("_lora_b"):
                    dump[(L, nm[: -len("_lora_b")], "lora_b")] = (
                        t.detach().cpu().clone()
                    )
        with open(args.init_dump_pkl, "wb") as f:
            pickle.dump(dump, f)
        print(f"  [ft] dumped {len(dump)} LoRA init tensors -> "
              f"{args.init_dump_pkl}", flush=True)

    source = JsonSFTTokenSource(
        path=args.dataset, tokenizer=args.hf_path,
        min_seq_len=32, max_seq_len=args.seq_len, loop=True,
    )

    losses: list[float] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        seqs = source.get_sequences(max_token_count=args.global_batch_tokens)
        if not seqs:
            print(f"  [ft] dataset exhausted at step {step}", flush=True)
            break
        step_tokens = sum(len(s) for s in seqs)
        stats = am.fwd_bwd(
            seqs,
            loss_scale_factor=1.0 / step_tokens,
            total_tokens_per_step=step_tokens,
        )
        am.step()
        _t.cuda.synchronize()
        avg = stats.total_loss / stats.total_tokens
        losses.append(float(avg))
        if step <= 3 or step % 5 == 0 or step == args.steps:
            print(
                f"  [ft] step {step:3d}/{args.steps}: loss={avg:.4f}  "
                f"elapsed={time.time()-t0:.1f}s  "
                f"max_alloc={_t.cuda.max_memory_allocated()/(1<<30):.1f}GiB",
                flush=True,
            )

    with open(args.losses_out_pkl, "wb") as f:
        pickle.dump(losses, f)
    print(f"  [ft] wrote {len(losses)} losses -> {args.losses_out_pkl}",
          flush=True)


# ---------------------------------------------------------------------------
# HF PEFT worker.
# ---------------------------------------------------------------------------


def _hf_peft_worker_main():
    """Sibling subprocess. Loads the HF model, applies PEFT LoRA,
    overwrites lora_A/lora_B with FT-side inits, then trains on the
    same JSONL with the same hyperparams."""
    import torch as _t
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    p = argparse.ArgumentParser(prog="hf-peft-worker", add_help=False)
    p.add_argument("--hf-peft-worker", action="store_true")
    p.add_argument("--hf-path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--init-pkl", required=True)
    p.add_argument("--losses-out-pkl", required=True)
    p.add_argument("--targets", required=True,
                    help="JSON dict of FT->HF target name pairs")
    p.add_argument("--lora-rank", type=int, required=True)
    p.add_argument("--lora-alpha", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--global-batch-tokens", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(sys.argv[1:])

    ft_to_hf = json.loads(args.targets)
    DTYPE = _t.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path, torch_dtype=DTYPE, device_map=args.device,
        attn_implementation="sdpa",
    )
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", target_modules=list(ft_to_hf.values()),
        init_lora_weights=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    # Gradient checkpointing: required for 9B+ on 24 GiB GPUs. The
    # full autograd graph for 32 layers x seq_len=2048 OOMs without
    # it. PEFT exposes ``gradient_checkpointing_enable`` on the
    # wrapped model.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()

    with open(args.init_pkl, "rb") as f:
        init: dict = pickle.load(f)

    overwrote = 0
    with _t.no_grad():
        for (L, proj, kind), t in init.items():
            hf_name = ft_to_hf.get(proj)
            if hf_name is None:
                continue
            if hf_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parent = model.model.model.layers[L].self_attn
            else:
                parent = model.model.model.layers[L].mlp
            lora_layer = getattr(parent, hf_name)
            if kind == "lora_a":
                # FT (in, r) -> HF (r, in).
                w = lora_layer.lora_A["default"].weight
                w.data.copy_(t.t().to(w.dtype).to(w.device))
            else:
                # FT (r, out) -> HF (out, r).
                w = lora_layer.lora_B["default"].weight
                w.data.copy_(t.t().to(w.dtype).to(w.device))
            overwrote += 1
    print(f"  [hf-peft] overwrote {overwrote} LoRA tensors", flush=True)

    tok = AutoTokenizer.from_pretrained(args.hf_path)
    records: list[dict] = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"  [hf-peft] loaded {len(records)} SFT records", flush=True)

    def _make_seqs():
        for rec in records:
            q = rec.get("instruction", "") or ""
            a = rec.get("output", "") or ""
            if not q or not a:
                continue
            prompt_ids = tok.encode(f"Problem: {q}\nSolution: ",
                                     add_special_tokens=False)
            response_ids = tok.encode(a, add_special_tokens=False)
            total = prompt_ids + response_ids
            if len(total) < 32:
                continue
            if len(total) > args.seq_len:
                if len(prompt_ids) >= args.seq_len:
                    continue
                response_ids = response_ids[: args.seq_len - len(prompt_ids)]
                total = prompt_ids + response_ids
            yield total, len(prompt_ids)

    seq_iter = _make_seqs()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = _t.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
    )

    losses: list[float] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch: list[tuple[list[int], int]] = []
        total = 0
        while total < args.global_batch_tokens:
            try:
                toks, plen = next(seq_iter)
            except StopIteration:
                seq_iter = _make_seqs()
                continue
            batch.append((toks, plen))
            total += len(toks)
        opt.zero_grad(set_to_none=False)
        batch_loss = 0.0
        active_total = 0
        for toks, plen in batch:
            ids = _t.tensor(toks, dtype=_t.int64, device=args.device).unsqueeze(0)
            T = ids.shape[1]
            labels = _t.full((T,), -100, dtype=_t.int64, device=args.device)
            labels[: T - 1] = ids[0, 1:]
            labels[: plen] = -100
            active = int((labels != -100).sum().item())
            out = model(input_ids=ids, labels=labels.unsqueeze(0))
            (out.loss * active).backward()
            batch_loss += float(out.loss.item()) * active
            active_total += active
        for q in trainable:
            if q.grad is not None:
                q.grad.div_(active_total)
        opt.step()
        avg = batch_loss / max(1, active_total)
        losses.append(avg)
        if step <= 3 or step % 5 == 0 or step == args.steps:
            print(
                f"  [hf-peft] step {step:3d}/{args.steps}: loss={avg:.4f}  "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    with open(args.losses_out_pkl, "wb") as f:
        pickle.dump(losses, f)
    print(f"  [hf-peft] wrote {len(losses)} losses", flush=True)


# ---------------------------------------------------------------------------
# HF full-FT worker (parity counterpart for ``--mode full``).
# ---------------------------------------------------------------------------


def _hf_full_worker_main():
    """HF Transformers full-finetune sibling. AutoModelForCausalLM in
    bf16, plain AdamW on every parameter, same SFT JSONL as the FT
    worker, same LR / batch / steps. Used as the parity reference for
    ``--mode full``."""
    import torch as _t
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(prog="hf-full-worker", add_help=False)
    p.add_argument("--hf-full-worker", action="store_true")
    p.add_argument("--hf-path", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--losses-out-pkl", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--global-batch-tokens", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(sys.argv[1:])
    DTYPE = _t.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path, torch_dtype=DTYPE, device_map=args.device,
        attn_implementation="sdpa",
    )
    model.train()

    tok = AutoTokenizer.from_pretrained(args.hf_path)
    records: list[dict] = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"  [hf-full] loaded {len(records)} SFT records", flush=True)

    def _make_seqs():
        for rec in records:
            q = rec.get("instruction", "") or ""
            a = rec.get("output", "") or ""
            if not q or not a:
                continue
            prompt_ids = tok.encode(f"Problem: {q}\nSolution: ",
                                     add_special_tokens=False)
            response_ids = tok.encode(a, add_special_tokens=False)
            total = prompt_ids + response_ids
            if len(total) < 32:
                continue
            if len(total) > args.seq_len:
                if len(prompt_ids) >= args.seq_len:
                    continue
                response_ids = response_ids[: args.seq_len - len(prompt_ids)]
                total = prompt_ids + response_ids
            yield total, len(prompt_ids)

    seq_iter = _make_seqs()
    opt = _t.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        eps=1e-8, weight_decay=0.0,
    )

    losses: list[float] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch: list[tuple[list[int], int]] = []
        total = 0
        while total < args.global_batch_tokens:
            try:
                toks, plen = next(seq_iter)
            except StopIteration:
                seq_iter = _make_seqs()
                continue
            batch.append((toks, plen))
            total += len(toks)
        opt.zero_grad(set_to_none=False)
        batch_loss = 0.0
        active_total = 0
        for toks, plen in batch:
            ids = _t.tensor(toks, dtype=_t.int64, device=args.device).unsqueeze(0)
            T = ids.shape[1]
            labels = _t.full((T,), -100, dtype=_t.int64, device=args.device)
            labels[: T - 1] = ids[0, 1:]
            labels[: plen] = -100
            active = int((labels != -100).sum().item())
            out = model(input_ids=ids, labels=labels.unsqueeze(0))
            (out.loss * active).backward()
            batch_loss += float(out.loss.item()) * active
            active_total += active
        for q in model.parameters():
            if q.grad is not None:
                q.grad.div_(active_total)
        opt.step()
        avg = batch_loss / max(1, active_total)
        losses.append(avg)
        if step <= 3 or step % 5 == 0 or step == args.steps:
            print(
                f"  [hf-full] step {step:3d}/{args.steps}: loss={avg:.4f}  "
                f"elapsed={time.time()-t0:.1f}s  "
                f"max_alloc={_t.cuda.max_memory_allocated()/(1<<30):.1f}GiB",
                flush=True,
            )

    with open(args.losses_out_pkl, "wb") as f:
        pickle.dump(losses, f)
    print(f"  [hf-full] wrote {len(losses)} losses", flush=True)


# ---------------------------------------------------------------------------
# Per-arch driver.
# ---------------------------------------------------------------------------


def _run_one_arch(
    spec: ArchSpec, *, mode: str, dataset: str, steps: int,
    gpu_gib: float, host_gib: float, smoke: bool, tol: float | None,
) -> tuple[bool, str]:
    model_dir = os.path.join(ROOT, "models", spec.hf_dir)
    if not os.path.isdir(model_dir):
        return False, f"weights missing at {model_dir} (run download script)"

    out_dir = os.path.join(ROOT, "parity_results", f"e2e_{spec.name}_{mode}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {spec.name} ({mode}) ===")
    print(f"  hf_dir={spec.hf_dir}  steps={steps}  "
           f"seq_len={spec.seq_len}  global_batch_tokens={spec.global_batch_tokens}")
    print(f"  budgets: GPU={gpu_gib} GiB, host={host_gib} GiB")
    if spec.notes:
        print(f"  note: {spec.notes}")

    with tempfile.TemporaryDirectory() as td:
        ft_log = os.path.join(out_dir, "ft.log")
        ft_losses_pkl = os.path.join(td, "ft_losses.pkl")
        init_dump = os.path.join(td, "lora_init.pkl") if mode == "lora" else None

        ft_lr = spec.lr_lora if mode == "lora" else spec.lr_full
        ft_cmd = [
            sys.executable, os.path.abspath(__file__),
            "--ft-worker",
            "--hf-path", model_dir,
            "--dataset", dataset,
            "--mode", mode,
            "--seq-len", str(spec.seq_len),
            "--global-batch-tokens", str(spec.global_batch_tokens),
            "--steps", str(steps),
            "--gpu-gib", str(gpu_gib),
            "--host-gib", str(host_gib),
            "--lora-rank", str(spec.lora_rank),
            "--lora-alpha", str(spec.lora_alpha),
            "--lr", str(ft_lr),
            "--losses-out-pkl", ft_losses_pkl,
        ]
        if init_dump is not None:
            ft_cmd += ["--init-dump-pkl", init_dump]
        rc, _ = _stream(ft_cmd, log_path=ft_log)
        if rc != 0:
            return False, f"FT worker exited {rc}"
        if not os.path.isfile(ft_losses_pkl):
            return False, "FT worker did not write losses pkl"
        with open(ft_losses_pkl, "rb") as f:
            ft_losses: list[float] = pickle.load(f)
        if any(l != l or l in (float("inf"), float("-inf")) for l in ft_losses):
            return False, f"NaN/inf in FT losses: {ft_losses[:5]}..."

        if smoke:
            csv = os.path.join(out_dir, "loss_curve.csv")
            with open(csv, "w") as f:
                f.write("step,ft\n")
                for i, l in enumerate(ft_losses):
                    f.write(f"{i+1},{l:.6f}\n")
            return True, f"smoke OK: {len(ft_losses)} steps, no NaN/inf"

        if mode == "full":
            # Run HF Transformers full-FT in a sibling subprocess and
            # compare loss curves. Same data, same LR, same batch.
            hf_log = os.path.join(out_dir, "hf_full.log")
            hf_losses_pkl = os.path.join(td, "hf_full_losses.pkl")
            hf_cmd = [
                sys.executable, os.path.abspath(__file__),
                "--hf-full-worker",
                "--hf-path", model_dir,
                "--dataset", dataset,
                "--losses-out-pkl", hf_losses_pkl,
                "--lr", str(ft_lr),
                "--seq-len", str(spec.seq_len),
                "--global-batch-tokens", str(spec.global_batch_tokens),
                "--steps", str(steps),
            ]
            rc, _ = _stream(hf_cmd, log_path=hf_log)
            if rc != 0:
                return False, f"HF full worker exited {rc}"
            with open(hf_losses_pkl, "rb") as f:
                hf_losses: list[float] = pickle.load(f)
            if any(l != l or l in (float("inf"), float("-inf")) for l in hf_losses):
                return False, f"NaN/inf in HF full losses: {hf_losses[:5]}..."

            n = min(len(ft_losses), len(hf_losses))
            if n == 0:
                return False, "no overlapping steps"
            diffs = [abs(ft_losses[i] - hf_losses[i]) for i in range(n)]
            max_diff = max(diffs)
            eff_tol = tol if tol is not None else spec.tolerance

            csv = os.path.join(out_dir, "loss_curves.csv")
            with open(csv, "w") as f:
                f.write("step,ft,hf_full,abs_diff\n")
                for i in range(n):
                    f.write(f"{i+1},{ft_losses[i]:.6f},"
                              f"{hf_losses[i]:.6f},{diffs[i]:.6f}\n")
            summary = os.path.join(out_dir, "summary.md")
            with open(summary, "w") as f:
                f.write(f"# {spec.name} full-FT E2E vs HF Transformers\n\n")
                f.write(f"- steps: {n}\n- tolerance: {eff_tol}\n"
                          f"- max |Δ|: {max_diff:.4f}\n\n")
                f.write("| step | FT loss | HF loss | |Δ| |\n|---|---|---|---|\n")
                heads = list(range(min(3, n)))
                tails = list(range(max(0, n-3), n)) if n > 3 else []
                for i in heads + tails:
                    f.write(f"| {i+1} | {ft_losses[i]:.4f} | "
                              f"{hf_losses[i]:.4f} | {diffs[i]:.4f} |\n")

            if max_diff >= eff_tol:
                return False, (
                    f"full max|Δ|={max_diff:.4f} >= tol={eff_tol:.4f}; "
                    f"FT 1st/last={ft_losses[0]:.4f}/{ft_losses[-1]:.4f}; "
                    f"HF={hf_losses[0]:.4f}/{hf_losses[-1]:.4f}"
                )
            return True, (
                f"full max|Δ|={max_diff:.4f} < tol={eff_tol:.4f}; "
                f"FT 1st/last={ft_losses[0]:.4f}/{ft_losses[-1]:.4f}; "
                f"HF={hf_losses[0]:.4f}/{hf_losses[-1]:.4f}"
            )

        # ---- LoRA mode: run HF PEFT and compare.
        assert init_dump is not None
        if not os.path.isfile(init_dump):
            return False, "FT worker did not dump LoRA init"

        hf_log = os.path.join(out_dir, "hf_peft.log")
        hf_losses_pkl = os.path.join(td, "hf_losses.pkl")
        hf_cmd = [
            sys.executable, os.path.abspath(__file__),
            "--hf-peft-worker",
            "--hf-path", model_dir,
            "--dataset", dataset,
            "--init-pkl", init_dump,
            "--losses-out-pkl", hf_losses_pkl,
            "--targets", json.dumps(spec.ft_to_hf),
            "--lora-rank", str(spec.lora_rank),
            "--lora-alpha", str(spec.lora_alpha),
            "--lr", str(ft_lr),
            "--seq-len", str(spec.seq_len),
            "--global-batch-tokens", str(spec.global_batch_tokens),
            "--steps", str(steps),
        ]
        rc, _ = _stream(hf_cmd, log_path=hf_log)
        if rc != 0:
            return False, f"HF PEFT worker exited {rc}"
        with open(hf_losses_pkl, "rb") as f:
            hf_losses: list[float] = pickle.load(f)
        if any(l != l or l in (float("inf"), float("-inf")) for l in hf_losses):
            return False, f"NaN/inf in HF PEFT losses: {hf_losses[:5]}..."

        n = min(len(ft_losses), len(hf_losses))
        if n == 0:
            return False, "no overlapping steps"
        diffs = [abs(ft_losses[i] - hf_losses[i]) for i in range(n)]
        max_diff = max(diffs)
        eff_tol = tol if tol is not None else spec.tolerance

        csv = os.path.join(out_dir, "loss_curves.csv")
        with open(csv, "w") as f:
            f.write("step,ft,hf_peft,abs_diff\n")
            for i in range(n):
                f.write(f"{i+1},{ft_losses[i]:.6f},"
                          f"{hf_losses[i]:.6f},{diffs[i]:.6f}\n")

        summary = os.path.join(out_dir, "summary.md")
        with open(summary, "w") as f:
            f.write(f"# {spec.name} LoRA E2E vs HF PEFT\n\n")
            f.write(f"- steps: {n}\n")
            f.write(f"- tolerance: {eff_tol}\n")
            f.write(f"- max |Δ|: {max_diff:.4f}\n\n")
            f.write("| step | FT loss | HF PEFT loss | |Δ| |\n|---|---|---|---|\n")
            heads = list(range(min(3, n)))
            tails = list(range(max(0, n-3), n)) if n > 3 else []
            for i in heads + tails:
                f.write(f"| {i+1} | {ft_losses[i]:.4f} | "
                          f"{hf_losses[i]:.4f} | {diffs[i]:.4f} |\n")

        if max_diff >= eff_tol:
            return False, (
                f"max|Δ|={max_diff:.4f} >= tol={eff_tol:.4f}; "
                f"FT 1st/last={ft_losses[0]:.4f}/{ft_losses[-1]:.4f}; "
                f"HF={hf_losses[0]:.4f}/{hf_losses[-1]:.4f}"
            )
        return True, (
            f"max|Δ|={max_diff:.4f} < tol={eff_tol:.4f}; "
            f"FT 1st/last={ft_losses[0]:.4f}/{ft_losses[-1]:.4f}; "
            f"HF={hf_losses[0]:.4f}/{hf_losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> int:
    # Sub-mode dispatch: when invoked with --ft-worker / --hf-peft-worker,
    # we are the sibling subprocess.
    if "--ft-worker" in sys.argv:
        _ft_worker_main()
        return 0
    if "--hf-peft-worker" in sys.argv:
        _hf_peft_worker_main()
        return 0
    if "--hf-full-worker" in sys.argv:
        _hf_full_worker_main()
        return 0

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arch", default=None,
                    help="One of: " + ", ".join(ARCHES))
    p.add_argument("--all", action="store_true",
                    help="Run every configured arch (skips ones with "
                         "missing weights).")
    p.add_argument("--list", action="store_true",
                    help="Print configured arches and exit.")
    p.add_argument("--mode", choices=("lora", "full", "smoke"), default="lora",
                    help="lora=parity vs HF PEFT; full=FT only "
                         "loss-decrease; smoke=5 steps NaN check.")
    p.add_argument(
        "--dataset", default="datasets/mathinstruct.jsonl",
        help=("SFT JSONL produced by download.py. Default: MathInstruct "
              "(richer than gsm8k for FT loss-curve comparisons)."),
    )
    p.add_argument("--steps", type=int, default=None,
                    help="Override per-arch step count.")
    p.add_argument("--gpu-gib", type=float, default=None)
    p.add_argument("--host-gib", type=float, default=None)
    p.add_argument("--tol", type=float, default=None,
                    help="Override per-arch tolerance for max |Δ|.")
    p.add_argument("--keep-going", action="store_true",
                    help="With --all, continue past failures.")
    args = p.parse_args()

    if args.list:
        print("Configured arches:")
        for k, s in ARCHES.items():
            print(f"  {k:18s}  hf_dir={s.hf_dir:24s}  "
                   f"gpu={s.gpu_gib:.0f}GiB  host={s.host_gib:.0f}GiB  "
                   f"steps={s.steps}  seq_len={s.seq_len}  "
                   f"batch_tokens={s.global_batch_tokens}  "
                   f"targets={list(s.ft_to_hf)}")
            if s.notes:
                print(f"    note: {s.notes}")
        return 0

    if not args.arch and not args.all:
        p.error("pass --arch <name> or --all (or --list)")
    if args.arch and args.arch not in ARCHES:
        p.error(f"unknown arch {args.arch!r}; --list to see options")

    selected = list(ARCHES.values()) if args.all else [ARCHES[args.arch]]
    smoke = (args.mode == "smoke")
    train_mode = "lora" if smoke else args.mode

    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(ROOT, dataset_path)
    if not os.path.isfile(dataset_path):
        print(f"\nERROR: dataset not found: {dataset_path}\n"
                f"Run scripts/download_test_models.sh, or:\n"
                f"  python download.py dataset TIGER-Lab/MathInstruct "
                f"--target datasets/mathinstruct.jsonl",
                file=sys.stderr)
        return 2

    results: list[tuple[str, bool, str]] = []
    overall_pass = True

    for spec in selected:
        steps = args.steps or (5 if smoke else spec.steps)
        gpu_gib = args.gpu_gib if args.gpu_gib is not None else spec.gpu_gib
        host_gib = args.host_gib if args.host_gib is not None else spec.host_gib

        try:
            ok, reason = _run_one_arch(
                spec, mode=train_mode, dataset=dataset_path, steps=steps,
                gpu_gib=gpu_gib, host_gib=host_gib, smoke=smoke,
                tol=args.tol,
            )
        except Exception as e:
            ok, reason = False, f"exception: {e}"
            print(f"\n[exception] {spec.name}: {e}", flush=True)

        flag = "PASS" if ok else "FAIL"
        print(f"\n  [{flag}] {spec.name}: {reason}")
        results.append((spec.name, ok, reason))
        if not ok:
            overall_pass = False
            if args.all and not args.keep_going:
                print("\nAborting --all on first failure (use --keep-going to continue)")
                break

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok, note in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name:18s}  {note}")

    out_md = os.path.join(ROOT, "parity_results",
                            f"sweep_{train_mode}_summary.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(f"# E2E sweep ({train_mode})\n\n")
        f.write(f"- dataset: `{args.dataset}`\n\n")
        f.write("| arch | result | note |\n|---|---|---|\n")
        for name, ok, note in results:
            flag = "PASS" if ok else "FAIL"
            f.write(f"| {name} | **{flag}** | {note} |\n")
    print(f"\nFull sweep summary: {out_md}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
