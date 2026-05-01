"""Inner script for bwd_self_consistency.py — runs one fwd+bwd pass
at a given chunk size and pickles out the LoRA grads.

Used by ``bwd_self_consistency.py`` as a subprocess so each pass
gets a clean GPU memory pool. Don't invoke directly.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _Seq:
    def __init__(self, t: torch.Tensor, seq_id: int = 0):
        self.tokens = t
        T = len(t)
        self.targets = torch.empty(T, dtype=torch.int64)
        self.targets[:-1] = t[1:]
        self.targets[-1] = -100
        self.per_token_loss = torch.zeros(T, dtype=torch.float32)
        self.seq_id = seq_id
        self.active_token_count = int((self.targets != -100).sum().item())

    def __len__(self) -> int:
        return len(self.tokens)


def _capture_lora_grads(am) -> dict[str, torch.Tensor]:
    """Pull all LoRA grads (A and B for every linear-attn / dense-attn /
    MoE / FFN linear) out of the engine and clone them to CPU.

    Returns a dict keyed by ``layer_id__param_name``. Only non-frozen
    grad tensors are captured (LoRA A and B params, depending on
    spec).

    Wait for compute stream first so all bwd kernels have flushed.
    """
    am.streams.compute.synchronize()
    out: dict[str, torch.Tensor] = {}
    for layer in am.backbone:
        lid = layer.layer_id
        host_grads = am.buffers.host_grads[lid]
        for name, g in host_grads.items():
            if g is None:
                continue
            # Clone to detach from the engine's host buffer (which
            # may be re-used by next round's bwd accumulation).
            out[f"layer{lid}__{name}"] = g.detach().clone()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--max-gpu-gib", type=float, default=18.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-init", default=None,
                    help="If provided, load the LoRA A/B init from this "
                         "pickle. Otherwise capture this pass's init "
                         "and write it out (for the next pass to load).")
    args = ap.parse_args()

    print(f"  inner: model={args.model}  T={args.target_tokens}  chunk={args.chunk_size}")

    # Tokenize.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.fixture) as f:
        text = f.read()
    bos = tok.bos_token_id
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if bos is not None:
        ids = [bos] + ids
    ids = ids[: args.target_tokens]

    # Engine.
    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    n_chunks = (len(ids) + args.chunk_size - 1) // args.chunk_size
    round_tokens = n_chunks * args.chunk_size

    am = from_pretrained(
        args.model,
        optimizer=AdamW(
            AdamWHyperparams(
                lr=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
            ),
            state_dtype=torch.bfloat16,
        ),
        max_seq_len=round_tokens,
        max_global_batch_tokens=round_tokens,
        max_gpu_mem_bytes=int(args.max_gpu_gib * (1 << 30)),
        max_host_mem_bytes=int(80.0 * (1 << 30)),
        device="cuda:0",
        leeway_gpu_mem_bytes=int(2 * (1 << 30)),
        leeway_host_mem_bytes=int(8 * (1 << 30)),
        max_chunk_size=args.chunk_size,
        min_chunk_size=args.chunk_size,
        lora_rank=8,
        lora_alpha=16.0,
        strict=False, verbose=False,
    )

    # LoRA init: ref pass captures + writes; test pass loads.
    if args.lora_init is not None:
        with open(args.lora_init, "rb") as f:
            init: dict = pickle.load(f)
        # Inject LoRA A/B init into the engine's host_params.
        # Format: { 'layer{lid}__{name}' : tensor }.
        with torch.no_grad():
            n_loaded = 0
            for layer in am.backbone:
                lid = layer.layer_id
                for name, p in am.buffers.host_params[lid].items():
                    key = f"layer{lid}__{name}"
                    if key in init:
                        p.copy_(init[key].to(p.dtype))
                        n_loaded += 1
        print(f"  inner: loaded {n_loaded} LoRA init tensors from {args.lora_init}")
    else:
        # Capture ALL host_params (the LoRA A/B values plus base
        # params; only the LoRA A/B are randomized at __init__,
        # base params come from the safetensors load and are
        # deterministic). Save just the trainable subset to keep the
        # bundle small and ensure we only re-init what differs.
        init: dict[str, torch.Tensor] = {}
        for layer in am.backbone:
            lid = layer.layer_id
            for name, p in am.buffers.host_params[lid].items():
                init[f"layer{lid}__{name}"] = p.detach().clone()
        with open(args.out + ".lora_init", "wb") as f:
            pickle.dump(init, f)
        print(f"  inner: captured {len(init)} init tensors -> {args.out}.lora_init")

    # One fwd+bwd pass.
    seq = _Seq(torch.tensor(ids, dtype=torch.int64))
    active = max(seq.active_token_count, 1)
    stats = am.fwd_bwd(
        [seq],
        loss_scale_factor=1.0 / active,
        total_tokens_per_step=active,
        verbose=False,
    )
    am.streams.compute.synchronize()
    loss = stats.total_loss / max(stats.total_tokens, 1)
    print(f"  inner: tokens={stats.total_tokens}  loss={loss:.4f}")

    # Capture grads. Note: do NOT call am.step() — we want raw grads
    # before the optimizer update.
    grads = _capture_lora_grads(am)
    with open(args.out, "wb") as f:
        pickle.dump(grads, f)
    print(f"  inner: wrote {len(grads)} grads -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
