"""FT replay: load an HF capture bundle, run FT multi-chunk fwd on the
same input_ids, write a parallel bundle.

Why save instead of compare in-process: with both models touching
24 GB GPU at 32k tokens, holding HF logits + FT logits + diff
workspace simultaneously OOMs. Splitting the runs lets each process
own the full GPU.

Usage:
    python tests/multi_chunk_dense_parity/ft_replay.py \\
        --hf-capture tests/multi_chunk_logs/Llama-3.2-1B__hf.pt \\
        --chunk-size 8192 \\
        --out tests/multi_chunk_logs/Llama-3.2-1B__ft_chunk8192.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-capture", required=True,
                    help="Path to HF bundle from hf_capture.py")
    ap.add_argument("--chunk-size", type=int, required=True,
                    help="FT max_chunk_size (force multi-chunk path)")
    ap.add_argument("--out", required=True, help="Output .pt path for FT bundle")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-gpu-gib", type=float, default=14.0)
    args = ap.parse_args()

    print(f"=== FT replay: chunk_size={args.chunk_size} ===")
    bundle_in = torch.load(args.hf_capture, map_location="cpu", weights_only=False)
    model_path = bundle_in["model"]
    ids = bundle_in["input_ids"].tolist()
    print(f"  loaded {args.hf_capture} (model={model_path}, T={len(ids)})")

    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.engine.schedule import prepare_training_chunks
    from flextrain.ops import flextrain_rmsnorm_fwd

    n_chunks_for_seq = (len(ids) + args.chunk_size - 1) // args.chunk_size
    round_tokens = n_chunks_for_seq * args.chunk_size

    print(f"  loading FT model (chunk_size={args.chunk_size}, max_gpu={args.max_gpu_gib} GiB) ...", flush=True)
    am = from_pretrained(
        model_path,
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
        device=args.device,
        leeway_gpu_mem_bytes=int(2 * (1 << 30)),
        leeway_host_mem_bytes=int(8 * (1 << 30)),
        max_chunk_size=args.chunk_size,
        min_chunk_size=args.chunk_size,
        strict=False, verbose=False,
    )

    class _Seq:
        def __init__(self, t):
            self.tokens = t
            T = len(t)
            self.targets = torch.zeros(T, dtype=torch.int64)
            self.per_token_loss = torch.zeros(T, dtype=torch.float32)
            self.seq_id = 0
            self.active_token_count = T
        def __len__(self):
            return len(self.tokens)

    seq = _Seq(torch.tensor(ids, dtype=torch.int64))
    prepared = prepare_training_chunks(
        [seq],
        max_chunk_size=am.working_set.max_chunk_size,
        device=am.device,
        policy=am.chunk_policy,
    )
    chunk_sizes = [c.meta.total_q for c in prepared.chunks]
    print(f"  prepared {len(prepared.chunks)} chunks (sizes: {chunk_sizes})", flush=True)

    am._allocate_moe_chunk_scratch(prepared)
    am.events.clear_per_round()
    plan = am._plan_save_levels(prepared)
    am.streams.compute.synchronize()
    am._setup_round(prepared, plan)

    print("  running FT forward ...", flush=True)
    t0 = time.time()
    am._forward_pass(prepared, plan)
    am.streams.compute.synchronize()
    print(f"  FT fwd done in {time.time()-t0:.1f}s", flush=True)

    head_weights = am.buffers.gpu_head_params
    head_cfg = am.head.cfg
    rms_eps = float(head_cfg.rms_norm_eps)

    # Per-chunk head pass + immediate offload to CPU. Each chunk's
    # logits move to CPU before the next allocates -- keeps GPU peak
    # at one chunk's worth (e.g. 2 GiB at chunk=8192, V=128k).
    logit_parts_cpu: list[torch.Tensor] = []
    for c in prepared.chunks:
        x_c = am.buffers.transitions[c.id]
        head_proj_in, _rstd = flextrain_rmsnorm_fwd(
            x_c, W=head_weights["w_final_norm"], rms_norm_eps=rms_eps,
        )
        logits_c = torch.mm(head_proj_in, head_weights["w_head_proj"]).to(torch.bfloat16)
        logit_parts_cpu.append(logits_c.cpu())
        # Drop the residual reference so the engine pool can recycle
        # the slot (it won't free actively, but it removes the explicit
        # hold so the next chunk's head_proj_in / matmul out aren't
        # blocked).
        am.buffers.transitions[c.id] = None
        del x_c, head_proj_in, logits_c
        torch.cuda.empty_cache()

    logits = torch.cat(logit_parts_cpu, dim=0).contiguous()

    bundle_out = {
        "model": model_path,
        "fixture": bundle_in.get("fixture"),
        "input_ids": bundle_in["input_ids"],
        "logits": logits,                              # (T, V) bf16 cpu
        "chunk_size": args.chunk_size,
        "num_chunks": len(prepared.chunks),
        "chunk_sizes": chunk_sizes,
        "num_input_tokens": len(ids),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(bundle_out, args.out)
    print(f"  saved -> {args.out}  ({os.path.getsize(args.out) / (1<<20):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
