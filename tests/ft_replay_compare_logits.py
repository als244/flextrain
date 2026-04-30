"""FT replay + compare against an HF logit capture.

Loads the bundle saved by ``tests/hf_capture_logits.py`` (token ids +
HF logits), reconstructs the same FT engine for the named model,
runs FT forward on those exact tokens, and prints a per-position
diff vs the HF reference logits.

Usage:

    PYTHONPATH=. python tests/ft_replay_compare_logits.py \\
        --capture hf_capture_27b.pt \\
        --model models/Qwen3.5-27B \\
        --gpu-gib 22.5 --host-gib 110

If ``--model`` is omitted, falls back to ``bundle["model_path_arg"]``
which records the HF-side model dir (handy when paths are identical
across machines).

Notes
-----
* Uses LoRA-mode FT engine to keep the GPU footprint small (frozen
  base = same fwd as the un-adapted model; adapters init to zero).
  The forward we compare is the pretrained-base forward.
* If the engine refuses to fit at the requested GPU budget, retry
  with smaller ``--max-chunk-size`` or larger ``--gpu-gib``.
* Compares logits in bf16 → float for the diff math.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _ft_forward(
    model_path: str, ids: list[int], *,
    gpu_gib: float, host_gib: float,
    max_chunk_size: int | None,
    use_lora: bool,
) -> torch.Tensor:
    """Build FT engine and run a single forward over ``ids``. Returns
    per-position logits (T, V) on GPU."""
    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.engine.schedule import prepare_training_chunks
    from flextrain.ops import flextrain_rmsnorm_fwd

    extra = {}
    if use_lora:
        extra.update(dict(lora_targets="all", lora_rank=8, lora_alpha=8.0))
    if max_chunk_size is not None:
        extra.update(dict(
            max_chunk_size=max_chunk_size,
            min_chunk_size=max_chunk_size,
        ))

    am = from_pretrained(
        model_path,
        optimizer=AdamW(
            AdamWHyperparams(
                lr=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
            ),
            state_dtype=torch.float32 if use_lora else torch.bfloat16,
        ),
        max_seq_len=max(len(ids) + 8, 1024),
        max_global_batch_tokens=max(len(ids) + 8, 1024),
        max_gpu_mem_bytes=int(gpu_gib * (1 << 30)),
        max_host_mem_bytes=int(host_gib * (1 << 30)),
        device="cuda:0",
        leeway_gpu_mem_bytes=int(2 * (1 << 30)),
        leeway_host_mem_bytes=int(10 * (1 << 30)),
        strict=False, verbose=False,
        **extra,
    )

    class _Seq:
        def __init__(self, t):
            self.tokens = t
            T = len(t)
            self.targets = torch.zeros(T, dtype=torch.int64)
            self.per_token_loss = torch.zeros(T, dtype=torch.float32)
            self.seq_id = 0
        def __len__(self):
            return len(self.tokens)

    seq = _Seq(torch.tensor(ids, dtype=torch.int64))
    prepared = prepare_training_chunks(
        [seq], max_chunk_size=am.working_set.max_chunk_size,
        device=am.device, policy=am.chunk_policy,
    )
    print(
        f"  FT chunks: {len(prepared.chunks)}  "
        f"sizes={[c.meta.total_q for c in prepared.chunks]}",
        flush=True,
    )
    am._allocate_moe_chunk_scratch(prepared)
    am.events.clear_per_round()
    plan = am._plan_save_levels(prepared)
    am.streams.compute.synchronize()
    am._setup_round(prepared, plan)
    am._forward_pass(prepared, plan)
    am.streams.compute.synchronize()

    # final-norm + lm_head over ALL positions (concat per-chunk
    # transitions back to (T, d_model)).
    head_w = am.buffers.gpu_head_params
    rms_eps = float(am.head.cfg.rms_norm_eps)
    parts = [am.buffers.transitions[c.id] for c in prepared.chunks]
    x = torch.cat(parts, dim=0).contiguous()
    head_proj_in, _rstd = flextrain_rmsnorm_fwd(
        x, W=head_w["w_final_norm"], rms_norm_eps=rms_eps,
    )
    logits = torch.mm(head_proj_in, head_w["w_head_proj"]).contiguous()
    return logits.to(torch.bfloat16)


def _stats(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    if ref.shape != got.shape:
        print(f"  {name:24s} SHAPE MISMATCH ref={tuple(ref.shape)} got={tuple(got.shape)}")
        return
    diff = (ref.float() - got.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ref_norm = float(ref.float().norm().item())
    rel = max_abs / max(ref_norm, 1e-12)
    bf16_eps = 5e-3
    flag = "OK" if rel < bf16_eps else ("DIVERGE" if rel < 0.1 else "BAD")
    print(
        f"  {name:24s} max|Δ|={max_abs:9.4f}  mean|Δ|={mean_abs:9.4f}  "
        f"rel={rel:.3e}  [{flag}]",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True,
                    help="Path to bundle saved by hf_capture_logits.py.")
    ap.add_argument("--model", default=None,
                    help="FT-side model dir. Defaults to bundle's model_path_arg.")
    ap.add_argument("--gpu-gib", type=float, default=22.5)
    ap.add_argument("--host-gib", type=float, default=110.0)
    ap.add_argument("--max-chunk-size", type=int, default=None,
                    help="Force this chunk size (else planner picks).")
    ap.add_argument("--no-lora", action="store_true",
                    help="Build FT engine in full-FT mode (heavier baseline; "
                         "use LoRA mode by default for low GPU footprint).")
    args = ap.parse_args()

    bundle = torch.load(args.capture, map_location="cpu", weights_only=False)
    model_path = args.model or bundle.get("model_path_arg") or bundle.get("model")
    print(f"=== Replay+Compare ===")
    print(f"  capture: {args.capture}")
    print(f"  model:   {bundle.get('model', '<missing>')}  (dir: {model_path})")
    print(f"  prompt:  {bundle.get('prompt', '<missing>')!r}")
    print(f"  T={bundle['input_ids'].numel()}  V={bundle['vocab_size']}")
    print(f"  HF dtype-used-for-fwd: {bundle.get('dtype_used_for_fwd', 'unknown')}")
    print()

    ids = bundle["input_ids"].tolist()
    hf_logits = bundle["logits"]  # CPU bf16 (T, V)

    print("Running FT forward ...")
    ft_logits = _ft_forward(
        model_path, ids,
        gpu_gib=args.gpu_gib, host_gib=args.host_gib,
        max_chunk_size=args.max_chunk_size,
        use_lora=not args.no_lora,
    )
    ft_logits_cpu = ft_logits.detach().to("cpu").contiguous()

    print()
    print("=== Diff ===")
    _stats("logits", hf_logits, ft_logits_cpu)

    # Per-position max|Δ| histogram.
    diff = (hf_logits.float() - ft_logits_cpu.float()).abs()
    per_pos_max = diff.max(dim=-1).values  # (T,)
    print(f"  per-position max|Δ|: min={per_pos_max.min().item():.4f}  "
          f"median={per_pos_max.median().item():.4f}  "
          f"max={per_pos_max.max().item():.4f}")

    # Argmax agreement.
    hf_arg = hf_logits.argmax(dim=-1)
    ft_arg = ft_logits_cpu.argmax(dim=-1)
    agree = (hf_arg == ft_arg).float().mean().item()
    print(f"  argmax agreement: {agree*100:.2f}% ({(hf_arg == ft_arg).sum().item()}/{len(ids)})")

    last_hf = int(hf_logits[-1].argmax().item())
    last_ft = int(ft_logits_cpu[-1].argmax().item())
    print(f"  last-position argmax — HF={last_hf}  FT={last_ft}  "
          f"{'MATCH' if last_hf == last_ft else 'DIFFER'}")

    # Cross-entropy loss vs prompt tokens (next-token CE).
    if len(ids) > 1:
        targets = torch.tensor(ids[1:], dtype=torch.int64)
        ce_hf = torch.nn.functional.cross_entropy(
            hf_logits[:-1].float(), targets, reduction="mean",
        )
        ce_ft = torch.nn.functional.cross_entropy(
            ft_logits_cpu[:-1].float(), targets, reduction="mean",
        )
        print(f"  next-token CE loss — HF={ce_hf.item():.4f}  FT={ce_ft.item():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
