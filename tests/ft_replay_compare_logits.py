"""FT replay + compare against an HF logit capture.

Loads the bundle saved by ``tests/hf_capture_logits.py`` (token ids +
HF logits over a prompt + greedy-generated continuation), reconstructs
the FT engine for the same model, runs ONE FT forward over the full
sequence, and prints a per-position diff vs the HF reference logits.

Per-region breakdown:
* ``prompt`` slice (positions 0..prompt_T-1): logits over the user
  prompt — identical to a non-generation parity test.
* ``generated`` slice (positions prompt_T..T-1): logits over the
  greedy-generated continuation. Cumulative numerical drift after
  many autoregressive steps shows up here.

Also reports next-token argmax agreement HF vs FT, the position
where they first diverge, and CE loss against the next-token targets
implied by ``input_ids``.

Usage:

    PYTHONPATH=. python tests/ft_replay_compare_logits.py \\
        --capture hf_capture_27b.pt \\
        --model models/Qwen3.5-27B \\
        --gpu-gib 22.5 --host-gib 110
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
    # ``transitions`` may be a view into the activation ring whose
    # underlying storage gets recycled — clone immediately so the
    # post-fwd ops (rmsnorm, mm, .to('cpu')) read a private copy.
    head_w = am.buffers.gpu_head_params
    rms_eps = float(am.head.cfg.rms_norm_eps)
    parts = [
        am.buffers.transitions[c.id].detach().clone()
        for c in prepared.chunks
    ]
    print(f"  parts[0].shape={tuple(parts[0].shape)} dtype={parts[0].dtype} device={parts[0].device}", flush=True)
    x = torch.cat(parts, dim=0).contiguous() if len(parts) > 1 else parts[0]
    print(f"  x.shape={tuple(x.shape)} contig={x.is_contiguous()}", flush=True)
    torch.cuda.synchronize()
    head_proj_in, _rstd = flextrain_rmsnorm_fwd(
        x, W=head_w["w_final_norm"], rms_norm_eps=rms_eps,
    )
    print(f"  head_proj_in.shape={tuple(head_proj_in.shape)}", flush=True)
    torch.cuda.synchronize()
    logits = torch.mm(head_proj_in, head_w["w_head_proj"]).contiguous()
    print(f"  logits.shape={tuple(logits.shape)} dtype={logits.dtype}", flush=True)
    torch.cuda.synchronize()
    out = logits.detach().clone().contiguous()
    torch.cuda.synchronize()
    return out


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
    T = int(bundle["input_ids"].numel())
    prompt_T = int(bundle.get("prompt_T", T))
    n_gen = int(bundle.get("n_generated", T - prompt_T))
    V = int(bundle["vocab_size"])
    print(f"  T={T}  prompt_T={prompt_T}  generated={n_gen}  V={V}")
    print(f"  HF dtype-used-for-fwd: {bundle.get('dtype_used_for_fwd', 'unknown')}")
    if "full_decoded" in bundle:
        print()
        print("  === HF prompt + generated (decoded) ===")
        print(bundle["full_decoded"])
        print("  === END ===")
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
    # All comparisons on GPU. ``ft_logits`` lives on cuda:0 with FT's
    # stream ownership; pulling it to CPU triggered cudaErrorInvalidValue
    # for unclear reasons. Push the (small bf16) HF capture to GPU
    # instead, do the math on GPU, and only return scalar summaries.
    torch.cuda.synchronize()
    hf_dev = hf_logits.to("cuda:0").contiguous()

    def _print_stats_gpu(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
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

    print()
    print("=== Diff (full sequence) ===")
    _print_stats_gpu("logits[full]", hf_dev, ft_logits)

    if prompt_T > 0:
        print()
        print("=== Diff (prompt slice [0, prompt_T)) ===")
        _print_stats_gpu(
            "logits[prompt]",
            hf_dev[:prompt_T], ft_logits[:prompt_T],
        )

    if n_gen > 0:
        print()
        print("=== Diff (generated slice [prompt_T, T)) ===")
        gen_hf = hf_dev[prompt_T:]
        gen_ft = ft_logits[prompt_T:]
        _print_stats_gpu("logits[generated]", gen_hf, gen_ft)

        # Per-position drift across generated region.
        diff_g = (gen_hf.float() - gen_ft.float()).abs()
        per_pos_max = diff_g.max(dim=-1).values  # (n_gen,)
        per_pos_max_cpu = per_pos_max.cpu()
        if n_gen >= 4:
            buckets = 4
            print(f"  per-position max|Δ| across generated region "
                  f"({n_gen} positions, {buckets} bins):")
            seg = max(1, n_gen // buckets)
            for b in range(buckets):
                s = b * seg
                e = (b + 1) * seg if b < buckets - 1 else n_gen
                if s >= n_gen:
                    break
                bucket_max = per_pos_max_cpu[s:e].max().item()
                bucket_med = per_pos_max_cpu[s:e].median().item()
                print(
                    f"    [{s:4d}, {e:4d})  max|Δ|={bucket_max:7.4f}  "
                    f"median|Δ|={bucket_med:7.4f}",
                    flush=True,
                )

    # Argmax agreement.
    hf_arg = hf_dev.argmax(dim=-1).cpu()
    ft_arg = ft_logits.argmax(dim=-1).cpu()
    agree_full = (hf_arg == ft_arg).float().mean().item()
    print()
    print(f"  argmax agreement [full]:      "
          f"{agree_full*100:.2f}% ({(hf_arg == ft_arg).sum().item()}/{T})")
    if prompt_T > 0:
        agree_p = (hf_arg[:prompt_T] == ft_arg[:prompt_T]).float().mean().item()
        print(f"  argmax agreement [prompt]:    "
              f"{agree_p*100:.2f}% ({(hf_arg[:prompt_T] == ft_arg[:prompt_T]).sum().item()}/{prompt_T})")
    if n_gen > 0:
        agree_g = (hf_arg[prompt_T:] == ft_arg[prompt_T:]).float().mean().item()
        print(f"  argmax agreement [generated]: "
              f"{agree_g*100:.2f}% ({(hf_arg[prompt_T:] == ft_arg[prompt_T:]).sum().item()}/{n_gen})")
        diff_mask = (hf_arg[prompt_T:] != ft_arg[prompt_T:])
        if diff_mask.any():
            first_diff = int(diff_mask.nonzero()[0].item())
            print(f"  first generated-region argmax mismatch at "
                  f"prompt_T + {first_diff} = {prompt_T + first_diff} "
                  f"(HF={hf_arg[prompt_T + first_diff].item()} "
                  f"FT={ft_arg[prompt_T + first_diff].item()})")

    # CE losses (GPU).
    if T > 1:
        targets_dev = torch.tensor(ids[1:], dtype=torch.int64, device="cuda:0")
        ce_hf = torch.nn.functional.cross_entropy(
            hf_dev[:-1].float(), targets_dev, reduction="mean",
        ).item()
        ce_ft = torch.nn.functional.cross_entropy(
            ft_logits[:-1].float(), targets_dev, reduction="mean",
        ).item()
        print()
        print(f"  next-token CE [full]:      HF={ce_hf:.4f}  FT={ce_ft:.4f}")
        if prompt_T > 1:
            tg_p = targets_dev[: prompt_T - 1]
            ce_hf_p = torch.nn.functional.cross_entropy(
                hf_dev[: prompt_T - 1].float(), tg_p, reduction="mean",
            ).item()
            ce_ft_p = torch.nn.functional.cross_entropy(
                ft_logits[: prompt_T - 1].float(), tg_p, reduction="mean",
            ).item()
            print(f"  next-token CE [prompt]:    HF={ce_hf_p:.4f}  FT={ce_ft_p:.4f}")
        if n_gen > 0:
            tg_g = torch.tensor(ids[prompt_T:], dtype=torch.int64, device="cuda:0")
            ce_hf_g = torch.nn.functional.cross_entropy(
                hf_dev[prompt_T - 1 : T - 1].float(), tg_g, reduction="mean",
            ).item()
            ce_ft_g = torch.nn.functional.cross_entropy(
                ft_logits[prompt_T - 1 : T - 1].float(), tg_g, reduction="mean",
            ).item()
            print(f"  next-token CE [generated]: HF={ce_hf_g:.4f}  FT={ce_ft_g:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
