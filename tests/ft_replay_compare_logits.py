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
    ft_logits_cpu = ft_logits.detach().to("cpu").contiguous()

    print()
    print("=== Diff (full sequence) ===")
    _stats("logits[full]", hf_logits, ft_logits_cpu)

    if prompt_T > 0:
        print()
        print("=== Diff (prompt slice [0, prompt_T)) ===")
        _stats(
            "logits[prompt]",
            hf_logits[:prompt_T], ft_logits_cpu[:prompt_T],
        )

    if n_gen > 0:
        print()
        print("=== Diff (generated slice [prompt_T, T)) ===")
        gen_hf = hf_logits[prompt_T:]
        gen_ft = ft_logits_cpu[prompt_T:]
        _stats("logits[generated]", gen_hf, gen_ft)

        # Per-position drift across the generated region — useful for
        # spotting cumulative drift over autoregressive steps.
        diff = (gen_hf.float() - gen_ft.float()).abs()
        per_pos_max = diff.max(dim=-1).values  # (n_gen,)
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
                bucket_max = per_pos_max[s:e].max().item()
                bucket_med = per_pos_max[s:e].median().item()
                print(
                    f"    [{s:4d}, {e:4d})  max|Δ|={bucket_max:7.4f}  "
                    f"median|Δ|={bucket_med:7.4f}",
                    flush=True,
                )

    # Argmax agreement (full).
    hf_arg = hf_logits.argmax(dim=-1)
    ft_arg = ft_logits_cpu.argmax(dim=-1)
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
        # First position in the generated region where argmax differs.
        diff_mask = (hf_arg[prompt_T:] != ft_arg[prompt_T:])
        first_diff = int(diff_mask.nonzero()[0].item()) if diff_mask.any() else -1
        if first_diff >= 0:
            print(f"  first generated-region argmax mismatch at "
                  f"prompt_T + {first_diff} = {prompt_T + first_diff} "
                  f"(HF={hf_arg[prompt_T + first_diff].item()} "
                  f"FT={ft_arg[prompt_T + first_diff].item()})")

    # Next-token CE.
    if T > 1:
        targets = torch.tensor(ids[1:], dtype=torch.int64)
        ce_hf = torch.nn.functional.cross_entropy(
            hf_logits[:-1].float(), targets, reduction="mean",
        )
        ce_ft = torch.nn.functional.cross_entropy(
            ft_logits_cpu[:-1].float(), targets, reduction="mean",
        )
        print()
        print(f"  next-token CE [full]:      HF={ce_hf.item():.4f}  FT={ce_ft.item():.4f}")
        if prompt_T > 1:
            tg_p = torch.tensor(ids[1:prompt_T], dtype=torch.int64)
            ce_hf_p = torch.nn.functional.cross_entropy(
                hf_logits[:prompt_T-1].float(), tg_p, reduction="mean",
            )
            ce_ft_p = torch.nn.functional.cross_entropy(
                ft_logits_cpu[:prompt_T-1].float(), tg_p, reduction="mean",
            )
            print(
                f"  next-token CE [prompt]:    "
                f"HF={ce_hf_p.item():.4f}  FT={ce_ft_p.item():.4f}"
            )
        if n_gen > 0:
            # Generated-region CE = how well each side predicts each
            # actual generated token from the prefix-up-to-that-point.
            tg_g = torch.tensor(ids[prompt_T:], dtype=torch.int64)
            # The position predicting ids[prompt_T] is logits[prompt_T - 1];
            # predicting ids[T-1] is logits[T-2]. So the slice is
            # logits[prompt_T - 1 : T - 1].
            ce_hf_g = torch.nn.functional.cross_entropy(
                hf_logits[prompt_T - 1 : T - 1].float(), tg_g, reduction="mean",
            )
            ce_ft_g = torch.nn.functional.cross_entropy(
                ft_logits_cpu[prompt_T - 1 : T - 1].float(), tg_g, reduction="mean",
            )
            print(
                f"  next-token CE [generated]: "
                f"HF={ce_hf_g.item():.4f}  FT={ce_ft_g.item():.4f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
