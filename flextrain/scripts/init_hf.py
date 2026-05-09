"""Init CLI — random-init a model from dims and export it as an HF
``transformers``-compatible checkpoint directory.

Output ``--out`` directory contains:

* ``config.json``        — produced by the arch's ``flextrain_to_hf_config``
                           (round-trippable through ``hf_config_to_flextrain``)
* ``model.safetensors``  — single-shard weights from ``ActiveModel.save_hf``

Usage::

    # individual flags
    python -m flextrain init --arch llama --d-model 768 --n-layers 12 \\
        --out checkpoints/llama_124M_init

    # JSON-driven (orig/model_dims.json schema)
    python -m flextrain init --arch llama \\
        --dims-json orig/model_dims.json --model-name nanogpt_124M \\
        --out checkpoints/nanogpt_124M_init

    # MoE
    python -m flextrain init --arch olmoe \\
        --d-model 1024 --n-layers 4 --num-routed-experts 8 --top-k 2 \\
        --out checkpoints/olmoe_toy_init

The dims-flags interface is the same as ``flextrain pretrain``: required
keys are ``--d-model`` and ``--n-layers``; ``n_heads``, ``n_kv_heads``,
and ``expert_dim`` get auto-derived (Llama SwiGLU convention) when
omitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def _cmd_init(args: argparse.Namespace) -> int:
    """Subcommand wiring: parse dims, build with from_dims, write
    config.json + model.safetensors."""
    # Reuse the dims-resolution helpers from the pretrain CLI — same
    # surface (--dims-json + --model-name OR individual flags + auto-
    # derived defaults).
    from flextrain.scripts.pretrain import (
        _parse_dims_flags, _load_dims_from_json,
    )
    from flextrain.io.arch import get_arch_module
    import flextrain
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.io.hf_weights import select_arch

    if args.dims_json:
        if not args.model_name:
            print("[init] --dims-json also needs --model-name",
                  file=sys.stderr)
            return 2
        dims = _load_dims_from_json(args.dims_json, args.model_name)
    else:
        dims = _parse_dims_flags(args)
        if "d_model" not in dims or "n_layers" not in dims:
            print(
                "[init] need --d-model and --n-layers (or --dims-json + "
                "--model-name) when --dims-json absent",
                file=sys.stderr,
            )
            return 2

    arch_mod = get_arch_module(args.arch)
    if not hasattr(arch_mod, "flextrain_to_hf_config"):
        print(
            f"[init] arch {args.arch!r} has no flextrain_to_hf_config — "
            f"can't emit HF checkpoint",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the model with from_dims (random init at the requested
    # seed/std; no training, just the buffer-allocate + init pass).
    print(f"[init] building {args.arch} model (seed={args.init_seed}, "
          f"std={args.init_std})...", flush=True)
    am = flextrain.from_dims(
        dims, arch=args.arch,
        # AdamW is required by from_dims (the engine sizes optimizer-
        # state buffers up-front from the Optimizer's state_spec). We
        # won't take a step, so the lr value doesn't matter.
        optimizer=AdamW(AdamWHyperparams(lr=1e-4)),
        max_seq_len=args.max_seq_len,
        max_global_batch_tokens=args.max_seq_len,
        max_gpu_mem_bytes=int(args.max_gpu_gib * (1 << 30)),
        max_host_mem_bytes=int(args.max_host_gib * (1 << 30)),
        init_seed=args.init_seed, init_std=args.init_std,
    )

    # 2. Compose the HF-side config dict, then look up the matching
    # ArchSpec via ``select_arch`` — that's the same dispatch
    # ``from_pretrained`` uses to find a weight map.
    hyperparams = dict(arch_mod.default_hyperparams())
    # Mirror the from_dims expansion of dims["layer_pattern"] →
    # hyperparams["layer_types"], so config.json carries the schedule.
    if (
        not hyperparams.get("layer_types")
        and dims.get("layer_pattern")
    ):
        from flextrain.io.arch import expand_layer_pattern
        hyperparams["layer_types"] = expand_layer_pattern(
            dims["layer_pattern"], int(dims["n_layers"]),
        )
    hf_config = arch_mod.flextrain_to_hf_config(dims, hyperparams)
    arch_spec = select_arch(hf_config)

    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps(hf_config, indent=2) + "\n")
    print(f"[init] wrote {cfg_path}", flush=True)

    # 3. Export host master params to a single-shard safetensors. The
    # HF-side post-load fixup hooks (Q/K halved→pair etc.) are
    # symmetric on save (handled inside export_hf_safetensors); FP
    # transforms in the ArchSpec emit each tensor in its HF-native
    # layout / dtype.
    #
    # Hybrid-attn arches (qwen3_5, qwen3_5_moe, qwen3_next) currently
    # emit ALL ArchSpec layer entries per layer, ignoring whether the
    # layer is linear-attn or full-attn — the export raises
    # KeyError on the missing layer-type-specific tensors. That's a
    # pre-existing flextrain export limitation; until it's fixed, the
    # init CLI writes config.json but skips safetensors for hybrid
    # arches.
    hybrid = any(arch_id in arch_spec.hf_arch_ids for arch_id in (
        "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM", "Qwen3NextForCausalLM",
        "Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForConditionalGeneration",
    ))
    if hybrid:
        print(
            "[init] hybrid-attn arch detected — skipping safetensors export "
            "(pre-existing limitation in flextrain.export_hf_safetensors). "
            "config.json was written.",
            flush=True,
        )
        return 0
    out_path = am.save_hf(str(out_dir), arch=arch_spec)
    print(f"[init] wrote {out_path}", flush=True)
    return 0


def add_argparse_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``init`` subcommand into ``flextrain.cli``."""
    p = sub.add_parser(
        "init",
        help="random-init a model from dims and export as HF "
             "config.json + model.safetensors",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Same dims interface as `pretrain` (--dims-json + --model-name OR
    # individual --d-model / --n-layers / etc).
    p.add_argument("--arch", required=True,
                   help="short arch name (llama, mistral, qwen2, qwen3, "
                        "olmoe, qwen3_moe, gemma2)")
    p.add_argument("--dims-json")
    p.add_argument("--model-name")
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--n-kv-heads", type=int)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--expert-dim", type=int)
    p.add_argument("--num-routed-experts", type=int)
    p.add_argument("--top-k", type=int)
    p.add_argument("--shared-expert-dim", type=int,
                   help="Per-shared-expert FFN dim (Qwen3-Next, Qwen3.5-MoE)")
    p.add_argument("--layer-pattern",
                   help="Hybrid-attn schedule shorthand (e.g. '1F1L', "
                        "'1F47L'). Codes: F=full, L=linear, S=sliding. "
                        "Pattern repeats to fill n_layers.")

    p.add_argument("--out", required=True,
                   help="output directory (config.json + model.safetensors)")
    p.add_argument("--init-seed", type=int, default=42)
    p.add_argument("--init-std", type=float, default=0.02)
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="seq-len capacity baked into the working-set "
                        "solver (just sizes the activation buffer; weights "
                        "are seq-len agnostic)")
    p.add_argument("--max-gpu-gib", type=int, default=22)
    p.add_argument("--max-host-gib", type=int, default=80)
    p.set_defaults(func=_cmd_init)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="flextrain init")
    sub = p.add_subparsers(dest="cmd", required=True)
    add_argparse_subparser(sub)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
