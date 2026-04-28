"""Standalone CLI for fetching HF models / datasets to explicit local paths.

Two subcommands:

    python download.py model meta-llama/Llama-3.1-8B --target models/Llama-3.1-8B
    python download.py dataset HuggingFaceH4/no_robots --target datasets/no_robots.jsonl

Designed for the air-gapped-compute-node workflow: run this on a login
node with internet, then pass the resulting paths to ``train.py`` on the
compute node.

Standalone import policy
------------------------
This file MUST be runnable on head / login nodes without GPU and
without a working CUDA / Triton stack. The actual download logic lives
in ``flextrain/io/download.py``, but importing that submodule normally
triggers ``flextrain/__init__.py`` (which pulls in the engine, ops
kernels, the matmul-dispatcher workspace init, etc.). To avoid that,
we load the download module's source file directly via ``importlib``,
bypassing the package's top-level side effects.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys


def _load_download_module():
    """Import flextrain/io/download.py without triggering
    ``flextrain/__init__.py`` or ``flextrain/io/__init__.py``. Those
    init files import the engine + ops kernels + matmul_dispatcher,
    which fail on head nodes without a GPU. The download module
    itself only uses the stdlib + ``huggingface_hub`` + ``datasets``
    (the latter two lazy-imported inside the functions), so once it's
    loaded directly there are no further GPU dependencies.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "flextrain", "io", "download.py")
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"download module source not found at {src!r}. "
            "Run from the repo root."
        )
    spec = importlib.util.spec_from_file_location(
        "flextrain_download_standalone", src,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build module spec for {src!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_dl = _load_download_module()
download_model = _dl.download_model
download_dataset = _dl.download_dataset


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download.py",
        description=(
            "Download an HF model snapshot or dataset to a local target path. "
            "Run this on a node WITH internet, then point train.py at the "
            "resulting paths from a node WITHOUT internet."
        ),
    )
    sub = p.add_subparsers(dest="kind", required=True)

    m = sub.add_parser(
        "model",
        help="Snapshot an HF model repo to a local directory.",
    )
    m.add_argument(
        "repo_id",
        help='HF repo id, e.g. "meta-llama/Llama-3.1-8B".',
    )
    m.add_argument(
        "--target", required=True,
        help="Local directory to mirror into (will be created).",
    )
    m.add_argument(
        "--revision", default=None,
        help="Optional git revision / branch / tag.",
    )
    m.add_argument(
        "--allow-patterns", nargs="+", default=None,
        help=(
            "Restrict the snapshot to these glob patterns "
            "(e.g. --allow-patterns '*.safetensors' '*.json'). "
            "Useful to skip pytorch_model.bin shards if both formats are present."
        ),
    )
    m.add_argument(
        "--force", action="store_true",
        help="Re-download even if the target looks complete.",
    )

    d = sub.add_parser(
        "dataset",
        help="Materialize an HF dataset (or download a JSONL URL) to a local file.",
    )
    d.add_argument(
        "spec",
        help=(
            'HF dataset id (e.g. "HuggingFaceH4/no_robots"), or an '
            "http(s) URL pointing to a JSONL file."
        ),
    )
    d.add_argument(
        "--target", required=True,
        help="Local .jsonl path to write (parent dirs will be created).",
    )
    d.add_argument(
        "--split", default="train",
        help="HF dataset split. Default: train.",
    )
    d.add_argument(
        "--force", action="store_true",
        help="Re-download even if the target already exists.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.kind == "model":
        out = download_model(
            args.repo_id,
            args.target,
            revision=args.revision,
            allow_patterns=args.allow_patterns,
            force=args.force,
        )
        print(out)
        return 0
    if args.kind == "dataset":
        out = download_dataset(
            args.spec,
            args.target,
            split=args.split,
            force=args.force,
        )
        print(out)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
