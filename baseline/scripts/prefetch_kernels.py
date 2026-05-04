#!/usr/bin/env python3
"""Pre-fetch HF kernel-hub kernels into the local HF cache.

Run this on a node WITH internet (login node, head node, jump host)
*before* launching jobs on a compute node that has no internet access.
The kernels package's lazy-load path queries the HF API for available
revisions even when a local cache exists, so the compute node has to
either be able to reach HuggingFace or have the API responses cached
in advance — this script populates the cache so the lazy-load
short-circuits.

Usage:
    # On a login node (with internet):
    conda activate baseline_core
    python baseline/scripts/prefetch_kernels.py

    # Then on the compute node (no internet), ensure HF_HOME or
    # HF_HUB_CACHE points at the same directory the prefetch wrote to
    # (typically ~/.cache/huggingface/, which is usually on a shared
    # home filesystem under SLURM).

The transformers ExpertsInterface integrations that use
``kernels.lazy_load_kernel`` and need to be pre-fetched:

    sonicmoe   ->  kernels-community/sonic-moe

Add more entries to ``KERNELS`` below if a future backend needs other
HF-hub kernels.
"""

from __future__ import annotations

import os
import sys

# (kernel-name-as-passed-to-lazy_load_kernel, repo_id) pairs.
# Right now only sonic-moe is wired up via HF transformers; this list
# stays small on purpose.
KERNELS: list[tuple[str, str]] = [
    ("sonic-moe", "kernels-community/sonic-moe"),
]


def main() -> int:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print(
            "[prefetch_kernels] HF_HUB_OFFLINE=1 is set; nothing to fetch.\n"
            "                   Run this on a node with internet first, then\n"
            "                   re-run your job on the offline compute node.",
            file=sys.stderr,
        )
        return 1

    try:
        from kernels import get_kernel
    except ImportError:
        print(
            "[prefetch_kernels] `kernels` package not installed in this env.\n"
            "                   Activate the baseline_core conda env first:\n"
            "                     conda activate baseline_core",
            file=sys.stderr,
        )
        return 2

    print(
        f"[prefetch_kernels] HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface (default)')}",
        flush=True,
    )

    failed: list[tuple[str, str]] = []
    for name, repo_id in KERNELS:
        print(f"[prefetch_kernels] fetching {repo_id} ({name})...", flush=True)
        try:
            kernel = get_kernel(repo_id)
        except Exception as exc:  # noqa: BLE001 — surface any failure
            print(f"[prefetch_kernels] FAILED {repo_id}: {exc!r}", flush=True)
            failed.append((repo_id, repr(exc)))
            continue
        path = getattr(kernel, "__file__", None) or repr(kernel)
        print(f"[prefetch_kernels] OK     {repo_id} -> {path}", flush=True)

    if failed:
        print(
            f"\n[prefetch_kernels] {len(failed)} kernel(s) failed to fetch:",
            file=sys.stderr,
        )
        for repo_id, err in failed:
            print(f"  {repo_id}: {err}", file=sys.stderr)
        return 1

    print(
        "\n[prefetch_kernels] all kernels cached.\n"
        "                   On the offline compute node, ensure HF_HOME / HF_HUB_CACHE\n"
        "                   resolves to the same directory and HF_HUB_OFFLINE=1 is set.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
