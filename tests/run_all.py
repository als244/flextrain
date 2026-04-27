"""Run every test module under tests/. CPU-only.

Usage:
    conda activate flextrain && PYTHONPATH=. python tests/run_all.py
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    modules = []
    for fname in sorted(os.listdir(HERE)):
        if not fname.startswith("test_") or not fname.endswith(".py"):
            continue
        modules.append("tests." + fname[:-3])

    failed: list[str] = []
    for modname in modules:
        print(f"\n=== {modname} ===", flush=True)
        try:
            m = importlib.import_module(modname)
            run = getattr(m, "_run_all", None)
            if run is None:
                print(f"  (skip: no _run_all in {modname})")
                continue
            run()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failed.append(modname)
        finally:
            # Cross-test isolation: unregister every cudaHostRegister'd
            # pointer tracked by LocalPinnedHostBackend instances in
            # this process. Stops GC'd-but-not-finalized BufferManagers
            # from leaving stale pinnings that poison the next module's
            # DMA path (cudaErrorHostMemoryAlreadyRegistered -> sticky
            # cudaErrorInvalidValue on copy_).
            try:
                import gc
                gc.collect()
                from flextrain.engine import (
                    unregister_all_process_pinned_memory,
                )
                unregister_all_process_pinned_memory()
            except Exception:  # noqa: BLE001
                # Never let cleanup failure mask a real test failure.
                pass

    print("\n" + "=" * 40)
    if failed:
        print(f"FAILED: {len(failed)} module(s): {failed}")
        return 1
    print(f"All {len(modules)} test modules passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
