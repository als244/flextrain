"""GPU + host memory capacity introspection.

Ported from ``orig/awsm_transformer/query_memory.py``. The working-set
solver needs accurate "how much memory can I actually claim" numbers, so
this module uses the same precision-first approach orig did:

* GPU:  prefer ``torch.cuda.mem_get_info`` (free side of the tuple).
        Fall back to ``nvidia-smi`` (NVIDIA), then ``rocm-smi`` /
        ``/sys/class/drm`` (AMD).
* Host: prefer Slurm allocation (``SLURM_MEM_PER_NODE`` /
        ``SLURM_MEM_PER_CPU``), then cgroup limits (v2 ``memory.max`` or
        v1 ``memory.limit_in_bytes``), with cgroup or process-tree usage
        subtracted, finally falling back to ``psutil.virtual_memory``.

Intentionally kept self-contained inside ``flextrain/core/`` so
:mod:`flextrain.core.working_set` doesn't need to reach into ``orig``.
"""

from __future__ import annotations

import os
import re
import subprocess

import torch


# ===========================================================================
# GPU memory
# ===========================================================================


def get_available_gpu_memory(device_id: int = 0) -> int:
    """Free GPU memory in bytes for ``device_id`` (NVIDIA + AMD support)."""
    if torch.cuda.is_available():
        try:
            free_memory, _total_memory = torch.cuda.mem_get_info(device_id)
            return int(free_memory)
        except Exception:
            pass

    # NVIDIA fallback: nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                f"--id={device_id}",
            ],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip()) * 1024 * 1024
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass

    # AMD ROCm fallback: sysfs (lighter) then rocm-smi
    try:
        amd_total = f"/sys/class/drm/card{device_id}/device/mem_info_vram_total"
        amd_used = f"/sys/class/drm/card{device_id}/device/mem_info_vram_used"
        if os.path.exists(amd_total) and os.path.exists(amd_used):
            with open(amd_total, "r") as f:
                total = int(f.read().strip())
            with open(amd_used, "r") as f:
                used = int(f.read().strip())
            return total - used

        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, check=True,
        )
        import json
        data = json.loads(result.stdout)
        for key, val in data.items():
            if f"card{device_id}" in key.lower() or key == str(device_id):
                if (
                    "VRAM Total Memory (B)" in val
                    and "VRAM Total Used Memory (B)" in val
                ):
                    return int(val["VRAM Total Memory (B)"]) - int(
                        val["VRAM Total Used Memory (B)"]
                    )
    except Exception:
        pass

    print(
        f"Warning: Could not determine GPU memory for device {device_id}. "
        "Returning 0."
    )
    return 0


# ===========================================================================
# Host memory
# ===========================================================================


def get_available_host_memory() -> int:
    """Available host memory in bytes, respecting Slurm / cgroup limits.

    Priority:
        1. ``SLURM_MEM_PER_NODE`` / ``SLURM_MEM_PER_CPU * SLURM_JOB_CPUS_PER_NODE``
           (minus current cgroup usage if available, else process-tree RSS).
        2. cgroup v2 ``memory.max`` or v1 ``memory.limit_in_bytes`` minus
           cgroup-reported usage.
        3. ``psutil.virtual_memory().available`` (or ``/proc/meminfo`` fallback).

    Always clamped at the system-available number, so an over-permissive
    cgroup limit on a smaller host can't lie.
    """
    system_available = _get_system_available_memory()

    slurm_limit = _get_slurm_memory_limit()
    if slurm_limit:
        cgroup_usage = _get_cgroup_memory_usage()
        if cgroup_usage is not None:
            return max(0, slurm_limit - cgroup_usage)
        process_usage = _get_process_tree_rss()
        if process_usage is not None:
            return max(0, slurm_limit - process_usage)
        return slurm_limit

    cgroup_limit = _get_cgroup_memory_limit()
    cgroup_usage = _get_cgroup_memory_usage()
    if cgroup_limit and cgroup_usage is not None:
        cgroup_available = cgroup_limit - cgroup_usage
        return min(max(0, cgroup_available), system_available)

    return system_available


def _get_slurm_memory_limit() -> int | None:
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_per_node:
        try:
            return int(mem_per_node) * 1024 * 1024
        except ValueError:
            pass

    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus_on_node = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if mem_per_cpu and cpus_on_node:
        try:
            match = re.match(r"(\d+)", cpus_on_node)
            if match:
                num_cpus = int(match.group(1))
                return int(mem_per_cpu) * 1024 * 1024 * num_cpus
        except ValueError:
            pass
    return None


def _resolve_cgroup_memory_paths() -> tuple[str | None, str | None]:
    """Returns ``(version, path)`` -- ``version`` is ``"v1"``/``"v2"``/None."""
    if os.path.isfile("/sys/fs/cgroup/memory.current"):
        return ("v2", "/sys/fs/cgroup")

    if os.path.isfile("/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        return ("v1", "/sys/fs/cgroup/memory")

    path = _find_cgroup_path("memory")
    if path:
        if os.path.isfile(os.path.join(path, "memory.current")):
            return ("v2", path)
        if os.path.isfile(os.path.join(path, "memory.usage_in_bytes")):
            return ("v1", path)

    return (None, None)


def _get_cgroup_memory_limit() -> int | None:
    version, path = _resolve_cgroup_memory_paths()
    if version is None:
        return None
    try:
        if version == "v2":
            limit_file = os.path.join(path, "memory.max")
            if os.path.isfile(limit_file):
                with open(limit_file, "r") as f:
                    val = f.read().strip()
                    if val != "max":
                        return int(val)
        elif version == "v1":
            limit_file = os.path.join(path, "memory.limit_in_bytes")
            if os.path.isfile(limit_file):
                with open(limit_file, "r") as f:
                    val = int(f.read().strip())
                    # Filter "unlimited" sentinels (kernels report ~2^63-1).
                    if val < 2 ** 60:
                        return val
    except (IOError, ValueError):
        pass
    return None


def _get_cgroup_memory_usage() -> int | None:
    """Used (anon + kernel) bytes; subtracts file cache because Linux treats
    page cache as 'used' but it's reclaimable on demand."""
    version, path = _resolve_cgroup_memory_paths()
    if version is None:
        return None

    usage = None
    cache = 0
    try:
        if version == "v2":
            current_file = os.path.join(path, "memory.current")
            stat_file = os.path.join(path, "memory.stat")
            if os.path.isfile(current_file):
                with open(current_file, "r") as f:
                    usage = int(f.read().strip())
                if os.path.isfile(stat_file):
                    with open(stat_file, "r") as f:
                        for line in f:
                            if line.startswith("file "):
                                cache = int(line.split()[1])
                                break
        elif version == "v1":
            usage_file = os.path.join(path, "memory.usage_in_bytes")
            stat_file = os.path.join(path, "memory.stat")
            if os.path.isfile(usage_file):
                with open(usage_file, "r") as f:
                    usage = int(f.read().strip())
                if os.path.isfile(stat_file):
                    with open(stat_file, "r") as f:
                        for line in f:
                            if line.startswith("total_cache") or line.startswith("cache"):
                                cache = int(line.split()[1])
                                if line.startswith("total_cache"):
                                    break
    except (IOError, ValueError):
        pass

    if usage is not None:
        return max(0, usage - cache)
    return None


def _find_cgroup_path(controller: str) -> str | None:
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 3:
                    if controller in parts[1] or (
                        parts[1] == "" and parts[0] == "0"
                    ):
                        rel_path = parts[2].lstrip("/")
                        v2_path = os.path.join("/sys/fs/cgroup", rel_path)
                        v1_path = os.path.join(
                            f"/sys/fs/cgroup/{controller}", rel_path
                        )
                        if os.path.exists(v2_path):
                            return v2_path
                        if os.path.exists(v1_path):
                            return v1_path
    except (IOError, OSError):
        pass
    return None


def _get_process_tree_rss() -> int | None:
    """RSS for current process + children. Used as a fallback to estimate
    cgroup usage when only the limit is known."""
    try:
        import psutil  # type: ignore[import-not-found]
        process = psutil.Process()
        usage = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                usage += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return usage
    except ImportError:
        pass

    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (IOError, ValueError):
        pass
    return None


def _get_system_available_memory() -> int:
    try:
        import psutil  # type: ignore[import-not-found]
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass

    if os.path.isfile("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo: dict[str, int] = {}
                for line in f:
                    parts = line.split()
                    meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
                if "MemAvailable" in meminfo:
                    return meminfo["MemAvailable"]
                return meminfo.get("MemFree", 0) + meminfo.get("Cached", 0)
        except (IOError, ValueError):
            pass
    return 0
