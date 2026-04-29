"""Working-set sizing -- v2-native (no orig dependency).

Picks the chunk size, tokens-per-round, and how many transformer layers'
worth of weights / grads / opt state to keep on the GPU vs. host. The math
is a port of ``orig/working_set.py:determine_working_set_config`` -- same
heuristics, same outputs -- but built on v2-owned helpers in
:mod:`flextrain.core._sizing` and :mod:`flextrain.core._memory` so we
no longer reach into ``orig/`` at runtime.

What changed from the orig wrapper
----------------------------------
* No more :func:`orig.awsm_transformer.get_hardware_env` call: the
  :class:`HardwareCost`-style numbers (peak TFLOPS, PCIe GB/s) come in
  as inputs from the much faster
  :func:`flextrain.core.hw_probe.probe_hardware`. Working set internally
  derives the layer transfer duration (= bytes / GB/s) and uses
  ``peak_tflops`` directly for the AI bound; the per-component matmul
  report is no longer consulted (it was only needed for finer-grained
  scheduling that the DP solver doesn't use).
* Helper modules: ``_sizing`` for byte counts, ``_memory`` for capacity
  introspection (slurm / cgroup / psutil).

The :class:`WorkingSetConfig` dataclass shape is unchanged so all
existing engine callers (``BufferManager``, ``ActiveModel``,
``flextrain.bench.parity``) keep working unchanged.

Heterogeneous backbones
-----------------------
When the schema-driven helper :func:`size_working_set_for_engine` is used
(callers that already have :class:`Layer` objects in hand), per-layer
quantities (param/grad/opt bytes, act-slot byte size) are reduced via
``max(...)`` across layers. That matches the engine's
:class:`BufferManager`, which sizes every ring slot for the worst-case
layer so any layer fits any slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from . import _memory, _sizing
from ._sizing import (
    backbone_layer_size_bytes,
    context_size_bytes,
    embedding_size_bytes,
    full_act_slot_size_bytes,
    get_divisors,
    head_size_bytes,
    layer_matmul_flops_per_token,
    min_act_slot_size_bytes,
    prev_high_div,
    round_to_nearest,
    torch_dtype_from_name,
    transformer_saved_act_sizes,
)


# ===========================================================================
# Tunables. Mirror orig:9-15.
# ===========================================================================


# Bias the AI-bound chunk-size pick toward bigger matmuls -- during
# training we want compute well above the roofline so the param/grad
# transfers can hide behind it. Orig defaulted to 2.
ARITH_BOUND_FACTOR = 2

DEFAULT_LEEWAY_GPU_BYTES = 2 * (1 << 30)
DEFAULT_LEEWAY_HOST_BYTES = 10 * (1 << 30)


# ===========================================================================
# WorkingSetConfig (public dataclass; consumed by ``BufferManager``,
# ``ActiveModel``, parity bench, etc.)
# ===========================================================================


@dataclass(frozen=True)
class WorkingSetConfig:
    """Result of :func:`determine_working_set_config`. Field names match
    orig's working_set return tuple exactly so engine consumers can stay
    unchanged."""

    # Data sizing (paper §3.2)
    target_round_tokens: int  # TR
    max_chunk_size: int  # TC
    max_training_chunks: int
    max_total_round_tokens: int
    target_num_rounds: int

    # Memory partitioning (paper §3.3) -- NA derived engine-side from
    # ``gpu_act_buffer_size // act_slot_size_bytes``.
    n_gpu_layers: int  # NP
    n_gpu_grads: int  # NG
    n_gpu_opt_layers: int

    # Activation buffer byte budgets
    gpu_act_buffer_size: int
    host_act_buffer_size: int

    # Budgets echoed back (used for logging + the engine's safety asserts)
    available_gpu_memory_bytes: int
    available_host_memory_bytes: int
    leeway_gpu_memory_bytes: int
    leeway_host_memory_bytes: int
    max_seq_len: int

    # Hardware report. Populated minimally now that we no longer run the
    # full orig probe -- present as ``Mapping[str, Any]`` for back-compat
    # with callers that read ``hw_env["transfer_report"][...]``. New
    # callers should use :func:`flextrain.core.hw_probe.probe_hardware`
    # directly for measured TFLOPS / PCIe GB/s.
    hardware_env: Mapping[str, Any] = field(default_factory=dict)

    # Untyped echo of the internal solve dict for debugging.
    raw: Mapping[str, Any] = field(default_factory=dict)


# ===========================================================================
# Public helpers for engine consumers.
# ===========================================================================


def derive_n_gpu_act_slots(
    gpu_act_buffer_size: int, act_slot_size_bytes: int
) -> int:
    """Number of GPU activation-ring slots = ``buffer // slot_size``.

    Mirrors ``orig/active_model.py:453`` and matches what
    :class:`BufferManager` derives engine-side.
    """
    if act_slot_size_bytes <= 0:
        raise ValueError("act_slot_size_bytes must be positive")
    return gpu_act_buffer_size // act_slot_size_bytes


# ===========================================================================
# Internal: model memory-requirement breakdown (orig:18-121).
# ===========================================================================


@dataclass(frozen=True)
class _BackboneSizes:
    weight_bytes: int
    master_bytes: int
    grad_bytes: int
    opt_bytes: int


@dataclass(frozen=True)
class _BaselineModelMemory:
    required_gpu_bytes: int
    required_host_bytes: int
    embed_bytes: int
    head_bytes: int
    backbone: _BackboneSizes


def _baseline_model_memory(
    model_dims: Mapping,
    num_local_layers: int,
    *,
    training_config: Mapping | None,
    has_embed: bool,
    has_head: bool,
    lora_active: bool = False,
    layer_param_specs: Sequence | None = None,
    embed_param_spec: object = None,
    head_param_spec: object = None,
) -> _BaselineModelMemory:
    """How many bytes one full-state copy of the model needs on host
    (training state for every layer + endpoints) and GPU (one full
    layer of weights + grads, plus the embed/head training state).

    Matches orig:18-121. The opt-state size is computed from
    ``training_config["opt_choice"]``: AdamW = 2x opt-dtype tensors per
    param, Muon = 1x (plus a per-layer adamw fixup for routers + norms
    that Muon can't update -- orig:99-100).

    Spec-driven sizing
    ------------------
    When ``layer_param_specs`` / ``embed_param_spec`` / ``head_param_spec``
    are supplied, we compute the actual byte budget from those specs
    via ``ParamSpec.byte_size`` (which already excludes frozen tensors
    from grad / opt-state roles). This is the correct path for hybrid
    setups (LoRA = some tensors frozen, MLP weights still trainable;
    Muon = only 2-D weights get a single buffer; etc.). When the specs
    are not provided we fall back to the model_dims-based generic
    estimator (used by callers that haven't built layers yet).

    ``lora_active`` only affects the legacy generic estimator path —
    when specs are supplied, ``frozen=True`` on the spec is the source
    of truth and ``lora_active`` is ignored.
    """
    required_gpu_bytes = 0
    required_host_bytes = 0
    embed_bytes = 0
    head_bytes = 0

    grad_dims: dict | None = None
    opt_dims: dict | None = None
    opt_mult = 0
    opt_choice: str | None = None
    master_dims: dict | None = None

    if training_config is not None:
        master_dims = _override_dtypes(
            model_dims, training_config["master_weight_dtype"]
        )
        grad_dims = _override_dtypes(
            model_dims, training_config["grad_dtype"]
        )
        opt_choice = training_config["opt_choice"]
        if opt_choice == "AdamW":
            opt_mult = 2
        elif opt_choice == "Muon":
            opt_mult = 1
        else:
            raise ValueError(
                f"Invalid opt_choice {opt_choice!r}: must be AdamW or Muon"
            )
        opt_dims = _override_dtypes(
            model_dims, training_config["opt_dtype"]
        )

    # ---- Endpoints: embed + head full training state lives on GPU
    if has_embed and grad_dims is not None:
        if embed_param_spec is not None:
            # Spec-driven: respect frozen / per-tensor dtypes.
            emb_master = embed_param_spec.byte_size(model_dims, role="master")
            emb_grad = embed_param_spec.byte_size(model_dims, role="grad")
            emb_opt = opt_mult * embed_param_spec.byte_size(
                model_dims, role="opt_state"
            )
        else:
            emb_master = embedding_size_bytes(master_dims)
            if lora_active:
                emb_grad = 0
                emb_opt = 0
            else:
                emb_grad = embedding_size_bytes(grad_dims)
                # Endpoints always use AdamW (orig:62)
                emb_opt = 2 * embedding_size_bytes(opt_dims)
        required_gpu_bytes += emb_master + emb_grad + emb_opt
        required_host_bytes += emb_master + emb_grad + emb_opt
        embed_bytes = emb_master + emb_grad + emb_opt

    if has_head and grad_dims is not None:
        if head_param_spec is not None:
            head_master = head_param_spec.byte_size(model_dims, role="master")
            head_grad = head_param_spec.byte_size(model_dims, role="grad")
            head_opt = opt_mult * head_param_spec.byte_size(
                model_dims, role="opt_state"
            )
        else:
            head_master = head_size_bytes(master_dims)
            if lora_active:
                head_grad = 0
                head_opt = 0
            else:
                head_grad = head_size_bytes(grad_dims)
                head_opt = 2 * head_size_bytes(opt_dims)
        required_gpu_bytes += head_master + head_grad + head_opt
        required_host_bytes += head_master + head_grad + head_opt
        head_bytes = head_master + head_grad + head_opt

    # ---- Backbone: training state in host, +1 layer weights+grads on GPU
    if training_config is not None and num_local_layers > 0:
        if layer_param_specs:
            # Spec-driven: take the per-layer maximum across (possibly
            # heterogeneous) layer types. ``ParamSpec.byte_size`` honors
            # ``TensorSpec.frozen`` for grad/opt roles, which is what
            # gives us correct LoRA accounting.
            weight_b = max(
                ps.byte_size(model_dims, role="compute")
                for ps in layer_param_specs
            )
            master_b = max(
                ps.byte_size(model_dims, role="master")
                for ps in layer_param_specs
            )
            grad_b = max(
                ps.byte_size(model_dims, role="grad")
                for ps in layer_param_specs
            )
            opt_b = opt_mult * max(
                ps.byte_size(model_dims, role="opt_state")
                for ps in layer_param_specs
            )
        else:
            weight_b = backbone_layer_size_bytes(model_dims)
            master_b = backbone_layer_size_bytes(master_dims)
            if lora_active:
                grad_b = 0
                opt_b = 0
            else:
                grad_b = backbone_layer_size_bytes(grad_dims)
                opt_b = opt_mult * backbone_layer_size_bytes(opt_dims)

                # Muon can't update routers and norms; those tensors fall back
                # to AdamW-style (2x state). Orig:99-100 adds the missing copy.
                if opt_choice == "Muon":
                    extra_dtype = torch_dtype_from_name(training_config["opt_dtype"])
                    d = model_dims["d_model"]
                    # 2 norms + (router weight matrix per routed expert)
                    opt_b += extra_dtype.itemsize * (
                        2 * d + model_dims["num_routed_experts"] * d
                    )

        backbone = _BackboneSizes(
            weight_bytes=weight_b,
            master_bytes=master_b,
            grad_bytes=grad_b,
            opt_bytes=opt_b,
        )
        required_host_bytes += num_local_layers * (master_b + grad_b + opt_b)
        # +1 layer of weights + grads in GPU memory (the rest of the
        # GPU activation+opt budget gets sliced up below).
        required_gpu_bytes += master_b + grad_b
    elif num_local_layers > 0:
        weight_b = backbone_layer_size_bytes(model_dims)
        backbone = _BackboneSizes(
            weight_bytes=weight_b, master_bytes=weight_b,
            grad_bytes=0, opt_bytes=0,
        )
        required_host_bytes += num_local_layers * weight_b
        required_gpu_bytes += weight_b
    else:
        backbone = _BackboneSizes(
            weight_bytes=0, master_bytes=0, grad_bytes=0, opt_bytes=0,
        )

    return _BaselineModelMemory(
        required_gpu_bytes=required_gpu_bytes,
        required_host_bytes=required_host_bytes,
        embed_bytes=embed_bytes,
        head_bytes=head_bytes,
        backbone=backbone,
    )


def _override_dtypes(model_dims: Mapping, dtype_name: str) -> dict:
    """Return a deep-ish copy of ``model_dims`` with every entry of the
    ``datatypes`` sub-dict replaced by ``dtype_name``. Mirrors orig's
    ``copy.deepcopy + override`` pattern."""
    out: dict = dict(model_dims)
    out["datatypes"] = {k: dtype_name for k in model_dims["datatypes"]}
    return out


def _baseline_gpu_activation_memory(
    model_dims: Mapping,
    max_seq_len: int,
    chunk_size: int,
    num_chunks: int,
    *,
    training_config: Mapping | None,
) -> int:
    """Per-round GPU activation overhead: transition table + (fwd + bwd)
    context windows + per-chunk attn/MLP scratch space. Mirrors orig:168-212.

    Counts the workspace each in-flight chunk needs on the device beyond
    the activation ring slots themselves. The MoE branch follows orig:194-200.
    """
    d = model_dims["d_model"]
    residual_dt = torch_dtype_from_name(model_dims["datatypes"]["residual"])
    sz_act = residual_dt.itemsize

    tokens_per_round = num_chunks * chunk_size

    # Transition table (residual stream snapshots between chunks).
    bytes_used = tokens_per_round * d * sz_act

    # Forward + backward context windows (K + V cache).
    ctx_window = max(chunk_size, max_seq_len)
    ctx_bytes = context_size_bytes(model_dims, ctx_window)
    bytes_used += ctx_bytes
    if training_config is not None:
        bytes_used += ctx_bytes  # bwd-side window

    # Per-chunk attention workspace: dQ, dK, dV accumulation copies.
    n_h = model_dims["n_heads"]
    n_kv = model_dims["n_kv_heads"]
    hd = model_dims["head_dim"]
    attn_workspace = (
        chunk_size * (4 * n_h * hd + 4 * n_kv * hd) * sz_act
    )

    # Per-chunk linear-attention bwd workspace (Qwen3-Next / Qwen3.5* /
    # Qwen3.6* hybrid layers). Not all archs have linear-attn layers,
    # but when they do they allocate substantial transient buffers in
    # ``GatedDeltaNetBlock.bwd`` that the dense-transformer accounting
    # above doesn't capture:
    #   * ``d_post_conv`` (cat of dq/dk/dv pre):   T * conv_dim * 2
    #   * ``d_conv_in`` (conv1d input grad):       T * conv_dim * 2
    #     (this lives in FLA's causal_conv1d_bwd output buffer; counted
    #     in fla_conv_scratch below.)
    #   * ``d_qkvz`` (cat output for proj wgrad):  T * proj_qkvz_dim * 2
    #   * FLA bwd internal scratch (rough bound).
    # The old ``q_n``/``k_n`` fp32 promotion (2 * T * key_dim * 4
    # bytes) is gone — q_n/k_n are now saved post-l2norm in the slot
    # (lin_q/lin_k tier-2) and read directly in bwd; no recompute.
    # We size for the worst case: every in-flight chunk could be on a
    # linear-attn layer simultaneously. ``conv_dim = 2*key_dim + value_dim``
    # and ``proj_qkvz_dim = 2*key_dim + 2*value_dim``.
    num_v_heads = model_dims.get("num_v_heads") or model_dims.get(
        "linear_num_v_heads"
    )
    head_v_dim = model_dims.get("head_v_dim") or model_dims.get(
        "linear_head_v_dim"
    )
    num_k_heads = model_dims.get("num_k_heads") or model_dims.get(
        "linear_num_k_heads"
    )
    head_k_dim = model_dims.get("head_k_dim") or model_dims.get(
        "linear_head_k_dim"
    )
    if num_v_heads and head_v_dim and num_k_heads and head_k_dim:
        key_dim = num_k_heads * head_k_dim
        value_dim = num_v_heads * head_v_dim
        conv_dim = 2 * key_dim + value_dim
        proj_qkvz_dim = 2 * key_dim + 2 * value_dim
        fp32 = torch.float32.itemsize
        # FLA's chunk_gated_delta_rule_fwd / _bwd allocates substantial
        # internal scratch tensors (see fla/ops/common/chunk_delta_h.py
        # and chunk_o.py). The dominant ones are per-FLA-chunk state
        # tensors of shape (B, NT, HV, V, K) where NT = T / FLA_CHUNK
        # (FLA's internal chunk size is 64 for the gated-delta-rule):
        #   * h, v_new (fwd)
        #   * dh, dv2 (bwd)
        # plus several T*value_dim and T*key_dim sized tensors (o, dq,
        # dk, dv, w, u, A) that don't depend on the FLA chunk axis.
        FLA_INTERNAL_CHUNK = 64
        nt_chunks = max(1, chunk_size // FLA_INTERNAL_CHUNK)
        # Per-state-tensor bytes; multiply by ~4 for the worst case
        # (h + v_new in fwd OR dh + dv2 in bwd, plus headroom).
        per_state = (
            nt_chunks * num_v_heads * head_v_dim * head_k_dim * sz_act
        )
        # Per-token tensor bytes (o / dq / dk / dv / w / u / A and
        # similar). ~6 of these are live simultaneously in bwd.
        per_token_scratch = chunk_size * value_dim * sz_act
        fla_scratch = 4 * per_state + 6 * per_token_scratch
        lin_attn_fwd_workspace = (
            fla_scratch                                # FLA fwd internals
            + chunk_size * value_dim * sz_act          # FLA core_out
        )
        lin_attn_bwd_workspace = (
            chunk_size * conv_dim * sz_act             # d_post_conv (pre-silu-bwd cat)
            + chunk_size * conv_dim * sz_act           # d_post_conv_pre_silu (silu_bwd output)
            + chunk_size * conv_dim * sz_act           # FLA conv1d_bwd dx output
            + chunk_size * proj_qkvz_dim * sz_act      # d_qkvz (pre-projection-wgrad cat)
            + fla_scratch                              # FLA gated-delta-rule bwd internals
        )
        # Take ``max`` of fwd vs bwd (only one runs at a time per layer)
        # and ``max`` against the dense full-attn workspace (each layer
        # is either full-attn or linear-attn, not both simultaneously
        # in flight on the same chunk).
        attn_workspace = max(
            attn_workspace,
            lin_attn_fwd_workspace,
            lin_attn_bwd_workspace,
        )

    # Per-chunk MLP workspace: depends on whether routed experts exist.
    expert_dim = model_dims["expert_dim"]
    top_k = model_dims["top_k"]
    num_routed = model_dims["num_routed_experts"]
    if num_routed > 0:
        # attn norm output, scattered X, scattered upstream
        mlp_workspace = (
            chunk_size * (d + 2 * top_k * d) * sz_act
        )
        # intra-expert backprop scratch (avg tokens per expert estimate)
        avg_tokens = int(chunk_size * top_k / num_routed)
        mlp_workspace += 2 * avg_tokens * 4 * expert_dim * sz_act
    else:
        # dense bwd: (d_act_upstream, fwd_act, dx1_up, dx3_up)
        mlp_workspace = chunk_size * 4 * expert_dim * sz_act

    # Residual scratch (1 chunk at a time on device during bwd).
    resid_workspace = chunk_size * d * sz_act

    bytes_used += resid_workspace + max(attn_workspace, mlp_workspace)
    return bytes_used


# ===========================================================================
# Public: dict-driven solver (preserves the orig signature so all
# existing callers keep working).
# ===========================================================================


def determine_working_set_config(
    model_dims: Mapping,
    max_seq_len: int,
    max_global_batch_tokens: int,
    *,
    training_config: Mapping | None = None,
    has_embed: bool = True,
    has_head: bool = True,
    num_local_layers: int | None = None,
    chunk_size: int | None = None,
    max_gpu_mem_bytes: int | None = None,
    max_host_mem_bytes: int | None = None,
    leeway_gpu_mem_bytes: int = DEFAULT_LEEWAY_GPU_BYTES,
    leeway_host_mem_bytes: int = DEFAULT_LEEWAY_HOST_BYTES,
    verbose: bool = False,
    device_id: int = 0,
    min_tokens_per_round_limit: int | None = None,
    max_tokens_per_round_limit: int | None = None,
    fixed_seq_len: bool = False,
    min_chunk_size: int | None = None,
    max_chunk_size: int | None = None,
    peak_tflops: float | None = None,
    pcie_bw_gbps: float | None = None,
    mem_bw_gbps: float | None = None,
    lora_active: bool = False,
    layer_param_specs: Sequence | None = None,
    embed_param_spec: object = None,
    head_param_spec: object = None,
    layer_schemas: Sequence | None = None,
) -> WorkingSetConfig:
    """Solve the working set: pick chunk size, tokens-per-round, and
    GPU-resident layer counts. Native v2 implementation -- no ``orig`` import.

    Parameters mirror the orig signature; the two new keyword-only knobs
    ``peak_tflops`` and ``pcie_bw_gbps`` accept measured numbers from
    :func:`flextrain.core.hw_probe.probe_hardware`. When omitted, the
    solver runs a quick probe internally (unless we're on a non-CUDA
    box, in which case it falls back to conservative scalars and prints
    a warning).

    Heterogeneous backbones: this dict-based path treats every layer as
    identical (using ``model_dims`` once). The schema-driven entrypoint
    :func:`size_working_set_for_engine` is the right tool when the
    backbone has mixed layer types -- it uses ``max(...)`` per-layer
    quantity so every reserved slot fits the worst-case layer.

    ``layer_schemas`` (optional): when callers have built the actual
    backbone layers before solving the working set (the
    ``from_pretrained`` flow does), pass the per-layer
    :class:`ActivationSchema` list here. The solver replaces the
    hand-coded :func:`transformer_saved_act_sizes` level-0 estimate
    with the maximum ``schema.home_size_bytes(num_tokens, dims, 0)``
    across layers, which captures arch-specific tier-0 fields (e.g.
    linear-attn ``lin_z`` / ``lin_*_rstd``) that the dense-transformer
    heuristic doesn't model. Without this, hybrid linear+full
    architectures (Qwen3-Next / Qwen3.5 / Qwen3.5-MoE / Qwen3.6 /
    Qwen3.6-MoE) under-size the host activation buffer by 1.5-2x and
    ``plan_from_solution`` raises at the first round.
    """
    def _min_act_slot_bytes(c: int) -> int:
        """Per-slot home bytes at level 0 — schema-driven when
        ``layer_schemas`` is provided, falling back to the dense-
        transformer estimate otherwise. We take ``max`` across layers
        because ``BufferManager`` sizes every host slot to the
        worst-case layer."""
        if layer_schemas is not None and len(layer_schemas) > 0:
            return max(
                int(s.home_size_bytes(c, model_dims, 0))
                for s in layer_schemas
            )
        return int(transformer_saved_act_sizes(model_dims, c)[0])

    def _full_act_slot_bytes(c: int) -> int:
        """Per-slot device bytes at the FULL save level — bytes the GPU
        activation ring actually reserves per slot. Schema-driven max
        across layers when available so hybrid linear+full backbones
        size the slot correctly: BufferManager uses
        ``max(schema.device_size_bytes(...) for schema in layer_schemas)``
        (see flextrain/engine/buffers.py:522), and the working_set's
        chunk-selection arithmetic must agree or the engine raises
        "gpu_act_buffer_size too small for a single activation slot"
        immediately after construction.
        """
        if layer_schemas is not None and len(layer_schemas) > 0:
            return max(
                int(s.device_size_bytes(c, model_dims))
                for s in layer_schemas
            )
        return int(full_act_slot_size_bytes(model_dims, c))
    if num_local_layers is None:
        num_local_layers = model_dims["n_layers"]

    if verbose:
        print("[Working Set Log] Probing hardware capacity + bandwidth...", flush=True)

    available_gpu = _memory.get_available_gpu_memory(device_id)
    available_host = _memory.get_available_host_memory()

    if peak_tflops is None or pcie_bw_gbps is None or mem_bw_gbps is None:
        from .hw_probe import probe_hardware  # local import keeps this module importable on no-CUDA boxes
        try:
            res = probe_hardware(device=f"cuda:{device_id}")
            if peak_tflops is None:
                peak_tflops = res.hw_cost.peak_tflops
            if pcie_bw_gbps is None:
                pcie_bw_gbps = res.hw_cost.pcie_bw_gbps
            if mem_bw_gbps is None:
                mem_bw_gbps = res.mem_bw_gbps
        except Exception as exc:
            print(
                f"Warning: hardware probe failed ({exc!r}); falling back to "
                "60 TFLOPS / 20 GB/s PCIe / 1000 GB/s mem-bw placeholders. "
                "Pass measured peak_tflops/pcie_bw_gbps/mem_bw_gbps to silence this.",
                flush=True,
            )
            if peak_tflops is None:
                peak_tflops = 60.0
            if pcie_bw_gbps is None:
                pcie_bw_gbps = 20.0
            if mem_bw_gbps is None:
                mem_bw_gbps = 1000.0

    if verbose:
        print(
            f"[Working Set Log] Raw Observed Available GPU Memory Capacity of "
            f"{available_gpu / (1 << 30):.2f}GiB and Host Memory Capacity of "
            f"{available_host / (1 << 30):.2f}GiB",
            flush=True,
        )
        if max_gpu_mem_bytes is not None:
            print(
                f"[Working Set Log] Inputted Max GPU Memory of "
                f"{max_gpu_mem_bytes / (1 << 30):.2f}GiB",
                flush=True,
            )
        if max_host_mem_bytes is not None:
            print(
                f"[Working Set Log] Inputted Max Host Memory of "
                f"{max_host_mem_bytes / (1 << 30):.2f}GiB",
                flush=True,
            )
        print(
            f"[Working Set Log] Using Leeway of "
            f"{leeway_gpu_mem_bytes / (1 << 30):.2f}GiB for GPU Memory and "
            f"{leeway_host_mem_bytes / (1 << 30):.2f}GiB for Host Memory",
            flush=True,
        )

    if max_gpu_mem_bytes is None:
        max_gpu_mem_bytes = available_gpu
    elif max_gpu_mem_bytes > available_gpu:
        print(
            f"Inputted max_gpu_mem_bytes ({max_gpu_mem_bytes}) is greater "
            f"than available_gpu_memory_capacity_bytes ({available_gpu}), "
            f"setting max gpu bytes to {available_gpu}",
            flush=True,
        )
        max_gpu_mem_bytes = available_gpu

    if max_host_mem_bytes is None:
        max_host_mem_bytes = available_host
    elif max_host_mem_bytes > available_host:
        print(
            f"Inputted max_host_mem_bytes ({max_host_mem_bytes}) is greater "
            f"than available_host_memory_capacity_bytes ({available_host}), "
            f"setting max host bytes to {available_host}",
            flush=True,
        )
        max_host_mem_bytes = available_host

    max_gpu_mem_bytes -= leeway_gpu_mem_bytes
    if max_gpu_mem_bytes < 0:
        raise ValueError("max_gpu_mem_bytes is less than 0 after accounting for leeway")
    max_host_mem_bytes -= leeway_host_mem_bytes
    if max_host_mem_bytes < 0:
        raise ValueError("max_host_mem_bytes is less than 0 after accounting for leeway")

    baseline = _baseline_model_memory(
        model_dims, num_local_layers,
        training_config=training_config,
        has_embed=has_embed, has_head=has_head,
        lora_active=lora_active,
        layer_param_specs=layer_param_specs,
        embed_param_spec=embed_param_spec,
        head_param_spec=head_param_spec,
    )
    if max_gpu_mem_bytes < baseline.required_gpu_bytes:
        raise ValueError(
            f"max_gpu_mem_bytes ({max_gpu_mem_bytes / (1 << 30):,.3f}GiB) is "
            "less than required minimum baseline_gpu_bytes "
            f"({baseline.required_gpu_bytes / (1 << 30):,.2f}GiB)"
        )
    if max_host_mem_bytes < baseline.required_host_bytes:
        raise ValueError(
            f"max_host_mem_bytes ({max_host_mem_bytes / (1 << 30):,.3f}GiB) is "
            "less than required minimum baseline_host_bytes "
            f"({baseline.required_host_bytes / (1 << 30):,.2f}GiB)"
        )

    remaining_gpu_mem_bytes = max_gpu_mem_bytes - baseline.required_gpu_bytes
    remaining_host_mem_bytes = max_host_mem_bytes - baseline.required_host_bytes

    if verbose:
        print(
            f"[Working Set Log] After Baseline Model Memory Requirements and "
            f"Accounting for Set Memory Bounds, Determined: Remaining GPU "
            f"Memory of {remaining_gpu_mem_bytes / (1 << 30):,.2f}GiB and "
            f"Remaining Host Memory of "
            f"{remaining_host_mem_bytes / (1 << 30):,.2f}GiB",
            flush=True,
        )

    # ---- Rough upper bound on tokens per round (orig:286-314) ----------
    remaining_total_mem = remaining_gpu_mem_bytes + remaining_host_mem_bytes
    d_model = model_dims["d_model"]
    ctx_dim = model_dims["head_dim"] * model_dims["n_kv_heads"]
    residual_dt = torch_dtype_from_name(model_dims["datatypes"]["residual"])

    # 100% intra-layer recompute, no kv recompute -- aggregate-memory bound.
    recomp_lim_max_tokens = remaining_total_mem / (
        (d_model + 2 * ctx_dim) * num_local_layers * residual_dt.itemsize
    )
    # GPU-only constraint (transition table + fwd+bwd ctx windows).
    gpu_lim_max_tokens = (
        remaining_gpu_mem_bytes
        - max_seq_len * 4 * ctx_dim * residual_dt.itemsize
    ) / (d_model * residual_dt.itemsize)

    max_tokens_per_round = int(min(recomp_lim_max_tokens, gpu_lim_max_tokens))

    if verbose:
        print(
            f"[Working Set Log] Orig max tok per round: {max_tokens_per_round}\n"
            f"\tMax global batch tokens: {max_global_batch_tokens}",
            flush=True,
        )

    max_tokens_per_round = min(max_tokens_per_round, max_global_batch_tokens)
    if max_tokens_per_round < max_seq_len:
        raise ValueError(
            f"Could not find a valid configuration for seq len {max_seq_len}; "
            f"estimating max tokens per round to be {max_tokens_per_round}"
        )

    if verbose:
        print(
            f"[Working Set Log] Determined Max Tokens Per Round of "
            f"{max_tokens_per_round} based on aggregate available memory of "
            f"{remaining_total_mem / (1 << 30):.2f}GiB, and GPU memory of "
            f"{remaining_gpu_mem_bytes / (1 << 30):.2f}GiB",
            flush=True,
        )

    # ---- Compute-bound target tokens per round (orig:319-393) ----------
    # Layer transfer: bytes / pcie_gbps. Inbound + outbound overlap on
    # different streams in steady state, so use the unidirectional rate.
    layer_bytes_inbound = baseline.backbone.weight_bytes
    layer_bytes_outbound = baseline.backbone.grad_bytes
    transfer_bandwidth_bytes_per_sec = pcie_bw_gbps * 1e9

    layer_transfer_duration_sec = layer_bytes_inbound / transfer_bandwidth_bytes_per_sec
    grad_transfer_duration_sec = (
        layer_bytes_outbound / transfer_bandwidth_bytes_per_sec
        if layer_bytes_outbound > 0 else 0.0
    )

    # Want layer compute >= layer in-transfer + grad out-transfer (orig:341).
    min_layer_computation_time = layer_transfer_duration_sec + grad_transfer_duration_sec

    if verbose:
        print(
            f"[Working Set Log] Observed Layer Transfer Duration of "
            f"{layer_transfer_duration_sec * 1e3:.2f} ms, "
            f"Estimated Peak TFLOPS: {peak_tflops:.2f}, "
            f"PCIe BW: {pcie_bw_gbps:.2f} GB/s",
            flush=True,
        )

    matmul_flops_per_token = layer_matmul_flops_per_token(model_dims)

    # Conservative attention-flops correction when seq len is fixed.
    attn_flops_min_est = 0
    if fixed_seq_len:
        attn_factor = 0.5 if model_dims["is_causal"] else 1.0
        attn_flops_min_est = (
            attn_factor * 4 * max_seq_len * max_seq_len
            * model_dims["head_dim"] * model_dims["n_heads"]
        )

    target_layer_flops = min_layer_computation_time * peak_tflops * 1e12
    target_tokens_per_round = math.ceil(
        (target_layer_flops - attn_flops_min_est) / matmul_flops_per_token
    )

    if verbose:
        print(
            f"[Working Set Log] Baseline Target Tokens Per Round for "
            f"Sufficient Computation Time: {target_tokens_per_round}",
            flush=True,
        )
    compute_lim_tokens_per_round = target_tokens_per_round

    if fixed_seq_len:
        target_tokens_per_round = max(max_seq_len, target_tokens_per_round)

    # Aggregate per-token activation cost across all layers.
    #
    # ``full_agg_act_bpt`` and ``min_act_bpt`` set the upper / lower
    # ``target_tokens_per_round`` ceilings (line 731 below picks
    # ``min_save_tokens_per_round`` as a hard cap). For hybrid
    # linear+full attention backbones (Qwen3-Next / Qwen3.5* / Qwen3.6*),
    # the dense-transformer ``min_act_slot_size_bytes`` heuristic
    # under-counts level-0 bytes by 2-3x because it doesn't model the
    # linear-attn ``lin_z`` / ``lin_*_rstd`` fields. That sets the
    # ceiling 2-3x too high → ``_pick_chunk_size`` then rejects every
    # candidate at the host check inside the loop because total round
    # bytes ≈ ``target * n_layers * bpt`` is constant in chunk size.
    #
    # When schema-driven sizing is available (``layer_schemas``), use
    # the worst-case ``max(home_size_bytes(1) for s in schemas) *
    # n_layers`` — same metric the in-loop host check uses, so the
    # ceiling and the per-option check are consistent.
    if layer_schemas is not None and len(layer_schemas) > 0:
        min_bpt_per_layer = max(
            int(s.home_size_bytes(1, model_dims, 0)) for s in layer_schemas
        )
        min_act_bpt = num_local_layers * min_bpt_per_layer
    else:
        min_act_bpt = num_local_layers * min_act_slot_size_bytes(model_dims, 1)
    full_agg_act_bpt = num_local_layers * full_act_slot_size_bytes(model_dims, 1)
    full_save_tokens_per_round = remaining_total_mem // full_agg_act_bpt
    min_save_tokens_per_round = remaining_total_mem // min_act_bpt

    if verbose:
        print(
            f"[Working Set Log] Based on aggregate available memory to save "
            f"all activations must use <= {full_save_tokens_per_round} tokens "
            f"per round and to save minimum activations must use <= "
            f"{min_save_tokens_per_round} tokens per round",
            flush=True,
        )
        print(
            f"[Working Set Log] Comparing prior tokens per round: "
            f"{target_tokens_per_round} with max seq len: {max_seq_len}, "
            f"full save tokens per round: {full_save_tokens_per_round}, "
            f"min save tokens per round: {min_save_tokens_per_round} and "
            f"max tokens per round: {max_tokens_per_round}",
            flush=True,
        )

    target_tokens_per_round = min(min_save_tokens_per_round, target_tokens_per_round)
    if target_tokens_per_round < max_seq_len:
        raise ValueError(
            f"Error: Could not find a valid configuration for seq len {max_seq_len}; "
            f"estimated max tokens with min activations to be {min_save_tokens_per_round}"
        )

    target_tokens_per_round = min(max_tokens_per_round, target_tokens_per_round)
    if min_tokens_per_round_limit is not None:
        target_tokens_per_round = max(min_tokens_per_round_limit, target_tokens_per_round)
    if max_tokens_per_round_limit is not None:
        target_tokens_per_round = min(max_tokens_per_round_limit, target_tokens_per_round)

    if fixed_seq_len:
        target_tokens_per_round = max(
            max_seq_len, round_to_nearest(target_tokens_per_round, max_seq_len)
        )
        if target_tokens_per_round > max_tokens_per_round:
            target_tokens_per_round -= max_seq_len
            if (
                target_tokens_per_round > max_tokens_per_round
                or target_tokens_per_round == 0
            ):
                raise ValueError(
                    f"Error: Could not find a valid configuration for fixed "
                    f"seq len {fixed_seq_len}; estimated max tokens per round "
                    f"to be {target_tokens_per_round}"
                )
    else:
        target_tokens_per_round = prev_high_div(target_tokens_per_round)

    if min_chunk_size is not None:
        target_tokens_per_round = max(min_chunk_size, target_tokens_per_round)

    if verbose:
        print(
            f"[Working Set Log] Comparing prior tokens per round: "
            f"{target_tokens_per_round} with min chunk size: {min_chunk_size} "
            f"and max global batch tokens: {max_global_batch_tokens}",
            flush=True,
        )

    target_tokens_per_round = min(max_global_batch_tokens, target_tokens_per_round)

    # ---- Min chunk size from MLP arithmetic intensity (orig:433-448) ----
    # H = peak_compute_flops / peak_mem_bytes_per_sec  (the hardware's
    # arithmetic-intensity bound -- a chunk smaller than this gives a
    # memory-bound matmul). For Llama 3.1 8B on an H100 (~700 TFLOPS,
    # ~3000 GB/s) H ~= 230, giving a min chunk of ~495 tokens for the
    # K=14336, N=4096 MLP. Mirrors orig:434.
    K = model_dims["expert_dim"]
    N = model_dims["d_model"]
    H = (peak_tflops * 1e12) / (mem_bw_gbps * 1e9)
    if model_dims["num_routed_experts"] > 0:
        target_min_per_exp = (
            ARITH_BOUND_FACTOR * H * K * N / max(1.0, K * N - H * (K + N))
        )
        inv_sparsity = (
            model_dims["num_routed_experts"] / max(1, model_dims["top_k"])
        )
        init_target_min_chunk = inv_sparsity * target_min_per_exp
    else:
        init_target_min_chunk = (
            ARITH_BOUND_FACTOR * H * K * N
            / max(1.0, K * N - H * (K + N))
        )

    if verbose:
        print(
            f"[Working Set Log] Determined Initial Target Min Chunk Size Est "
            f"(based on Arithmetic Intensity bound x factor of "
            f"{ARITH_BOUND_FACTOR}) of: {init_target_min_chunk}",
            flush=True,
        )

    if min_chunk_size is not None:
        # Caller-supplied min_chunk_size overrides the AI-bound
        # heuristic. Useful for inference / unit-test forwards on
        # short inputs, where the throughput-oriented AI lower bound
        # would reject every divisor of the actual batch size.
        init_target_min_chunk = min_chunk_size

    init_chunk_size_options = sorted(get_divisors(target_tokens_per_round), reverse=True)

    if fixed_seq_len:
        max_seqs_per_round = target_tokens_per_round // max_seq_len
        chunk_size_options = [
            max_seq_len * i for i in range(max_seqs_per_round, 0, -1)
        ]
        seq_len_divs = sorted(get_divisors(max_seq_len), reverse=True)
        for d in seq_len_divs:
            chunk_size_options.append(d)
    else:
        chunk_size_options = init_chunk_size_options

    # Augment with divisors of ``max_global_batch_tokens`` that are <=
    # ``target_tokens_per_round``. These give a *zero-tail* round
    # structure (``round_tokens`` divides the step cleanly), which is
    # what we want when ``target_tokens_per_round`` doesn't itself
    # divide ``max_global_batch_tokens`` — the original divisor list
    # has the divisor-of-divisor pathology that causes the tail-filter
    # to reject every option.
    batch_divs = [
        d for d in get_divisors(max_global_batch_tokens)
        if d <= target_tokens_per_round
    ]
    merged = set(chunk_size_options) | set(batch_divs)
    chunk_size_options = sorted(merged, reverse=True)

    chunk_size_options = [
        d for d in chunk_size_options if d >= init_target_min_chunk
    ]

    if verbose:
        print(f"[Working Set Log] Chunk Size Options: {chunk_size_options}", flush=True)
        print(
            f"[Working Set Log] Before deciding chunk size, observe remaining "
            f"gpu mem bytes as: {remaining_gpu_mem_bytes}, target tokens per "
            f"round: {target_tokens_per_round}",
            flush=True,
        )

    # ---- Pick chunk size: greedy / break on first option that fits a
    # second full layer (matches orig:476-653). ----
    best_option = _pick_chunk_size(
        chunk_size_options=chunk_size_options,
        model_dims=model_dims,
        max_seq_len=max_seq_len,
        max_global_batch_tokens=max_global_batch_tokens,
        target_tokens_per_round=target_tokens_per_round,
        remaining_gpu_mem_bytes=remaining_gpu_mem_bytes,
        remaining_host_mem_bytes=remaining_host_mem_bytes,
        baseline=baseline,
        training_config=training_config,
        compute_lim_tokens_per_round=compute_lim_tokens_per_round,
        num_local_layers=num_local_layers,
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        verbose=verbose,
        min_act_slot_fn=_min_act_slot_bytes,
        full_act_slot_fn=_full_act_slot_bytes,
    )

    if best_option is None:
        raise ValueError(
            "Error: Not enough GPU memory to fit any valid chunk size large "
            "enough to fit at least 1 additional complete layer"
        )

    if verbose:
        print(f"[Working Set Log] Selected Best Option: {best_option}", flush=True)

    target_chunk_size = best_option["target_chunk_size"]
    target_num_chunks = best_option["target_num_chunks"]
    n_gpu_layers = best_option["n_gpu_layers"]
    n_gpu_grad_layers = best_option["n_gpu_grad_layers"]
    n_gpu_opt_layers = best_option["n_gpu_opt_layers"]
    gpu_act_workspace_size_bytes = best_option["gpu_act_workspace_size_bytes"]
    total_act_slots = best_option["total_act_slots"]
    gpu_act_slots = best_option["gpu_act_slots"]

    full_act_slot = _full_act_slot_bytes(target_chunk_size)
    gpu_act_buffer_size_bytes = gpu_act_workspace_size_bytes
    endpoint_bytes = baseline.embed_bytes + baseline.head_bytes

    baseline_act_gpu_memory = _baseline_gpu_activation_memory(
        model_dims, max_seq_len, target_chunk_size, target_num_chunks,
        training_config=training_config,
    )

    est_total_gpu_bytes = (
        baseline_act_gpu_memory
        + gpu_act_workspace_size_bytes
        + baseline.backbone.weight_bytes * n_gpu_layers
        + baseline.backbone.grad_bytes * n_gpu_grad_layers
        + endpoint_bytes
    )
    assert est_total_gpu_bytes <= max_gpu_mem_bytes, (
        f"GPU memory accounting blew the budget: "
        f"{est_total_gpu_bytes} > {max_gpu_mem_bytes}"
    )

    host_act_slots = total_act_slots - gpu_act_slots
    max_host_act_buffer = host_act_slots * full_act_slot
    host_act_buffer_size_bytes = min(max_host_act_buffer, remaining_host_mem_bytes)

    min_act_slot = _min_act_slot_bytes(target_chunk_size)

    est_total_host_bytes = host_act_buffer_size_bytes + baseline.required_host_bytes

    if verbose:
        print(
            f"[Working Set Log] Determined Target Max Chunk Size of "
            f"{target_chunk_size}, Target Tokens Per Round of "
            f"{target_tokens_per_round}\n"
            f"\tAct Slot Size: {full_act_slot / (1 << 20):.2f}MiB\n"
            f"\t# GPU Full Act Slots: {gpu_act_slots}\n"
            f"\t# Host Act Slots: {host_act_slots}\n"
            f"\t# GPU Act Buffer Size: "
            f"{gpu_act_buffer_size_bytes / (1 << 30):.2f}GiB\n"
            f"\t# Host Act Buffer Size: "
            f"{host_act_buffer_size_bytes / (1 << 30):.2f}GiB",
            flush=True,
        )
        print(
            f"[Working Set Log] Expected GPU Memory Usage: "
            f"{est_total_gpu_bytes / (1 << 30):.2f}GiB, Expected Host Memory "
            f"Usage: {est_total_host_bytes / (1 << 30):.2f}GiB",
            flush=True,
        )

    min_host_act_buffer = host_act_slots * min_act_slot
    assert host_act_buffer_size_bytes >= min_host_act_buffer
    assert est_total_host_bytes <= max_host_mem_bytes

    target_round_tokens = target_chunk_size * target_num_chunks

    raw = {
        "available_gpu_memory_bytes": available_gpu,
        "available_host_memory_bytes": available_host,
        "leeway_gpu_memory_bytes": leeway_gpu_mem_bytes,
        "leeway_host_memory_bytes": leeway_host_mem_bytes,
        "n_gpu_layers": min(n_gpu_layers, num_local_layers),
        "n_gpu_grads": min(n_gpu_grad_layers, num_local_layers),
        "n_gpu_opt_layers": min(n_gpu_opt_layers, num_local_layers),
        "max_training_chunks": target_num_chunks,
        "max_chunk_size": target_chunk_size,
        "max_seq_len": max_seq_len,
        "target_round_tokens": target_round_tokens,
        "target_num_rounds": math.ceil(max_global_batch_tokens / target_round_tokens),
        "max_total_round_tokens": max_tokens_per_round,
        "host_act_buffer_size": int(host_act_buffer_size_bytes),
        "gpu_act_buffer_size": int(gpu_act_buffer_size_bytes),
        "max_host_mem_gb": max_host_mem_bytes / (1 << 30),
        "max_gpu_mem_gb": max_gpu_mem_bytes / (1 << 30),
    }

    # Lightweight hardware_env: we no longer run the full orig probe, but
    # callers (cli.py legacy path) read these keys -- so populate them
    # from the inputs we have. New callers should consume probe_hardware
    # directly.
    hardware_env: dict[str, Any] = {
        "available_gpu_memory_capacity": available_gpu,
        "available_host_memory_capacity": available_host,
        "transfer_report": {
            "layer_concurrent_transfer_duration_sec": layer_transfer_duration_sec,
            "overall_unidirectional_concurrent_bandwidth_gb_per_sec": pcie_bw_gbps,
        },
        "matmul_report": {
            "overall_layer_matmul_throughput_tflops_per_sec": peak_tflops,
        },
        "basic_peak_tflops_est": peak_tflops,
        "basic_peak_mem_bandwidth_gb_per_sec": mem_bw_gbps,
    }

    return WorkingSetConfig(
        target_round_tokens=target_round_tokens,
        max_chunk_size=target_chunk_size,
        max_training_chunks=target_num_chunks,
        max_total_round_tokens=max_tokens_per_round,
        target_num_rounds=math.ceil(max_global_batch_tokens / target_round_tokens),
        n_gpu_layers=min(n_gpu_layers, num_local_layers),
        n_gpu_grads=min(n_gpu_grad_layers, num_local_layers),
        n_gpu_opt_layers=min(n_gpu_opt_layers, num_local_layers),
        gpu_act_buffer_size=int(gpu_act_buffer_size_bytes),
        host_act_buffer_size=int(host_act_buffer_size_bytes),
        available_gpu_memory_bytes=available_gpu,
        available_host_memory_bytes=available_host,
        leeway_gpu_memory_bytes=leeway_gpu_mem_bytes,
        leeway_host_memory_bytes=leeway_host_mem_bytes,
        max_seq_len=max_seq_len,
        hardware_env=hardware_env,
        raw=raw,
    )


def _pick_chunk_size(
    *,
    chunk_size_options: Sequence[int],
    model_dims: Mapping,
    max_seq_len: int,
    max_global_batch_tokens: int,
    target_tokens_per_round: int,
    remaining_gpu_mem_bytes: int,
    remaining_host_mem_bytes: int,
    baseline: _BaselineModelMemory,
    training_config: Mapping | None,
    compute_lim_tokens_per_round: int,
    num_local_layers: int,
    max_chunk_size: int | None,
    min_chunk_size: int | None,
    verbose: bool,
    min_act_slot_fn=None,
    full_act_slot_fn=None,
) -> dict | None:
    """Greedy chunk-size selection. Mirrors orig:476-653.

    For each candidate chunk size (largest first), tries to allocate:
    1. Baseline activation overhead (transition + ctx + workspace).
    2. One activation slot.
    3. One layer of opt state (often shares space with the act slot).
    4. A second activation slot, then second-layer weights/grads.
    5. As many full additional layers (weights+grads+act-slots) as fit.
    6. Fill remaining GPU memory with extra act slots (act workspace).

    Stops as soon as we find an option that fits a second full layer
    (the engine's overlap relies on at least 2 layers' worth of
    weights+grads being on-device at any time).
    """
    backbone = baseline.backbone

    def _search(tail_filter_enabled: bool) -> dict | None:
      best_option: dict | None = None

      for chunk_size in chunk_size_options:
        cur_gpu = remaining_gpu_mem_bytes
        if max_chunk_size is not None and chunk_size > max_chunk_size:
            continue
        if min_chunk_size is not None and chunk_size < min_chunk_size:
            break

        # Pick ``target_num_chunks``. The default is
        # ``target_tokens_per_round // chunk_size`` (largest nc that
        # fits inside the target round). But if ``chunk_size`` divides
        # ``max_global_batch_tokens`` cleanly, we can do better: pick
        # the largest nc such that (nc * chunk) divides batch — that
        # gives a *zero-tail* round structure. Without this, e.g.
        # chunk=131072 with target=478800 and batch=524288 picks
        # nc=3 → total_round=393216 → tail=131072 (25% of step!),
        # which the tail-round filter then rejects, leaving us with
        # only chunk=262144 nc=1 (the lone zero-tail option in the
        # divisor list of target=478800).
        default_nc = target_tokens_per_round // chunk_size
        target_num_chunks = default_nc
        if (
            chunk_size > 0
            and max_global_batch_tokens % chunk_size == 0
            and default_nc > 0
        ):
            # Largest nc <= default_nc such that nc divides
            # (max_global_batch_tokens // chunk_size).
            max_nc_in_batch = max_global_batch_tokens // chunk_size
            for nc_candidate in range(default_nc, 0, -1):
                if max_nc_in_batch % nc_candidate == 0:
                    target_num_chunks = nc_candidate
                    break
        temp_round_tokens = target_num_chunks * chunk_size
        final_round_tokens = max_global_batch_tokens % temp_round_tokens
        # Skip configs where the last round would be both (a) absolutely
        # small (per-round overhead dominates) AND (b) a meaningful
        # fraction of the step (its inefficiency isn't amortized).
        #
        # The two-clause version replaces orig's single absolute check
        # at orig:495-497, which rejected every divisor of e.g. a
        # ``target_tokens_per_round=5040`` candidate against a
        # ``max_global_batch_tokens=524288`` (=2^19) batch -- none of
        # 5040's divisors evenly divides 524288, so the remainder is
        # always nonzero and small. With many full rounds preceding it
        # (524288/5040 ~= 104) the per-step overhead is negligible.
        #
        # Clause (b) compares the tail to the *step*, not to one round:
        # the cost we care about is "tail wastes a meaningful fraction
        # of a step's compute", not "tail < 5% of one round" (the latter
        # would still reject the 5040/524288 case, since every divisor
        # of 5040 produces ``temp_round_tokens=5040`` and the same fixed
        # remainder, and the remainder is always >5% of 5040). The
        # divisor-list pathology kicks in exactly when
        # ``target_tokens_per_round`` doesn't divide
        # ``max_global_batch_tokens`` cleanly: e.g. tgt=25200 vs
        # batch=131072 (=2^17) gives ``131072 % 25200 = 5072`` for every
        # divisor of 25200, so a per-round threshold rejects all of them.
        #
        # On hardware where ``compute_lim_tokens_per_round`` is so large
        # that *every* candidate's ``final_round_tokens`` clears clause
        # (b) but fails clause (a), this filter rejects every option and
        # ``_pick_chunk_size`` returns None. The outer wrapper retries
        # with ``tail_filter_enabled=False`` to recover — the user gets
        # a tail-inefficient round rather than a hard failure.
        if (
            tail_filter_enabled
            and final_round_tokens > 0
            and final_round_tokens < 0.4 * compute_lim_tokens_per_round
            and final_round_tokens > 0.05 * max_global_batch_tokens
        ):
            continue

        baseline_act = _baseline_gpu_activation_memory(
            model_dims, max_seq_len, chunk_size, target_num_chunks,
            training_config=training_config,
        )
        cur_gpu -= baseline_act

        if full_act_slot_fn is not None:
            full_act = int(full_act_slot_fn(chunk_size))
        else:
            full_act = full_act_slot_size_bytes(model_dims, chunk_size)
        if cur_gpu < full_act:
            continue

        # Reserve at least one activation slot.
        gpu_act_workspace = full_act
        cur_gpu -= gpu_act_workspace
        n_gpu_layers = 1
        n_gpu_grad_layers = 1

        # +1 layer of opt state (often fits inside the slot we just reserved).
        if gpu_act_workspace < backbone.opt_bytes:
            extra_opt = backbone.opt_bytes - gpu_act_workspace
            if cur_gpu < extra_opt:
                continue
            gpu_act_workspace += extra_opt
            cur_gpu -= extra_opt

        # Prioritize 2 weight + 2 grad layers BEFORE the 2nd act slot.
        # The AdaWS pipeline overlaps layer N+1's weight prefetch with
        # layer N's compute, and layer N's grad eviction with layer
        # N-1's bwd compute — both need n_gpu_*_layers >= 2 to hide
        # PCIe latency. A 2nd act slot only buys per-chunk overlap on
        # a single layer (less impactful when num_chunks is small),
        # and adding it greedily at large chunks (e.g. Qwen3.6-35B-A3B
        # at chunk=262144 where slot ~ 20-25 GiB) consumes nearly the
        # whole post-baseline budget and leaves nothing for the 2nd
        # weight/grad — yielding a save plan that runs entirely at
        # level 0 with 60-70% of the step spent in recompute.
        if cur_gpu >= backbone.weight_bytes:
            n_gpu_layers = 2
            cur_gpu -= backbone.weight_bytes
        if cur_gpu >= backbone.grad_bytes:
            n_gpu_grad_layers = 2
            cur_gpu -= backbone.grad_bytes

        # Now try a 2nd act slot if there's still room.
        temp_slots = gpu_act_workspace // full_act
        if temp_slots < 2 and cur_gpu >= full_act:
            gpu_act_workspace += full_act
            cur_gpu -= full_act

        # Fill act slots up through layer 1, then layer 2, with whatever
        # still fits (orig:553-581).
        temp_slots = gpu_act_workspace // full_act
        if temp_slots < target_num_chunks:
            need = target_num_chunks - temp_slots
            need_bytes = need * full_act
            if cur_gpu < need_bytes:
                fits = cur_gpu // full_act
                gpu_act_workspace += fits * full_act
                cur_gpu -= fits * full_act
            else:
                gpu_act_workspace += need_bytes
                cur_gpu -= need_bytes

        temp_slots = gpu_act_workspace // full_act
        if temp_slots < 2 * target_num_chunks:
            need = 2 * target_num_chunks - temp_slots
            need_bytes = need * full_act
            if cur_gpu < need_bytes:
                fits = cur_gpu // full_act
                gpu_act_workspace += fits * full_act
                cur_gpu -= fits * full_act
            else:
                gpu_act_workspace += need_bytes
                cur_gpu -= need_bytes

        # As many full additional layers (weights+grads+act-slots-per-chunk) as fit.
        if full_act_slot_fn is not None:
            multi_chunk_slot_bytes = int(
                full_act_slot_fn(chunk_size * target_num_chunks)
            )
        else:
            multi_chunk_slot_bytes = full_act_slot_size_bytes(
                model_dims, chunk_size * target_num_chunks
            )
        addl_full_layer_bytes = (
            backbone.weight_bytes + backbone.grad_bytes
            + multi_chunk_slot_bytes
        )
        addl_complete = int(min(
            num_local_layers - 1,
            cur_gpu // addl_full_layer_bytes if addl_full_layer_bytes > 0 else 0,
        ))
        n_gpu_layers += addl_complete
        n_gpu_grad_layers += addl_complete
        complete_layers_bytes = addl_complete * addl_full_layer_bytes
        leftover = cur_gpu - complete_layers_bytes
        gpu_act_workspace += addl_complete * multi_chunk_slot_bytes

        # Final pass: prioritize getting to 2 layers/grads, then dump the
        # rest into act workspace (orig:603-610).
        if (
            n_gpu_layers < 2
            and n_gpu_layers < num_local_layers
            and leftover >= backbone.weight_bytes
        ):
            n_gpu_layers += 1
            leftover -= backbone.weight_bytes
        if (
            n_gpu_grad_layers < 2
            and n_gpu_grad_layers < num_local_layers
            and leftover >= backbone.grad_bytes
        ):
            n_gpu_grad_layers += 1
            leftover -= backbone.grad_bytes
        gpu_act_workspace += leftover

        n_gpu_opt_layers = int(min(
            num_local_layers,
            gpu_act_workspace // backbone.opt_bytes
            if backbone.opt_bytes > 0 else num_local_layers,
        ))

        total_act_slots = int(target_num_chunks * num_local_layers)
        gpu_act_slots = int(min(
            total_act_slots,
            gpu_act_workspace // full_act if full_act > 0 else 0,
        ))

        if min_act_slot_fn is not None:
            min_act_slot = int(min_act_slot_fn(chunk_size))
        else:
            min_act_slot = int(transformer_saved_act_sizes(model_dims, chunk_size)[0])
        if remaining_host_mem_bytes < min_act_slot * (total_act_slots - gpu_act_slots):
            continue

        option = {
            "target_chunk_size": chunk_size,
            "target_num_chunks": target_num_chunks,
            "n_gpu_layers": n_gpu_layers,
            "n_gpu_grad_layers": n_gpu_grad_layers,
            "n_gpu_opt_layers": n_gpu_opt_layers,
            "gpu_act_workspace_size_bytes": gpu_act_workspace,
            "gpu_act_slots": gpu_act_slots,
            "total_act_slots": total_act_slots,
            "act_slot_size_bytes": full_act,
        }

        # Healthy AdaWS pipelining requires all four resources to have
        # >= 2 GPU-resident copies: weights (so layer N+1 prefetch
        # overlaps layer N compute), grads (so layer N's grad eviction
        # overlaps layer N-1 bwd compute), opt state (so the optimizer
        # step overlaps the next round's first-layer prefetch), and act
        # slots (so chunk i+1's fwd overlaps chunk i's send-home /
        # bwd's chunk i-1 fetch-home). Chunks are iterated largest -> smallest,
        # so the first option that satisfies all four IS the largest
        # such chunk — take it and stop.
        healthy = (
            option["n_gpu_layers"] >= 2
            and option["n_gpu_grad_layers"] >= 2
            and option["n_gpu_opt_layers"] >= 2
            and option["gpu_act_slots"] >= 2
        )
        if healthy:
            return option

        if best_option is None:
            best_option = option
            continue

        # Otherwise: prefer the option whose minimum across the four
        # axes is highest (fewest pipelining shortfalls), with the
        # original per-axis "promote 1 -> >1" rules as the secondary
        # signal so we never regress on a measurable axis.
        def _floor(opt):
            return min(
                opt["n_gpu_layers"],
                opt["n_gpu_grad_layers"],
                opt["n_gpu_opt_layers"],
                opt["gpu_act_slots"],
            )

        if _floor(option) > _floor(best_option):
            best_option = option
            continue

        # Same floor — fall back to per-axis promotions (orig:639-652).
        if best_option["gpu_act_slots"] == 1 and option["gpu_act_slots"] > 1:
            best_option = option
        if (
            best_option["gpu_act_slots"] < best_option["target_num_chunks"]
            and option["gpu_act_slots"] >= option["target_num_chunks"]
        ):
            best_option = option
        if best_option["n_gpu_layers"] == 1 and option["n_gpu_layers"] > 1:
            best_option = option
        if best_option["n_gpu_grad_layers"] == 1 and option["n_gpu_grad_layers"] > 1:
            best_option = option
        if best_option["n_gpu_opt_layers"] == 1 and option["n_gpu_opt_layers"] > 1:
            best_option = option

      return best_option

    # First pass: tail filter on (orig behavior).
    chosen = _search(tail_filter_enabled=True)
    if chosen is not None:
        return chosen
    # Fallback: every option's tail tripped clause-(a) of the filter
    # against an enormous compute_lim. Retry with the filter disabled
    # so we accept a tail-inefficient round rather than fail outright.
    # ``chunk_size_options`` is already filtered by the AI-bound floor
    # upstream, so the fallback still respects ``init_target_min_chunk``.
    if verbose:
        print(
            "[Working Set Log] All chunk options rejected by tail-round "
            "filter; retrying with tail filter disabled (will accept "
            "smallest chunk above the arithmetic-intensity floor).",
            flush=True,
        )
    return _search(tail_filter_enabled=False)
