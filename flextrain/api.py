"""High-level convenience API.

The block-by-block / layer-by-layer construction in
``flextrain.engine.active_model.ActiveModel`` is the lowest level —
flexible but verbose. For SFT and continued-pretraining use cases on a
known HF model, the entry point you want is :func:`from_pretrained`,
which:

1. Reads the HF ``config.json``.
2. Looks up the registered :class:`flextrain.io.hf_weights.ArchSpec`
   for that ``architectures`` value.
3. Looks up the matching block builder (registered by the arch module).
4. Builds the embed / backbone / head, the working-set config, and
   the :class:`ActiveModel`.
5. Loads HF safetensors and applies any arch-specific weight
   permutations needed for FT's kernels (e.g. Llama Q/K halved→pair).

Example
-------
::

    import torch
    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    am = from_pretrained(
        "models/Llama-3.1-8B",
        optimizer=AdamW(AdamWHyperparams(lr=3e-5)),
        max_seq_len=2048,
        max_global_batch_tokens=2048,
        max_gpu_mem_bytes=int(24 * (1<<30)),
        max_host_mem_bytes=int(110 * (1<<30)),
        device="cuda:0",
    )
    # Then: standard fwd_bwd / step loop, or wrap with LoRA via the
    # ``lora_*`` kwargs (see below).

To switch to LoRA fine-tuning, pass ``lora_targets="all"`` (and
optionally ``lora_rank``, ``lora_alpha``, plus dtype overrides).

To extend FlexTrain to a new architecture
-----------------------------------------
See :doc:`docs/implementing.md`. The short version:

1. Build a layer/block class under ``flextrain/nn/layers/``.
2. Register an :class:`ArchSpec` (HF tensor-name map) under
   ``flextrain/io/arch/``.
3. Register a block-builder via :func:`register_block_builder` in the
   same module.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import torch

from flextrain.core.save_level import HardwareCost
from flextrain.core.working_set import determine_working_set_config
from flextrain.engine.active_model import ActiveModel
from flextrain.io.hf_weights import select_arch
from flextrain.optim.base import Optimizer


# ---------------------------------------------------------------------------
# Block-builder registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildContext:
    """Inputs each block builder receives.

    A builder turns ``(layer_idx, ctx)`` into one configured layer
    instance (a ``LlamaBlock``, ``OLMoEBlock``, ``Qwen3DenseBlock``,
    etc., possibly wrapped with :class:`LoRAWrapperLayer`).

    Fields
    ------
    hf_config
        The raw HF ``config.json`` dict (or ``PretrainedConfig``).
    dims
        FlexTrain ``dims`` map produced by the arch's
        ``hf_config_to_flextrain``. The builder may read it for shapes.
    hyperparams
        Per-layer hyperparams produced by the arch's
        ``hf_config_to_hyperparams`` (rope_base, rope_scaling,
        rms_norm_eps, sliding window, ...).
    compute_dtype, master_dtype, grad_dtype
        Per-role dtype overrides applied uniformly to all backbone
        layers. Default bf16 / bf16 / bf16. The block is free to
        override norms to fp32 internally.
    lora_targets, lora_rank, lora_alpha, lora_adapter_*_dtype
        If ``lora_targets`` is non-empty, the builder MUST wrap the
        base layer in :class:`LoRAWrapperLayer` with the given
        targets, rank, alpha and adapter dtype overrides.
    """
    hf_config: Mapping[str, Any]
    dims: Mapping[str, Any]
    hyperparams: Mapping[str, Any]
    compute_dtype: torch.dtype
    master_dtype: torch.dtype
    grad_dtype: torch.dtype
    norm_grad_dtype: torch.dtype
    lora_targets: object | None = None
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_adapter_compute_dtype: torch.dtype | None = None
    lora_adapter_master_dtype: torch.dtype | None = None
    lora_adapter_grad_dtype: torch.dtype | None = None
    lora_adapter_opt_state_dtype: torch.dtype | None = None


BlockBuilder = Callable[[int, BuildContext], object]
_BLOCK_BUILDER_REGISTRY: dict[str, BlockBuilder] = {}


def register_block_builder(hf_arch_ids: Iterable[str], builder: BlockBuilder) -> None:
    """Register a block builder for one or more HF architecture IDs.

    Idempotent — re-registering with the same builder is a no-op.
    Registering a different builder for an existing ID raises.
    """
    for arch_id in hf_arch_ids:
        existing = _BLOCK_BUILDER_REGISTRY.get(arch_id)
        if existing is not None and existing is not builder:
            raise ValueError(
                f"block builder for {arch_id!r} already registered "
                f"(was {existing!r}, new {builder!r})"
            )
        _BLOCK_BUILDER_REGISTRY[arch_id] = builder


def _select_block_builder(hf_config: Mapping) -> BlockBuilder:
    archs = (
        getattr(hf_config, "architectures", None)
        or hf_config.get("architectures")
        or []
    )
    for a in archs:
        if a in _BLOCK_BUILDER_REGISTRY:
            return _BLOCK_BUILDER_REGISTRY[a]
    raise ValueError(
        f"No block builder registered for any of {archs!r}. "
        f"Known: {sorted(_BLOCK_BUILDER_REGISTRY)}. "
        f"See flextrain/io/arch/<your_arch>.py for examples."
    )


# ---------------------------------------------------------------------------
# Helpers — embed / head construction
# ---------------------------------------------------------------------------


def _build_embed(dims: Mapping, *, compute, master, grad):
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    return TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=int(dims["vocab_size"]),
        d_model=int(dims["d_model"]),
        compute_dtype=compute, master_dtype=master, grad_dtype=grad,
    ))


def _build_head(dims: Mapping, hp: Mapping, *, compute, master, grad, norm_grad):
    from flextrain.nn.head import LMHead, LMHeadConfig
    return LMHead(LMHeadConfig(
        d_model=int(dims["d_model"]),
        vocab_size=int(dims["vocab_size"]),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-5)),
        head_chunk_size=int(hp.get("head_chunk_size", 512)),
        compute_dtype=compute, master_dtype=master, grad_dtype=grad,
        norm_grad_dtype=norm_grad,
    ))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def from_pretrained(
    model_path: str,
    *,
    optimizer: Optimizer,
    max_seq_len: int,
    max_global_batch_tokens: int,
    max_gpu_mem_bytes: int,
    max_host_mem_bytes: int,
    device: str = "cuda:0",
    leeway_gpu_mem_bytes: int = 2 * (1 << 30),
    leeway_host_mem_bytes: int = 4 * (1 << 30),
    compute_dtype: torch.dtype = torch.bfloat16,
    master_dtype: torch.dtype | None = None,
    grad_dtype: torch.dtype | None = None,
    norm_grad_dtype: torch.dtype = torch.float32,
    lora_targets: object | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_adapter_compute_dtype: torch.dtype | None = None,
    lora_adapter_master_dtype: torch.dtype | None = torch.float32,
    lora_adapter_grad_dtype: torch.dtype | None = torch.float32,
    lora_adapter_opt_state_dtype: torch.dtype | None = torch.float32,
    lora_init_seed: int = 20260424,
    lora_a_std: float = 0.02,
    hw_cost: HardwareCost | None = None,
    mem_bw_gbps: float | None = None,
    fixed_seq_len: bool = False,
    force_saved_act_level: int | None = None,
    head_chunk_size: int = 512,
    load_weights: bool = True,
    strict: bool = False,
    verbose: bool = False,
) -> ActiveModel:
    """Build a configured :class:`ActiveModel` for an HF model directory.

    Parameters
    ----------
    model_path
        Path to a directory containing ``config.json`` and HF
        safetensor shards (or ``pytorch_model.bin``).
    optimizer
        FlexTrain optimizer instance (e.g. ``AdamW(...)``).
    max_seq_len
        Hardware-side cap. Used by the working-set solver for
        activation buffer sizing.
    max_global_batch_tokens
        Tokens per step (across all sequences). The solver may decide
        to split into multiple gradient-accumulation rounds if this
        exceeds the GPU budget.
    max_gpu_mem_bytes, max_host_mem_bytes, leeway_*
        Memory caps the solver respects.
    compute_dtype, master_dtype, grad_dtype, norm_grad_dtype
        Per-role dtype overrides for the BASE block. Defaults match
        FlexTrain's standard bf16 stack.
    lora_targets
        ``None`` for full fine-tuning. ``"all"`` or a list of param
        names (e.g. ``("w_q","w_v")``) for LoRA. The default LoRA
        adapter dtype is fp32 for master/grad/opt-state (matches HF
        PEFT).
    hw_cost
        :class:`HardwareCost` for the DP-solver. Pass the result of
        :func:`flextrain.core.hw_probe.probe_hardware` for measured
        numbers, or construct one directly with known TFLOPS / PCIe
        values. ``None`` falls back to a conservative 60 TFLOPS /
        20 GB/s placeholder — fine for offline tests, but on real
        hardware this inflates compute-times ~10x and produces
        plans that skip recompute regardless of the actual
        compute/PCIe ratio. Real training runs should always pass
        a measured ``hw_cost``.
    force_saved_act_level
        Debug knob. If set, every (layer, chunk) pair that would have
        gone to a host slot is forced to this tier (clamped to each
        layer's ``schema.max_tier``). The on-device tail set by the
        engine still uses ``-1``. Useful for ablations / parity tests
        where you want a known save policy regardless of the DP
        solver's choice.
    load_weights
        If False, skip the ``am.load_hf`` call (random init).
    strict
        Forwarded to ``am.load_hf``.

    Returns
    -------
    A configured :class:`ActiveModel` with weights loaded and
    arch-specific permutations applied (callers should NOT need to
    apply Q/K halved→pair perms manually — the block builder does it).
    """
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"missing {cfg_path}")
    with open(cfg_path) as f:
        hf_config = json.load(f)

    # 1. Pick arch + block builder.
    arch = select_arch(hf_config)
    build_block = _select_block_builder(hf_config)

    # 2. Convert HF config → FT dims + hyperparams via the arch module.
    arch_module = _arch_module_for(hf_config)
    dims = dict(arch_module.hf_config_to_flextrain(hf_config))
    # Inject attn_dim / kv_dim if the arch's dims map omitted them — some
    # blocks (e.g. OLMoE QK-norm) expect them.
    if "attn_dim" not in dims:
        dims["attn_dim"] = int(dims["n_heads"]) * int(dims["head_dim"])
    if "kv_dim" not in dims:
        dims["kv_dim"] = int(dims["n_kv_heads"]) * int(dims["head_dim"])
    hyperparams = arch_module.hf_config_to_hyperparams(hf_config)

    # Sane dtype defaults.
    master_dtype = master_dtype or compute_dtype
    grad_dtype = grad_dtype or compute_dtype

    n_layers = int(dims["n_layers"])

    # 3. Build embed + head + backbone.
    embed = _build_embed(
        dims, compute=compute_dtype, master=master_dtype, grad=grad_dtype,
    )
    head = _build_head(
        {**dims, "head_chunk_size": head_chunk_size},
        hyperparams,
        compute=compute_dtype, master=master_dtype, grad=grad_dtype,
        norm_grad=norm_grad_dtype,
    )
    ctx = BuildContext(
        hf_config=hf_config, dims=dims, hyperparams=hyperparams,
        compute_dtype=compute_dtype, master_dtype=master_dtype,
        grad_dtype=grad_dtype, norm_grad_dtype=norm_grad_dtype,
        lora_targets=lora_targets, lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_adapter_compute_dtype=lora_adapter_compute_dtype,
        lora_adapter_master_dtype=lora_adapter_master_dtype,
        lora_adapter_grad_dtype=lora_adapter_grad_dtype,
        lora_adapter_opt_state_dtype=lora_adapter_opt_state_dtype,
    )
    backbone = [build_block(i, ctx) for i in range(n_layers)]

    # 4. Solve working set.
    opt_dtype_name = _dtype_name(getattr(optimizer, "state_spec", None))
    training_config = {
        "master_weight_dtype": _dtype_name_from_torch(master_dtype),
        "grad_dtype": _dtype_name_from_torch(grad_dtype),
        "opt_choice": type(optimizer).__name__,
        "opt_dtype": opt_dtype_name or "float32",
    }
    # If the caller provided a measured ``hw_cost``, hand its scalars to
    # the working-set solver so it doesn't run a redundant probe. If they
    # also pass ``mem_bw_gbps`` (from ``probe_hardware().mem_bw_gbps``),
    # we propagate that for the AI-bound chunk-size pick. Otherwise the
    # solver runs its own internal probe.
    ws_peak_tflops = hw_cost.peak_tflops if hw_cost is not None else None
    ws_pcie_bw_gbps = hw_cost.pcie_bw_gbps if hw_cost is not None else None
    working_set = determine_working_set_config(
        model_dims=dims,
        max_seq_len=max_seq_len,
        max_global_batch_tokens=max_global_batch_tokens,
        training_config=training_config,
        has_embed=True, has_head=True, num_local_layers=n_layers,
        max_gpu_mem_bytes=max_gpu_mem_bytes,
        max_host_mem_bytes=max_host_mem_bytes,
        leeway_gpu_mem_bytes=leeway_gpu_mem_bytes,
        leeway_host_mem_bytes=leeway_host_mem_bytes,
        peak_tflops=ws_peak_tflops,
        pcie_bw_gbps=ws_pcie_bw_gbps,
        mem_bw_gbps=mem_bw_gbps,
        fixed_seq_len=fixed_seq_len,
        verbose=verbose,
    )

    # 5. Build engine. ``hw_cost`` must come from the caller — either
    # a measured :func:`flextrain.core.hw_probe.probe_hardware` result
    # (the recommended path) or hand-set values for tests / unusual
    # hardware. We deliberately do NOT probe here: hardware benchmarking
    # is orthogonal to model construction and benefits from being
    # callable independently (cached across runs, swappable for tests,
    # etc.). The placeholder fallback only fires when the caller passes
    # nothing AND no GPU probe ran — which is fine for offline tests
    # but produces poor DP plans on real hardware (compute_times come
    # out ~10x slower than reality, letting level-3 "fit" everywhere
    # and producing the no-recompute symptom).
    if hw_cost is None:
        hw_cost = HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0)
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head, optimizer=optimizer,
        working_set=working_set,
        hw_cost=hw_cost,
        dims=dims, device=device,
        force_saved_act_level=force_saved_act_level,
    )

    # 6. Load + permute weights.
    if load_weights:
        am.load_hf(model_path, strict=strict)
        # Arch-specific post-load fixups (Q/K halved→pair, tied head, ...).
        post_load = getattr(arch_module, "post_load_permute", None)
        if post_load is not None:
            post_load(am, hf_config, dims, hyperparams)

    # 7. LoRA auto-init: ``A ~ N(0, lora_a_std)``, ``B = 0`` so the LoRA
    # delta starts at zero (model behaves identically to the base at
    # step 0 — required for clean transfer learning). Skip when we
    # didn't load weights (cold start) or LoRA isn't enabled.
    if lora_targets and load_weights:
        _init_lora_params(am, seed=lora_init_seed, a_std=lora_a_std)

    return am


def _init_lora_params(am: ActiveModel, *, seed: int, a_std: float) -> None:
    """Initialize all ``*_lora_a`` and ``*_lora_b`` host-resident
    parameters across the backbone, then refresh GPU residents so the
    next ``fwd_bwd`` sees the new values.

    Convention: ``A ~ N(0, a_std)``, ``B = 0``. Yields
    ``W_eff = W + (A @ B) * scale = W`` at step 0 (LoRA delta is zero
    until B picks up gradient). This matches HF PEFT's
    ``init_lora_weights="default"`` behavior."""
    import torch as _t
    g = _t.Generator(device="cpu").manual_seed(int(seed))
    n_layers = len(am.backbone)
    touched = 0
    for L in range(n_layers):
        host = am.buffers.host_params[L]
        for nm, t in host.items():
            if nm.endswith("_lora_a"):
                # Generate in fp32, cast to t.dtype to match adapter dtype.
                rand = _t.empty(t.shape, dtype=_t.float32).normal_(
                    mean=0.0, std=a_std, generator=g,
                )
                t.copy_(rand.to(t.dtype))
                touched += 1
            elif nm.endswith("_lora_b"):
                t.zero_()
                touched += 1
    am._refresh_gpu_residents()
    _t.cuda.synchronize()


def _arch_module_for(hf_config: Mapping):
    """Locate the Python module under ``flextrain/io/arch/`` whose
    registered ArchSpec handles this config. We use the module so we
    can call its ``hf_config_to_flextrain`` / ``..._hyperparams`` /
    ``post_load_permute`` (if present)."""
    import importlib
    archs = (
        getattr(hf_config, "architectures", None)
        or hf_config.get("architectures")
        or []
    )
    # Map the leading HF arch_id to a module name. This is a soft
    # convention: "LlamaForCausalLM" → flextrain.io.arch.llama. If your
    # arch module name doesn't match, register it explicitly via
    # _ARCH_MODULE_OVERRIDES.
    mod_name = _ARCH_MODULE_OVERRIDES.get(
        archs[0] if archs else "",
        _arch_module_name_default(archs[0] if archs else ""),
    )
    return importlib.import_module(f"flextrain.io.arch.{mod_name}")


def _arch_module_name_default(arch_id: str) -> str:
    s = arch_id.replace("ForCausalLM", "").replace("Model", "").lower()
    # E.g. "Qwen3MoeForCausalLM" -> "qwen3_moe", "Qwen3NextForCausalLM" -> "qwen3_next"
    out = []
    for i, c in enumerate(s):
        if c.isupper() and i > 0:
            out.append("_")
        out.append(c.lower())
    return "".join(out)


_ARCH_MODULE_OVERRIDES: dict[str, str] = {
    # Names that don't follow the simple arch-id → module-name mapping.
    "LlamaForCausalLM": "llama",
    "MistralForCausalLM": "mistral",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3MoeForCausalLM": "qwen3_moe",
    "Qwen3NextForCausalLM": "qwen3_next",
    "OlmoeForCausalLM": "olmoe",
    "Gemma2ForCausalLM": "gemma2",
    "Gemma3ForCausalLM": "gemma3",
}


def _dtype_name(state_spec) -> str | None:
    if state_spec is None:
        return None
    for t in getattr(state_spec, "tensors", ()):
        return _dtype_name_from_torch(t.dtype)
    return None


def _dtype_name_from_torch(dt: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(dt, str(dt).replace("torch.", ""))


__all__ = [
    "BuildContext",
    "BlockBuilder",
    "from_pretrained",
    "register_block_builder",
]
