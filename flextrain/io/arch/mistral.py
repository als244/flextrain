"""Mistral arch spec + builder.

Tensor naming and weight layout match Llama exactly; the only
differences from Llama are at the block level (sliding-window
attention vs full-context). So the ArchSpec mirrors Llama's, and the
post-load fixups are reused.
"""
from __future__ import annotations

from typing import Any

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


MISTRAL_ARCH = ArchSpec(
    hf_arch_ids=("MistralForCausalLM",),
    embed=(
        WeightMapEntry(
            flextrain_name="w_tok_embeddings",
            hf_name="model.embed_tokens.weight",
            transform=Transform.NONE,
        ),
    ),
    head=(
        WeightMapEntry(
            flextrain_name="w_final_norm",
            hf_name="model.norm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_head_proj",
            hf_name="lm_head.weight",
            transform=Transform.TRANSPOSE,
        ),
    ),
    layer=(
        WeightMapEntry(
            flextrain_name="w_attn_norm",
            hf_name="model.layers.{i}.input_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_q",
            hf_name="model.layers.{i}.self_attn.q_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_k",
            hf_name="model.layers.{i}.self_attn.k_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_v",
            hf_name="model.layers.{i}.self_attn.v_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_o",
            hf_name="model.layers.{i}.self_attn.o_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_ffn_norm",
            hf_name="model.layers.{i}.post_attention_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_1",
            hf_name="model.layers.{i}.mlp.gate_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_2",
            hf_name="model.layers.{i}.mlp.down_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_3",
            hf_name="model.layers.{i}.mlp.up_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
    ),
)
register_arch(MISTRAL_ARCH)


ARCH_NAME = "mistral"


_REQUIRED_DIMS = (
    "vocab_size", "n_layers", "d_model", "n_heads", "head_dim", "expert_dim",
)
_DEFAULT_DATATYPES = {
    "embed": "bfloat16", "head_proj": "bfloat16", "attn_proj": "bfloat16",
    "expert_proj": "bfloat16", "router": "bfloat16", "norm": "bfloat16",
    "residual": "bfloat16",
}


def expand_dims(dims) -> dict:
    """Mistral dims schema = Llama dims schema. Tensor layout is
    identical; only sliding-window attention differs (lives in
    hyperparams)."""
    out = dict(dims)
    missing = [k for k in _REQUIRED_DIMS if k not in out]
    if missing:
        raise KeyError(
            f"mistral dims missing required keys: {missing}. "
            f"Got keys: {sorted(out)}"
        )
    out.setdefault("n_kv_heads", out["n_heads"])
    out.setdefault("num_shared_experts", 1)
    out.setdefault("num_routed_experts", 0)
    out.setdefault("top_k", 0)
    out.setdefault("is_causal", True)
    out.setdefault("datatypes", dict(_DEFAULT_DATATYPES))
    out["attn_dim"] = int(out["n_heads"]) * int(out["head_dim"])
    out["kv_dim"] = int(out["n_kv_heads"]) * int(out["head_dim"])
    return out


def default_hyperparams() -> dict:
    """Mistral defaults: eps=1e-5, rope=10k. ``window_size_left=None``
    means full-context attention (Mistral-7B-v0.3 and later); pass an
    integer to opt into sliding-window."""
    return {
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "window_size_left": None,
        "window_size_right": 0,
    }


def hf_config_to_flextrain(hf_config: Any) -> dict:
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    return {
        "vocab_size": get("vocab_size"),
        "n_layers": get("num_hidden_layers"),
        "d_model": get("hidden_size"),
        "n_heads": get("num_attention_heads"),
        "n_kv_heads": get("num_key_value_heads") or get("num_attention_heads"),
        "head_dim": get("head_dim")
        or (get("hidden_size") // get("num_attention_heads")),
        "expert_dim": get("intermediate_size"),
        "num_shared_experts": 1,
        "num_routed_experts": 0,
        "top_k": 0,
        "is_causal": True,
        "datatypes": {
            "embed": "bfloat16",
            "head_proj": "bfloat16",
            "attn_proj": "bfloat16",
            "expert_proj": "bfloat16",
            "router": "bfloat16",
            "norm": "bfloat16",
            "residual": "bfloat16",
        },
    }


def hf_config_to_hyperparams(hf_config: Any) -> dict:
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    # Mistral defaults: rms eps 1e-5, rope_base 10000 (older) / 1e6 (Mistral-Nemo+).
    sw = get("sliding_window")
    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-5),
        "rope_theta": get("rope_theta", 10000.0),
        "rope_scaling": get("rope_scaling"),
        # ``sliding_window = null`` means full-context attention
        # (Mistral-7B-v0.3 and later). ``None`` propagates so the
        # builder can pick LlamaBlock instead of MistralBlock.
        "window_size_left": (int(sw) if sw is not None else None),
        "window_size_right": 0,
    }


def _mistral_block_builder(layer_idx: int, ctx) -> object:
    """Mistral with ``sliding_window != null`` uses MistralBlock (SWA);
    when ``sliding_window`` is null (Mistral-7B-v0.3 and later) we use
    LlamaBlock since the layer is full-context Llama-style."""
    dims = ctx.dims
    hp = ctx.hyperparams
    sw = hp.get("window_size_left")

    if sw is None:
        # Full-context: build a Llama-style block.
        from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
        block_cfg = LlamaBlockConfig(
            d_model=int(dims["d_model"]),
            n_heads=int(dims["n_heads"]),
            n_kv_heads=int(dims["n_kv_heads"]),
            head_dim=int(dims["head_dim"]),
            expert_dim=int(dims["expert_dim"]),
            rms_norm_eps=float(hp.get("rms_norm_eps", 1e-5)),
            rope_base=float(hp.get("rope_theta", 10000.0)),
            rope_scaling=hp.get("rope_scaling"),
            is_causal=True,
            compute_dtype=ctx.compute_dtype,
            master_dtype=ctx.master_dtype,
            grad_dtype=ctx.grad_dtype,
            norm_grad_dtype=ctx.norm_grad_dtype,
        )
        base = LlamaBlock(layer_id=layer_idx, cfg=block_cfg)
    else:
        from flextrain.nn.layers.mistral import MistralBlock, MistralBlockConfig
        block_cfg = MistralBlockConfig(
            d_model=int(dims["d_model"]),
            n_heads=int(dims["n_heads"]),
            n_kv_heads=int(dims["n_kv_heads"]),
            head_dim=int(dims["head_dim"]),
            expert_dim=int(dims["expert_dim"]),
            window_size_left=int(sw),
            rms_norm_eps=float(hp.get("rms_norm_eps", 1e-5)),
            rope_base=float(hp.get("rope_theta", 10000.0)),
            rope_scaling=hp.get("rope_scaling"),
            is_causal=True,
            compute_dtype=ctx.compute_dtype,
            master_dtype=ctx.master_dtype,
            grad_dtype=ctx.grad_dtype,
            norm_grad_dtype=ctx.norm_grad_dtype,
        )
        base = MistralBlock(layer_id=layer_idx, cfg=block_cfg)

    if not ctx.lora_targets:
        return base
    from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
    return LoRAWrapperLayer(
        base, lora_targets=ctx.lora_targets,
        rank=ctx.lora_rank, alpha=ctx.lora_alpha,
        dims=dict(dims, attn_dim=int(dims["n_heads"]) * int(dims["head_dim"]),
                  kv_dim=int(dims["n_kv_heads"]) * int(dims["head_dim"])),
        adapter_compute_dtype=ctx.lora_adapter_compute_dtype,
        adapter_master_dtype=ctx.lora_adapter_master_dtype,
        adapter_grad_dtype=ctx.lora_adapter_grad_dtype,
        adapter_opt_state_dtype=ctx.lora_adapter_opt_state_dtype,
    )


def post_load_permute(am, hf_config, dims, hyperparams):
    """Same Q/K halved→pair perm as Llama. Reuse Llama's hook."""
    from flextrain.io.arch.llama import post_load_permute as _llama_perm
    _llama_perm(am, hf_config, dims, hyperparams)


def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(("MistralForCausalLM",), _mistral_block_builder)


BLOCK_BUILDER = _mistral_block_builder


_register_builder()
