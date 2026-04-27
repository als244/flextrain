"""Gemma 3 HF <-> FlexTrain mapping.

Gemma 3 = Gemma 2 + per-head QK-norm. The post-load hook still does the
+1 shift on RMSNorm γ values (Gemma 3 inherits the convention). The
QK-norm weights are saved as ``γ - 1`` too.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _gemma3_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Shift every Gemma RMSNorm weight by +1 (Gemma's γ convention)."""
    for L in range(num_layers):
        for n in (
            "w_pre_attn_norm", "w_post_attn_norm",
            "w_pre_ffn_norm", "w_post_ffn_norm",
            "w_q_norm", "w_k_norm",
        ):
            t = dest.get((f"layer_{L}", n))
            if t is not None:
                t.add_(1.0)
    final = dest.get(("head", "w_final_norm"))
    if final is not None:
        final.add_(1.0)


GEMMA3_ARCH = ArchSpec(
    hf_arch_ids=("Gemma3ForCausalLM", "Gemma3ForConditionalGeneration"),
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
    ),
    layer=(
        WeightMapEntry(
            flextrain_name="w_pre_attn_norm",
            hf_name="model.layers.{i}.input_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_post_attn_norm",
            hf_name="model.layers.{i}.post_attention_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_q_norm",
            hf_name="model.layers.{i}.self_attn.q_norm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_k_norm",
            hf_name="model.layers.{i}.self_attn.k_norm.weight",
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
            flextrain_name="w_pre_ffn_norm",
            hf_name="model.layers.{i}.pre_feedforward_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_post_ffn_norm",
            hf_name="model.layers.{i}.post_feedforward_layernorm.weight",
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
    post_load_hook=_gemma3_post_load_hook,
)

register_arch(GEMMA3_ARCH)


def hf_config_to_flextrain(hf_config: Any) -> dict:
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    text_cfg = get("text_config")  # Gemma3 puts text-arch fields under text_config
    if text_cfg is not None:
        cfg = text_cfg
    else:
        cfg = hf_config
    cget = (
        (lambda k, default=None: getattr(cfg, k, default))
        if not isinstance(cfg, dict)
        else cfg.get
    )
    n_heads = cget("num_attention_heads")
    hidden = cget("hidden_size")
    head_dim = cget("head_dim") or (hidden // n_heads)
    return {
        "vocab_size": cget("vocab_size"),
        "n_layers": cget("num_hidden_layers"),
        "d_model": hidden,
        "n_heads": n_heads,
        "n_kv_heads": cget("num_key_value_heads") or n_heads,
        "head_dim": head_dim,
        "expert_dim": cget("intermediate_size"),
        "num_shared_experts": 1,
        "num_routed_experts": 0,
        "top_k": 0,
        "is_causal": True,
        "datatypes": {
            "embed": "bfloat16", "head_proj": "bfloat16",
            "attn_proj": "bfloat16", "expert_proj": "bfloat16",
            "router": "bfloat16", "norm": "bfloat16", "residual": "bfloat16",
        },
    }


def hf_config_to_hyperparams(hf_config: Any) -> dict:
    text_cfg = getattr(hf_config, "text_config", None) or hf_config
    get = (
        (lambda k, default=None: getattr(text_cfg, k, default))
        if not isinstance(text_cfg, dict)
        else text_cfg.get
    )
    rope_params = get("rope_parameters") or {}
    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-6),
        "rope_theta": (
            rope_params.get("full_attention", {}).get("rope_theta")
            or get("rope_theta", 1_000_000.0)
        ),
        "rope_local_base": (
            rope_params.get("sliding_attention", {}).get("rope_theta")
            or get("rope_local_base_freq", 10_000.0)
        ),
        "attn_logit_softcap": get("attn_logit_softcapping"),
        "final_logit_softcap": get("final_logit_softcapping"),
        "query_pre_attn_scalar": get("query_pre_attn_scalar"),
        "sliding_window": get("sliding_window"),
        "layer_types": get("layer_types"),
    }
