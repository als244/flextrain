"""Llama family (Llama 2, 3, 3.1, 3.2, 3.3) HF <-> FlexTrain mapping.

Mapping logic
-------------
All Llama variants share one HF architecture id (``LlamaForCausalLM``) and
one tensor-name layout. YARN long-context (Llama 3.1+) changes RoPE
parameters but NOT tensor names; we don't touch RoPE constants here, those
come through the config adapter.

FlexTrain weight names (for a backbone layer ``i``) are:
    w_attn_norm, w_q, w_k, w_v, w_o, w_ffn_norm, w_1, w_2, w_3
where the FFN is SwiGLU (w_1=gate, w_3=up, w_2=down).

HF counterpart tensors (under ``model.layers.{i}.`` prefix):
    input_layernorm.weight       <-> w_attn_norm
    self_attn.q_proj.weight      <-> w_q     (transpose)
    self_attn.k_proj.weight      <-> w_k     (transpose)
    self_attn.v_proj.weight      <-> w_v     (transpose)
    self_attn.o_proj.weight      <-> w_o     (transpose)
    post_attention_layernorm.weight <-> w_ffn_norm
    mlp.gate_proj.weight         <-> w_1     (transpose)
    mlp.up_proj.weight           <-> w_3     (transpose)
    mlp.down_proj.weight         <-> w_2     (transpose)

And at top level:
    model.embed_tokens.weight    <-> embed.w_tok_embeddings
    model.norm.weight            <-> head.w_final_norm
    lm_head.weight               <-> head.w_head_proj (transpose)

See docs/internal/PLAN.md "HF integration: config + weights, not compute" for the
rationale (we own the compute path; HF owns config + weights + tokenizer).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _llama_pre_export_hook(am, dst, num_layers: int) -> None:
    """Drop ``lm_head.weight`` from the export when the source had
    ``tie_word_embeddings: True`` (Llama-3.2-1B / 3B). FT mirrors the
    embedding into ``w_head_proj`` at load time and the unmirroring
    isn't needed: HF re-mirrors at the receiving end."""
    from flextrain.export._pre_export_helpers import read_tie_word_embeddings
    if read_tie_word_embeddings(am):
        dst.pop("lm_head.weight", None)


LLAMA_ARCH = ArchSpec(
    hf_arch_ids=("LlamaForCausalLM",),
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
            optional=True,
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
    pre_export_hook=_llama_pre_export_hook,
)

register_arch(LLAMA_ARCH)


def hf_config_to_flextrain(hf_config: Any) -> dict:
    """Translate a Llama ``transformers.LlamaConfig`` into a FlexTrain
    dims dict (same schema as ``orig/model_dims.json``).

    Notes
    -----
    * ``expert_dim`` in FlexTrain's dims dict is the FFN intermediate size
      for dense models (matching ``orig/model_dims.json["llama3_8B"]``
      which sets it to 14336).
    * RoPE parameters (``rope_theta``, ``rope_scaling``, YARN for Llama
      3.1+) belong in ``hyperparams`` alongside ``rms_norm_eps`` and the
      sliding-window fields; we return those separately so the layer can
      pick them up without reaching into the HF config.
    """
    # Accept either a HF PretrainedConfig or a raw dict.
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
        # Dense: zero routed experts. Keeps the dims dict shape compatible
        # with orig/model_dims.json so downstream code doesn't need to
        # branch on dense vs. MoE at this layer.
        "num_shared_experts": 1,
        "num_routed_experts": 0,
        "top_k": 0,
        "is_causal": True,
        # dtypes default to bf16 (matches orig/model_dims.json); user config
        # overrides are applied at a higher layer.
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
    """Extract runtime hyperparams (norm epsilon, RoPE constants, window
    sizes) the layer needs but that don't belong in ``model_dims``."""
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )

    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-5),
        "rope_theta": get("rope_theta", 10000.0),
        "rope_scaling": get("rope_scaling"),  # None, or dict for YARN
        # Llama has no sliding window; keep keys for protocol parity with
        # Mistral / GPT-OSS.
        "window_size_left": -1,
        "window_size_right": 0,
    }


# ---------------------------------------------------------------------------
# Block builder + post-load permutation hook used by ``flextrain.from_pretrained``.
# ---------------------------------------------------------------------------


def _llama_block_builder(layer_idx: int, ctx) -> object:
    import torch
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig

    dims = ctx.dims
    hp = ctx.hyperparams
    block_cfg = LlamaBlockConfig(
        d_model=int(dims["d_model"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=int(dims["n_kv_heads"]),
        head_dim=int(dims["head_dim"]),
        expert_dim=int(dims["expert_dim"]),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-5)),
        rope_base=float(hp.get("rope_theta", 500_000.0)),
        rope_scaling=hp.get("rope_scaling"),
        is_causal=True,
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )
    base = LlamaBlock(layer_id=layer_idx, cfg=block_cfg)
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
    """Llama-family post-load fixups: Q/K halved→pair RoPE permutation
    on base weights AND any matching LoRA-B column dim. Also handle
    ``tie_word_embeddings`` by mirroring embed weights into the head
    when the loaded ``w_head_proj`` is empty."""
    import torch

    # Q/K permutation. FlexTrain's RoPE kernel uses pair-interleave
    # layout (even=cos, odd=sin), but HF stores Q/K in halved layout
    # (first D/2 rotated by cos, second D/2 by sin). Permute the OUT
    # dim of w_q / w_k AND the matching dim of w_q_lora_b / w_k_lora_b.
    head_dim = int(dims["head_dim"])
    n_heads = int(dims["n_heads"])
    n_kv = int(dims["n_kv_heads"])
    n_layers = int(dims["n_layers"])
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    def _halved_to_pair_perm(dim: int, head_dim: int) -> torch.Tensor:
        half = head_dim // 2
        out = torch.empty(dim, dtype=torch.int64)
        for h in range(dim // head_dim):
            base = h * head_dim
            for i in range(half):
                out[base + 2 * i] = base + i
                out[base + 2 * i + 1] = base + half + i
        return out

    q_perm = _halved_to_pair_perm(attn_dim, head_dim)
    k_perm = _halved_to_pair_perm(kv_dim, head_dim)
    for i in range(n_layers):
        host = am.buffers.host_params[i]
        for name in ("w_q", "w_k"):
            if name in host:
                w = host[name]
                # w shape (d_model, attn_or_kv_dim) — permute the OUT dim.
                host[name].copy_(w[:, q_perm if name == "w_q" else k_perm])
        # LoRA B mirrors permutation along its column dim if present.
        for name, perm in (("w_q_lora_b", q_perm), ("w_k_lora_b", k_perm)):
            if name in host and host[name].dim() == 2:
                host[name].copy_(host[name][:, perm])

    # Tied embeddings (Llama-3.2-1B / 3B): if config says so AND head
    # was loaded as zeros, mirror embed.t() into head.
    tied = (
        getattr(hf_config, "tie_word_embeddings", None)
        if not isinstance(hf_config, dict)
        else hf_config.get("tie_word_embeddings")
    )
    if tied:
        head_w = am.buffers.host_head_params.get("w_head_proj")
        embed_w = am.buffers.host_embed_params.get("w_tok_embeddings")
        if (
            head_w is not None and embed_w is not None
            and float(head_w.abs().sum().item()) == 0.0
        ):
            head_w.copy_(embed_w.t())

    # Refresh GPU residents to pick up the permuted host weights.
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        if name in am.buffers.host_head_params:
            dev_t.copy_(am.buffers.host_head_params[name])
    import torch as _t
    _t.cuda.synchronize()


# Register the block builder so flextrain.api.from_pretrained finds it.
def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(("LlamaForCausalLM",), _llama_block_builder)


_register_builder()
