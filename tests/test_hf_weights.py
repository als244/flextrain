"""HF safetensors load / export round-trip on a fake 2-layer model.

We generate a tiny checkpoint in safetensors, load it through
:func:`load_hf_safetensors` into a dest mapping, then re-export and
diff against the source. This verifies:

* Shard iteration + single-file fallback
* Name templating with ``{i}`` for per-layer entries
* Transpose transform on both paths
* Strict-mode error on missing tensors
* Leftover reporting

Does NOT require a GPU or any real HF checkpoint.
"""

from __future__ import annotations

import os
import tempfile

import torch

from flextrain.io.arch.llama import LLAMA_ARCH
from flextrain.io.hf_weights import (
    Transform,
    export_hf_safetensors,
    load_hf_safetensors,
    select_arch,
)


def _make_fake_llama_checkpoint(out_dir: str, num_layers: int = 2) -> dict:
    """Create a single-shard safetensors file with all expected Llama
    tensor names. Returns the in-memory src dict so callers can cross-check.
    """
    from safetensors.torch import save_file

    d_model = 32
    n_heads = 4
    n_kv = 2
    head_dim = d_model // n_heads
    expert_dim = 64
    vocab = 256

    src: dict[str, torch.Tensor] = {}

    # Top-level tensors
    src["model.embed_tokens.weight"] = torch.randn(vocab, d_model)
    src["model.norm.weight"] = torch.randn(d_model)
    src["lm_head.weight"] = torch.randn(vocab, d_model)  # HF convention: (vocab, d_model)

    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        src[prefix + "input_layernorm.weight"] = torch.randn(d_model)
        src[prefix + "post_attention_layernorm.weight"] = torch.randn(d_model)
        # Attention. HF stores (out, in).
        src[prefix + "self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, d_model)
        src[prefix + "self_attn.k_proj.weight"] = torch.randn(n_kv * head_dim, d_model)
        src[prefix + "self_attn.v_proj.weight"] = torch.randn(n_kv * head_dim, d_model)
        src[prefix + "self_attn.o_proj.weight"] = torch.randn(d_model, n_heads * head_dim)
        # MLP SwiGLU
        src[prefix + "mlp.gate_proj.weight"] = torch.randn(expert_dim, d_model)
        src[prefix + "mlp.up_proj.weight"] = torch.randn(expert_dim, d_model)
        src[prefix + "mlp.down_proj.weight"] = torch.randn(d_model, expert_dim)

    path = os.path.join(out_dir, "model.safetensors")
    save_file(src, path)
    return src


def _allocate_dest(num_layers: int, src: dict) -> dict:
    """Build the (scope, name) -> host tensor mapping the engine would
    normally own. Sizes follow the transposed shapes FlexTrain expects."""
    dest: dict[tuple[str, str], torch.Tensor] = {}

    dest[("embed", "w_tok_embeddings")] = torch.zeros_like(
        src["model.embed_tokens.weight"]
    )
    dest[("head", "w_final_norm")] = torch.zeros_like(src["model.norm.weight"])
    # Head proj is transposed on load -> FlexTrain shape (d_model, vocab)
    vocab, d_model = src["lm_head.weight"].shape
    dest[("head", "w_head_proj")] = torch.zeros(d_model, vocab)

    for i in range(num_layers):
        prefix = f"model.layers.{i}."
        scope = f"layer_{i}"

        dest[(scope, "w_attn_norm")] = torch.zeros_like(
            src[prefix + "input_layernorm.weight"]
        )
        dest[(scope, "w_ffn_norm")] = torch.zeros_like(
            src[prefix + "post_attention_layernorm.weight"]
        )

        # Attention: HF (out, in) -> FlexTrain (in, out), so we need
        # dest tensors with transposed shapes.
        for hf_name, fx_name in [
            ("self_attn.q_proj.weight", "w_q"),
            ("self_attn.k_proj.weight", "w_k"),
            ("self_attn.v_proj.weight", "w_v"),
            ("self_attn.o_proj.weight", "w_o"),
            ("mlp.gate_proj.weight", "w_1"),
            ("mlp.down_proj.weight", "w_2"),
            ("mlp.up_proj.weight", "w_3"),
        ]:
            hf_t = src[prefix + hf_name]
            # Transposed shape
            dest[(scope, fx_name)] = torch.zeros(hf_t.shape[1], hf_t.shape[0])

    return dest


def test_load_round_trip_llama() -> None:
    num_layers = 2
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_fake_llama_checkpoint(tmp, num_layers=num_layers)
        dest = _allocate_dest(num_layers, src)

        leftover = load_hf_safetensors(
            hf_path=tmp,
            arch=LLAMA_ARCH,
            dest=dest,
            num_layers=num_layers,
            strict=True,
        )
        assert leftover == [], f"unexpected leftover HF tensors: {leftover}"

        # Verify a few round-trip values:

        # embed token table copied verbatim
        assert torch.allclose(
            dest[("embed", "w_tok_embeddings")], src["model.embed_tokens.weight"]
        )
        # final norm copied verbatim
        assert torch.allclose(dest[("head", "w_final_norm")], src["model.norm.weight"])
        # head proj transposed on load
        assert torch.allclose(
            dest[("head", "w_head_proj")], src["lm_head.weight"].T.contiguous()
        )
        # A layer matmul weight
        assert torch.allclose(
            dest[("layer_0", "w_q")],
            src["model.layers.0.self_attn.q_proj.weight"].T.contiguous(),
        )


def test_export_round_trip_llama() -> None:
    num_layers = 2
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_fake_llama_checkpoint(tmp, num_layers=num_layers)
        dest = _allocate_dest(num_layers, src)

        load_hf_safetensors(
            hf_path=tmp,
            arch=LLAMA_ARCH,
            dest=dest,
            num_layers=num_layers,
            strict=True,
        )

        out_dir = os.path.join(tmp, "exported")
        export_hf_safetensors(
            out_dir=out_dir,
            arch=LLAMA_ARCH,
            src=dest,
            num_layers=num_layers,
        )

        # Reload the exported file as HF weights and verify identity.
        from safetensors import safe_open

        with safe_open(
            os.path.join(out_dir, "model.safetensors"),
            framework="pt",
            device="cpu",
        ) as f:
            keys = set(f.keys())
            assert keys == set(src.keys())
            for k in src:
                assert torch.allclose(f.get_tensor(k), src[k]), (
                    f"mismatch on {k}"
                )


def test_select_arch() -> None:
    arch = select_arch({"architectures": ["LlamaForCausalLM"]})
    assert arch is LLAMA_ARCH


def test_strict_missing_raises() -> None:
    num_layers = 2
    with tempfile.TemporaryDirectory() as tmp:
        src = _make_fake_llama_checkpoint(tmp, num_layers=num_layers)
        # Delete one expected HF tensor before saving; we need to rebuild
        # the file with the tensor removed.
        from safetensors.torch import save_file

        partial = {
            k: v
            for k, v in src.items()
            if k != "model.layers.0.self_attn.q_proj.weight"
        }
        save_file(partial, os.path.join(tmp, "model.safetensors"))
        dest = _allocate_dest(num_layers, src)

        try:
            load_hf_safetensors(
                hf_path=tmp,
                arch=LLAMA_ARCH,
                dest=dest,
                num_layers=num_layers,
                strict=True,
            )
        except KeyError as e:
            assert "q_proj" in str(e)
        else:  # pragma: no cover
            raise AssertionError("strict load should raise on missing tensor")


def _run_all() -> None:
    tests = [
        test_select_arch,
        test_load_round_trip_llama,
        test_export_round_trip_llama,
        test_strict_missing_raises,
    ]
    for fn in tests:
        print(f"... {fn.__name__}", flush=True)
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    print(f"\nAll {len(tests)} HF weight I/O tests passed.")


if __name__ == "__main__":
    _run_all()
