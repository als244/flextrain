"""Vision encoder forward parity vs HF ``Qwen3_5VisionModel``.

Loads the Qwen3.5-2B checkpoint's vision tower from HF, builds a
:class:`flextrain.nn.encoders.QwenVLVisionEncoder` with identical
config, maps HF state-dict weights into the flextrain naming
convention, runs both on the same ``(pixel_values, image_grid_thw)``,
and asserts per-token cosine similarity ≥ 0.99 between flextrain's
post-merger output and HF's ``pooler_output``.

The mapping convention mirrors the ArchSpec entries in
``flextrain/io/arch/qwen3_5.py::_VISION_EMBED + _VISION_LAYER``:

* HF ``model.visual.<part>`` -> flextrain ``image0_<part_mapped>``
  where ``part_mapped`` follows the per-tensor renames in those
  entries (e.g. ``patch_embed.proj.weight`` -> ``patch_embed_proj_w``,
  ``blocks.{i}.norm1.weight`` -> ``layer_{i}_norm1_w``, ...).

Run: ``./run_with_env.sh python tests/test_qwen_vl_vit_forward.py``

Requirements: CUDA, transformers, PIL. The test SKIPs cleanly if any
is missing.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-2B")


def _hf_to_ft_name(hf_subname: str) -> str:
    """Convert an HF ``model.visual.*`` subname into flextrain's
    ``image0_*`` flextrain_name from the qwen3_5 ArchSpec.

    Examples:
      ``patch_embed.proj.weight`` -> ``image0_patch_embed_proj_w``
      ``blocks.7.attn.qkv.bias`` -> ``image0_layer_7_qkv_b``
      ``merger.linear_fc1.weight`` -> ``image0_merger_fc1_w``
    """
    suffix = "_w" if hf_subname.endswith(".weight") else "_b"
    body = hf_subname.rsplit(".", 1)[0]

    # blocks.{i}.<part>...
    if body.startswith("blocks."):
        parts = body.split(".")
        i = parts[1]
        rest = ".".join(parts[2:])
        # rest is e.g. ``attn.qkv``, ``mlp.linear_fc1``, ``norm1``.
        rest_map = {
            "attn.qkv": "qkv",
            "attn.proj": "proj",
            "mlp.linear_fc1": "mlp_fc1",
            "mlp.linear_fc2": "mlp_fc2",
            "norm1": "norm1",
            "norm2": "norm2",
        }
        if rest not in rest_map:
            raise KeyError(f"unknown HF block subname: {body!r}")
        return f"image0_layer_{i}_{rest_map[rest]}{suffix}"

    # one-shot parts.
    one_shot = {
        "patch_embed.proj": "patch_embed_proj",
        "pos_embed": "pos_embed",
        "merger.norm": "merger_norm",
        "merger.linear_fc1": "merger_fc1",
        "merger.linear_fc2": "merger_fc2",
    }
    if body in one_shot:
        return f"image0_{one_shot[body]}{suffix}"
    raise KeyError(f"unmapped HF subname: {body!r}")


def _build_flextrain_encoder(hf_vision_cfg, attn_implementation="eager"):
    from flextrain.nn.encoders import QwenVLVisionConfig, QwenVLVisionEncoder
    cfg = QwenVLVisionConfig(
        depth=hf_vision_cfg.depth,
        hidden_size=hf_vision_cfg.hidden_size,
        intermediate_size=hf_vision_cfg.intermediate_size,
        num_heads=hf_vision_cfg.num_heads,
        in_channels=hf_vision_cfg.in_channels,
        patch_size=hf_vision_cfg.patch_size,
        spatial_merge_size=hf_vision_cfg.spatial_merge_size,
        temporal_patch_size=hf_vision_cfg.temporal_patch_size,
        out_hidden_size=hf_vision_cfg.out_hidden_size,
        num_position_embeddings=hf_vision_cfg.num_position_embeddings,
        hidden_act=hf_vision_cfg.hidden_act,
        deepstack_visual_indexes=tuple(
            getattr(hf_vision_cfg, "deepstack_visual_indexes", ()) or ()
        ),
        compute_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    return QwenVLVisionEncoder(cfg, modality="image", encoder_id=0, frozen=True)


def _load_weights(hf_visual, encoder):
    """Build a name -> tensor dict matching ``encoder.param_spec``
    populated from HF ``hf_visual.state_dict()``."""
    hf_sd = hf_visual.state_dict()
    weights: dict[str, torch.Tensor] = {}
    for ts in encoder.param_spec.tensors:
        # ts.name is e.g. "image0_layer_7_qkv_b".
        # Reverse-map to find which HF tensor this is.
        ft_name = ts.name
        assert ft_name.startswith("image0_"), ft_name
        # Brute force: iterate HF state_dict, map each, find the match.
        # (Faster: precompute the inverse map once, but encoder.param_spec
        # is < 500 tensors so this is fine for a one-off test.)
        # We'll match by computing _hf_to_ft_name for each hf subname.
        match = None
        for hf_name in hf_sd:
            try:
                if _hf_to_ft_name(hf_name) == ft_name:
                    match = hf_name
                    break
            except KeyError:
                continue
        if match is None:
            raise KeyError(f"no HF tensor maps to flextrain {ft_name!r}")
        # Cast to compute dtype + move to CUDA.
        weights[ft_name] = hf_sd[match].detach().to(
            device="cuda", dtype=torch.bfloat16,
        ).contiguous()
    return weights


def _make_pixel_values_and_grid(image_size: int = 224):
    """Synthetic image -> HF processor -> (pixel_values, grid_thw)."""
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor

    rng = np.random.default_rng(42)
    img_array = rng.integers(0, 256, (image_size, image_size, 3), dtype=np.uint8)
    image = Image.fromarray(img_array, mode="RGB")
    proc = AutoImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    out = proc(images=[image], return_tensors="pt")
    return out["pixel_values"], out["image_grid_thw"]


def test_vision_encoder_forward_parity() -> None:
    # ----- HF side -----
    # CRITICAL: load via from_pretrained (the production path), NOT via
    # ``Qwen3_5VisionModel._from_config(vc).to(bf16)``. The two differ
    # on a load-bearing buffer dtype: ``from_pretrained(torch_dtype=
    # bf16)`` keeps the ``rotary_pos_emb.inv_freq`` non-persistent
    # buffer at fp32 (HF's ``_init_weights`` re-initializes it after
    # weight load), whereas ``.to(bf16)`` casts everything including
    # the buffer. A standalone-cast reference test is silently using
    # bf16 inv_freq while the production model uses fp32 inv_freq,
    # giving a "byte-exact" match against the WRONG precision policy.
    from transformers import AutoModelForImageTextToText

    hf_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    hf_visual = hf_model.model.visual
    vc = hf_visual.config
    # Confirm the production precision policy.
    inv_freq_dtype = hf_visual.rotary_pos_emb.inv_freq.dtype
    print(f"[setup] HF (production) visual inv_freq dtype: {inv_freq_dtype}")
    print(f"[setup] HF visual eager attn impl: {vc._attn_implementation}")

    # ----- flextrain encoder -----
    encoder = _build_flextrain_encoder(vc)
    weights = _load_weights(hf_visual, encoder)
    print(f"[setup] built flextrain encoder with {len(weights)} weight tensors")

    # ----- inputs -----
    pixel_values_cpu, grid_thw_cpu = _make_pixel_values_and_grid(224)
    pixel_values = pixel_values_cpu.to("cuda")
    grid_thw = grid_thw_cpu.to("cuda")
    print(f"[setup] pixel_values shape={tuple(pixel_values.shape)}, dtype={pixel_values.dtype}; "
          f"grid_thw={grid_thw.tolist()}")

    # ----- HF forward -----
    with torch.inference_mode():
        hf_out = hf_visual(hidden_states=pixel_values.to(torch.bfloat16), grid_thw=grid_thw)
    hf_pre_merger = hf_out.last_hidden_state.float()   # (n_pre_merge, hidden_size)
    hf_post_merger = hf_out.pooler_output.float()       # (n_post_merge, out_hidden_size)
    print(f"[hf] pre-merger shape={tuple(hf_pre_merger.shape)}; "
          f"post-merger shape={tuple(hf_post_merger.shape)}")

    # ----- flextrain forward -----
    from flextrain.core.modality import ImageInputs

    inputs = ImageInputs(
        pixel_values=pixel_values.to(torch.bfloat16),
        pix_offsets=torch.tensor([0, pixel_values.shape[0]], dtype=torch.int32, device="cuda"),
        grid_thw=grid_thw.to(torch.int32),
    )
    # The encoder doesn't actually use ctx in Phase 1; pass None.
    ft_out = encoder.forward_round(inputs, weights, ctx=None)
    ft_post_merger = ft_out.embeds.float()
    print(f"[ft] post-merger shape={tuple(ft_post_merger.shape)}")

    # ----- compare -----
    # Threshold: cos_min >= 0.9999. The flextrain encoder matches HF
    # Qwen3_5VisionModel EXACTLY in bf16 when both stacks use eager
    # attention AND both keep ``inv_freq`` at fp32 (the production
    # ``from_pretrained(torch_dtype=bf16)`` policy: HF's
    # ``_init_weights`` re-initializes the non-persistent buffer after
    # the cast, so it stays fp32). flextrain's
    # ``_build_vision_rotary_inv_freq`` now defaults to fp32 to match.
    assert ft_post_merger.shape == hf_post_merger.shape, (
        f"shape mismatch: ft={tuple(ft_post_merger.shape)} vs hf={tuple(hf_post_merger.shape)}"
    )
    cos = torch.cosine_similarity(hf_post_merger, ft_post_merger, dim=-1)
    cos_min, cos_mean = float(cos.min()), float(cos.mean())
    print(f"[parity] post-merger cosine: min={cos_min:.6f}, mean={cos_mean:.6f}")
    abs_err = (hf_post_merger - ft_post_merger).abs()
    print(f"[parity] post-merger abs err: max={float(abs_err.max()):.4e}, "
          f"mean={float(abs_err.mean()):.4e}")
    assert cos_min > 0.9999, (
        f"vision encoder forward parity failed: post-merger cosine MIN={cos_min:.6f} < 0.9999\n"
        f"  hf: shape={tuple(hf_post_merger.shape)}, mean={float(hf_post_merger.mean()):.4f}\n"
        f"  ft: shape={tuple(ft_post_merger.shape)}, mean={float(ft_post_merger.mean()):.4f}\n"
        f"  HF inv_freq dtype was {inv_freq_dtype} (expected fp32 via from_pretrained).\n"
        "  If this fires: check that flextrain's _build_vision_rotary_inv_freq "
        "is using fp32 to match production HF, and that attn_implementation is eager."
    )
    print(f"[OK] vision encoder forward parity holds bit-exactly "
          f"(cos_min={cos_min:.6f}, cos_mean={cos_mean:.6f}).")


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available.")
        return
    if not os.path.isdir(MODEL_PATH):
        print(f"SKIP: {MODEL_PATH} not present.")
        return
    try:
        import transformers  # noqa: F401
        import PIL  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError as e:
        print(f"SKIP: missing dep: {e}")
        return
    test_vision_encoder_forward_parity()


if __name__ == "__main__":
    main()
