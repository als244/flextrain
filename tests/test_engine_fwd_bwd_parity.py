"""Engine-level forward+backward parity vs HF transformers.

Drives the full ``flextrain.from_pretrained`` → ``am.fwd_bwd`` path,
NOT a manual block-by-block driver. Compares scalar loss + per-block
weight gradients against HF. This is the test the user requested as
"end-to-end via the engine": it validates everything wired up by
``from_pretrained`` — the block builder, ``post_load_permute``, multi-
prefix HF safetensor loading (for Gemma3-ConditionalGeneration), the
LM-head softcap (Gemma 2 only), tied LM-head mirroring, the embed
forward+backward, the head fwd+loss+bwd, and the new dual-residual
backward implementation inside ``Gemma{2,3}Block``.

Parameterized over (model_name, hf_arch_id). Skips models whose HF
weights aren't present locally. The smaller models (Gemma 3 1B,
Gemma 2 2B) fit comfortably; 4B/12B/9B are gated behind a memory
heuristic.
"""
from __future__ import annotations

import gc
import os
import sys
from typing import Dict, List

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.test_gemma3_block_parity import _compare, _diffstats
from tests.test_gemma3_full_fwd_bwd_parity import (
    _HF_PARAM_SUFFIX, _hf_fwd_bwd_capture, _map_hf_grad_to_ft_layout,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MODELS_ROOT = os.path.join(ROOT, "models")

# Engine path goes through the chunk-scheduler + working-set solver +
# DP save-tier picker. Numerical noise floor is the same shape as the
# manual-driver test but with slightly more drift from the engine's
# additional cast points. Keep same thresholds.
PARITY_COS_TOL = 0.98
PARITY_SIGN_TOL = 0.92
PARITY_REL_L2_TOL = 2e-1
LOSS_REL_TOL = 1e-2
TINY_GRAD_REF_SCALE = 3e-4
TINY_GRAD_COS_TOL = 0.5

# Per-model spec: (dir, hf arch_id, hf_grad_prefix, hf_safetensor_prefix,
# memory_needed_gb).
_SPECS = {
    "Gemma2-2B": {
        "dir": "Gemma-2-2B-Instruct",
        "arch_id": "Gemma2ForCausalLM",
        "hf_grad_prefix": "model.layers",
        "hf_safetensor_prefix": "model",
        "memory_gb": 10,
    },
    "Gemma2-9B": {
        "dir": "Gemma-2-9B-Instruct",
        "arch_id": "Gemma2ForCausalLM",
        "hf_grad_prefix": "model.layers",
        "hf_safetensor_prefix": "model",
        # 9B at bf16 needs ~18 GB params + ~18 GB grads on the HF side
        # alone, before flextrain's buffers and activations — closer
        # to 40 GB total. Skipped on 32 GB cards; runs on 48 GB+.
        "memory_gb": 38,
    },
    "Gemma3-1B": {
        "dir": "Gemma-3-1B-Instruct",
        "arch_id": "Gemma3ForCausalLM",
        "hf_grad_prefix": "model.layers",
        "hf_safetensor_prefix": "model",
        "memory_gb": 8,
    },
    "Gemma3-4B": {
        "dir": "Gemma-3-4B-Instruct",
        "arch_id": "Gemma3ForConditionalGeneration",
        "hf_grad_prefix": "model.language_model.layers",
        "hf_safetensor_prefix": "language_model.model",
        "memory_gb": 18,
    },
}


# Gemma 2 has the same weight-mapping conventions as Gemma 3 (transpose
# on linears, halved→pair permute on Q/K, +1 on RMSNorm γ). It does NOT
# use QK-norm, so we skip the q_norm/k_norm entries when present.
_GEMMA2_PARAM_SUFFIX = {
    k: v for k, v in _HF_PARAM_SUFFIX.items()
    if k not in ("w_q_norm", "w_k_norm")
}


def _maybe_skip(spec: Dict, model_name: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    md = os.path.join(MODELS_ROOT, spec["dir"])
    if not os.path.isdir(md):
        pytest.skip(f"{spec['dir']} not present under models/")
    needs = spec["memory_gb"]
    free_b, total_b = torch.cuda.mem_get_info()
    if free_b < needs * 2**30:
        pytest.skip(
            f"{model_name} parity needs ~{needs} GB free; "
            f"have {free_b / 2**30:.1f} of {total_b / 2**30:.1f} GB"
        )


def _ft_engine_fwd_bwd(
    *, model_dir: str, input_ids: torch.Tensor,
) -> Dict:
    """Run one ``am.fwd_bwd`` step via the engine path. Capture loss
    + per-block weight grads from the engine's host-side grad mirror.
    """
    from flextrain.api import from_pretrained
    from flextrain.bench.parity import _Seq
    from flextrain.optim import AdamW, AdamWHyperparams

    T = int(input_ids.shape[0])
    # Build SFT targets: targets[i] = tokens[i+1] for i in [0, T-2];
    # targets[T-1] = -100 (no label for the last position). Matches HF's
    # internal shift inside ``Gemma{2,3}ForCausalLM.forward(labels=)``.
    tokens = input_ids.clone().to(DEVICE)
    targets = torch.full_like(tokens, -100)
    targets[:-1] = tokens[1:]

    opt = AdamW(
        AdamWHyperparams(lr=1e-5, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0),
    )
    am = from_pretrained(
        model_dir, optimizer=opt,
        # Cap working set tightly so the test runs fast.
        max_seq_len=max(128, T),
        max_global_batch_tokens=max(512, T * 2),
        max_gpu_mem_bytes=int(24 * 2**30),
        max_host_mem_bytes=int(60 * 2**30),
    )

    seq = _Seq(tokens)
    seq.targets = targets

    active = int((targets != -100).sum().item())
    # HF default is ``reduction='mean'`` over active labels. To get the
    # same gradient magnitudes, pass ``loss_scale_factor = 1/active``.
    stats = am.fwd_bwd([seq], loss_scale_factor=1.0 / active, verbose=False)
    loss_value = stats.total_loss / active

    # Make sure all backward ops finish + grads land on host.
    torch.cuda.synchronize()

    n_layers = len(am.backbone)
    grads_per_layer: List[Dict[str, torch.Tensor]] = []
    for i in range(n_layers):
        host = am.buffers.host_grads[i]
        grads_per_layer.append({
            name: t.detach().to("cpu", copy=True)
            for name, t in host.items()
        })

    result = {
        "loss": float(loss_value),
        "grads_per_layer": grads_per_layer,
        "n_layers": n_layers,
    }
    # Tear down so the next parameterization gets a clean CUDA state
    # (pinned memory + the param/grad rings would otherwise persist
    # and trip the next load with cudaErrorInvalidValue).
    try:
        am.buffers.destroy()
    except Exception:
        pass
    del am
    gc.collect()
    torch.cuda.empty_cache()
    return result


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="engine parity requires CUDA",
)


@pytest.mark.parametrize(
    "model_name", ["Gemma2-2B", "Gemma3-1B", "Gemma3-4B", "Gemma2-9B"],
)
def test_engine_fwd_bwd_parity(model_name: str) -> None:
    spec = _SPECS[model_name]
    _maybe_skip(spec, model_name)
    model_dir = os.path.join(MODELS_ROOT, spec["dir"])

    from transformers import AutoConfig, AutoTokenizer

    hf_cfg = AutoConfig.from_pretrained(model_dir)
    text_cfg = hf_cfg.get_text_config()

    tok = AutoTokenizer.from_pretrained(model_dir)
    prompt = (
        "Programming is the art of constructing precise instructions for a "
        "computer. The smallest detail in a program can completely change "
        "the result."
    )
    input_ids = tok(prompt, return_tensors="pt").input_ids.squeeze(0)
    input_ids = input_ids[:32]  # keep tight for memory

    # === HF capture ===
    print(f"\n[{model_name}] HF fwd+bwd...", flush=True)
    hf = _hf_fwd_bwd_capture(model_dir, spec["arch_id"], input_ids)
    print(f"[{model_name}] HF loss={hf['loss']:.6f}", flush=True)

    # === Flextrain via engine ===
    print(f"[{model_name}] flextrain engine fwd+bwd...", flush=True)
    ft = _ft_engine_fwd_bwd(model_dir=model_dir, input_ids=input_ids)
    print(f"[{model_name}] FT loss={ft['loss']:.6f}", flush=True)

    rel_loss_err = abs(ft["loss"] - hf["loss"]) / max(abs(hf["loss"]), 1e-9)
    print(
        f"[{model_name}] loss: hf={hf['loss']:.6f} ft={ft['loss']:.6f} "
        f"rel_err={rel_loss_err:.3e}",
        flush=True,
    )
    assert rel_loss_err < LOSS_REL_TOL, (
        f"loss parity broken: hf={hf['loss']} ft={ft['loss']}"
    )

    # === Per-layer weight grads ===
    n_layers = ft["n_layers"]
    head_dim = int(text_cfg.head_dim)
    n_heads = int(text_cfg.num_attention_heads)
    n_kv_heads = int(text_cfg.num_key_value_heads)
    hf_grad_prefix = spec["hf_grad_prefix"]
    is_gemma2 = model_name.startswith("Gemma2")
    param_map = _GEMMA2_PARAM_SUFFIX if is_gemma2 else _HF_PARAM_SUFFIX

    fail: List[str] = []
    for i in range(n_layers):
        ft_grads = ft["grads_per_layer"][i]
        for ft_name, hf_suffix in param_map.items():
            grad_key = "g_" + ft_name[2:]
            if grad_key not in ft_grads:
                continue
            hf_key = f"{hf_grad_prefix}.{i}.{hf_suffix}"
            hf_grad = hf["grads"].get(hf_key)
            if hf_grad is None:
                fail.append(f"L{i}.{ft_name}: no HF grad at {hf_key!r}")
                continue
            hf_grad_in_ft = _map_hf_grad_to_ft_layout(
                ft_name, hf_grad.to(DEVICE),
                head_dim, n_heads, n_kv_heads,
            )
            ft_grad = ft_grads[grad_key].to(DEVICE)
            s = _diffstats(ft_grad, hf_grad_in_ft)
            if s["ref_scale"] < TINY_GRAD_REF_SCALE:
                if s["cos"] < TINY_GRAD_COS_TOL:
                    fail.append(
                        f"g[L{i:02d}].{ft_name}: tiny-grad "
                        f"cos={s['cos']:.4f} (ref_scale={s['ref_scale']:.2e})"
                    )
                continue
            try:
                _compare(
                    f"g[L{i:02d}].{ft_name}", ft_grad, hf_grad_in_ft,
                    cos_tol=PARITY_COS_TOL,
                    sign_tol=PARITY_SIGN_TOL,
                    rel_l2_tol=PARITY_REL_L2_TOL,
                )
            except AssertionError as e:
                fail.append(str(e))
            if i in (0, n_layers // 2, n_layers - 1) and ft_name in (
                "w_q", "w_pre_attn_norm", "w_2",
            ):
                print(
                    f"  L{i:02d}.{ft_name:18s} "
                    f"cos={s['cos']:.6f} sign={s['sign_match']:.4f} "
                    f"rel_l2={s['rel_l2']:.3e}",
                    flush=True,
                )

    if fail:
        msg = "\n".join(fail[:25])
        more = f"\n... and {len(fail) - 25} more" if len(fail) > 25 else ""
        raise AssertionError(
            f"{model_name} engine parity failures ({len(fail)}):\n{msg}{more}"
        )
