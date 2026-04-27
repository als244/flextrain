"""Llama-1B with forced save_level=0 — test if recompute bug also
affects 1B."""

import os
import sys
import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import ModelShape, _Seq, _flextrain_step
from tests.test_llama32_1b_parity import (  # noqa: E402
    _permute_qk_for_pair_interleave, _pull_step_batches,
    _build_flextrain_engine_for_1b,
)

DEVICE = "cuda:0"

def main(force_level):
    hf_path = os.path.join(ROOT, "models", "Llama-3.2-1B")
    shape = ModelShape(
        d_model=2048, n_layers=16, n_heads=32, n_kv_heads=8,
        head_dim=64, expert_dim=8192, vocab_size=128256,
        rms_norm_eps=1e-5, rope_base=500000.0,
    )
    batches = _pull_step_batches(
        hf_path, n_steps=3, target_tokens_per_step=2048,
    )
    am = _build_flextrain_engine_for_1b(shape, 5e-5, DEVICE)
    am.force_saved_act_level = force_level
    am.load_hf(hf_path, strict=False)
    for i in range(shape.n_layers):
        for name in ("w_q", "w_k"):
            w = am.buffers.host_params[i][name]
            am.buffers.host_params[i][name].copy_(
                _permute_qk_for_pair_interleave(w, shape.head_dim)
            )
    hh = am.buffers.host_head_params["w_head_proj"]
    if hh.abs().sum().item() == 0:
        hh.copy_(am.buffers.host_embed_params["w_tok_embeddings"].T)
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    for step_i, batch in enumerate(batches):
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        loss = _flextrain_step(am, seqs)
        print(f"  [save={force_level}] step {step_i}: loss = {loss:.4f}")

    am.buffers.destroy()
    del am
    import gc; gc.collect()
    try:
        from flextrain.engine import unregister_all_process_pinned_memory
        unregister_all_process_pinned_memory()
    except Exception:
        pass
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\n=== Llama-1B default save (baseline) ===")
    main(None)
    print("\n=== Llama-1B force save=0 ===")
    main(0)
    print("\n=== Llama-1B force save=2 ===")
    main(2)
