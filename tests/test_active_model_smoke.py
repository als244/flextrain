"""Smoke tests for :class:`flextrain.engine.active_model.ActiveModel`.

Tiny Llama-style config, random-init, short training sequences. We
verify:

* ``fwd_bwd`` completes without error (forward + head + backward +
  embed-backward loop).
* Returned loss is finite and in the expected range for a uniform
  random init (~log(vocab_size)).
* Gradients land in host / GPU grad buffers.
* ``step()`` actually mutates host master weights.
* After ``step()``, a subsequent ``fwd_bwd`` still runs (ring state
  survived).

This is NOT a numerical parity test — that's 3K. Here we just prove
the engine's scheduling loop runs end-to-end on the 3090 with real
compute underneath.

Cross-test isolation
--------------------
``tests/run_all.py`` calls
:func:`flextrain.engine.unregister_all_process_pinned_memory` between
modules so each test starts with a clean CUDA host-pin registry. Test
modules that create a :class:`BufferManager` don't need to clean up
their own pinned memory — the runner does it for them.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


class FakeSeq:
    """Duck-type match for the scheduler's Sequence expectation."""

    def __init__(self, seq_id: int, n: int, vocab_size: int = 256) -> None:
        torch.manual_seed(seq_id * 100 + 1)
        self.seq_id = seq_id
        self.tokens = torch.randint(0, vocab_size, (n,), dtype=torch.int64)
        self.targets = torch.roll(self.tokens, -1)
        self.per_token_loss = torch.zeros(n, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.tokens)


def _build_minimal_model():
    """Build a tiny Llama backbone + embed + head."""
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    d_model = 64
    n_heads = 4
    n_kv_heads = 2
    head_dim = 16
    expert_dim = 128
    vocab_size = 256
    n_layers = 2

    cfg = LlamaBlockConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, expert_dim=expert_dim,
        rms_norm_eps=1e-5, rope_base=10000.0, is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [LlamaBlock(layer_id=i, cfg=cfg) for i in range(n_layers)]
    embed = TokenEmbedLayer(
        TokenEmbedConfig(
            vocab_size=vocab_size, d_model=d_model,
            compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        )
    )
    head = LMHead(
        LMHeadConfig(
            d_model=d_model, vocab_size=vocab_size,
            rms_norm_eps=1e-5, head_chunk_size=128,
            compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
            norm_grad_dtype=torch.float32,
        )
    )
    dims = dict(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, expert_dim=expert_dim, vocab_size=vocab_size,
    )
    working_set = WorkingSetConfig(
        target_round_tokens=256, max_chunk_size=128, max_training_chunks=4,
        max_total_round_tokens=512, target_num_rounds=1,
        n_gpu_layers=n_layers, n_gpu_grads=n_layers, n_gpu_opt_layers=n_layers,
        gpu_act_buffer_size=4 * 1024 * 1024,
        host_act_buffer_size=0,
        available_gpu_memory_bytes=1 << 30,
        available_host_memory_bytes=1 << 32,
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=256, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(
        peak_tflops=10.0, pcie_bw_gbps=10.0, practical_efficiency_factor=1.0,
    )
    opt = AdamW(AdamWHyperparams())

    return dict(
        embed=embed, backbone=backbone, head=head, dims=dims,
        working_set=working_set, hw_cost=hw_cost, opt=opt,
    )


def _build_active_model(parts):
    from flextrain.engine.active_model import ActiveModel
    return ActiveModel(
        embed=parts["embed"], backbone=parts["backbone"], head=parts["head"],
        optimizer=parts["opt"],
        working_set=parts["working_set"], hw_cost=parts["hw_cost"],
        dims=parts["dims"], device=DEVICE,
    )


def _random_init_host_weights(bm) -> None:
    torch.manual_seed(7777)
    for lp in bm.host_params:
        for name, t in lp.items():
            if "norm" in name:
                t.fill_(1.0)
            else:
                t.normal_(mean=0.0, std=0.02)
    for t in bm.host_embed_params.values():
        t.normal_(mean=0.0, std=0.02)
    for name, t in bm.host_head_params.items():
        if "norm" in name:
            t.fill_(1.0)
        else:
            t.normal_(mean=0.0, std=0.02)


def _sync_host_to_device(am, parts):
    for slot_idx in range(parts["working_set"].n_gpu_layers):
        am.buffers.fetch_layer_params(slot_idx, slot_idx, non_blocking=False)
    for name, dev_t in am.buffers.gpu_embed_params.items():
        dev_t.copy_(am.buffers.host_embed_params[name])
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


def test_fwd_bwd_smoke() -> None:
    """fwd_bwd runs end-to-end and produces sensible loss + non-zero grads."""
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("active_model smoke test requires CUDA")
    parts = _build_minimal_model()
    am = _build_active_model(parts)
    _random_init_host_weights(am.buffers)
    _sync_host_to_device(am, parts)

    seqs = [FakeSeq(0, 96), FakeSeq(1, 96)]
    stats = am.fwd_bwd(
        seqs,
        loss_scale_factor=1.0 / sum(len(s) for s in seqs),
        verbose=True,
    )
    avg = stats.total_loss / stats.total_tokens
    print(
        f"  fwd_bwd: rounds={stats.rounds} tokens={stats.total_tokens} "
        f"total_loss={stats.total_loss:.4f} avg={avg:.4f}"
    )
    assert stats.total_tokens == 192
    assert 0.1 < avg < 20.0, f"avg loss out of range: {avg}"

    assert am.buffers.gpu_head_grads["g_head_proj"].abs().sum().item() > 0.0
    assert (
        am.buffers.gpu_embed_grads["g_tok_embeddings"].abs().sum().item() > 0.0
    )
    gpu_l0_grad = am.buffers.gpu_grad_slot(0, parts["backbone"][0].param_spec)
    assert gpu_l0_grad["g_q"].abs().sum().item() > 0.0

    am.buffers.destroy()


def test_fwd_bwd_step_fwd_bwd() -> None:
    """fwd_bwd -> step -> fwd_bwd succeeds, and step actually mutates
    the host master weights."""
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("active_model smoke test requires CUDA")
    parts = _build_minimal_model()
    am = _build_active_model(parts)
    _random_init_host_weights(am.buffers)
    _sync_host_to_device(am, parts)

    seqs = [FakeSeq(0, 96), FakeSeq(1, 96)]
    stats1 = am.fwd_bwd(seqs, loss_scale_factor=1.0 / 192)

    before = am.buffers.host_params[0]["w_q"].float().clone()
    ret = am.step()
    assert ret == 0, f"step returned {ret}"
    after = am.buffers.host_params[0]["w_q"].float()
    delta = (after - before).norm().item()
    print(
        f"  after step: host w_q delta norm = {delta:.4e} (should be > 0)"
    )
    assert delta > 0.0

    stats2 = am.fwd_bwd(seqs, loss_scale_factor=1.0 / 192)
    a1 = stats1.total_loss / stats1.total_tokens
    a2 = stats2.total_loss / stats2.total_tokens
    print(f"  avg_loss pre-step={a1:.4f}  post-step={a2:.4f}")
    assert 0.1 < a2 < 20.0

    am.buffers.destroy()


def _run_all() -> None:
    tests = [
        ("test_fwd_bwd_smoke", test_fwd_bwd_smoke),
        ("test_fwd_bwd_step_fwd_bwd", test_fwd_bwd_step_fwd_bwd),
    ]
    for name, fn in tests:
        print(f"  - {name}")
        fn()
        # Per-test cleanup: drop any backend registrations still alive
        # after the test's BufferManager was GC'd but before the next
        # test allocates.
        import gc
        gc.collect()
        try:
            from flextrain.engine import unregister_all_process_pinned_memory
            unregister_all_process_pinned_memory()
        except Exception:
            pass
    print(f"  OK ({len(tests)} tests)")


if __name__ == "__main__":
    _run_all()
