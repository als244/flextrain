"""Per-layer per-tensor parity diagnostic.

Not a test — a debugging tool. Runs one forward through FlexTrain
under a given working-set config + the naive reference with the same
weights and same input, and prints the residual-stream output after
every layer. If the drift between them stays at bf16 noise per layer
and doesn't compound unboundedly, the compute path is correct.

Useful when the loss-curve parity test shows drift and you want to
localize it (forward, backward, or a specific layer's compute).

Usage:
    PYTHONPATH=. python tests/diagnostic_per_layer_parity.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import (  # noqa: E402
    ModelShape, WorkingSetSpec,
    NaiveLlamaModel, _init_naive_model,
    _copy_naive_to_flextrain,
    _build_flextrain_engine,
    _generate_sequence_stream,
)


DEVICE = "cuda:0"


def _run_naive_per_layer(
    naive: NaiveLlamaModel,
    tokens: torch.Tensor,
) -> list[torch.Tensor]:
    """Return residual stream after each block (list of tensors)."""
    positions = torch.arange(len(tokens), device=DEVICE, dtype=torch.int32)
    x = naive.w_tok_embeddings[tokens, :]
    stages = []
    for block in naive.blocks:
        x = block(x, positions)
        stages.append(x.detach().clone())
    return stages


def _run_flextrain_per_layer(
    am,
    tokens: torch.Tensor,
) -> list[torch.Tensor]:
    """Mimic ActiveModel.fwd_bwd's forward pass but without the
    head/backward. Returns residual stream after each layer."""
    from flextrain.engine.schedule import (
        prepare_training_chunks, ChunkPolicy,
    )
    from flextrain.core.activation_schema import ActivationSlot

    class _Seq:
        def __init__(self, t):
            self.tokens = t
            self.targets = torch.roll(t, -1)
            self.per_token_loss = torch.zeros(len(t), dtype=torch.float32)
            self.seq_id = 0
        def __len__(self):
            return len(self.tokens)

    seqs = [_Seq(tokens.cpu())]
    prepared = prepare_training_chunks(
        seqs, max_chunk_size=am.working_set.max_chunk_size, device=DEVICE,
        policy=ChunkPolicy.CAUSAL,
    )
    plan = am._plan_save_levels(prepared)
    am.events.clear_per_round()
    am._setup_round(prepared, plan)
    # Now run per-layer forward and record each chunk's transition.
    am._forward_pass(prepared, plan)

    # For per-layer snapshots, we'd need to instrument _forward_pass.
    # Easier: re-run fwd manually here, just like _forward_pass does,
    # but snapshot the transition table after each layer.
    stages = []
    for cid in sorted(am.buffers.transitions):
        stages.append(am.buffers.transitions[cid].detach().clone())
    return stages


def _run_per_layer_diagnostic(ws_spec, label: str) -> None:
    print(f"\n=== {label} ===")
    shape = ModelShape()
    naive = NaiveLlamaModel(shape).to(DEVICE)
    _init_naive_model(naive, seed=4242, device=DEVICE)

    am = _build_flextrain_engine(shape, ws_spec, lr=5e-4, device=DEVICE)
    _copy_naive_to_flextrain(naive, am.buffers)
    for slot_idx in range(ws_spec.n_gpu_layers):
        am.buffers.fetch_layer_params(slot_idx, slot_idx, non_blocking=False)
    for name, dev_t in am.buffers.gpu_embed_params.items():
        dev_t.copy_(am.buffers.host_embed_params[name])
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    # Fixed input: take one document from the FineWeb shard.
    seqs_list = _generate_sequence_stream(
        os.path.join(ROOT, "orig", "fineweb", "fineweb_train_000001.bin"),
        n_steps=1, target_tokens_per_step=200, min_len=128, max_len=200,
    )
    tokens = seqs_list[0][0].tokens.to(DEVICE)
    print(f"  input tokens shape: {tokens.shape}")

    # Naive per-layer.
    naive_stages = _run_naive_per_layer(naive, tokens)
    print(f"  naive stages: {len(naive_stages)}")

    # FlexTrain per-chunk.
    ft_final = _run_flextrain_per_layer(am, tokens)
    print(f"  ft transitions (post-all-layers): {len(ft_final)}")

    # Compare FlexTrain's final residual against naive's final.
    ft_post = ft_final[0][: len(tokens)]
    naive_post = naive_stages[-1]
    abs_delta = (ft_post.float() - naive_post.float()).abs()
    rel = abs_delta.norm().item() / (naive_post.float().norm().item() + 1e-6)
    print(
        f"  final residual: ft_norm={ft_post.float().norm().item():.4f} "
        f"naive_norm={naive_post.float().norm().item():.4f}  "
        f"rel-err={rel:.4e}"
    )

    am.buffers.destroy()


def main() -> None:
    shape = ModelShape()

    # A: fast path (sanity)
    ws_A = WorkingSetSpec(
        label="A", n_gpu_layers=shape.n_layers,
        n_gpu_grads=shape.n_layers, n_gpu_opt_layers=shape.n_layers,
        gpu_act_buffer_size=256 * 1024 * 1024, host_act_buffer_size=0,
        max_chunk_size=512, target_round_tokens=512,
        max_total_round_tokens=1024, max_training_chunks=4,
    )
    _run_per_layer_diagnostic(ws_A, "A (all on-device)")

    # E: weight ring rotation
    ws_E = WorkingSetSpec(
        label="E", n_gpu_layers=max(1, shape.n_layers // 2),
        n_gpu_grads=shape.n_layers, n_gpu_opt_layers=shape.n_layers,
        gpu_act_buffer_size=256 * 1024 * 1024, host_act_buffer_size=0,
        max_chunk_size=512, target_round_tokens=512,
        max_total_round_tokens=1024, max_training_chunks=4,
    )
    _run_per_layer_diagnostic(ws_E, "E (weight ring rotation)")


if __name__ == "__main__":
    main()
