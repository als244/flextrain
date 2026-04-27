"""Loss-curve parity: FlexTrain vs naive PyTorch, across working-set
configs.

The heavy lifting lives in :mod:`flextrain.bench.parity` (reusable).
This test file just wires up eight working-set configs and asserts
every FlexTrain trajectory matches naive + matches every other FT
config in its windowed-mean.

To sweep different model sizes or working-set combos in the future,
import the harness directly::

    from flextrain.bench import (
        ModelShape, WorkingSetSpec, LossCurveParityConfig,
        run_loss_curve_parity,
    )
    cfg = LossCurveParityConfig(
        shape=ModelShape(d_model=1024, n_layers=12),
        working_sets=[...],
        n_steps=500,
        shard_path=".../fineweb_train_000001.bin",
    )
    result = run_loss_curve_parity(cfg)
    result.print_summary()
    result.assert_all_match(windowed_atol=0.10)

Runs on RTX 3090 under the ``flextrain`` conda env. ~2-3 min total.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench import (  # noqa: E402
    LossCurveParityConfig,
    ModelShape,
    WorkingSetSpec,
    run_loss_curve_parity,
)


# ---------------------------------------------------------------------------
# Eight working-set configs. Each stresses a different engine path.
# Every config uses the SAME gpu_act_buffer_size (large enough to hold
# the opt-state ring) so we exercise only one dimension at a time.
# ---------------------------------------------------------------------------


def _mib(n: int) -> int:
    return n * 1024 * 1024


def _est_act_buffer_bytes(shape: ModelShape) -> int:
    """Size the GPU activation buffer so it's bigger than both
    (a) our realistic forward-activation ring, and
    (b) the opt-state ring during step() (= n_layers × per-layer AdamW
        opt-state bytes).

    Returns a byte count rounded up to 64 MiB.
    """
    # Per-layer AdamW: 2 fp32 tensors per param.
    # Llama block params: 4× d_model² (QKV,O) + 3× d_model·expert_dim
    # + 2× d_model (norm scales).
    per_layer_params = (
        4 * shape.d_model * shape.d_model
        + 3 * shape.d_model * shape.expert_dim
        + 2 * shape.d_model
    )
    per_layer_opt_bytes = per_layer_params * 4 * 2  # 2× fp32
    # When N_opt = n_layers all layers' opt states fit in the ring.
    total_opt_bytes = shape.n_layers * per_layer_opt_bytes
    # 2× headroom + round up to 64 MiB.
    target = total_opt_bytes * 2
    return ((target + _mib(64) - 1) // _mib(64)) * _mib(64)


def _working_sets(shape: ModelShape) -> list[WorkingSetSpec]:
    act_mib = _est_act_buffer_bytes(shape)
    return [
        WorkingSetSpec(
            label="A. fast path (all on-device, 1 chunk/round)",
            n_gpu_layers=shape.n_layers,
            n_gpu_grads=shape.n_layers,
            n_gpu_opt_layers=shape.n_layers,
            gpu_act_buffer_size=act_mib,
            host_act_buffer_size=0,
            max_chunk_size=512,
            target_round_tokens=512,
            max_total_round_tokens=1024,
            max_training_chunks=4,
        ),
        WorkingSetSpec(
            label="B. multi-chunk (many chunks/round, on-device)",
            n_gpu_layers=shape.n_layers,
            n_gpu_grads=shape.n_layers,
            n_gpu_opt_layers=shape.n_layers,
            gpu_act_buffer_size=act_mib,
            host_act_buffer_size=0,
            max_chunk_size=80,
            target_round_tokens=512,
            max_total_round_tokens=1024,
            max_training_chunks=16,
        ),
        WorkingSetSpec(
            label="C. multi-round (2+ rounds/step, on-device)",
            n_gpu_layers=shape.n_layers,
            n_gpu_grads=shape.n_layers,
            n_gpu_opt_layers=shape.n_layers,
            gpu_act_buffer_size=act_mib,
            host_act_buffer_size=0,
            max_chunk_size=256,
            target_round_tokens=192,  # forces 2+ rounds per step
            max_total_round_tokens=384,
            max_training_chunks=8,
        ),
        WorkingSetSpec(
            label="D. host offload pressure (tight act ring)",
            n_gpu_layers=shape.n_layers,
            n_gpu_grads=shape.n_layers,
            n_gpu_opt_layers=max(1, shape.n_layers // 2),  # opt ring halves
            gpu_act_buffer_size=act_mib,
            host_act_buffer_size=max(_mib(512), act_mib * 2),
            max_chunk_size=128,
            target_round_tokens=512,
            max_total_round_tokens=512,
            max_training_chunks=8,
        ),
        WorkingSetSpec(
            label="E. weight ring rotation (N_P < n_layers)",
            n_gpu_layers=max(1, shape.n_layers // 2),
            n_gpu_grads=shape.n_layers,
            n_gpu_opt_layers=shape.n_layers,
            gpu_act_buffer_size=act_mib,
            host_act_buffer_size=0,
            max_chunk_size=256,
            target_round_tokens=512,
            max_total_round_tokens=512,
            max_training_chunks=4,
        ),
        WorkingSetSpec(
            label="F. grad ring rotation (N_G < n_layers)",
            n_gpu_layers=shape.n_layers,
            n_gpu_grads=max(1, shape.n_layers // 2),
            n_gpu_opt_layers=shape.n_layers,
            gpu_act_buffer_size=act_mib,
            host_act_buffer_size=0,
            max_chunk_size=256,
            target_round_tokens=512,
            max_total_round_tokens=512,
            max_training_chunks=4,
        ),
        WorkingSetSpec(
            label="G. opt-state ring rotation (N_opt < n_layers)",
            n_gpu_layers=shape.n_layers,
            n_gpu_grads=shape.n_layers,
            n_gpu_opt_layers=max(1, shape.n_layers // 2),
            gpu_act_buffer_size=act_mib,
            host_act_buffer_size=0,
            max_chunk_size=256,
            target_round_tokens=512,
            max_total_round_tokens=512,
            max_training_chunks=4,
        ),
        WorkingSetSpec(
            label="H. sequence spans chunks (KV refresh)",
            n_gpu_layers=shape.n_layers,
            n_gpu_grads=shape.n_layers,
            n_gpu_opt_layers=shape.n_layers,
            gpu_act_buffer_size=act_mib,
            host_act_buffer_size=0,
            max_chunk_size=96,  # forces 200-token seqs into 3 chunks
            target_round_tokens=1024,
            max_total_round_tokens=1024,
            max_training_chunks=16,
        ),
    ]


def _run_one_setting(
    *, label: str, shape: ModelShape, n_steps: int, lr: float,
    atol: float,
) -> dict:
    cfg = LossCurveParityConfig(
        shape=shape,
        working_sets=_working_sets(shape),
        n_steps=n_steps,
        target_tokens_per_step=384,
        min_seq_len=64,
        max_seq_len=256,
        lr=lr,
        init_seed=4242,
        shard_path=os.path.join(
            ROOT, "orig", "fineweb", "fineweb_train_000001.bin"
        ),
        device="cuda:0",
    )
    print(f"\n\n####################################################")
    print(f"#  SETTING: {label}")
    print(f"#  n_steps={n_steps}, lr={lr}, d_model={shape.d_model}, "
          f"n_layers={shape.n_layers}")
    print(f"####################################################")
    result = run_loss_curve_parity(cfg)
    result.print_summary()

    finals = {"naive": sum(result.naive_curve[-3:]) / 3}
    for label_, curve in result.ft_curves.items():
        finals[label_] = sum(curve[-3:]) / 3

    result.assert_all_match(window=10, windowed_atol=atol)
    return finals


def test_loss_curve_parity_all_configs() -> None:
    """Stress-matrix parity sweep.

    Three increasing-stress settings on three orthogonal axes (more
    steps / bigger model / higher LR). If any FT config diverges from
    naive beyond the windowed tolerance at any setting, that's a real
    scheduling / update bug. bf16 noise scales with step-count × LR;
    tolerances are scaled accordingly.
    """
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("requires CUDA")

    base_shape = ModelShape()
    big_shape = ModelShape(
        d_model=768, n_heads=12, head_dim=64, n_layers=8,
        expert_dim=1536,
    )

    settings = []
    # Setting 1: baseline (what we validated at lr=5e-4, 100 steps).
    settings.append(
        ("S1: baseline (100 steps, lr=5e-4, d=512, L=6)",
         _run_one_setting(
             label="S1: baseline (100 steps, lr=5e-4, d=512, L=6)",
             shape=base_shape, n_steps=100, lr=5e-4, atol=0.10,
         )),
    )
    # Setting 2: bigger model (d=768, L=8) at same LR/steps.
    settings.append(
        ("S2: bigger model (100 steps, lr=5e-4, d=768, L=8)",
         _run_one_setting(
             label="S2: bigger model (100 steps, lr=5e-4, d=768, L=8)",
             shape=big_shape, n_steps=100, lr=5e-4, atol=0.10,
         )),
    )
    # Setting 3: 2x steps + 2x LR on the baseline.
    settings.append(
        ("S3: stress (200 steps, lr=1e-3, d=512, L=6)",
         _run_one_setting(
             label="S3: stress (200 steps, lr=1e-3, d=512, L=6)",
             shape=base_shape, n_steps=200, lr=1e-3, atol=0.20,
         )),
    )

    # --- Final cross-setting summary ---
    print("\n\n" + "=" * 104)
    print("  CROSS-SETTING FINAL LOSS (avg of last 3 steps)")
    print("=" * 104)
    labels = list(settings[0][1].keys())
    header = f"  {'config':<52}" + "".join(
        f"{s[0][:24]:>26}" for s in settings
    )
    print(header)
    print("  " + "-" * 102)
    for lbl in labels:
        row = f"  {lbl[:50]:<52}"
        for _, finals in settings:
            row += f"{finals[lbl]:>26.4f}"
        print(row)
    print("=" * 104)
    print("  (every FT config should land within a few percent of naive "
          "at every setting)")


def _run_all() -> None:
    test_loss_curve_parity_all_configs()
    print("\n  OK")


if __name__ == "__main__":
    _run_all()
