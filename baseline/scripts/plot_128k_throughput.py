#!/usr/bin/env python3
"""Plot 128K-context training-throughput figure for the paper.

4 panels (one per model) x 7 bars per panel (one per baseline). Y-axis is
tokens/second at sequence length 128K. The "FlexTrain @ 40G" bar is the
GPU-memory-constrained variant of our system (rendered with diagonal
hatching to mark it as the constrained twin of "FlexTrain"); both
FlexTrain bars are coloured the same to read as "ours".

Edit `DATA` below with measurements pulled from
`baseline/runs/<sweep>/throughput.csv`. Use `None` for any (model,
baseline) combination that OOMed — those slots render as a hatched
empty bar with an "OOM" annotation so the panel structure stays
intact.

Usage:
    # default output: baseline/runs/figures/throughput_128k.pdf
    python baseline/scripts/plot_128k_throughput.py

    # Custom output (PDF for paper, PNG for previews):
    python baseline/scripts/plot_128k_throughput.py --output paper/figures/fig_throughput_128k.pdf

    # Open interactively:
    python baseline/scripts/plot_128k_throughput.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Layout: model order across panels, baseline order along the X axis.
# ---------------------------------------------------------------------------

# Top-left, top-right, bottom-left, bottom-right of the 2x2 grid.
MODELS: list[str] = [
    "Llama3-8B",
    "Qwen3-30B-A3B",
    "Qwen3.6-27B",
    "Qwen3.6-35B-A3B",
]

# Baseline columns in plot order. FlexTrain entries are kept last so the
# "ours" cluster sits at the right edge of every panel.
BASELINES: list[str] = [
    "TorchTitan",
    "MegaTrain",
    "ALST",
    "TRL_DeepSpeed",
    "TRL_FSDP",
    "FlexTrain @ 40G",
    "FlexTrain",
]

# Color per baseline. Standard matplotlib tab10 palette for the prior
# work; tab:red for both FlexTrain entries (the @ 40G one is hatched
# below to distinguish it from the unconstrained variant).
COLORS: dict[str, str] = {
    "TorchTitan":      "#1f77b4",  # tab:blue
    "MegaTrain":       "#2ca02c",  # tab:green
    "ALST":            "#ff7f0e",  # tab:orange
    "TRL_DeepSpeed":   "#9467bd",  # tab:purple
    "TRL_FSDP":        "#8c564b",  # tab:brown
    "FlexTrain @ 40G": "#d62728",  # tab:red, will be hatched
    "FlexTrain":       "#d62728",  # tab:red, solid
}

# Bars rendered with hatching (in addition to their fill color). Used to
# distinguish constrained variants from the unconstrained version.
HATCHED: set[str] = {"FlexTrain @ 40G"}

OOM_FACE = "#eaeaea"
OOM_EDGE = "#999999"
OOM_HATCH = "////"

# ---------------------------------------------------------------------------
# Data: tokens/second at 128K seq length. Edit me.
#   - Use None for OOMs / unsupported (model, baseline) combos.
#   - Numbers are tokens/sec, NOT tokens/sec/GPU (single-GPU baseline).
# ---------------------------------------------------------------------------

DATA: dict[str, dict[str, float | None]] = {
    "Llama3-8B": {
        "TorchTitan":      None,
        "MegaTrain":       None,
        "ALST":            None,
        "TRL_DeepSpeed":   None,
        "TRL_FSDP":        None,
        "FlexTrain @ 40G": None,
        "FlexTrain":       None,
    },
    "Qwen3-30B-A3B": {
        "TorchTitan":      None,
        "MegaTrain":       None,
        "ALST":            None,
        "TRL_DeepSpeed":   None,
        "TRL_FSDP":        None,
        "FlexTrain @ 40G": None,
        "FlexTrain":       None,
    },
    "Qwen3.6-27B": {
        "TorchTitan":      None,
        "MegaTrain":       None,
        "ALST":            None,
        "TRL_DeepSpeed":   None,
        "TRL_FSDP":        None,
        "FlexTrain @ 40G": None,
        "FlexTrain":       None,
    },
    "Qwen3.6-35B-A3B": {
        "TorchTitan":      None,
        "MegaTrain":       None,
        "ALST":            None,
        "TRL_DeepSpeed":   None,
        "TRL_FSDP":        None,
        "FlexTrain @ 40G": None,
        "FlexTrain":       None,
    },
}

# ---------------------------------------------------------------------------
# Figure rendering.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("baseline/runs/figures/throughput_128k.pdf"),
        help=(
            "Output figure path. Format inferred from extension "
            "(.pdf for paper, .png for previews)."
        ),
    )
    parser.add_argument(
        "--show", action="store_true", help="Display interactively after saving.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Raster DPI when saving PNG.",
    )
    return parser.parse_args()


def _validate_data() -> None:
    """Fail loud if DATA is missing keys we'll dereference during render."""
    for model in MODELS:
        if model not in DATA:
            raise SystemExit(f"DATA missing model key: {model!r}")
        for b in BASELINES:
            if b not in DATA[model]:
                raise SystemExit(f"DATA[{model!r}] missing baseline key: {b!r}")


def _bar_panel(ax: plt.Axes, model: str, values: dict[str, float | None]) -> None:
    """Render one panel: bar chart of tokens/sec across BASELINES for ``model``."""
    x = np.arange(len(BASELINES))
    heights = [values[b] for b in BASELINES]

    # Per-panel y-scale: use the max present bar (or 1.0 fallback) to size
    # the OOM placeholder bars proportionally.
    visible_max = max((v for v in heights if v is not None), default=1.0)
    if visible_max <= 0:
        visible_max = 1.0
    oom_height = visible_max * 0.05  # 5% of panel max — visible but small.

    for xi, h, b in zip(x, heights, BASELINES):
        if h is None:
            ax.bar(
                xi, oom_height,
                color=OOM_FACE, edgecolor=OOM_EDGE,
                hatch=OOM_HATCH, linewidth=0.5,
            )
            ax.text(
                xi, oom_height + visible_max * 0.01, "OOM",
                ha="center", va="bottom", fontsize=8, color="#666",
            )
            continue
        hatch = "//" if b in HATCHED else None
        ax.bar(
            xi, h,
            color=COLORS[b], edgecolor="black",
            linewidth=0.5, hatch=hatch,
        )
        # Numeric value above each bar — paper readers want the number.
        ax.text(
            xi, h + visible_max * 0.01, f"{h:,.0f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_title(model, fontsize=12)
    ax.set_ylabel("tokens / sec")
    ax.set_xticks(x)
    ax.set_xticklabels(BASELINES, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, visible_max * 1.20)  # headroom for value labels.
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)


def main() -> int:
    args = parse_args()
    _validate_data()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, model in zip(axes.flat, MODELS):
        _bar_panel(ax, model, DATA[model])

    fig.suptitle(
        "Single-GPU training throughput at 128K context length",
        fontsize=14, y=1.02,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
    print(f"wrote {args.output}", flush=True)

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
