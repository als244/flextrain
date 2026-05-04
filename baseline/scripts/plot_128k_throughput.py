#!/usr/bin/env python3
"""Plot 128K-context training-throughput figure for the paper.

1x4 grid (one panel per model) x 7 bars per panel (one per baseline).
Sized for the top-of-page-1 NeurIPS figure that spans the full text
width. Y-axis is tokens/second at sequence length 128K. The
"FlexTrain @ 40G" bar is the GPU-memory-constrained variant of our
system (rendered with diagonal hatching to mark it as the constrained
twin of "FlexTrain"); both FlexTrain bars are coloured the same so
"ours" reads as one cluster at the right edge of every panel.

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

# Plausible-looking placeholder numbers so ``--demo`` produces a
# preview that exercises every visual feature (colored bars, hatched
# FlexTrain @ 40G, OOM placeholders, value labels, per-panel y-scale).
# These are NOT real measurements; replace ``DATA`` above with the
# actual sweep results before publishing the figure.
DEMO_DATA: dict[str, dict[str, float | None]] = {
    "Llama3-8B": {
        "TorchTitan":      8500,
        "MegaTrain":       7200,
        "ALST":            9100,
        "TRL_DeepSpeed":   6800,
        "TRL_FSDP":        7500,
        "FlexTrain @ 40G": 12400,
        "FlexTrain":       14200,
    },
    "Qwen3-30B-A3B": {
        "TorchTitan":      None,
        "MegaTrain":       3800,
        "ALST":            4100,
        "TRL_DeepSpeed":   None,
        "TRL_FSDP":        3950,
        "FlexTrain @ 40G": 6200,
        "FlexTrain":       7800,
    },
    "Qwen3.6-27B": {
        "TorchTitan":      4200,
        "MegaTrain":       3600,
        "ALST":            None,
        "TRL_DeepSpeed":   3100,
        "TRL_FSDP":        3500,
        "FlexTrain @ 40G": 5400,
        "FlexTrain":       6800,
    },
    "Qwen3.6-35B-A3B": {
        "TorchTitan":      None,
        "MegaTrain":       2800,
        "ALST":            3100,
        "TRL_DeepSpeed":   None,
        "TRL_FSDP":        None,
        "FlexTrain @ 40G": 4900,
        "FlexTrain":       6100,
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
        "--demo",
        action="store_true",
        help=(
            "Render with hard-coded sample numbers (DEMO_DATA) so you can "
            "preview the layout without editing DATA. Sample numbers are "
            "plausible but NOT real measurements; do not use for the paper."
        ),
    )
    parser.add_argument(
        "--show", action="store_true", help="Display interactively after saving.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Raster DPI when saving PNG.",
    )
    return parser.parse_args()


def _validate_data(data: dict[str, dict[str, float | None]]) -> None:
    """Fail loud if ``data`` is missing keys we'll dereference during render."""
    for model in MODELS:
        if model not in data:
            raise SystemExit(f"data missing model key: {model!r}")
        for b in BASELINES:
            if b not in data[model]:
                raise SystemExit(f"data[{model!r}] missing baseline key: {b!r}")


def _bar_panel(
    ax: plt.Axes,
    model: str,
    values: dict[str, float | None],
) -> None:
    """Render one panel: bar chart of tokens/sec across BASELINES for ``model``.

    Y-axis is scaled independently per model (model sizes give very
    different absolute throughputs at 128K, so a shared scale would
    squash the smaller-throughput panels). Each panel keeps its own
    "tokens / sec" label and tick numbers so cross-panel comparisons
    require reading the scale, not eyeballing bar heights.
    """
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
                ha="center", va="bottom", fontsize=7, color="#666",
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
            ha="center", va="bottom", fontsize=7,
        )

    ax.set_title(model, fontsize=11)
    ax.set_ylabel("tokens / sec", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(BASELINES, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, visible_max * 1.20)  # headroom for value labels.
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)


def main() -> int:
    args = parse_args()
    data = DEMO_DATA if args.demo else DATA
    if args.demo:
        print(
            "[plot] --demo: rendering with placeholder DEMO_DATA "
            "(NOT real measurements)",
            flush=True,
        )
    _validate_data(data)

    # 1x4 row sized for a NeurIPS top-of-page figure: aim for the
    # final \\includegraphics[width=\\textwidth]{...} to render with
    # ~1:4 aspect. Each panel ~3.5in wide gives 7 bars + rotated
    # labels enough room without overflowing.
    #
    # sharey=False is explicit: each model has its own y-axis scale
    # because the absolute throughputs differ ~3x across the lineup
    # (Llama3-8B vs Qwen3.6-35B-A3B at 128K context).
    fig, axes = plt.subplots(
        1, 4, figsize=(14, 3.8), constrained_layout=True, sharey=False,
    )
    for ax, model in zip(axes, MODELS):
        _bar_panel(ax, model, data[model])

    fig.suptitle(
        "Single-GPU training throughput at 128K context length (tokens/sec)",
        fontsize=12, y=1.04,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
    print(f"wrote {args.output}", flush=True)

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
