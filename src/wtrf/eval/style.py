"""Shared plot styling. Colour follows the model, never its rank, so every report agrees."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#9a9893"

# Validated categorical slots (CVD-safe as a set of three on all pairs).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MODEL_COLOURS = {"cnn": SERIES[0], "resnet": SERIES[1], "transformer": SERIES[2]}

# Single-hue sequential ramp for magnitudes (confusion matrices).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "wtrf_blue", ["#f4f8fd", "#9cc2ee", "#2a78d6", "#123a6b"]
)


def colour_for(model_name: str, fallback_index: int = 0) -> str:
    return MODEL_COLOURS.get(model_name, SERIES[fallback_index % len(SERIES)])


def figure(width: float = 7.0, height: float = 4.2) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def decorate(fig: plt.Figure, ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, pad=10)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)
    fig.tight_layout()


def finish(fig: plt.Figure, ax: plt.Axes, title: str, path: str | Path) -> None:
    decorate(fig, ax, title)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
