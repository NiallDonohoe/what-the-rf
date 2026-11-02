"""How big is the sim-to-real gap? Accuracy on synthetic test vs real captures, per class.

One of only two places that touch the real-capture set (the other is ablation).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wtrf.data.dataset import IQDataset


@dataclass
class GapRow:
    cls: str
    synth_acc: float
    synth_n: int
    real_acc: float
    real_n: int
    real_sessions: int

    @property
    def gap(self) -> float:
        return self.synth_acc - self.real_acc


def _acc(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return float((pred[mask] == y[mask]).mean()) if mask.any() else float("nan")


def gap_table(
    synth: IQDataset, synth_pred: np.ndarray, real: IQDataset, real_pred: np.ndarray
) -> list[GapRow]:
    """Per-class rows for classes present in the real set, plus an "ALL" row."""
    rows = []
    present = sorted(set(real.y.tolist()))
    groups = real.groups if real.groups is not None else np.zeros(len(real), dtype=object)
    for k in present:
        sm, rm = synth.y == k, real.y == k
        rows.append(
            GapRow(
                real.classes[k],
                _acc(synth_pred, synth.y, sm),
                int(sm.sum()),
                _acc(real_pred, real.y, rm),
                int(rm.sum()),
                len(set(groups[rm].tolist())),
            )
        )
    # Synthetic "ALL" is restricted to the classes the real set covers, so the gap is like-for-like.
    sm = np.isin(synth.y, present)
    rows.append(
        GapRow(
            "ALL",
            _acc(synth_pred, synth.y, sm),
            int(sm.sum()),
            float((real_pred == real.y).mean()),
            len(real),
            len(set(groups.tolist())),
        )
    )
    return rows


def write_csv(rows: list[GapRow], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "synth_acc", "synth_n", "real_acc", "real_n", "real_sessions", "gap"])
        for r in rows:
            w.writerow(
                [
                    r.cls,
                    f"{r.synth_acc:.4f}",
                    r.synth_n,
                    f"{r.real_acc:.4f}",
                    r.real_n,
                    r.real_sessions,
                    f"{r.gap:.4f}",
                ]
            )


def plot(rows: list[GapRow], title: str, path: Path) -> None:
    from wtrf.eval import style

    fig, ax = style.figure(1.5 + 1.0 * len(rows), 4.2)
    x = np.arange(len(rows))
    w = 0.38
    for offset, colour, name, vals, ns in (
        (
            -w / 2 - 0.01,
            style.SERIES[0],
            "synthetic test",
            [r.synth_acc for r in rows],
            [r.synth_n for r in rows],
        ),
        (
            w / 2 + 0.01,
            style.SERIES[1],
            "real captures",
            [r.real_acc for r in rows],
            [r.real_n for r in rows],
        ),
    ):
        bars = ax.bar(x + offset, vals, w, color=colour, label=name)
        for bar, n in zip(bars, ns, strict=True):
            ax.annotate(
                f"n={n}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=6,
                color=style.TEXT_2,
            )
    ax.set_xticks(x, [r.cls for r in rows])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=8, labelcolor=style.TEXT_2, loc="upper right")
    style.finish(fig, ax, title, path)
