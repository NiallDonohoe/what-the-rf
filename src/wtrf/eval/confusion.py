"""Which classes get mixed up? Row-normalised confusion matrices at high and low SNR."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def confusion_matrix(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts, rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def row_normalise(cm: np.ndarray) -> np.ndarray:
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals > 0)


def write_csv(cm: np.ndarray, classes: Sequence[str], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *classes])
        for cls, row in zip(classes, cm, strict=True):
            w.writerow([cls, *row.tolist()])


def plot(cm: np.ndarray, classes: Sequence[str], title: str, path: Path) -> None:
    from wtrf.eval import style

    norm = row_normalise(cm)
    k = len(classes)
    fig, ax = style.figure(1.0 + 0.62 * k, 0.8 + 0.55 * k)
    ax.grid(False)
    ax.imshow(norm, cmap=style.SEQUENTIAL, vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if norm[i, j] >= 0.05:
                ink = "#ffffff" if norm[i, j] > 0.55 else style.TEXT
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7, color=ink)
    ax.set_xticks(range(k), classes, rotation=45, ha="right")
    ax.set_yticks(range(k), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    style.finish(fig, ax, f"{title}\n{int(cm.sum())} windows, rows sum to 1", path)
