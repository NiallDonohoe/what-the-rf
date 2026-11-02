"""How does accuracy fall as SNR drops? One line per model, spread across seeds."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SnrCurve = list[tuple[float, float, int]]  # (snr_db, accuracy, n_windows)


def accuracy_by_snr(y: np.ndarray, pred: np.ndarray, snr: np.ndarray) -> SnrCurve:
    """Accuracy per SNR bucket. Windows with unknown (NaN) SNR are skipped."""
    out = []
    for s in np.unique(snr[~np.isnan(snr)]):
        mask = snr == s
        out.append((float(s), float((pred[mask] == y[mask]).mean()), int(mask.sum())))
    return out


def accuracy_at(curve: SnrCurve, snr_db: float) -> tuple[float, float, int]:
    """The bucket nearest `snr_db`: (bucket snr, accuracy, n)."""
    if not curve:
        raise ValueError("empty SNR curve")
    return min(curve, key=lambda row: abs(row[0] - snr_db))


def write_csv(curves: dict[str, list[SnrCurve]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed_index", "snr_db", "accuracy", "n_windows"])
        for label, seeds in curves.items():
            for i, curve in enumerate(seeds):
                for snr, acc, n in curve:
                    w.writerow([label, i, snr, f"{acc:.4f}", n])


def plot(curves: dict[str, list[SnrCurve]], colours: dict[str, str], path: Path) -> None:
    from wtrf.eval import style

    fig, ax = style.figure()
    ends: list[tuple[float, float, str]] = []
    for label, seeds in curves.items():
        snrs = np.array([row[0] for row in seeds[0]])
        accs = np.array([[row[1] for row in c] for c in seeds])
        mean = accs.mean(axis=0)
        colour = colours[label]
        if len(seeds) > 1:
            ax.fill_between(
                snrs, accs.min(axis=0), accs.max(axis=0), color=colour, alpha=0.15, lw=0
            )
        ax.plot(snrs, mean, color=colour, lw=2, marker="o", ms=4, label=f"{label} (n={len(seeds)})")
        ends.append((float(snrs[-1]), float(mean[-1]), label))
    # Direct labels at line ends, pushed apart so close lines stay readable.
    min_gap, last_y = 0.045, -1.0
    for x, y, label in sorted(ends, key=lambda e: e[1]):
        y = max(y, last_y + min_gap)
        last_y = y
        ax.annotate(
            label,
            (x, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=style.TEXT_2,
            annotation_clip=False,
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=style.TEXT_2, loc="lower right")
    note = "Line: mean over seeds. Band: min to max."
    style.finish(fig, ax, f"Accuracy vs SNR\n{note}", path)
