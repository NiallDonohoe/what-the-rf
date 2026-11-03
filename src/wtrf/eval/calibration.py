"""Is the model's confidence honest? Reliability diagram, ECE and temperature scaling.

Temperature is fitted on the validation split after training and stored with the
checkpoint; "calibrated probabilities" everywhere else means softmax(logits / T).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

MIN_BIN_COUNT = 10  # reliability-diagram bins with fewer windows are too noisy to draw


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def reliability(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per confidence bin: (mean confidence, accuracy, count)."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    safe = np.maximum(count, 1)
    mean_conf = np.bincount(idx, weights=conf, minlength=n_bins) / safe
    acc = np.bincount(idx, weights=correct.astype(float), minlength=n_bins) / safe
    return mean_conf, acc, count


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: count-weighted |accuracy - confidence| over bins."""
    conf, acc, count = reliability(probs, y, n_bins)
    return float(np.sum(count * np.abs(acc - conf)) / max(count.sum(), 1))


def fit_temperature(logits: np.ndarray, y: np.ndarray, iters: int = 200) -> float:
    """Single scalar T minimising validation NLL (Guo et al., 2017)."""
    lg = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = loss_fn(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def write_csv(rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "bin", "mean_confidence", "accuracy", "n_windows"])
        for name, (conf, acc, count) in rows.items():
            for i, (c, a, n) in enumerate(zip(conf, acc, count, strict=True)):
                w.writerow([name, i, f"{c:.4f}", f"{a:.4f}", int(n)])


def plot(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray],
    calibrated: tuple[np.ndarray, np.ndarray, np.ndarray],
    ece_raw: float,
    ece_cal: float,
    title: str,
    path: Path,
) -> None:
    from wtrf.eval import style

    fig, ax = style.figure(5.2, 4.6)
    ax.plot([0, 1], [0, 1], color=style.REFERENCE, lw=1, ls="--", label="perfect calibration")
    for (conf, acc, count), colour, name, e in (
        (raw, style.SERIES[1], "raw softmax", ece_raw),
        (calibrated, style.SERIES[0], "temperature scaled", ece_cal),
    ):
        keep = count >= MIN_BIN_COUNT
        ax.plot(
            conf[keep],
            acc[keep],
            color=colour,
            lw=2,
            marker="o",
            ms=4,
            label=f"{name} (ECE {e:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=8, labelcolor=style.TEXT_2, loc="upper left")
    ax.text(
        0.98,
        0.02,
        f"bins with < {MIN_BIN_COUNT} windows not drawn",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=style.TEXT_2,
    )
    style.finish(fig, ax, title, path)
