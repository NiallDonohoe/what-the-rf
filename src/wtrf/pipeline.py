"""Classify and explain a capture. The CLI commands call these functions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wtrf.data import sigmf
from wtrf.data.dataset import capture_windows
from wtrf.eval.snr_curves import accuracy_at
from wtrf.explain import Explanation, Reliability, explain
from wtrf.features import Features, measure
from wtrf.models.train import Run


@dataclass
class CaptureMeta:
    sample_rate_hz: float
    model_rate_hz: float
    gain_db: float | None
    sdr: str | None
    n_windows: int


@dataclass
class Classification:
    run_id: str
    classes: tuple[str, ...]
    probs: np.ndarray  # calibrated, averaged over windows
    features: Features
    meta: CaptureMeta
    reliability: Reliability | None

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        order = np.argsort(self.probs)[::-1][:k]
        return [(self.classes[i], float(self.probs[i])) for i in order]


def load_reliability(run: Run, snr_db: float) -> Reliability | None:
    """Accuracy at the nearest SNR bucket, from the run's evaluation (wtrf eval)."""
    path = run.dir / "snr_accuracy.json"
    if not path.exists():
        return None
    curve = [(r["snr_db"], r["accuracy"], r["n_windows"]) for r in json.loads(path.read_text())]
    snr, acc, n = accuracy_at(curve, snr_db)
    return Reliability(snr, acc, n)


def classify_capture(run: Run, capture: str | Path | sigmf.Capture) -> Classification:
    cap = capture if isinstance(capture, sigmf.Capture) else sigmf.read(capture)
    windows, fs, iq = capture_windows(cap, run.n_samples, run.cfg.captures)
    probs = run.probabilities(windows).mean(axis=0)
    offset = float(cap.glob.get("wtrf:signal_offset_hz", 0.0))
    feats = measure(iq, fs, cap.centre_freq + offset)
    meta = CaptureMeta(
        sample_rate_hz=cap.sample_rate,
        model_rate_hz=fs,
        gain_db=cap.gain_db,
        sdr=cap.sdr,
        n_windows=len(windows),
    )
    return Classification(
        run.id, run.classes, probs, feats, meta, load_reliability(run, feats.snr_db_est)
    )


def explain_classification(c: Classification, run: Run) -> Explanation:
    return explain(c.top(3), c.features, c.meta.n_windows, c.reliability, run.cfg.explain.threshold)
