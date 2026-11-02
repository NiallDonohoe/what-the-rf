"""The one tensor format every source shares, plus splitting.

An IQDataset holds float32 IQ of shape (M, 2, N), integer labels into `classes`,
SNR in dB (NaN when unknown) and an optional group id (capture session).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wtrf.config import CaptureConfig, Config, DataConfig
from wtrf.data import radioml, sigmf, synth
from wtrf.data.labels import canonical, resolve_classes


@dataclass
class IQDataset:
    x: np.ndarray
    y: np.ndarray
    snr: np.ndarray
    classes: tuple[str, ...]
    groups: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> IQDataset:
        groups = None if self.groups is None else self.groups[idx]
        return IQDataset(self.x[idx], self.y[idx], self.snr[idx], self.classes, groups)


def normalize(x: np.ndarray) -> np.ndarray:
    """Scale each example to unit RMS power. Applied to every source and after augmenting."""
    power = np.mean(x.astype(np.float64) ** 2, axis=(-2, -1), keepdims=True) * 2
    return (x / np.sqrt(power + 1e-12)).astype(np.float32)


def stratified_split(
    y: np.ndarray, snr: np.ndarray, val: float, test: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices so every (class, SNR) cell is divided in the same proportions."""
    rng = np.random.default_rng(seed)
    parts: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] = ([], [], [])
    keys = np.stack([y, np.nan_to_num(snr, nan=-999)], axis=1)
    for key in np.unique(keys, axis=0):
        idx = np.flatnonzero((keys == key).all(axis=1))
        rng.shuffle(idx)
        n_test, n_val = round(len(idx) * test), round(len(idx) * val)
        parts[2].append(idx[:n_test])
        parts[1].append(idx[n_test : n_test + n_val])
        parts[0].append(idx[n_test + n_val :])
    train, val_idx, test_idx = (np.sort(np.concatenate(p)) for p in parts)
    return train, val_idx, test_idx


def group_split(
    groups: np.ndarray, val: float, test: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by group (capture session) so near-duplicate windows never straddle splits."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_test, n_val = round(len(unique) * test), round(len(unique) * val)
    test_g, val_g = set(unique[:n_test]), set(unique[n_test : n_test + n_val])
    test_idx = np.flatnonzero([g in test_g for g in groups])
    val_idx = np.flatnonzero([g in val_g for g in groups])
    train = np.setdiff1d(np.arange(len(groups)), np.concatenate([test_idx, val_idx]))
    return train, val_idx, test_idx


def build_synth(cfg: DataConfig, seed: int) -> IQDataset:
    classes = resolve_classes(cfg.classes)
    rng = np.random.default_rng(seed)
    xs, ys, snrs = [], [], []
    for label, cls in enumerate(classes):
        for snr in cfg.snrs:
            xs.append(synth.generate(cls, cfg.per_class_per_snr, snr, rng, cfg.n_samples))
            ys.append(np.full(cfg.per_class_per_snr, label))
            snrs.append(np.full(cfg.per_class_per_snr, snr))
    return IQDataset(
        normalize(np.concatenate(xs)),
        np.concatenate(ys).astype(np.int64),
        np.concatenate(snrs).astype(np.float32),
        classes,
    )


def build_dataset(cfg: DataConfig, seed: int) -> IQDataset:
    if cfg.source == "synth":
        return build_synth(cfg, seed)
    classes = resolve_classes(cfg.classes)
    x, y, snr = radioml.load(cfg.radioml_path, classes)
    if x.shape[-1] != cfg.n_samples:
        raise ValueError(f"RadioML windows are {x.shape[-1]} samples, config says {cfg.n_samples}")
    return IQDataset(normalize(x), y, snr, classes)


def build_splits(cfg: Config) -> dict[str, IQDataset]:
    """Deterministic train/val/test from a config. Split happens before any augmentation."""
    ds = build_dataset(cfg.data, cfg.seed)
    tr, va, te = stratified_split(ds.y, ds.snr, cfg.data.split.val, cfg.data.split.test, cfg.seed)
    return {"train": ds.subset(tr), "val": ds.subset(va), "test": ds.subset(te)}


def capture_windows(
    cap: sigmf.Capture, n_samples: int, cfg: CaptureConfig
) -> tuple[np.ndarray, float, np.ndarray]:
    """Preprocess and window one capture. Returns (windows, model-rate fs, complex stream)."""
    iq, fs = sigmf.preprocess(cap, cfg.target_rate)
    return normalize(sigmf.window(iq, n_samples, cfg.hop)), fs, iq


def load_capture_set(
    cfg: CaptureConfig, classes: tuple[str, ...], n_samples: int
) -> IQDataset:
    """All labelled captures in cfg.dir whose class the model knows, as one windowed set.

    The real-capture set is a test set only: never train or tune on it.
    """
    from wtrf.features import estimate_snr_db

    xs, ys, snrs, groups = [], [], [], []
    for base in sigmf.find_captures(cfg.dir):
        cap = sigmf.read(base)
        if cap.label is None:
            continue
        cls = canonical(cap.label, "capture")
        if cls not in classes:
            continue
        w, fs, iq = capture_windows(cap, n_samples, cfg)
        xs.append(w)
        ys.append(np.full(len(w), classes.index(cls)))
        snrs.append(np.full(len(w), estimate_snr_db(iq, fs)))
        groups.append(np.full(len(w), cap.session, dtype=object))
    if not xs:
        return IQDataset(
            np.zeros((0, 2, n_samples), np.float32),
            np.zeros(0, np.int64),
            np.zeros(0, np.float32),
            classes,
            np.zeros(0, dtype=object),
        )
    return IQDataset(
        np.concatenate(xs),
        np.concatenate(ys).astype(np.int64),
        np.concatenate(snrs).astype(np.float32),
        classes,
        np.concatenate(groups),
    )
