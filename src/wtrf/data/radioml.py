"""RadioML 2016.10a loader.

The dataset is CC BY-NC-SA 4.0 (DeepSig). It is downloaded with
scripts/download_radioml.py and never committed. The file is a Python pickle, so
only load a copy you obtained from DeepSig: unpickling runs code.

Layout: dict[(modulation, snr_db)] -> float32 array (1000, 2, 128).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from wtrf.data.labels import RADIOML_TO_CANONICAL


def load(path: str | Path, classes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, snr) for the RadioML modulations that map onto `classes`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_radioml.py` first (see README)."
        )
    with path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")  # noqa: S301 - trusted public dataset

    xs, ys, snrs = [], [], []
    for (mod, snr), arr in sorted(data.items()):
        cls = RADIOML_TO_CANONICAL.get(mod)
        if cls is None or cls not in classes:
            continue
        xs.append(np.asarray(arr, dtype=np.float32))
        ys.append(np.full(len(arr), classes.index(cls)))
        snrs.append(np.full(len(arr), float(snr)))
    if not xs:
        raise ValueError(f"no RadioML modulations in {path} map onto {classes}")
    return (
        np.concatenate(xs),
        np.concatenate(ys).astype(np.int64),
        np.concatenate(snrs).astype(np.float32),
    )
