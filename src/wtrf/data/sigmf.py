"""Read and write SigMF captures (.sigmf-meta + .sigmf-data).

wtrf stores its own fields under the `wtrf:` namespace, declared in core:extensions:
  wtrf:label            ground-truth capture label (see labels.CAPTURE_TO_CANONICAL)
  wtrf:session          capture session id; splits are made by session, never by window
  wtrf:gain_db          receiver gain
  wtrf:signal_offset_hz optional offset of the signal from the tuned centre frequency
The SDR model goes in core:hw and the tuned frequency in captures[0].core:frequency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

_DTYPES: dict[str, tuple[np.dtype, float]] = {
    "cf32_le": (np.dtype("<f4"), 1.0),
    "ci16_le": (np.dtype("<i2"), 1 / 32768),
    "ci8": (np.dtype("i1"), 1 / 128),
    "cu8": (np.dtype("u1"), 1 / 128),  # RTL-SDR native; offset by 127.5
}


@dataclass
class Capture:
    iq: np.ndarray  # complex64, shape (N,)
    meta: dict[str, Any]
    path: Path | None = None

    @property
    def glob(self) -> dict[str, Any]:
        return dict(self.meta.get("global", {}))

    @property
    def sample_rate(self) -> float:
        return float(self.glob["core:sample_rate"])

    @property
    def centre_freq(self) -> float:
        caps = self.meta.get("captures") or [{}]
        return float(caps[0].get("core:frequency", 0.0))

    @property
    def label(self) -> str | None:
        if "wtrf:label" in self.glob:
            return str(self.glob["wtrf:label"])
        for ann in self.meta.get("annotations", []):
            if "core:label" in ann:
                return str(ann["core:label"])
        return None

    @property
    def session(self) -> str:
        default = self.path.stem if self.path else "unknown"
        return str(self.glob.get("wtrf:session", default))

    @property
    def gain_db(self) -> float | None:
        g = self.glob.get("wtrf:gain_db")
        return None if g is None else float(g)

    @property
    def sdr(self) -> str | None:
        return self.glob.get("core:hw")

    @property
    def duration_s(self) -> float:
        return len(self.iq) / self.sample_rate


def _base(path: str | Path) -> Path:
    p = Path(path)
    for suffix in (".sigmf-meta", ".sigmf-data", ".sigmf"):
        if p.name.endswith(suffix):
            return p.with_name(p.name[: -len(suffix)])
    return p


def read(path: str | Path) -> Capture:
    base = _base(path)
    meta = json.loads(base.with_name(base.name + ".sigmf-meta").read_text())
    datatype = meta["global"]["core:datatype"]
    if datatype not in _DTYPES:
        raise ValueError(f"unsupported SigMF datatype {datatype!r}; use one of {list(_DTYPES)}")
    dtype, scale = _DTYPES[datatype]
    raw = np.fromfile(base.with_name(base.name + ".sigmf-data"), dtype=dtype).astype(np.float32)
    if datatype == "cu8":
        raw -= 127.5
    raw *= scale
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    return Capture(iq=iq, meta=meta, path=base)


def write(
    path: str | Path,
    iq: np.ndarray,
    sample_rate: float,
    centre_freq: float,
    *,
    label: str | None = None,
    session: str | None = None,
    gain_db: float | None = None,
    sdr: str | None = None,
    datatype: str = "cf32_le",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a SigMF pair and return the base path (without extension)."""
    if datatype not in _DTYPES:
        raise ValueError(f"unsupported SigMF datatype {datatype!r}")
    base = _base(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    dtype, scale = _DTYPES[datatype]
    inter = np.empty(2 * len(iq), dtype=np.float64)
    inter[0::2], inter[1::2] = np.real(iq), np.imag(iq)
    if datatype == "cf32_le":
        out = inter.astype(dtype)
    else:
        offset = 127.5 if datatype == "cu8" else 0.0
        info = np.iinfo(dtype)
        out = np.clip(np.round(inter / scale + offset), info.min, info.max).astype(dtype)
    out.tofile(base.with_name(base.name + ".sigmf-data"))

    glob: dict[str, Any] = {
        "core:datatype": datatype,
        "core:sample_rate": float(sample_rate),
        "core:version": "1.0.0",
        "core:extensions": [{"name": "wtrf", "version": "0.1.0", "optional": True}],
    }
    if sdr:
        glob["core:hw"] = sdr
    for key, value in (("wtrf:label", label), ("wtrf:session", session), ("wtrf:gain_db", gain_db)):
        if value is not None:
            glob[key] = value
    glob.update(extra or {})
    meta = {
        "global": glob,
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": float(centre_freq),
                "core:datetime": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
        ],
        "annotations": (
            [{"core:sample_start": 0, "core:sample_count": len(iq), "core:label": label}]
            if label
            else []
        ),
    }
    base.with_name(base.name + ".sigmf-meta").write_text(json.dumps(meta, indent=2))
    return base


def preprocess(cap: Capture, target_rate: float | None) -> tuple[np.ndarray, float]:
    """Shift the signal to 0 Hz (if wtrf:signal_offset_hz is set) and resample.

    Returns (complex iq, sample rate). With target_rate=None the capture rate is kept.
    """
    iq = cap.iq.astype(np.complex128)
    fs = cap.sample_rate
    offset = float(cap.glob.get("wtrf:signal_offset_hz", 0.0))
    if offset:
        iq = iq * np.exp(-2j * np.pi * offset / fs * np.arange(len(iq)))
    if target_rate and not np.isclose(target_rate, fs):
        ratio = Fraction(target_rate / fs).limit_denominator(1000)
        iq = signal.resample_poly(iq, ratio.numerator, ratio.denominator)
        fs = fs * ratio.numerator / ratio.denominator
    return iq, fs


def window(iq: np.ndarray, n: int, hop: int | None = None) -> np.ndarray:
    """Cut complex iq into float32 windows of shape (W, 2, n). Partial tail is dropped."""
    hop = hop or n
    if len(iq) < n:
        raise ValueError(f"capture has {len(iq)} samples, fewer than one window of {n}")
    starts = np.arange(0, len(iq) - n + 1, hop)
    z = np.stack([iq[s : s + n] for s in starts])
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


def find_captures(directory: str | Path) -> list[Path]:
    return sorted(_base(p) for p in Path(directory).glob("*.sigmf-meta"))
