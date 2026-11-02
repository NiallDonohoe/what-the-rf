"""Class taxonomy: the single mapping between RadioML, synthetic and capture labels.

Canonical class names are what the models predict. Synthetic data uses canonical
names directly; RadioML and real captures are mapped onto them here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Sequence

# Phase 1: classes that overlap with RadioML 2016.10a.
PHASE1: tuple[str, ...] = ("BPSK", "QPSK", "8PSK", "QAM16", "GFSK", "AM-DSB", "WBFM")

# Phase 3: classes you can capture off-air with your own SDR.
PHASE3: tuple[str, ...] = ("WBFM", "AFSK-FM", "8GFSK", "OOK")

TAXONOMIES: dict[str, tuple[str, ...]] = {"phase1": PHASE1, "phase3": PHASE3}

ALL_CLASSES: tuple[str, ...] = tuple(dict.fromkeys(PHASE1 + PHASE3))

# RadioML 2016.10a modulation names -> canonical. Unlisted RadioML classes
# (AM-SSB, CPFSK, PAM4, QAM64) are dropped at load time.
RADIOML_TO_CANONICAL: dict[str, str] = {
    "BPSK": "BPSK",
    "QPSK": "QPSK",
    "8PSK": "8PSK",
    "QAM16": "QAM16",
    "GFSK": "GFSK",
    "AM-DSB": "AM-DSB",
    "WBFM": "WBFM",
}

# Ground-truth labels written into SigMF captures -> canonical.
CAPTURE_TO_CANONICAL: dict[str, str] = {
    "fm_broadcast": "WBFM",
    "aprs": "AFSK-FM",
    "ft8": "8GFSK",
    "cw": "OOK",
}


def resolve_classes(spec: str | Sequence[str]) -> tuple[str, ...]:
    """Turn a taxonomy name ("phase1") or an explicit list into a class tuple."""
    if isinstance(spec, str):
        if spec not in TAXONOMIES:
            raise ValueError(f"unknown taxonomy {spec!r}; choose from {sorted(TAXONOMIES)}")
        return TAXONOMIES[spec]
    unknown = [c for c in spec if c not in ALL_CLASSES]
    if unknown:
        raise ValueError(f"unknown classes {unknown}; known: {ALL_CLASSES}")
    return tuple(spec)


def canonical(name: str, source: str) -> str:
    """Map a label from `source` ("radioml", "capture" or "synth") to its canonical name."""
    if source == "synth" or name in ALL_CLASSES:
        if name not in ALL_CLASSES:
            raise KeyError(f"unknown synthetic class {name!r}")
        return name
    table = {"radioml": RADIOML_TO_CANONICAL, "capture": CAPTURE_TO_CANONICAL}[source]
    if name not in table:
        raise KeyError(f"no canonical class for {source} label {name!r}")
    return table[name]
