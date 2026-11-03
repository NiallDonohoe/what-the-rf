"""Plain-English explanation of a classification, built only from measured facts.

Deterministic: the same probabilities and features always give the same text. It names
modulations, never protocols or uses, because the classifier only knows modulations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from wtrf.features import Features

# What tells each class apart; describes modulations, never uses.
SIGNATURES: dict[str, str] = {
    "BPSK": "a two-point phase constellation",
    "QPSK": "four phase states at constant amplitude",
    "8PSK": "eight phase states at constant amplitude",
    "QAM16": "amplitude as well as phase changes (16 points)",
    "GFSK": "a constant envelope with two smoothed frequency states",
    "AM-DSB": "an envelope that follows the message, with a carrier at the centre",
    "WBFM": "a constant envelope with wide, continuous frequency deviation",
    "AFSK-FM": "two audio tones carried on a narrow FM signal",
    "8GFSK": "eight discrete, evenly spaced tones",
    "OOK": "a carrier that switches fully on and off",
}


@dataclass
class Reliability:
    """The model's accuracy at the nearest evaluated SNR bucket (synthetic test set)."""

    snr_bucket_db: float
    accuracy: float
    n_windows: int


@dataclass
class Explanation:
    verdict: str
    evidence: list[str]
    confidence_note: str
    alternatives: list[str]
    uncertain: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _hz(f: float) -> str:
    for unit, scale in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if abs(f) >= scale:
            return f"{f / scale:.4g} {unit}"
    return f"{f:.4g} Hz"


def explain(
    top: list[tuple[str, float]],
    features: Features,
    n_windows: int,
    reliability: Reliability | None,
    threshold: float,
) -> Explanation:
    """Explain a top-k result. Below `threshold` the verdict and note say "uncertain"."""
    label, prob = top[0]
    uncertain = prob < threshold
    verdict = (
        f"Uncertain: the best guess is {label}, but the classifier is not confident."
        if uncertain
        else f"The signal is most likely {label}."
    )
    evidence = [
        f"Occupied bandwidth is about {_hz(features.occupied_bandwidth_hz)} "
        f"around {_hz(features.centre_freq_hz)}.",
        f"Estimated SNR is {features.snr_db_est:.1f} dB "
        f"over {features.duration_s:.2f} s ({n_windows} windows).",
    ]
    if label in SIGNATURES:
        evidence.append(f"{label} is characterised by {SIGNATURES[label]}.")
    note = f"Top probability is {prob:.0%} (threshold {threshold:.0%})"
    note += ", so this result is uncertain." if uncertain else ", above the threshold."
    if reliability:
        r = reliability
        note += (
            f" On the test set the model was {r.accuracy:.0%} accurate at "
            f"{r.snr_bucket_db:+g} dB SNR (n={r.n_windows})."
        )
    alternatives = [
        f"{alt} ({p:.0%}): would show {SIGNATURES.get(alt, 'a different structure')}."
        for alt, p in top[1:]
    ]
    return Explanation(verdict, evidence, note, alternatives, uncertain)
