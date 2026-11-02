"""Channel and receiver impairments used to close the sim-to-real gap.

Each augmentation acts on a complex batch (B, N) and is applied per example with
probability `p`. `apply` always runs the selected ones in ORDER, so "all together"
is one fixed pipeline and every ablation row is comparable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel

from wtrf.data.synth import add_noise, from_iq, to_iq

Rng = np.random.Generator

# Physical order: multipath channel, then receiver oscillator and clock, then noise.
ORDER: tuple[str, ...] = ("fading", "cfo", "drift", "noise")


class AugmentConfig(BaseModel):
    use: list[str] = []
    p: float = 0.8
    cfo_max: float = 0.02  # cycles/sample
    noise_snr: tuple[float, float] = (-4.0, 20.0)  # dB, drawn uniformly
    fading_taps: int = 4
    fading_decay: float = 1.0  # exponential power-delay profile, per tap
    drift_max: float = 0.005  # fractional sample-rate error


def cfo(z: np.ndarray, rng: Rng, cfg: AugmentConfig) -> np.ndarray:
    f = rng.uniform(-cfg.cfo_max, cfg.cfo_max, size=(z.shape[0], 1))
    return z * np.exp(2j * np.pi * f * np.arange(z.shape[-1])[None, :])


def noise(z: np.ndarray, rng: Rng, cfg: AugmentConfig) -> np.ndarray:
    lo, hi = cfg.noise_snr
    return add_noise(z, rng.uniform(lo, hi, size=z.shape[0]), rng)


def fading(z: np.ndarray, rng: Rng, cfg: AugmentConfig) -> np.ndarray:
    """Rayleigh multipath with an exponential power-delay profile (sample-spaced taps)."""
    k = cfg.fading_taps
    profile = np.exp(-cfg.fading_decay * np.arange(k))
    taps = rng.standard_normal((z.shape[0], k)) + 1j * rng.standard_normal((z.shape[0], k))
    taps *= np.sqrt(profile / profile.sum() / 2)
    out = np.zeros_like(z)
    for d in range(k):
        out[:, d:] += taps[:, d : d + 1] * z[:, : z.shape[-1] - d]
    return out


def drift(z: np.ndarray, rng: Rng, cfg: AugmentConfig) -> np.ndarray:
    """Sample clock offset plus a random fractional timing offset (linear interpolation)."""
    b, n = z.shape
    rate = 1 + rng.uniform(-cfg.drift_max, cfg.drift_max, size=(b, 1))
    t = rng.uniform(0, 1, size=(b, 1)) + np.arange(n)[None, :] * rate
    t = np.clip(t, 0, n - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = t - i0
    take = np.take_along_axis
    return (1 - frac) * take(z, i0, axis=-1) + frac * take(z, i1, axis=-1)


AUGMENTATIONS: dict[str, Callable[[np.ndarray, Rng, AugmentConfig], np.ndarray]] = {
    "fading": fading,
    "cfo": cfo,
    "drift": drift,
    "noise": noise,
}


def apply(x: np.ndarray, names: Sequence[str], cfg: AugmentConfig, rng: Rng) -> np.ndarray:
    """Augment a float32 batch (B, 2, N); returns the same shape."""
    unknown = set(names) - set(AUGMENTATIONS)
    if unknown:
        raise ValueError(f"unknown augmentations {sorted(unknown)}; known: {list(ORDER)}")
    if not names:
        return x
    z = from_iq(x)
    for name in ORDER:
        if name in names:
            mask = np.asarray(rng.random(z.shape[0]) < cfg.p)
            if mask.any():
                z[mask] = AUGMENTATIONS[name](z[mask], rng, cfg)
    return to_iq(z)
