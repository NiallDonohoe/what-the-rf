"""Synthetic IQ generator (NumPy, seeded).

Every generator returns complex baseband at unit average power. `generate` adds a
random carrier phase and AWGN at the requested SNR, where SNR is signal power over
noise power across the full sample bandwidth (the RadioML convention).

Frequencies are in cycles per sample, so the same generator serves any sample rate.
Phase 1 digital classes use 8 samples per symbol to match RadioML 2016.10a.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import signal

SPS = 8
_MARGIN = 64  # extra samples generated so filter transients can be cropped off

Rng = np.random.Generator


def _unit_power(z: np.ndarray) -> np.ndarray:
    return z / np.sqrt(np.mean(np.abs(z) ** 2, axis=-1, keepdims=True) + 1e-12)


def _rrc_taps(sps: int, beta: float = 0.35, span: int = 8) -> np.ndarray:
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    taps = np.empty_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 + beta * (4 / np.pi - 1)
        elif np.isclose(abs(ti), 1 / (4 * beta)):
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            taps[i] = num / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return taps / np.sqrt(np.sum(taps**2))


def _gaussian_taps(sps: int, bt: float, span: int = 4) -> np.ndarray:
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    alpha = np.sqrt(np.log(2) / 2) / bt
    taps = np.exp(-((np.pi * t / alpha) ** 2))
    return taps / taps.sum()


def _lowpass_noise(rng: Rng, n: int, length: int, cutoff: float) -> np.ndarray:
    """Real band-limited noise, unit std, cutoff in cycles/sample."""
    taps = signal.firwin(63, cutoff * 2)
    x = signal.lfilter(taps, 1.0, rng.standard_normal((n, length + 63)), axis=-1)[:, 63:]
    return x / (x.std(axis=-1, keepdims=True) + 1e-12)


def _crop(z: np.ndarray, length: int, rng: Rng) -> np.ndarray:
    start = rng.integers(0, z.shape[-1] - length + 1, size=z.shape[0])
    idx = start[:, None] + np.arange(length)[None, :]
    return np.take_along_axis(z, idx, axis=-1)


def _linear(constellation: np.ndarray) -> Callable[[int, int, Rng], np.ndarray]:
    def gen(n: int, length: int, rng: Rng) -> np.ndarray:
        n_sym = (length + _MARGIN) // SPS + 8
        symbols = constellation[rng.integers(0, len(constellation), size=(n, n_sym))]
        up = np.zeros((n, n_sym * SPS), dtype=complex)
        up[:, ::SPS] = symbols
        z = signal.lfilter(_rrc_taps(SPS), 1.0, up, axis=-1)
        return _crop(z[:, 4 * SPS :], length, rng)

    return gen


def _fsk(levels: int, sps: int, h: float, bt: float) -> Callable[[int, int, Rng], np.ndarray]:
    """Continuous-phase (G)FSK. Tone spacing is h / sps cycles/sample."""

    def gen(n: int, length: int, rng: Rng) -> np.ndarray:
        n_sym = (length + _MARGIN) // sps + 4
        sym = rng.integers(0, levels, size=(n, n_sym)) - (levels - 1) / 2
        freq = np.repeat(sym, sps, axis=-1)
        freq = signal.lfilter(_gaussian_taps(sps, bt), 1.0, freq, axis=-1)
        phase = 2 * np.pi * np.cumsum(freq * h / sps, axis=-1)
        return _crop(np.exp(1j * phase), length, rng)

    return gen


def _am_dsb(n: int, length: int, rng: Rng) -> np.ndarray:
    m = _lowpass_noise(rng, n, length + _MARGIN, cutoff=0.05)
    m = m / np.max(np.abs(m), axis=-1, keepdims=True)
    depth = rng.uniform(0.3, 0.9, size=(n, 1))
    return _crop((1 + depth * m).astype(complex), length, rng)


def _wbfm(n: int, length: int, rng: Rng) -> np.ndarray:
    m = _lowpass_noise(rng, n, length + _MARGIN, cutoff=0.03)
    deviation = rng.uniform(0.05, 0.12, size=(n, 1))  # cycles/sample per unit std
    return _crop(np.exp(2j * np.pi * np.cumsum(deviation * m, axis=-1)), length, rng)


def _afsk_fm(n: int, length: int, rng: Rng) -> np.ndarray:
    """Bell 202-style AFSK (mark/space tones in a 6:11 ratio) on narrowband FM."""
    bit_len = 16
    n_bits = (length + _MARGIN) // bit_len + 2
    bits = rng.integers(0, 2, size=(n, n_bits))
    tone = np.where(np.repeat(bits, bit_len, axis=-1) == 1, 0.03, 0.055)
    audio = np.sin(2 * np.pi * np.cumsum(tone, axis=-1))
    deviation = rng.uniform(0.02, 0.04, size=(n, 1))
    return _crop(np.exp(2j * np.pi * np.cumsum(deviation * audio, axis=-1)), length, rng)


def _ook(n: int, length: int, rng: Rng) -> np.ndarray:
    """Morse-like on-off keying: dots, dashes and gaps with soft keying edges."""
    unit = 12
    total = length + _MARGIN
    env = np.zeros((n, total))
    for i in range(n):
        pos = int(rng.integers(0, 3 * unit))
        while pos < total:
            on = unit * (1 if rng.random() < 0.5 else 3)
            env[i, pos : pos + on] = 1.0
            pos += on + unit * int(rng.choice([1, 1, 3, 7]))
    env = signal.lfilter(np.hanning(5) / np.hanning(5).sum(), 1.0, env, axis=-1)
    offset = rng.uniform(-0.05, 0.05, size=(n, 1))
    tone = np.exp(2j * np.pi * offset * np.arange(total)[None, :])
    return _crop(env * tone, length, rng)


def _qam16() -> np.ndarray:
    pts = np.array([complex(i, q) for i in (-3, -1, 1, 3) for q in (-3, -1, 1, 3)])
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


GENERATORS: dict[str, Callable[[int, int, Rng], np.ndarray]] = {
    "BPSK": _linear(np.array([-1.0, 1.0], dtype=complex)),
    "QPSK": _linear(np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))),
    "8PSK": _linear(np.exp(2j * np.pi * np.arange(8) / 8)),
    "QAM16": _linear(_qam16()),
    "GFSK": _fsk(levels=2, sps=SPS, h=0.5, bt=0.35),
    "AM-DSB": _am_dsb,
    "WBFM": _wbfm,
    "AFSK-FM": _afsk_fm,
    "8GFSK": _fsk(levels=8, sps=16, h=1.0, bt=2.0),  # FT8-like: 8 tones, spacing = baud
    "OOK": _ook,
}


def clean(cls: str, n: int, rng: Rng, n_samples: int = 128) -> np.ndarray:
    """`n` noiseless examples of `cls` as complex (n, n_samples) at unit power."""
    if cls not in GENERATORS:
        raise KeyError(f"no synthetic generator for {cls!r}")
    z = GENERATORS[cls](n, n_samples, rng)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(n, 1)))
    return _unit_power(z * phase)


def add_noise(z: np.ndarray, snr_db: float | np.ndarray, rng: Rng) -> np.ndarray:
    """AWGN at `snr_db` relative to the per-example signal power."""
    power = np.mean(np.abs(z) ** 2, axis=-1, keepdims=True)
    snr = np.asarray(snr_db, dtype=float)
    if snr.ndim:
        snr = snr.reshape(-1, 1)
    sigma = np.sqrt(power * 10 ** (-snr / 10) / 2)
    return z + sigma * (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape))


def to_iq(z: np.ndarray) -> np.ndarray:
    """Complex (..., N) -> float32 (..., 2, N)."""
    return np.stack([z.real, z.imag], axis=-2).astype(np.float32)


def from_iq(x: np.ndarray) -> np.ndarray:
    """Float (..., 2, N) -> complex (..., N)."""
    return x[..., 0, :] + 1j * x[..., 1, :]


def generate(cls: str, n: int, snr_db: float, rng: Rng, n_samples: int = 128) -> np.ndarray:
    """`n` noisy examples of `cls` as float32 (n, 2, n_samples)."""
    return to_iq(add_noise(clean(cls, n, rng, n_samples), snr_db, rng))
