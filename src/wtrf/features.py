"""Measured signal features: bandwidth, centre frequency, SNR and spectrogram.

All estimates come from a Welch PSD. The noise floor is a low percentile of the PSD,
so the estimates degrade when the signal fills most of the sample bandwidth.
SNR is signal power over noise power across the full sample bandwidth, the same
definition the synthetic generator and RadioML use, so it indexes the SNR curves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal


@dataclass
class Features:
    centre_freq_hz: float
    occupied_bandwidth_hz: float
    snr_db_est: float
    duration_s: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def psd(iq: np.ndarray, fs: float, nperseg: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided PSD, frequencies ascending from -fs/2."""
    nperseg = min(nperseg, len(iq))
    f, p = signal.welch(iq, fs=fs, nperseg=nperseg, return_onesided=False, detrend=False)
    return np.fft.fftshift(f), np.fft.fftshift(p)


def _noise_floor(p: np.ndarray) -> float:
    return float(np.percentile(p, 20))


def estimate_snr_db(iq: np.ndarray, fs: float) -> float:
    _, p = psd(iq, fs)
    floor = _noise_floor(p)
    total, noise = p.mean(), floor
    return float(10 * np.log10(max(total - noise, 1e-12) / max(noise, 1e-12)))


def occupied_band(iq: np.ndarray, fs: float, fraction: float = 0.99) -> tuple[float, float]:
    """(centre offset Hz, bandwidth Hz) holding `fraction` of the above-floor power."""
    f, p = psd(iq, fs)
    excess = np.clip(p - _noise_floor(p), 0, None)
    if excess.sum() <= 0:
        return 0.0, 0.0
    cdf = np.cumsum(excess) / excess.sum()
    tail = (1 - fraction) / 2
    lo = f[int(np.searchsorted(cdf, tail))]
    hi = f[min(int(np.searchsorted(cdf, 1 - tail)), len(f) - 1)]
    df = f[1] - f[0]
    centre = float(np.sum(f * excess) / excess.sum())
    return centre, float(hi - lo + df)


def measure(iq: np.ndarray, fs: float, centre_freq_hz: float = 0.0) -> Features:
    offset, bw = occupied_band(iq, fs)
    return Features(
        centre_freq_hz=centre_freq_hz + offset,
        occupied_bandwidth_hz=bw,
        snr_db_est=estimate_snr_db(iq, fs),
        duration_s=len(iq) / fs,
    )


def spectrogram(
    iq: np.ndarray, fs: float, nperseg: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(freqs Hz ascending, times s, power dB [freq, time])."""
    nperseg = min(nperseg, len(iq))
    f, t, s = signal.spectrogram(
        iq, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, return_onesided=False, detrend=False
    )
    return np.fft.fftshift(f), t, 10 * np.log10(np.fft.fftshift(s, axes=0) + 1e-20)
