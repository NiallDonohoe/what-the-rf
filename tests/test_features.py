import numpy as np

from wtrf.data import synth
from wtrf.features import estimate_snr_db, measure, spectrogram


def test_tone_centre_frequency(rng):
    fs, f0 = 48_000.0, 6_000.0
    z = synth.add_noise(np.exp(2j * np.pi * f0 / fs * np.arange(48_000))[None], 20.0, rng)[0]
    feats = measure(z, fs, centre_freq_hz=145.0e6)
    assert abs(feats.centre_freq_hz - (145.0e6 + f0)) < 500
    assert feats.occupied_bandwidth_hz < 2_000
    assert abs(feats.duration_s - 1.0) < 1e-9


def test_snr_estimate_tracks_truth_for_narrowband_signal(rng):
    z = synth.clean("AFSK-FM", 1, rng, 65_536)
    for snr in (0.0, 10.0, 20.0):
        est = estimate_snr_db(synth.add_noise(z, snr, rng)[0], 1.0)
        assert abs(est - snr) < 3.0


def test_spectrogram_shape(rng):
    f, t, s = spectrogram(synth.clean("WBFM", 1, rng, 4096)[0], 1.0)
    assert s.shape == (len(f), len(t)) and np.all(np.diff(f) > 0)
