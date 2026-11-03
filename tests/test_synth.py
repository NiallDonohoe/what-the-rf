import numpy as np
import pytest

from wtrf.data import synth
from wtrf.data.labels import ALL_CLASSES
from wtrf.features import occupied_band


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_shape_dtype_and_unit_power(cls, rng):
    x = synth.generate(cls, 8, 10.0, rng, n_samples=128)
    assert x.shape == (8, 2, 128) and x.dtype == np.float32
    clean = synth.clean(cls, 8, rng)
    assert np.allclose(np.mean(np.abs(clean) ** 2, axis=-1), 1.0, atol=1e-3)


def test_seeded_generation_is_reproducible():
    a = synth.generate("QPSK", 4, 0.0, np.random.default_rng(7))
    b = synth.generate("QPSK", 4, 0.0, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_noise_matches_requested_snr(rng):
    z = synth.clean("QPSK", 200, rng, 1024)
    noisy = synth.add_noise(z, 0.0, rng)
    snr = 10 * np.log10(np.mean(np.abs(z) ** 2) / np.mean(np.abs(noisy - z) ** 2))
    assert abs(snr) < 0.3


def _envelope_cv(cls, rng):
    env = np.abs(synth.clean(cls, 64, rng, 512))
    return float(np.mean(env.std(-1) / env.mean(-1)))


@pytest.mark.parametrize("cls", ["GFSK", "WBFM", "AFSK-FM", "8GFSK"])
def test_frequency_modulations_have_constant_envelope(cls, rng):
    assert _envelope_cv(cls, rng) < 1e-6


@pytest.mark.parametrize("cls", ["QAM16", "AM-DSB", "OOK"])
def test_amplitude_modulations_vary_in_envelope(cls, rng):
    assert _envelope_cv(cls, rng) > 0.15


def test_am_dsb_has_a_carrier_at_dc(rng):
    z = synth.clean("AM-DSB", 32, rng, 1024)
    spec = np.mean(np.abs(np.fft.fft(z, axis=-1)) ** 2, axis=0)
    assert spec[0] > 50 * np.median(spec)


def test_occupied_bandwidths_match_generator_design(rng):
    def bw(cls):
        z = synth.clean(cls, 1, rng, 8192)[0]
        return occupied_band(z, 1.0)[1]

    assert bw("WBFM") > bw("GFSK")
    assert 0.4 < bw("8GFSK") < 0.6  # 8 tones spaced 1/16 cycles/sample
    assert 0.05 < bw("AM-DSB") < 0.15  # message band-limited to 0.05 either side
