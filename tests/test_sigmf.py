import json

import numpy as np
import pytest

from wtrf.config import CaptureConfig
from wtrf.data import sigmf
from wtrf.data.dataset import load_capture_set


@pytest.mark.parametrize("datatype,tol", [("cf32_le", 1e-7), ("ci16_le", 1e-4), ("cu8", 1e-2)])
def test_round_trip(tmp_path, rng, datatype, tol):
    iq = (0.5 * (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)) / 3).astype(
        np.complex64
    )
    base = sigmf.write(
        tmp_path / "cap",
        iq,
        2.4e6,
        145.8e6,
        label="aprs",
        session="s1",
        gain_db=30.0,
        sdr="RTL-SDR v4",
        datatype=datatype,
    )
    cap = sigmf.read(str(base) + ".sigmf-meta")
    assert np.allclose(cap.iq, iq, atol=tol)
    assert (cap.sample_rate, cap.centre_freq, cap.label) == (2.4e6, 145.8e6, "aprs")
    assert (cap.session, cap.gain_db, cap.sdr) == ("s1", 30.0, "RTL-SDR v4")
    meta = json.loads((tmp_path / "cap.sigmf-meta").read_text())
    assert meta["global"]["core:version"] == "1.0.0"


def test_window_and_preprocess(tmp_path, rng):
    iq = np.exp(2j * np.pi * 0.01 * np.arange(4800)).astype(np.complex64)
    sigmf.write(tmp_path / "a", iq, 48_000, 0.0, label="cw", session="s1")
    cap = sigmf.read(tmp_path / "a")
    stream, fs = sigmf.preprocess(cap, 24_000)
    assert fs == 24_000 and abs(len(stream) - 2400) <= 1
    w = sigmf.window(stream, 128)
    assert w.shape == (len(stream) // 128, 2, 128)


def test_capture_set_groups_by_session_and_skips_unknown_classes(tmp_path, rng):
    for name, label, session in (("a", "cw", "s1"), ("b", "cw", "s2"), ("c", "aprs", "s3")):
        sigmf.write(
            tmp_path / name,
            rng.standard_normal(1024).astype(np.complex64),
            48_000,
            0.0,
            label=label,
            session=session,
        )
    ds = load_capture_set(CaptureConfig(dir=str(tmp_path)), ("OOK",), 128)
    assert len(ds) == 16 and set(ds.groups) == {"s1", "s2"}
