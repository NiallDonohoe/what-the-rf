import pytest

from wtrf.data.labels import ALL_CLASSES
from wtrf.explain import SIGNATURES, Reliability, explain
from wtrf.features import Features

FEATURES = Features(
    centre_freq_hz=144.8e6, occupied_bandwidth_hz=12_000, snr_db_est=14.2, duration_s=3.5
)
TOP = ["AFSK-FM", "GFSK", "WBFM"]


def make(probs, threshold=0.6, reliability=None):
    return explain(list(zip(TOP, probs, strict=False)), FEATURES, 1312, reliability, threshold)


def test_confident_result():
    out = make([0.81, 0.12, 0.04], reliability=Reliability(14.0, 0.93, 480))
    assert not out.uncertain
    assert out.verdict == "The signal is most likely AFSK-FM."
    assert "81%" in out.confidence_note and "93% accurate at +14 dB" in out.confidence_note
    assert len(out.evidence) == 3 and len(out.alternatives) == 2


def test_evidence_cites_measured_values():
    text = " ".join(make([0.81, 0.12, 0.04]).evidence)
    for value in ("12 kHz", "144.8 MHz", "14.2 dB", "3.50 s", "1312 windows"):
        assert value in text


@pytest.mark.parametrize("probs", [[0.45, 0.35, 0.20], [0.34, 0.33, 0.33]])
def test_low_confidence_and_out_of_distribution_results_say_uncertain(probs):
    out = make(probs)
    assert out.uncertain
    assert out.verdict.startswith("Uncertain")
    assert "uncertain" in out.confidence_note


def test_threshold_comes_from_config_value():
    assert make([0.7, 0.2, 0.1], threshold=0.8).uncertain
    assert not make([0.7, 0.2, 0.1], threshold=0.6).uncertain


def test_every_class_has_a_signature():
    assert set(ALL_CLASSES) <= set(SIGNATURES)


def test_names_modulations_never_uses():
    out = make([0.81, 0.12, 0.04])
    text = " ".join([out.verdict, *out.evidence, out.confidence_note, *out.alternatives])
    for use in ("APRS", "packet", "broadcast", "station"):
        assert use.lower() not in text.lower()
