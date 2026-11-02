import pytest

from wtrf.data import synth
from wtrf.data.labels import (
    ALL_CLASSES,
    CAPTURE_TO_CANONICAL,
    RADIOML_TO_CANONICAL,
    TAXONOMIES,
    canonical,
    resolve_classes,
)


def test_every_taxonomy_class_is_known_and_generatable():
    for classes in TAXONOMIES.values():
        for cls in classes:
            assert cls in ALL_CLASSES
            assert cls in synth.GENERATORS


def test_mappings_land_on_canonical_classes():
    for table in (RADIOML_TO_CANONICAL, CAPTURE_TO_CANONICAL):
        assert set(table.values()) <= set(ALL_CLASSES)


def test_phase1_is_covered_by_radioml_and_phase3_by_captures():
    assert set(TAXONOMIES["phase1"]) <= set(RADIOML_TO_CANONICAL.values())
    assert set(TAXONOMIES["phase3"]) <= set(CAPTURE_TO_CANONICAL.values())


def test_canonical_and_resolve():
    assert canonical("aprs", "capture") == "AFSK-FM"
    assert canonical("QAM16", "radioml") == "QAM16"
    assert resolve_classes("phase1") == TAXONOMIES["phase1"]
    with pytest.raises(KeyError):
        canonical("not-a-label", "capture")
    with pytest.raises(ValueError):
        resolve_classes(["BPSK", "NOPE"])
