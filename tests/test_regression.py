"""Golden-set regression: accuracy on a few labelled real captures must stay above a floor.

Freeze the golden set once phase 3 captures exist (see README). Until then this skips.
tests/golden/floor.json: {"run": "<path to run dir>", "floor": 0.8}
tests/golden/*.sigmf-meta + .sigmf-data: the labelled captures.
"""

import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.skipif(not (GOLDEN / "floor.json").exists(), reason="golden set not frozen yet")
def test_golden_captures_stay_above_floor():
    from wtrf.data.dataset import load_capture_set
    from wtrf.models.train import load_run

    spec = json.loads((GOLDEN / "floor.json").read_text())
    run = load_run(GOLDEN / spec["run"])
    cfg = run.cfg.captures.model_copy(update={"dir": str(GOLDEN)})
    real = load_capture_set(cfg, run.classes, run.n_samples)
    assert len(real) > 0
    acc = float(np.mean(run.probabilities(real.x).argmax(1) == real.y))
    assert acc >= spec["floor"], f"golden accuracy {acc:.3f} fell below floor {spec['floor']}"
