from __future__ import annotations

import numpy as np
import pytest

from wtrf.config import Config


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def tiny_cfg() -> Config:
    """A config small enough to train in a couple of seconds on a laptop CPU."""
    return Config.model_validate(
        {
            "name": "tiny",
            "seed": 0,
            "model": {"name": "cnn", "params": {"channels": [16, 16], "kernel": 5}},
            "data": {
                "classes": ["BPSK", "AM-DSB", "WBFM"],
                "snrs": [10.0, 18.0],
                "per_class_per_snr": 40,
                "n_samples": 64,
            },
            "train": {"epochs": 2, "batch_size": 32, "device": "cpu"},
        }
    )
