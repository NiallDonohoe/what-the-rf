"""Experiment config: one YAML per experiment, validated here and saved with every run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from wtrf.data.augment import AugmentConfig


class ModelConfig(BaseModel):
    name: str = "cnn"
    params: dict[str, Any] = {}


class SplitConfig(BaseModel):
    val: float = 0.15
    test: float = 0.15


class DataConfig(BaseModel):
    source: Literal["synth", "radioml"] = "synth"
    classes: str | list[str] = "phase1"
    n_samples: int = 128
    # synth only
    snrs: list[float] = Field(default_factory=lambda: [float(s) for s in range(-10, 19, 4)])
    per_class_per_snr: int = 300
    # radioml only
    radioml_path: str = "data/RML2016.10a_dict.pkl"
    split: SplitConfig = SplitConfig()


class CaptureConfig(BaseModel):
    """How real SigMF captures become model input. Used by real_vs_synth, ablation, classify."""

    dir: str = "captures"
    target_rate: float | None = None  # resample to this rate (Hz) before windowing
    hop: int | None = None  # window hop in samples; default = n_samples (no overlap)


class TrainConfig(BaseModel):
    epochs: int = 15
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4
    device: str = "auto"  # auto | cpu | cuda | mps


class EvalConfig(BaseModel):
    snr_low: float = -6.0
    snr_high: float = 10.0
    ece_bins: int = 15


class ExplainConfig(BaseModel):
    threshold: float = 0.6  # below this top-1 probability the explanation says "uncertain"


class Config(BaseModel):
    name: str
    seed: int = 0
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    augment: AugmentConfig = AugmentConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    captures: CaptureConfig = CaptureConfig()
    explain: ExplainConfig = ExplainConfig()
    tracking: Literal["none", "mlflow"] = "none"


def load_config(path: str | Path) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text()))


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
