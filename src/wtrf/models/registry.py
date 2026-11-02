"""Model name -> constructor. Every model maps (batch, 2, N) to logits over the class list."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from wtrf.models.cnn import CNN
from wtrf.models.resnet import ResNet1d
from wtrf.models.transformer import PatchTransformer

MODELS: dict[str, Callable[..., nn.Module]] = {
    "cnn": CNN,
    "resnet": ResNet1d,
    "transformer": PatchTransformer,
}


def build_model(name: str, n_classes: int, n_samples: int, **params: Any) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](n_classes=n_classes, n_samples=n_samples, **params)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
