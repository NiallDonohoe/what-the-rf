"""Baseline 1D CNN: a few conv blocks, batch norm, global pooling, linear head."""

from __future__ import annotations

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_samples: int,
        channels: tuple[int, ...] = (64, 64, 128, 128),
        kernel: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 2
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
