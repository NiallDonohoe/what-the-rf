"""1D ResNet: residual blocks with downsampling between stages, same head as the CNN."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, kernel: int = 5) -> None:
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip: nn.Module = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class ResNet1d(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_samples: int,
        widths: tuple[int, ...] = (32, 64, 128),
        blocks_per_stage: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, widths[0], 7, padding=3, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        c_in = widths[0]
        for i, width in enumerate(widths):
            for b in range(blocks_per_stage):
                stride = 2 if (b == 0 and i > 0) else 1
                stages.append(BasicBlock1d(c_in, width, stride))
                c_in = width
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))
