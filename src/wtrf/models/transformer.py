"""Small patch-based transformer: patch the IQ sequence, small encoder, mean-pooled head."""

from __future__ import annotations

import torch
from torch import nn


class PatchTransformer(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_samples: int,
        patch: int = 8,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_samples % patch:
            raise ValueError(f"n_samples={n_samples} is not divisible by patch={patch}")
        self.patch = patch
        self.embed = nn.Linear(2 * patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_samples // patch, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=4 * dim, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n = x.shape
        # (B, 2, N) -> (B, N/patch, 2*patch): each token is one patch of I and Q samples.
        tokens = x.reshape(b, c, n // self.patch, self.patch).permute(0, 2, 1, 3).flatten(2)
        h = self.encoder(self.embed(tokens) + self.pos)
        return self.head(self.norm(h.mean(dim=1)))
