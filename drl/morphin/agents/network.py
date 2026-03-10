from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
