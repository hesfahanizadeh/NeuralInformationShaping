from typing import Iterable, List

import torch
from torch import nn


class SimpleClassifier(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dims: Iterable[int], out_dim, dropout_rate=0.1
    ) -> None:
        super(SimpleClassifier, self).__init__()

        layers: List[nn.Module] = []
        prev_size = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_size = hidden_dim

        layers.append(nn.Linear(prev_size, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
