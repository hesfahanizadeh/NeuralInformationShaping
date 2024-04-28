from abc import ABC, abstractmethod
from typing import List, Iterable

import torch.nn as nn
import torch


# Encoder class
class Encoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.in_dim = None
        self.out_dim = None
        self.hidden_dims = None

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class DenseEncoder(Encoder):
    """
    This class represents a dense encoder module.
    """
    def __init__(self, in_dim, hidden_dims, out_dim, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        prev_size = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(nn.ReLU())
            prev_size = hidden_dim

        layers.append(nn.Linear(prev_size, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.layers(x)
        return x


class DenseEncoder2(Encoder):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super(DenseEncoder2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        prev_size = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(nn.ReLU())
            prev_size = hidden_dim

        layers.append(nn.Linear(prev_size, out_dim))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class DenseEncoder3(Encoder):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], out_dim: int):
        super(DenseEncoder3, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        prev_size = in_dim

        for hidden_size in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, out_dim))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class FlexibleDenseEncoder(Encoder):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], out_dim: int) -> None:
        super(FlexibleDenseEncoder, self).__init__()
        self.in_size = in_dim
        self.out_size = out_dim

        layers = []
        prev_size = in_dim

        for hidden_size in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, out_dim))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


# Encoder class
class MI_CalculatorModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.in_dim = None
        self.hidden_dims = None

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass


class MINE_Model(MI_CalculatorModel):
    def __init__(self, in_dim, hidden_dims):
        super(MINE_Model, self).__init__()
        self.input_sizes = in_dim
        self.hidden_sizes = hidden_dims

        layers = []
        prev_size = in_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_size, hidden_dim))
            layers.append(nn.ReLU())
            prev_size = hidden_dim

        layers.append(nn.Linear(prev_size, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, *inputs) -> torch.Tensor:
        x: torch.Tensor
        input_list: List[torch.Tensor] = list()
        for x in inputs:
            input_list.append(x.float().view(x.size(0), -1))
        cat = torch.cat(input_list, -1)
        return self.layers(cat)
