
from typing import Any, List, Optional

import numpy as np

from .engine import Tensor
from .init import he_initialization

class Module:
    def __init__(self):
        self._parameters = {}

    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class Linear(Module):
    def __init__(self, inp_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.weights = Tensor(he_initialization((output_dim, inp_dim)))
        if bias:
            self.biases = Tensor(he_initialization((output_dim, )))
        else:
            self.biases = None

    def forward(self, t: Tensor) -> Tensor:
        if self.biases is None:
            return t.matmul(self.weights.T)
        return t.matmul(self.weights.T) + self.biases

    def parameters(self):
        if self.biases is None:
            return [self.weights]
        return [self.weights, self.biases]


class MLP(Module):
    """A simple Multi Layer Perceptron."""

    def __init__(self, inp_dim: int, hidden_size: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.l1 = Linear(inp_dim, hidden_size, bias=bias)
        self.l2 = Linear(hidden_size, output_dim, bias=bias)

    def forward(self, t: Tensor) -> Tensor:
        t = self.l1(t)
        # print("PHGRAD: After l1", t.shape, t)
        t = t.relu()
        # print("PHGRAD: After relu", t.shape, t)
        t = self.l2(t)
        # print("PHGRAD: After l2", t.shape, t)
        return t
    
    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()
