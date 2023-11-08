
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
