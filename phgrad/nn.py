
from typing import Any, List, Optional

import numpy as np

from .engine import PensorTensor

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
        self.weights = PensorTensor(np.random.randn(output_dim, inp_dim))
        if bias:
            self.biases = PensorTensor(np.random.randn(output_dim))
        else:
            self.biases = None

    def forward(self, t: PensorTensor) -> PensorTensor:
        if self.biases is None:
            return t.matmul(self.weights.T)
        return t.matmul(self.weights.T) + self.biases

    def parameters(self):
        if self.biases is None:
            return [self.weights]
        return [self.weights, self.biases]
