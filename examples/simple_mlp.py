import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phgrad.engine import Tensor
from phgrad.nn import Linear, Module

# We now have cuda support!
device = "cuda"


class MLP(Module):
    """A simple Multi Layer Perceptron."""

    def __init__(
        self,
        inp_dim: int,
        hidden_size: int,
        output_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.l1 = Linear(inp_dim, hidden_size, bias=bias)
        self.l2 = Linear(hidden_size, output_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        return x


model = MLP(784, 256, 10).to(device)
x = Tensor(np.random.randn(32, 784), device=device)
y = model(x).mean()
y.backward()
