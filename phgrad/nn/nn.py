from typing import Optional

import numpy as np

from phgrad.engine import Tensor
from phgrad.init import he_initialization

from .base import Module, Parameter


class Linear(Module):
    def __init__(self, inp_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.weights = Parameter(
            he_initialization((output_dim, inp_dim)).astype(np.float32)
        )

        self.biases: Optional[Tensor] = None

        if bias:
            self.biases = Parameter(he_initialization((output_dim,)).astype(np.float32))

    def forward(self, t: Tensor) -> Tensor:
        if self.biases is None:
            return t.matmul(self.weights.T)

        return t.matmul(self.weights.T) + self.biases

    def __repr__(self) -> str:
        return f"Linear(in_dim={self.weights.shape[1]}, out_dim={self.weights.shape[0]}, bias={self.biases is not None})"


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

    def forward(self, t: Tensor) -> Tensor:
        t = self.l1(t)
        t = t.relu()
        t = self.l2(t)
        return t


class Dropout(Module):
    """Dropout layer."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, t: Tensor) -> Tensor:
        return t.dropout(self.p, self.training)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = Parameter(he_initialization((num_embeddings, embedding_dim)))

    def forward(self, indexes: Tensor) -> Tensor:
        return self.embeddings.take(indexes)
