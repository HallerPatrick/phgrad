from typing import Optional

from phgrad.engine import Tensor
from phgrad.init import he_initialization

from .base import Module

class Linear(Module):
    def __init__(self, inp_dim: int, output_dim: int, bias: bool = True, device="cpu"):
        super().__init__(device=device)
        self.weights = Tensor(he_initialization((output_dim, inp_dim)), device=device)

        self.biases: Optional[Tensor] = None

        if bias:
            self.biases = Tensor(he_initialization((output_dim,)), device)

    def forward(self, t: Tensor) -> Tensor:
        if self.biases is None:
            return t.matmul(self.weights.T)
        return t.matmul(self.weights.T) + self.biases

class MLP(Module):
    """A simple Multi Layer Perceptron."""

    def __init__(
        self, inp_dim: int, hidden_size: int, output_dim: int, bias: bool = True, device="cpu"
    ):
        super().__init__(device=device)
        self.l1 = Linear(inp_dim, hidden_size, bias=bias, device=device)
        self.l2 = Linear(hidden_size, output_dim, bias=bias, device=device)

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
        self.embeddings = Tensor(he_initialization((num_embeddings, embedding_dim)))

    def forward(self, indexes: Tensor) -> Tensor:
        dims = list(indexes.shape)
        dims.append(self.embedding_dim)
        flatten_indexes = indexes.flatten()
        return self.embeddings.take(flatten_indexes).reshape(dims)

