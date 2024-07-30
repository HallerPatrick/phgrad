
from phgrad import Tensor
from phgrad.nn import Module

class LayerNorm(Module):
    def __init__(self, d_model: int, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = Tensor.ones((d_model, ))
            self.beta = Tensor.zeros((d_model, ))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1)
        # We dont have keepdims so we need to do this
        mean = mean.unsqueeze(-1)
        std = x.std(dim=-1)

