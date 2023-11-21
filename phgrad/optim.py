from typing import List

import numpy as np

from .engine import Tensor


class Optimizer:
    def __init__(self, params: List[Tensor], *args, **kwargs):
        self.params = [p for p in params if p.requires_grad]

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.grad = 0


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= p.grad * self.lr


class Adam(Optimizer):
    def __init__(
        self, params: List[Tensor], lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # TODO: This is only CPU implementation
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def l1_regularization(parameters: List[Tensor], lambda_reg: float):
    """
    Compute the L1 regularization term.

    Args:
    - parameters (list of numpy arrays): The model parameters (weights).
    - lambda_reg (float): The regularization strength.

    Returns:
    - float: The L1 regularization term to be added to the loss.
    """
    l1_penalty = sum(np.sum(np.abs(param.data)) for param in parameters)
    return lambda_reg * l1_penalty
