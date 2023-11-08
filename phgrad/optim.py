from typing import List

import numpy as np

from .engine import Tensor

class SGD:

    def __init__(self, params: List[Tensor], lr=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

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
