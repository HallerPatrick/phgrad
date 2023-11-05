from typing import List

from .engine import PensorTensor

class SGD:

    def __init__(self, params: List[PensorTensor], lr=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None