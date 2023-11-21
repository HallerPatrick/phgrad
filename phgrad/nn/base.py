from typing import Any

from phgrad.engine import Tensor


class Module:
    """Base class for all modules."""

    def __init__(self, device="cpu"):
        self.training = True
        self.device = device

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def modules(self):
        """Return all first level modules."""
        for module in self.__dict__.values():
            if isinstance(module, Module):
                yield module

    def _parameters(self):
        for parameter in self.__dict__.values():
            if issubclass(type(parameter), Module):
                yield from parameter.parameters()

            if isinstance(parameter, Tensor):
                yield parameter

    def to(self, device):
        """Move all parameters to the specified device."""
        for parameter in self._parameters():
            parameter.to(device, in_place=True)

        return self

    def train(self):
        """Set the module in training mode.
        As far as I am concerned, this is only useful for Dropout or some other
        layers that behave differently during training and evaluation and
        has nothing to do with saving gradients.
        """
        self.training = True

        for module in self.modules():
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules():
            module.eval()

    def parameters(self):
        """Return all parameters of the module."""
        return list(self._parameters())
