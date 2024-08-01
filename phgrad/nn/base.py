from typing import Any

from phgrad.engine import Tensor

from .utils import string_format_module_tree


class Module:
    """Base class for all modules."""

    def __init__(self):
        self.training = True

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

            if isinstance(parameter, Parameter):
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

    def _build_module_repr(self):
        """Build the string representation of all modules and submodules."""
        module_repr = []
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                module_repr.append(f"{name}={module}")
        return ", ".join(module_repr)

    def __repr__(self):
        return string_format_module_tree(self.__build_module_tree_repr())

    def state_dict(self):
        """Return the state of the module."""
        return self.__build_module_tree_state_dict()

    def __build_module_tree_state_dict(self):
        def _build_module_tree(module: Module, tree: dict):
            for name, submodule in module.__dict__.items():
                if isinstance(submodule, Module):
                    tree[name] = {}
                    _build_module_tree(submodule, tree[name])
                elif isinstance(submodule, Parameter):
                    tree[name] = submodule.data

        tree = {}
        _build_module_tree(self, tree)
        return tree

    def __build_module_tree_repr(self):
        def _build_module_tree(module: Module, tree: dict):
            for name, submodule in module.__dict__.items():
                module_string = module.__class__.__name__
                if isinstance(submodule, Module):
                    tree[f"{module_string}.{name}"] = {}
                    _build_module_tree(submodule, tree[f"{module_string}.{name}"])
                elif isinstance(submodule, Parameter):
                    tree[f"{module_string}.{name}"] = submodule.data.shape

        tree = {}
        _build_module_tree(self, tree)
        return tree


class Parameter(Tensor):
    """A parameter is a tensor that is meant to be learned."""

    def __init__(self, data, requires_grad=True):
        super().__init__(data)
        self.requires_grad = requires_grad
