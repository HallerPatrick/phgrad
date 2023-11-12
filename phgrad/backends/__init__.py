
from typing import Dict, Type, Callable
from argparse import Namespace
import importlib

from functools import lru_cache


def apply_tensorfication(
    fn: Callable, tensor_type: Type, backend_tensor_type: Type
) -> Callable:
    def wrapper(*args, **kwargs):
        val = fn(*args, **kwargs)

        if isinstance(val, tuple):
            fn_result, ctx, is_differentiable = val
        else:
            # Factory operation, has no context
            return val

        # TODO: What should we check here?
        # assert isinstance(
        #     fn_result, backend_tensor_type
        # ), f"Function {fn} must return a {backend_tensor_type}, got {type(fn_result)}"

        if isinstance(args[0], tensor_type):
            tensor = tensor_type(
                fn_result, requires_grad=args[0].requires_grad, _backend=args[0].backend
            )
        elif len(args) > 1 and isinstance(args[1], tensor_type):
            tensor = tensor_type(
                fn_result, requires_grad=args[1].requires_grad, _backend=args[1].backend
            )
        else:
            # TODO: Do we need check for is_differentiable if we passed another tensor to op?
            tensor = tensor_type(fn_result, requires_grad=is_differentiable)

        if is_differentiable:
            tensor.ctx = ctx
        return tensor

    return wrapper

class BackendNamespace(Namespace):
    def __getattr__(self, item):
        """Forward all unknown attributes to the backend."""
        raise AttributeError(f"Backend '{self.name}' has no attribute '{item}'")


@lru_cache(maxsize=2)
def backend_from_device(device: str, tensor_type: Type):
    """Load backend ops based on device.

    Args:
        device (str): Device to load backend for.
        tensor_type (Type): Tensor type to use (Cant import Tensor directly, due to circular imports)

    Returns:
        Namespace: Backend namespace.

    Note:
        * This function is cached to prevent loading backend multiple times.
        * We lose type hints for backend functions, but we provide them in
        the Tensor frontend anyway.
    """

    assert device in ["cpu", "cuda"], f"Unknown device {device}"
    
    backend = importlib.import_module(f"phgrad.backends.{device}")
    backend_ops: Dict[str, Callable] = backend.ops_map
    for attr, func in backend_ops.items():
        backend_ops[attr] = apply_tensorfication(func, tensor_type, backend.BackendTensor)

    factories = backend.factories
    for attr, func in backend.factories.items():
        factories[attr] = apply_tensorfication(func, tensor_type, backend.BackendTensor)

    backend_namespace = BackendNamespace(**backend_ops, **factories, **backend.funcs, name=device)
    return backend_namespace


__all__ = ["backend_from_device"]
