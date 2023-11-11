from typing import Callable, Type
from argparse import Namespace
from . import cpu

from functools import lru_cache

def apply_tensorfication(fn, tensor_type: Type, backend_tensor_type: Type):
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

        tensor = tensor_type(fn_result, requires_grad=is_differentiable)
        if is_differentiable:
            tensor.ctx = ctx
        return tensor

    return wrapper


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

    if device == "cpu":
        ops = cpu.ops_map
        for attr, func in ops.items():
            ops[attr] = apply_tensorfication(func, tensor_type, cpu.BackendTensor)

        factories = cpu.factories
        for attr, func in cpu.factories.items():
            factories[attr] = apply_tensorfication(func, tensor_type, cpu.BackendTensor)

        backend_namespace =  Namespace(**ops, **factories, **cpu.funcs)
        return backend_namespace
    
    raise ValueError(f"Unknown device {device}")

__all__ = ["backend_from_device"]
