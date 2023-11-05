from functools import partialmethod
from typing import List, Tuple

import numpy.typing as npt
import numpy as np

from .engine import PensorTensor


class Function:
    def __init__(self, *tensors: Tuple[npt.NDArray]) -> None:
        self.prev: Tuple[npt.NDArray] = tensors
        self._precomputed_tensors: List[Tuple[npt.NDArray]] = []

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        raise NotImplementedError

    def save_precomputed_tensors(self, *tensors: Tuple[npt.NDArray]) -> None:
        return self._precomputed_tensors.extend(tensors)

    def apply(self, arg, *x, **kwargs):
        # support the args in both orders
        if isinstance(arg, PensorTensor):
            op: "Function" = self
            x = [arg] + list(x)
        else:
            op: "Function" = arg
            x = [self] + list(x)
        tt = x[0]

        converted_x = []
        for arg in x:
            if isinstance(arg, PensorTensor):
                # TODO: check dtype, and what types can be used in combination
                # if arg.dtype != tt.dtype:
                #     raise TypeError(
                #         f"Cannot apply {op} to tensors of different dtypes: {tt.dtype} and {arg.dtype}"
                #     )
                converted_x.append(arg)
            else:
                converted_x.append(PensorTensor(np.array([arg], dtype=tt.dtype), requires_grad=False))
        ctx = op(*converted_x)
        ret = PensorTensor(op.forward(ctx, *[t.data for t in converted_x], **kwargs))
        if ret.requires_grad:
            ret.ctx = ctx
        return ret
    
    def __str__(self) -> str:
        return f"<op.{self.__class__.__name__}>"
    
    def __repr__(self) -> str:
        return self.__str__()


class Add(Function):
    """Addition function.

    Function:
    f(x, y) = x + y
    d/dx f(x, y) = 1
    d/dy f(x, y) = 1
    """

    @staticmethod
    def forward(ctx, *args, **_):
        """Addition of two tensors."""
        ctx.save_precomputed_tensors(*args)
        return args[0] + args[1]

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        return grad_output, grad_output


class Sum(Function):
    """Sum function.

    Function:
    f(x) = sum(x)
    d/dx f(x) = 1
    """

    @staticmethod
    def forward(ctx, args, **_):
        """Sum of all elements in a tensor."""
        ctx.save_precomputed_tensors(args)
        result = np.array([args.sum()])
        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input_tensor,) = ctx._precomputed_tensors
        return grad_output * np.ones_like(input_tensor)
    
class Mean(Function):
    """Mean function.
    
    Function:
    f(x) = mean(x)
    d/dx f(x) = 1 / len(x)
    """

    @staticmethod
    def forward(ctx, args, **_):
        """Mean of all elements in a tensor."""
        ctx.save_precomputed_tensors(args)
        result = np.array([args.mean()])
        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input_tensor,) = ctx._precomputed_tensors
        return grad_output * np.ones_like(input_tensor) / len(input_tensor)
    
class Max(Function):
    """Max function.
    
    Function:
    f(x) = max(x)
    d/dx f(x) = 1
    """

    @staticmethod
    def forward(ctx, args, **_):
        """Max of all elements in a tensor."""
        ctx.save_precomputed_tensors(args)
        result = np.array([args.max()])
        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input_tensor,) = ctx._precomputed_tensors
        return grad_output * np.ones_like(input_tensor)


class Mul(Function):
    """Multiplication function.

    Function:
    f(x, y) = x * y
    d/dx f(x, y) = y
    d/dy f(x, y) = x
    """

    @staticmethod
    def forward(ctx, *args, **_):
        """Multiplication of two tensors."""
        ctx.save_precomputed_tensors(*args)
        return args[0] * args[1]

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        return (
            grad_output * ctx._precomputed_tensors[1],
            grad_output * ctx._precomputed_tensors[0],
        )


class Sub(Function):
    """Subtraction function.

    Function:
    f(x, y) = x - y
    d/dx f(x, y) = 1
    d/dy f(x, y) = -1
    """

    @staticmethod
    def forward(ctx, *args, **_):
        """Subtraction of two tensors."""
        ctx.save_precomputed_tensors(*args)
        return args[0] - args[1]

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        return grad_output, -grad_output


class Neg(Function):
    @staticmethod
    def forward(ctx, *args, **_):
        """Negation of a tensor."""
        ctx.save_precomputed_tensors(*args)
        return -args[0]

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        return -grad_output


class Div(Function):
    """Division function.

    Function:
    f(x, y) = x / y
    d/dx f(x, y) = 1 / y
    d/dy f(x, y) = -x / y^2
    """

    @staticmethod
    def forward(ctx, *args, **_):
        """Division of two tensors."""
        ctx.save_precomputed_tensors(*args)
        return args[0] / args[1]

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        return (
            grad_output / ctx._precomputed_tensors[1],
            -grad_output
            * ctx._precomputed_tensors[0]
            / ctx._precomputed_tensors[1] ** 2,
        )


# class Pow(Function):
#     @staticmethod
#     def forward(ctx, *args, **_):
#         """Power of two tensors."""
#         ctx.save_precomputed_tensors(*args)
#         return args[0] ** args[1]

#     @staticmethod
#     def backward(ctx, grad_output: npt.NDArray):
#         return grad_output * ctx._precomputed_tensors[1] * ctx._precomputed_tensors[
#             0
#         ] ** (ctx._precomputed_tensors[1] - 1), grad_output * ctx._precomputed_tensors[
#             0
#         ] ** ctx._precomputed_tensors[
#             1
#         ] * np.log(
#             ctx._precomputed_tensors[0]
#         )


class MatMul(Function):
    @staticmethod
    def forward(ctx, *args, **_):
        """Matrix multiplication of two tensors."""
        ctx.save_precomputed_tensors(*args)
        return args[0] @ args[1]

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        input, weight = ctx._precomputed_tensors
        grad_input = grad_output @ np.swapaxes(weight, -2, -1)
        grad_weight = np.swapaxes(input, -2, -1) @ grad_output
        return grad_input, grad_weight

Dot = MatMul

class Log(Function):
    @staticmethod
    def forward(ctx, *args, **_):
        """Log of a tensor."""
        ctx.save_precomputed_tensors(*args)
        return np.log(args[0])

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx._precomputed_tensors
        return grad_output / input


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, *args, **_):
        """Log softmax of a tensor."""
        ctx.save_precomputed_tensors(*args)
        x = args[0]
        x_off = x - np.max(x)
        ctx.x_off = x_off
        return x_off - np.log(np.exp(x_off).sum())

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx._precomputed_tensors
        x_off = ctx.x_off
        return grad_output - np.exp(x_off) * grad_output.sum()

class Softmax(Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Softmax of a tensor."""
        ctx.save_precomputed_tensors(*args)
        x = args[0]
        return np.exp(x) / np.exp(x).sum()

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx._precomputed_tensors
        return grad_output * np.exp(input) * (1 - np.exp(input).sum())


class ReLU(Function):
    @staticmethod
    def forward(ctx, *args, **_):
        """ReLU of a tensor."""
        ctx.save_precomputed_tensors(*args)
        x = args[0]
        return np.maximum(x, 0)

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx._precomputed_tensors
        return grad_output * (input > 0)
    
class Transpose(Function):

    @staticmethod
    def forward(ctx, x, order):
        ctx.save_precomputed_tensors(order)
        # TODO: Not sure how to handle this
        order = order[0]
        ctx.order = order
        # Float to int
        order = [int(o) for o in order]
        return np.transpose(x, order)

    @staticmethod
    def backward(ctx, x):
        order = ctx._precomputed_tensors[0]
        return np.transpose(x, tuple(np.argsort(ctx.order)))
    
class Take(Function):
    """Take function.
    
    Function:
    f(x, y) = x.take(y)
    d/dx f(x, y) = 1
    d/dy f(x, y) = 1
    """
    @staticmethod
    def forward(ctx: "Function", *args, **kwargs):
        """Take of a tensor."""
        ctx.save_precomputed_tensors(*args)
        axis = kwargs.get("dim", 0)
        ctx.axis = axis
        x = args[0]
        result = np.take(x, args[1], axis=axis)
        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input, indices) = ctx._precomputed_tensors
        axis = ctx.axis
        grad_input = np.zeros_like(input, dtype=np.float32)

        # Reshape grad_output to match input shape
        num_dims = len(input.shape)
        indices_shape = [slice(None) for _ in range(num_dims)]
        indices_shape[axis] = indices
        indices_shape = tuple(indices_shape)

        np.add.at(grad_input, indices_shape, grad_output)

        return grad_input


def register(name, fxn):
    setattr(PensorTensor, name, partialmethod(fxn.apply, fxn))


register("add", Add)
register("sum", Sum)
register("mean", Mean)
register("mul", Mul)
register("sub", Sub)
register("div", Div)
# register("pow", Pow)
register("dot", Dot)
register("matmul", MatMul)

register("log", Log)
register("log_softmax", LogSoftmax)
register("softmax", Softmax)
register("relu", ReLU)
register("neg", Neg)


register("take", Take)
register("transpose", Transpose)
