"""Define the operations that can be applied to tensors.

Our current CPU implementation is based on numpy, which already provides
a lot of operations. However, we need to define the gradient of these
operations. This is done in this module.

We define a class for each operation, which inherits from the Function
class. This class defines the forward and backward pass of the operation.
The forward pass is called when the operation is applied to a tensor.
The backward pass is called when the gradient is computed.

All functions operate with numpy arrays. Therefore all function signatures
and return types are defined as numpy arrays. We wrap the function calls 
in a Tensor object, which also stores the gradient and the operation that
created the tensor.

For typing we generate a stub file on the fly, which is used by the type
checker to check the types of the functions. The stub files then contain
the signature with the tensor type instead of the numpy array type.
"""

from typing import List, Tuple

import numpy.typing as npt
import numpy as np

from .engine import Tensor
from .utils import register


class Function:
    def __init__(self, *tensors: Tuple[npt.NDArray]) -> None:
        self.prev = tensors
        self._precomputed_tensors: List[Tuple[npt.NDArray]] = []

    def save_precomputed_tensors(self, *tensors: Tuple[npt.NDArray]) -> None:
        return self._precomputed_tensors.extend(tensors)

    def apply(self, arg, *x, **kwargs):
        # support the args in both orders

        if isinstance(arg, Tensor):
            op_function: "Function" = self
            x = [arg] + list(x)
        else:
            op_function: "Function" = arg
            x = [self] + list(x)
        tt = x[0]

        converted_x = []
        for arg in x:
            if isinstance(arg, Tensor):
                # TODO: check dtype, and what types can be used in combination
                # if arg.dtype != tt.dtype:
                #     raise TypeError(
                #         f"Cannot apply {op} to tensors of different dtypes: {tt.dtype} and {arg.dtype}"
                #     )
                converted_x.append(arg)
            else:
                converted_x.append(
                    Tensor(np.array([arg], dtype=tt.dtype), requires_grad=False)
                )
        ctx = op_function(*converted_x)
        ret = Tensor(op_function.forward(ctx, *[t.data for t in converted_x], **kwargs))
        if ret.requires_grad:
            ret.ctx = ctx
        return ret

    def __str__(self) -> str:
        return f"<op.{self.__class__.__name__}>"

    def __repr__(self) -> str:
        return self.__str__()


def unbroadcast(grad, original_shape):
    """Numpy like any other tensor library does support broadcasting,
    which can happen for any binary operation. This function undoes
    the broadcasting that was done by numpy to have the same shape
    for the gradient as for the input tensor.

    Reduce the gradient tensor to the original tensor shape by
    summing along the broadcasted dimensions.

    Args:
        out (npt.NDArray): The gradient of the output tensor.
        input_shape (Tuple[int]): The shape of the input tensor.

    Returns:
        np.ndarray: The gradient tensor reduced to the original tensor shape.
    """
    # First, we need to expand the original shape to the same number of dimensions as the grad
    # by adding singleton dimensions at the beginning
    shape_diff = len(grad.shape) - len(original_shape)
    padded_original_shape = (1,) * shape_diff + original_shape

    # Identify dimensions that were broadcasted
    axes_to_sum = [
        i
        for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, padded_original_shape))
        if orig_dim == 1 and grad_dim != 1
    ]

    # Sum the gradient along these dimensions
    for axis in sorted(axes_to_sum, reverse=True):
        grad = grad.sum(axis=axis, keepdims=True)

    # Remove singleton dimensions that were added to match the original shape
    if shape_diff > 0:
        grad = grad.reshape(original_shape)

    return grad


class Add(Function):
    """Addition function.

    Function:
    f(x, y) = x + y
    d/dx f(x, y) = 1
    d/dy f(x, y) = 1
    """

    @staticmethod
    def forward(ctx, self: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        """Addition of two tensors."""
        ctx.save_precomputed_tensors(self, tensor)
        return self + tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        x, y = ctx._precomputed_tensors
        return unbroadcast(grad_output, x.shape), unbroadcast(grad_output, y.shape)


class Mul(Function):
    """Multiplication function.

    Function:
    f(x, y) = x * y
    d/dx f(x, y) = y
    d/dy f(x, y) = x
    """

    @staticmethod
    def forward(ctx, self: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        """Multiplication of two tensors."""
        ctx.save_precomputed_tensors(self, tensor)
        return self * tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        x, y = ctx._precomputed_tensors
        return unbroadcast(grad_output * y, x.shape), unbroadcast(
            grad_output * x, y.shape
        )


class Sub(Function):
    """Subtraction function.

    Function:
    f(x, y) = x - y
    d/dx f(x, y) = 1
    d/dy f(x, y) = -1
    """

    @staticmethod
    def forward(ctx, self: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        """Subtraction of two tensors."""
        ctx.save_precomputed_tensors(self, tensor)
        return self - tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        x, y = ctx._precomputed_tensors
        return unbroadcast(grad_output, x.shape), unbroadcast(-grad_output, y.shape)


class Div(Function):
    """Division function.

    Function:
    f(x, y) = x / y
    d/dx f(x, y) = 1 / y
    d/dy f(x, y) = -x / y^2
    """

    @staticmethod
    def forward(ctx, self: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        """Division of two tensors."""
        ctx.save_precomputed_tensors(self, tensor)
        return self / tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        x, y = ctx._precomputed_tensors
        return unbroadcast(grad_output / y, x.shape), unbroadcast(
            -grad_output * x / y**2, y.shape
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


class Exp(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Exponential of a tensor."""
        ret = np.exp(self.clip(-88, 88))
        ctx.save_precomputed_tensors(ret)
        return ret

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx._precomputed_tensors
        return grad_output * input


class Sum(Function):
    """Sum function.

    Function:
    f(x) = sum(x)
    d/dx f(x) = 1
    """

    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Sum of all elements in a tensor."""
        ctx.save_precomputed_tensors(self)
        result = np.array([self.sum()])
        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input_tensor,) = ctx._precomputed_tensors
        return grad_output * np.ones_like(input_tensor)


class Neg(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Negation of a tensor."""
        ctx.save_precomputed_tensors(self)
        return -self

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        return -grad_output


class Mean(Function):
    """Mean function.

    Function:
    f(x) = mean(x)
    d/dx f(x) = 1 / len(x)
    """

    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Mean of all elements in a tensor."""
        ctx.save_precomputed_tensors(self)
        result = np.array([self.mean()])
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
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Max of all elements in a tensor."""
        ctx.save_precomputed_tensors(self)
        result = np.array([self.max()])
        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input_tensor,) = ctx._precomputed_tensors
        return grad_output * np.ones_like(input_tensor)


class MatMul(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        """Matrix multiplication of two tensors."""
        ctx.save_precomputed_tensors(self, tensor)
        return self @ tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        input, weight = ctx._precomputed_tensors
        grad_input = grad_output @ np.swapaxes(weight, -2, -1)
        grad_weight = np.swapaxes(input, -2, -1) @ grad_output
        return grad_input, grad_weight


Dot = MatMul


class Log(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Log of a tensor."""
        ctx.save_precomputed_tensors(self)
        return np.log(self)

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx._precomputed_tensors
        return grad_output / input


# class LogSoftmax(Function):
#     @staticmethod
#     def forward(ctx, self: np.ndarray, dim: int = 0) -> np.ndarray:
#         """Log softmax of a tensor."""
#         ctx.save_precomputed_tensors(self)
#         ctx.dim = dim
#         x_off = self - self.max(dim, keepdims=True)
#         ctx.x_off = x_off
#         return x_off - np.log(np.exp(x_off).sum(dim, keepdims=True))

#     @staticmethod
#     def backward(ctx, grad_output: npt.NDArray):
#         x_off = ctx.x_off
#         dim = ctx.dim
#         return grad_output - np.exp(x_off) * grad_output.sum(dim, keepdims=True)


# class LogSoftmax(Function):
#     @staticmethod
#     def forward(ctx, self: np.ndarray, dim: int = 0) -> np.ndarray:
#         """Log softmax of a tensor."""
#         ctx.save_precomputed_tensors(self)
#         ctx.dim = dim
#         # input_float = self.astype(np.float64)

#         # x_off = input_float - input_float.max(dim, keepdims=True)

#         # # Clipping x_off to prevent overflow in exp
#         # x_off_clipped = np.clip(x_off, -np.inf, np.log(np.finfo(x_off.dtype).max))

#         # softmax_output = np.exp(x_off_clipped) / np.exp(x_off_clipped).sum(
#         #     dim, keepdims=True
#         # )
#         # ctx.softmax_output = softmax_output
#         # # print("Softmax Output:", softmax_output)
#         # return np.log(softmax_output)
#         x_max = self.max(axis=dim, keepdims=True)
#         shifted_logits = self - x_max
#         exp_shifted = np.exp(shifted_logits)
#         log_softmax_result = shifted_logits - np.log(np.sum(exp_shifted, axis=dim, keepdims=True))
#         return log_softmax_result

#     @staticmethod
#     def backward(ctx, grad_output: npt.NDArray):
#         softmax_output = ctx.softmax_output
#         dim = ctx.dim
#         return grad_output - softmax_output * grad_output.sum(dim, keepdims=True)
class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, input: np.ndarray, dim: int = 0) -> np.ndarray:
        """Log softmax of a tensor."""
        ctx.save_precomputed_tensors(input)
        ctx.dim = dim

        # Shift the input for numerical stability
        x_max = input.max(axis=dim, keepdims=True)
        shifted_logits = input - x_max
        log_softmax_output = shifted_logits - np.log(np.sum(np.exp(shifted_logits), axis=dim, keepdims=True))
        
        return log_softmax_output

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        """Backward pass for log softmax."""
        (input,) = ctx._precomputed_tensors
        dim = ctx.dim

        # Compute softmax
        x_max = input.max(axis=dim, keepdims=True)
        shifted_logits = input - x_max
        softmax_output = np.exp(shifted_logits) / np.sum(np.exp(shifted_logits), axis=dim, keepdims=True)

        # Compute gradient
        grad_input = grad_output - softmax_output * np.sum(grad_output, axis=dim, keepdims=True)

        return grad_input

class Softmax(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Softmax of a tensor."""
        ctx.save_precomputed_tensors(self)
        return np.exp(self) / np.exp(self).sum()

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx._precomputed_tensors
        return grad_output * np.exp(input) * (1 - np.exp(input).sum())


class ReLU(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """ReLU of a tensor."""
        ctx.save_precomputed_tensors(self)
        return np.maximum(self, 0)

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx._precomputed_tensors
        return grad_output * (input > 0)


class Transpose(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray, order) -> np.ndarray:
        ctx.save_precomputed_tensors(order)
        # TODO: Not sure how to handle this
        order = order[0]
        ctx.order = order
        # Float to int
        order = [int(o) for o in order]
        return np.transpose(self, order)

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
    def forward(
        ctx: "Function", self: np.ndarray, tensor: np.ndarray, dim=0
    ) -> np.ndarray:
        """Take of a tensor."""
        ctx.save_precomputed_tensors(self, tensor)
        ctx.axis = dim
        return np.take(self, tensor, axis=ctx.axis)

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


def register_tensor_op(name, op):
    register(name, op, Tensor)


# register("pow", Pow)
register_tensor_op("add", Add)
register_tensor_op("sum", Sum)
register_tensor_op("mean", Mean)
register_tensor_op("max", Max)
register_tensor_op("mul", Mul)
register_tensor_op("sub", Sub)
register_tensor_op("div", Div)
register_tensor_op("dot", Dot)
register_tensor_op("matmul", MatMul)
register_tensor_op("exp", Exp)

register_tensor_op("log", Log)
register_tensor_op("log_softmax", LogSoftmax)
register_tensor_op("softmax", Softmax)
register_tensor_op("relu", ReLU)
register_tensor_op("neg", Neg)


register_tensor_op("take", Take)
register_tensor_op("transpose", Transpose)
