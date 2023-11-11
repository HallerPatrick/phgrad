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

from typing import List, Tuple, Optional

from functools import partial

import numpy.typing as npt
import numpy as np

from ..engine import Tensor

class Function:
    """Our CPU backend. Mostly based on numpy"""

    __slots__ = ("prev", "forward_context")

    def __init__(self, *tensors: Tuple[npt.NDArray]) -> None:
        self.prev = tensors
        self.forward_context: List[Tuple[npt.NDArray]] = []

    def save_forward_context(self, *tensors: Tuple[npt.NDArray]) -> None:
        return self.forward_context.extend(tensors)
    
    @staticmethod
    def apply(self, arg, *x, **kwargs):

        if isinstance(arg, Tensor):
            op_function: "Function" = self
            x = [arg] + list(x)
        else:
            op_function: "Function" = arg
            x = [self] + list(x)

        converted_x = []
        # TODO: We should not convert everything blindly to a tensor
        # For now only convert numpy arrays
        for arg in x:
            if isinstance(arg, Tensor):
                # TODO: check dtype, and what types can be used in combination
                # if arg.dtype != tt.dtype:
                #     raise TypeError(
                #         f"Cannot apply {op} to tensors of different dtypes: {tt.dtype} and {arg.dtype}"
                #     )
                converted_x.append(arg)
            elif isinstance(arg, np.ndarray):
                # TODO; We need to find a better way to check for dtypes and verification, for now we dont do it
                converted_x.append(Tensor(np.array(arg), requires_grad=False))
                # converted_x.append(
                #     Tensor(np.array(arg, dtype=tt.dtype), requires_grad=False)
                # )
            else:
                converted_x.append(arg)

        ctx = op_function(*converted_x)

        # Why are we even converting to a tensor in the first place?
        passing_args = []
        for t in converted_x:
            if isinstance(t, Tensor):
                passing_args.append(t.data)
            else:
                passing_args.append(t)
        
        ret = Tensor(op_function.forward(ctx, *passing_args, **kwargs))
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
        ctx.save_forward_context(self, tensor)
        return self + tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        x, y = ctx.forward_context
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
        ctx.save_forward_context(self, tensor)
        return self * tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        x, y = ctx.forward_context
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
        ctx.save_forward_context(self, tensor)
        return self - tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        x, y = ctx.forward_context
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
        ctx.save_forward_context(self, tensor)
        return self / tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        x, y = ctx.forward_context
        return unbroadcast(grad_output / y, x.shape), unbroadcast(
            -grad_output * x / y**2, y.shape
        )


# class Pow(Function):
#     @staticmethod
#     def forward(ctx, *args, **_):
#         """Power of two tensors."""
#         ctx.save_forward_context(*args)
#         return args[0] ** args[1]

#     @staticmethod
#     def backward(ctx, grad_output: npt.NDArray):
#         return grad_output * ctx.forward_context[1] * ctx.forward_context[
#             0
#         ] ** (ctx.forward_context[1] - 1), grad_output * ctx.forward_context[
#             0
#         ] ** ctx.forward_context[
#             1
#         ] * np.log(
#             ctx.forward_context[0]
#         )


class Exp(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Exponential of a tensor."""
        ret = np.exp(self.clip(-88, 88))
        ctx.save_forward_context(ret)
        return ret

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx.forward_context
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
        ctx.save_forward_context(self)
        result = np.array([self.sum()])
        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input_tensor,) = ctx.forward_context
        return grad_output * np.ones_like(input_tensor)


class Neg(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Negation of a tensor."""
        ctx.save_forward_context(self)
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
    def forward(ctx, self: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
        """Mean of all elements in a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim

        if dim is None:
            result = np.array([self.mean()])
        else:
            result = self.mean(axis=dim, keepdims=True)

        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input_tensor,) = ctx.forward_context
        dim = ctx.dim

        if dim is not None:
            shape = np.array(input_tensor.shape)
            shape[dim] = 1
            grad = grad_output / np.prod(shape)
            return np.broadcast_to(grad, input_tensor.shape)

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
        ctx.save_forward_context(self)
        result = np.array([self.max()])
        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input_tensor,) = ctx.forward_context
        return grad_output * np.ones_like(input_tensor)


class MatMul(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        """Matrix multiplication of two tensors."""
        ctx.save_forward_context(self, tensor)
        # TODO: Handle overflow
        return self @ tensor

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        input, weight = ctx.forward_context
        grad_input = grad_output @ np.swapaxes(weight, -2, -1)
        grad_weight = np.swapaxes(input, -2, -1) @ grad_output
        return grad_input, grad_weight


class Log(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Log of a tensor."""
        ctx.save_forward_context(self)
        return np.log(self)

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx.forward_context
        return grad_output / input


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray, dim: int = 0) -> np.ndarray:
        """Log softmax of a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim

        # Shift the input for numerical stability
        x_max = self.max(axis=dim, keepdims=True)
        # TODO: Handle nan values
        shifted_logits = self - x_max
        log_softmax_output = shifted_logits - np.log(
            np.sum(np.exp(shifted_logits), axis=dim, keepdims=True)
        )

        return log_softmax_output

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        """Backward pass for log softmax."""
        (input,) = ctx.forward_context
        dim = ctx.dim

        # Compute softmax
        x_max = input.max(axis=dim, keepdims=True)
        shifted_logits = input - x_max
        softmax_output = np.exp(shifted_logits) / np.sum(
            np.exp(shifted_logits), axis=dim, keepdims=True
        )

        # Compute gradient
        grad_input = grad_output - softmax_output * np.sum(
            grad_output, axis=dim, keepdims=True
        )

        return grad_input


class Softmax(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray, dim: int = 0) -> np.ndarray:
        """Softmax of a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim
        return np.exp(self) / np.exp(self).sum(axis=dim)

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx.forward_context
        dim = ctx.dim
        return grad_output * np.exp(input) * (1 - np.exp(input).sum(axis=dim))


class ReLU(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """ReLU of a tensor."""
        ctx.save_forward_context(self)
        return np.maximum(self, 0)

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        (input,) = ctx.forward_context
        return grad_output * (input > 0)

class Sigmoid(Function):

    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Sigmoid of a tensor."""
        ctx.save_forward_context(self)
        result =  1 / (1 + np.exp(-self))
        ctx.result = result
        return result

    @staticmethod
    def backward(ctx, grad_output: npt.NDArray):
        result = ctx.result
        return grad_output  * result * (1 - result)


class Transpose(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray, order) -> np.ndarray:
        ctx.save_forward_context(order)
        ctx.order = order
        return np.transpose(self, order)

    @staticmethod
    def backward(ctx, x):
        return np.transpose(x, tuple(np.argsort(ctx.order)))


class Reshape(Function):
    @staticmethod
    def forward(ctx, self: np.ndarray, shape: Tuple[int]) -> np.ndarray:
        ctx.save_forward_context(self.shape)
        return np.reshape(self, shape)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input_shape = ctx.forward_context[0]
        return np.reshape(grad_output, input_shape)

class Flatten(Function):

    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        ctx.save_forward_context(self.shape)
        return np.reshape(self, (self.shape[0], -1))

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input_shape = ctx.forward_context[0]
        return np.reshape(grad_output, input_shape)


class Take(Function):
    """Take function without a dimension parameter.

    NOTE: The current implementation only works with flat indices

    """

    @staticmethod
    def forward(ctx, input: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Take elements from a tensor using indices."""
        ctx.save_forward_context(input, indices)
        # Assume indices are for the first dimension
        return input[indices]

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input, indices = ctx.forward_context
        grad_input = np.zeros_like(input, dtype=np.float32)

        # Iterate over each index and add the corresponding gradient
        for idx, grad in zip(indices, grad_output):
            grad_input[idx] += grad

        return grad_input

class Dropout(Function):

    @staticmethod
    def forward(ctx, self: np.ndarray, p: float, training: bool) -> np.ndarray:
        """Dropout function."""
        ctx.save_forward_context(p, training)
        if training:
            mask = np.random.binomial(1, p, size=self.shape)
        else:
            mask = None

        ctx.mask = mask
        
        if training:
            return self * mask
        else:
            return self

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        _, training = ctx.forward_context

        if training:
            return grad_output * ctx.mask
        else:
            return grad_output

class Cat(Function):

    @staticmethod
    def forward(ctx, self: np.ndarray, tensors: Tuple[Tensor], dim: int = 0):
        assert isinstance(tensors, tuple), "Tensors must be a tuple"
        ctx.save_forward_context(self, tensors)
        all_tensors = [self, *tensors]
        ctx.shapes = [t.shape for t in all_tensors]
        return np.concatenate([t.data for t in all_tensors], axis=ctx.axis)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        grads = np.split(grad_output, np.cumsum(ctx.shapes)[:-1], axis=ctx.axis)
        return tuple(grads)


def attach_op(function: Function):
    return partial(function.apply, function)

ops_map = {
    "add": attach_op(Add),
    "sum": attach_op(Sum),
    "neg": attach_op(Neg),
    "mean": attach_op(Mean),
    "max": attach_op(Max),
    "mul": attach_op(Mul),
    "sub": attach_op(Sub),
    "div": attach_op(Div),
    "matmul": attach_op(MatMul),
    "exp": attach_op(Exp),
    "log": attach_op(Log),
    "log_softmax": attach_op(LogSoftmax),
    "softmax": attach_op(Softmax),
    "relu": attach_op(ReLU),
    "sigmoid": attach_op(Sigmoid),
    "dropout": attach_op(Dropout),
    "take": attach_op(Take),
    "transpose": attach_op(Transpose),
    "reshape": attach_op(Reshape),
    "flatten": attach_op(Flatten),
    "cat": attach_op(Cat)
}

