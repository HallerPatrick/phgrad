"""Define the operations that can be applied to tensors.

Our current CPU implementation is based on numpy, which already provides
a lot of operations. However, we need to define the gradient of these
operations. This is done in this module.

We define a class for each operation, which inherits from the Function
class. This class defines the forward and backward pass of the operation.
The forward pass is called when the operation is applied to a tensor.
The backward pass is called when the gradient is computed.

All functions operate with numpy arrays. Therefore all function signatures
and return types are defined as numpy arrays. 
"""

import time
from typing import Any, List, Tuple, Optional, Type, Union

from functools import partial

import numpy as np

from .. import debug

# from ..types import DType, _DType
from .. import types

BackendTensor = np.ndarray


def init_data(data: Any, type: types.DType) -> BackendTensor:
    backend_type = to_backend_type(type)
    if isinstance(data, np.ndarray):
        if data.dtype == backend_type:
            return data
        return data.astype(backend_type)
    try:
        data = np.array(data, dtype=backend_type)
    except Exception as error:
        raise ValueError(f"Cannot convert {type(data)} to CPU tensor (numpy). {error}")

    return data


def copy(data: BackendTensor) -> BackendTensor:
    return data.copy()


def numpy(data: BackendTensor) -> BackendTensor:
    return data


def to_dtype(data: BackendTensor, dtype: Type) -> BackendTensor:
    return data.astype(dtype)


def to_backend_type(frontend_type: types.DType) -> np.dtype:
    match frontend_type:
        case types.bool:
            return bool
        case types.float32:
            return np.float32
        case types.float64:
            return np.float64
        case types.int8:
            return np.int8
        case types.uint8:
            return np.uint8
        case types.int16:
            return np.int16
        case types.int32:
            return np.int32
        case types.int64:
            return np.int64
        case _:  # noqa
            raise ValueError(f"Unknown dtype {frontend_type}")


class CPUFunction:
    """Our CPU backend. Mostly based on numpy.

    We ensure that all tensor that are passed are unnamed.

    """

    __slots__ = ("prev", "forward_context")

    differentiable = True

    def __init__(self, *tensors: Tuple[np.ndarray]):
        self.prev = tensors
        self.forward_context: List[Tuple[np.ndarray]] = []

    def save_forward_context(self, *tensors: Tuple[np.ndarray]):
        return self.forward_context.extend(tensors)

    @staticmethod
    def apply(self_, arg, *x, **kwargs):
        # We need to check for the type of the first argument
        if isinstance(arg, CPUFunction):
            op_function: "CPUFunction" = arg
            x = [self_] + list(x)
        else:
            op_function: "CPUFunction" = self_
            x = [arg] + list(x)

        ctx = op_function(*x)

        # Why are we even converting to a tensor in the first place?
        passing_args = []
        for t in x:
            if isinstance(t, np.ndarray):
                passing_args.append(t)
            # TODO: Find a better way to find out if this is a tensor
            elif hasattr(t, "data"):
                passing_args.append(t.data)
            else:
                passing_args.append(t)

        if debug.DEBUG == 1:
            debug.func_calls[str(op_function)] += 1
            time_start = time.time()

        ret = op_function.forward(ctx, *passing_args, **kwargs)  # type: ignore

        if debug.DEBUG == 1:
            debug.forward_time[str(op_function)] += time.time() - time_start

        return ret, ctx, ctx.differentiable

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
        out (np.ndarray): The gradient of the output tensor.
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


class Add(CPUFunction):
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
    def backward(ctx, grad_output: np.ndarray):
        x, y = ctx.forward_context
        return unbroadcast(grad_output, x.shape), unbroadcast(grad_output, y.shape)


class Mul(CPUFunction):
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
    def backward(ctx, grad_output: np.ndarray):
        x, y = ctx.forward_context
        return unbroadcast(grad_output * y, x.shape), unbroadcast(
            grad_output * x, y.shape
        )


class Sub(CPUFunction):
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
    def backward(ctx, grad_output: np.ndarray):
        x, y = ctx.forward_context
        return unbroadcast(grad_output, x.shape), unbroadcast(-grad_output, y.shape)


class Div(CPUFunction):
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
    def backward(ctx, grad_output: np.ndarray):
        x, y = ctx.forward_context
        return unbroadcast(grad_output / y, x.shape), unbroadcast(
            -grad_output * x / y**2, y.shape
        )


class MatMul(CPUFunction):
    @staticmethod
    def forward(ctx, input: np.ndarray, weight: np.ndarray) -> np.ndarray:
        ctx.save_forward_context(input, weight)
        return np.matmul(input, weight)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input, weight = ctx.forward_context

        # Calculate grad_input
        if weight.ndim > 1:
            # For multi-dimensional weight, transpose the last two dimensions
            axes = np.arange(weight.ndim)
            axes[-1], axes[-2] = axes[-2], axes[-1]
            grad_input = np.matmul(grad_output, weight.transpose(axes))
        else:
            # For vector weight, simply use the vector as-is
            grad_input = np.matmul(grad_output, weight.T)

        # Calculate grad_weight
        if input.ndim > 1:
            # For multi-dimensional input, transpose the last two dimensions
            axes = np.arange(input.ndim)
            axes[-1], axes[-2] = axes[-2], axes[-1]
            grad_weight = np.matmul(input.transpose(axes), grad_output)
        else:
            # For vector input, simply use the vector as-is
            grad_weight = np.matmul(input.T, grad_output)

        return unbroadcast(grad_input, input.shape), unbroadcast(
            grad_weight, weight.shape
        )


# class Pow(Function):
#     @staticmethod
#     def forward(ctx, *args, **_):
#         """Power of two tensors."""
#         ctx.save_forward_context(*args)
#         return args[0] ** args[1]

#     @staticmethod
#     def backward(ctx, grad_output: np.ndarray):
#         return grad_output * ctx.forward_context[1] * ctx.forward_context[
#             0
#         ] ** (ctx.forward_context[1] - 1), grad_output * ctx.forward_context[
#             0
#         ] ** ctx.forward_context[
#             1
#         ] * np.log(
#             ctx.forward_context[0]
#         )


class Exp(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Exponential of a tensor."""
        ret = np.exp(self.clip(-88, 88))
        ctx.save_forward_context(ret)
        return ret

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        (input,) = ctx.forward_context
        return grad_output * input


class Sum(CPUFunction):
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
    def backward(ctx, grad_output: np.ndarray):
        (input_tensor,) = ctx.forward_context
        return grad_output * np.ones_like(input_tensor)


class Neg(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Negation of a tensor."""
        ctx.save_forward_context(self)
        return -self

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        return -grad_output


class Mean(CPUFunction):
    """Mean function.

    Function:
    f(x) = mean(x)
    d/dx f(x) = 1 / len(x)
    """

    @staticmethod
    def forward(ctx, self: np.ndarray, *, dim: Optional[int] = None) -> np.ndarray:
        """Mean of all elements in a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim

        # if dim is None:
        #     result = np.array([self.mean()])
        # else:
        #     result = self.mean(axis=dim, keepdims=True)
        #
        return self.mean(axis=dim, keepdims=True)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        (input_tensor,) = ctx.forward_context
        dim = ctx.dim

        if dim is not None:
            shape = np.array(input_tensor.shape)
            shape[dim] = 1
            grad = grad_output / np.prod(shape)
            return np.broadcast_to(grad, input_tensor.shape)

        return grad_output * np.ones_like(input_tensor) / len(input_tensor)


class Max(CPUFunction):
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
    def backward(ctx, grad_output: np.ndarray):
        (input_tensor,) = ctx.forward_context
        return grad_output * np.ones_like(input_tensor)


class Log(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Log of a tensor."""
        ctx.save_forward_context(self)
        return np.log(self)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        (input,) = ctx.forward_context
        return grad_output / input


class LogSoftmax(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray, *, dim: int = 0) -> np.ndarray:
        """Log softmax of a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim

        # Shift the input for numerical stability
        x_max = self.max(axis=dim, keepdims=True)
        shifted_logits = np.subtract(self, x_max, out=self, where=~np.isnan(self))

        ctx.softmax_output = np.exp(shifted_logits)
        ctx.softmax_sum = np.sum(ctx.softmax_output, axis=dim, keepdims=True)
        log_softmax_output = shifted_logits - np.log(ctx.softmax_sum)

        return log_softmax_output

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        """Backward pass for log softmax."""
        dim = ctx.dim

        # Compute softmax
        softmax_output = ctx.softmax_output / ctx.softmax_sum

        # Compute gradient
        grad_input = np.subtract(
            grad_output,
            softmax_output * np.sum(grad_output, axis=dim, keepdims=True),
            out=grad_output,
        )
        return grad_input


class Softmax(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray, *, dim: int = 0) -> np.ndarray:
        """Softmax of a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim
        max_val = np.max(self, axis=dim, keepdims=True)
        exps = np.exp(self - max_val)
        return exps / exps.sum(axis=dim, keepdims=True)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input = ctx.forward_context.pop()
        dim = ctx.dim
        return grad_output * np.exp(input) * (1 - np.exp(input).sum(axis=dim))


class ReLU(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """ReLU of a tensor."""
        ctx.save_forward_context(self)
        return np.maximum(self, 0)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input = ctx.forward_context.pop()
        return grad_output * (input > 0)


class Sigmoid(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Sigmoid of a tensor."""
        ctx.save_forward_context(self)
        result = 1 / (1 + np.exp(-self))
        ctx.result = result
        return result

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        return grad_output * ctx.result * (1 - ctx.result)


class TanH(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        """Tanh of a tensor."""
        ctx.save_forward_context(self)
        result = np.tanh(self)
        ctx.result = result
        return result

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        return grad_output * (1 - ctx.result**2)


class GetItem(CPUFunction):
    @staticmethod
    def forward(ctx, tensor: np.ndarray, *, indices) -> np.ndarray:
        """Get item using numpy-style indexing."""
        ctx.save_forward_context(tensor.shape, indices)
        return tensor[indices]

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        indices = ctx.forward_context.pop()
        input_shape = ctx.forward_context.pop()
        grad_input = np.zeros(input_shape, dtype=grad_output.dtype)

        # Placing the gradients back in the positions specified by the slice
        np.add.at(grad_input, indices, grad_output)
        return grad_input


class Transpose(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray, *, order) -> np.ndarray:
        ctx.save_forward_context(self)
        ctx.order = order
        return np.transpose(self, order)

    @staticmethod
    def backward(ctx, x):
        return np.transpose(x, tuple(np.argsort(ctx.order)))


class Reshape(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray, *, shape: Union[int, Tuple[int]]) -> np.ndarray:
        ctx.save_forward_context(self)
        return np.reshape(self, shape)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input = ctx.forward_context.pop()
        return np.reshape(grad_output, input.shape)


class Flatten(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray) -> np.ndarray:
        ctx.save_forward_context(self)
        res = np.reshape(self, (self.shape[0], -1))
        return res

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input = ctx.forward_context.pop()
        return np.reshape(grad_output, input.shape)


class Take(CPUFunction):
    """Take function without a dimension parameter.

    NOTE: The current implementation only works with flat indices
    """

    @staticmethod
    def forward(ctx, self: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Take elements from a tensor using indices."""
        assert (
            indices.dtype == np.int64 or indices.dtype == bool
        ), f"Indices must be of type int64 or bool, got {indices.dtype}"
        ctx.save_forward_context(self)
        ctx.indices = indices
        # Assume indices are for the first dimension
        return self[indices]

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        input = ctx.forward_context.pop()
        grad_input = np.zeros_like(input, dtype=np.float32)

        indices = ctx.indices

        # Iterate over each index and add the corresponding gradient
        for idx, grad in zip(indices, grad_output):
            grad_input[idx] += grad

        return grad_input


class Dropout(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray, *, p: float, training: bool) -> np.ndarray:
        """Dropout function."""
        ctx.save_forward_context(p)
        ctx.training = training
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
        if ctx.training:
            return grad_output * ctx.mask
        else:
            return grad_output


class Cat(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray, tensors: Tuple[np.ndarray], *, dim: int = 0):
        assert isinstance(tensors, tuple), "Tensors must be a tuple"
        ctx.save_forward_context(self, tensors)
        all_tensors = [self, *tensors]
        ctx.shapes = [t.shape for t in all_tensors]
        ctx.axis = dim
        return np.concatenate([t.data for t in all_tensors], axis=ctx.axis)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        # grads = np.split(grad_output, np.cumsum(ctx.shapes)[:-1], axis=ctx.axis)
        # return tuple(grads)

        axis = ctx.axis
        # Calculate the indices for splitting
        split_sizes = [shape[axis] for shape in ctx.shapes]
        split_indices = np.cumsum(split_sizes)[:-1]
        grads = np.split(grad_output, split_indices, axis=axis)
        return tuple(grads)


class Squeeze(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray, *, dim: int = 0):
        ctx.save_forward_context(self)
        ctx.dim = dim
        return np.squeeze(self, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        return np.expand_dims(grad_output, axis=ctx.dim)


class Unsqueeze(CPUFunction):
    @staticmethod
    def forward(ctx, self: np.ndarray, *, dim: int = 0):
        ctx.save_forward_context(self)
        ctx.dim = dim
        return np.expand_dims(self, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        return np.squeeze(grad_output, axis=ctx.dim)


class ArgMax(CPUFunction):
    differentiable = False

    @staticmethod
    def forward(ctx, self: np.ndarray, *, dim: int = 0) -> np.ndarray:
        return np.argmax(self, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        raise RuntimeError("ArgMax is not differentiable")


class CosineSimilarity(CPUFunction):
    @staticmethod
    def forward(
        ctx, self: np.ndarray, tensor: np.ndarray, *, dim: int = 0
    ) -> np.ndarray:
        ctx.save_forward_context(self, tensor)
        ctx.dim = dim

        # Compute cosine similarity
        dot_product = np.sum(self * tensor, axis=dim, keepdims=True)
        norm_a = np.sqrt(np.sum(self**2, axis=dim, keepdims=True))
        norm_b = np.sqrt(np.sum(tensor**2, axis=dim, keepdims=True))
        return dot_product / (norm_a * norm_b)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        self, tensor = ctx.forward_context
        # dim = ctx.dim
        return unbroadcast(grad_output * tensor, self.shape), unbroadcast(
            grad_output * self, tensor.shape
        )


class _Stack(CPUFunction):
    @staticmethod
    def forward(
        ctx, self: np.ndarray, tensors: Tuple[np.ndarray], *, dim: int = 0
    ) -> np.ndarray:
        assert isinstance(tensors, tuple), "Tensors must be a tuple"
        all_tensors = (self,) + tensors
        ctx.dim = dim
        ctx.num_tensors = len(all_tensors)
        ctx.input_shapes = [t.shape for t in all_tensors]
        return np.stack(all_tensors, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        grads = np.split(grad_output, ctx.num_tensors, axis=ctx.dim)

        # Reshape gradients to match input shapes
        reshaped_grads = [
            grad.reshape(shape) for grad, shape in zip(grads, ctx.input_shapes)
        ]

        # First gradient is for 'self', rest are for 'tensors'
        grad_self = reshaped_grads[0]
        grad_tensors = tuple(reshaped_grads[1:])

        return (grad_self, grad_tensors, None)


class Stack(CPUFunction):
    """Stack function.

    Function:
    f(x1, x2, ..., xn) = stack([x1, x2, ..., xn], axis)
    where x1, x2, ..., xn are arrays with the same shape.

    Gradient:
    The gradient with respect to each input is the corresponding slice of the output gradient.
    """

    @staticmethod
    def forward(
        ctx, self: np.ndarray, tensors: List[np.ndarray], dim: int = 0
    ) -> np.ndarray:
        """Stack a list of arrays along a new axis."""
        ctx.save_forward_context(self, *tensors, dim)
        tensor_data = [t.data for t in tensors]
        tensor_data.insert(0, self)
        return np.stack(tensor_data, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        tensors, axis = ctx.forward_context[:-1], ctx.forward_context[-1]
        grads = np.split(grad_output, len(tensors), axis=axis)
        return tuple(
            unbroadcast(grad, tensor.shape) for grad, tensor in zip(grads, tensors)
        )


class Scatter(CPUFunction):
    @staticmethod
    def forward(
        ctx, self: np.ndarray, indices: np.ndarray, values: np.ndarray, axis: int = 0
    ) -> np.ndarray:
        ctx.save_forward_context(self, indices, values, axis)

        # If indices is -1 match the indices with the shape of self
        # and put the values along the axis
        if isinstance(indices, int) and indices == -1:
            breakpoint()
            indices = np.indices(self.shape)
            indices = np.moveaxis(indices, 0, axis)
            breakpoint()

        breakpoint()
        np.put_along_axis(self, indices, values, axis)
        return self

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        self, indices, values, axis = ctx.forward_context
        return np.take(grad_output, indices, axis=axis)


# Factories
def eye(n: int, m: Optional[int] = None) -> np.ndarray:
    """Create an identity matrix.

    Args:
        n (int): The number of rows.
        m (int): The number of columns.
        requires_grad (bool, optional): Whether the tensor requires a gradient. Defaults to False.

    Returns:
        Tensor: The identity matrix.
    """
    return np.eye(n, m)


def ones(shape: Tuple[int]) -> np.ndarray:
    """Create a tensor of ones.

    Args:
        shape (Tuple[int]): The shape of the tensor.
        requires_grad (bool, optional): Whether the tensor requires a gradient. Defaults to False.

    Returns:
        Tensor: The tensor of ones.
    """
    return np.ones(shape)


def zeros(shape: Tuple[int]) -> np.ndarray:
    """Create a tensor of zeros.

    Args:
        shape (Tuple[int]): The shape of the tensor.
        requires_grad (bool, optional): Whether the tensor requires a gradient. Defaults to False.

    Returns:
        Tensor: The tensor of zeros.
    """
    return np.zeros(shape)


def arange(start: int, stop: int, step: int = 1) -> np.ndarray:
    return np.arange(start, stop, step)


def stack(tensors: Tuple[np.ndarray], dim: int = 0) -> np.ndarray:
    return np.stack(tensors, axis=dim)


def attach_op(function: Type[CPUFunction]):
    return partial(function.apply, function)


def scatter_add(
    tensor: BackendTensor,
    indices: BackendTensor,
    values: BackendTensor,
    axis: Optional[int] = None,
):
    return np.put_along_axis(tensor, indices, values, axis)


def move_to_backend(tensor: Any) -> BackendTensor:
    tensor_type = type(tensor).__module__

    if tensor_type == "cupy":
        return tensor.get()

    raise ValueError(f"Cannot move tensor of type {tensor_type} to the backend")


funcs = {
    "init_data": init_data,
    "copy": copy,
    "numpy": numpy,
    "scatter_add": scatter_add,
    "move_to_backend": move_to_backend,
}

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
    "tanh": attach_op(TanH),
    "sigmoid": attach_op(Sigmoid),
    "dropout": attach_op(Dropout),
    "getitem": attach_op(GetItem),
    "take": attach_op(Take),
    "transpose": attach_op(Transpose),
    "reshape": attach_op(Reshape),
    "flatten": attach_op(Flatten),
    "cat": attach_op(Cat),
    "argmax": attach_op(ArgMax),
    "squeeze": attach_op(Squeeze),
    "unsqueeze": attach_op(Unsqueeze),
    "cosine_similarity": attach_op(CosineSimilarity),
    "scatter": attach_op(Scatter),
    # "stack": attach_op(Stack),
}

factories = {
    "eye": eye,
    "ones": ones,
    "zeros": zeros,
    "arange": arange,
    "stack": stack,
}
