"""Define the operations that can be applied to tensors.

This is more or less a copy of the CPU backend but uses cupy instead of numpy.
"""

from typing import Any, List, Tuple, Optional, Type, Union

from functools import partial

try:
    import cupy as cp
except ImportError:
    raise ImportError("cupy not installed (pip install cupy-cuda12x)")

import numpy as np

BackendTensor = cp.ndarray

def init_data(data: Any):
    if isinstance(data, cp.ndarray):
        return data
    try:
        data = cp.array(data)
    except Exception as error:
        raise ValueError(f"Cannot convert {type(data)} to GPU tensor (cupy). {error}")
    
    return data

def copy(tensor: BackendTensor) -> BackendTensor:
    return cp.copy(tensor)

def to_dtype(tensor: BackendTensor, dtype: Type) -> BackendTensor:
    return tensor.astype(dtype)


class CudaFunction:
    """Our GPU (CUDA) backend. Mostly based on cupy"""

    __slots__ = ("prev", "forward_context")

    differentiable = True

    def __init__(self, *tensors: Tuple[cp.ndarray]):
        self.prev = tensors
        self.forward_context: List[Tuple[cp.ndarray]] = []

    def save_forward_context(self, *tensors: Tuple[cp.ndarray]):
        return self.forward_context.extend(tensors)
    
    @staticmethod
    def apply(self_, arg, *x, **kwargs):

        # We need to check for the type of the first argument
        if isinstance(arg, CudaFunction):
            op_function: "CudaFunction" = arg
            x = [self_] + list(x)
        else:
            op_function: "CudaFunction" = self_
            x = [arg] + list(x)

        ctx = op_function(*x)

        # Why are we even converting to a tensor in the first place?
        passing_args = []
        for t in x:
            if hasattr(t, "data"):
                passing_args.append(t.data)
            else:
                passing_args.append(t)
        
        ret = op_function.forward(ctx, *passing_args, **kwargs) # type: ignore
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
        out (cp.ndarray): The gradient of the output tensor.
        input_shape (Tuple[int]): The shape of the input tensor.

    Returns:
        cp.ndarray: The gradient tensor reduced to the original tensor shape.
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


class Add(CudaFunction):
    """Addition function.

    Function:
    f(x, y) = x + y
    d/dx f(x, y) = 1
    d/dy f(x, y) = 1
    """

    @staticmethod
    def forward(ctx, self: cp.ndarray, tensor: cp.ndarray) -> cp.ndarray:
        """Addition of two tensors."""
        ctx.save_forward_context(self, tensor)
        return self + tensor

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        x, y = ctx.forward_context
        return unbroadcast(grad_output, x.shape), unbroadcast(grad_output, y.shape)


class Mul(CudaFunction):
    """Multiplication function.

    Function:
    f(x, y) = x * y
    d/dx f(x, y) = y
    d/dy f(x, y) = x
    """

    @staticmethod
    def forward(ctx, self: cp.ndarray, tensor: cp.ndarray) -> cp.ndarray:
        """Multiplication of two tensors."""
        ctx.save_forward_context(self, tensor)
        return self * tensor

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        x, y = ctx.forward_context
        return unbroadcast(grad_output * y, x.shape), unbroadcast(
            grad_output * x, y.shape
        )


class Sub(CudaFunction):
    """Subtraction function.

    Function:
    f(x, y) = x - y
    d/dx f(x, y) = 1
    d/dy f(x, y) = -1
    """

    @staticmethod
    def forward(ctx, self: cp.ndarray, tensor: cp.ndarray) -> cp.ndarray:
        """Subtraction of two tensors."""
        ctx.save_forward_context(self, tensor)
        return self - tensor

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        x, y = ctx.forward_context
        return unbroadcast(grad_output, x.shape), unbroadcast(-grad_output, y.shape)


class Div(CudaFunction):
    """Division function.

    Function:
    f(x, y) = x / y
    d/dx f(x, y) = 1 / y
    d/dy f(x, y) = -x / y^2
    """

    @staticmethod
    def forward(ctx, self: cp.ndarray, tensor: cp.ndarray) -> cp.ndarray:
        """Division of two tensors."""
        ctx.save_forward_context(self, tensor)
        return self / tensor

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
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
#     def backward(ctx, grad_output: cp.ndarray):
#         return grad_output * ctx.forward_context[1] * ctx.forward_context[
#             0
#         ] ** (ctx.forward_context[1] - 1), grad_output * ctx.forward_context[
#             0
#         ] ** ctx.forward_context[
#             1
#         ] * cp.log(
#             ctx.forward_context[0]
#         )


class Exp(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray) -> cp.ndarray:
        """Exponential of a tensor."""
        ret = cp.exp(self.clip(-88, 88))
        ctx.save_forward_context(ret)
        return ret

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        (input,) = ctx.forward_context
        return grad_output * input


class Sum(CudaFunction):
    """Sum function.

    Function:
    f(x) = sum(x)
    d/dx f(x) = 1
    """

    @staticmethod
    def forward(ctx, self: cp.ndarray) -> cp.ndarray:
        """Sum of all elements in a tensor."""
        ctx.save_forward_context(self)
        result = cp.array([self.sum()])
        return result

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        (input_tensor,) = ctx.forward_context
        return grad_output * cp.ones_like(input_tensor)


class Neg(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray) -> cp.ndarray:
        """Negation of a tensor."""
        ctx.save_forward_context(self)
        return -self

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        return -grad_output


class Mean(CudaFunction):
    """Mean function.

    Function:
    f(x) = mean(x)
    d/dx f(x) = 1 / len(x)
    """

    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: Optional[int] = None) -> cp.ndarray:
        """Mean of all elements in a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim

        if dim is None:
            result = cp.array([self.mean()])
        else:
            result = self.mean(axis=dim, keepdims=True)

        return result

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        (input_tensor,) = ctx.forward_context
        dim = ctx.dim

        if dim is not None:
            shape = cp.array(input_tensor.shape)
            shape[dim] = 1
            grad = grad_output / cp.prod(shape)
            return cp.broadcast_to(grad, input_tensor.shape)

        return grad_output * cp.ones_like(input_tensor) / len(input_tensor)


class Max(CudaFunction):
    """Max function.

    Function:
    f(x) = max(x)
    d/dx f(x) = 1
    """

    @staticmethod
    def forward(ctx, self: cp.ndarray) -> cp.ndarray:
        """Max of all elements in a tensor."""
        ctx.save_forward_context(self)
        result = cp.array([self.max()])
        return result

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        (input_tensor,) = ctx.forward_context
        return grad_output * cp.ones_like(input_tensor)


class MatMul(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray, tensor: cp.ndarray) -> cp.ndarray:
        """Matrix multiplication of two tensors."""
        ctx.save_forward_context(self, tensor)
        # TODO: Handle overflow
        return self @ tensor

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        input, weight = ctx.forward_context
        grad_input = grad_output @ cp.swapaxes(weight, -2, -1)
        grad_weight = cp.swapaxes(input, -2, -1) @ grad_output
        return grad_input, grad_weight


class Log(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray) -> cp.ndarray:
        """Log of a tensor."""
        ctx.save_forward_context(self)
        return cp.log(self)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        (input,) = ctx.forward_context
        return grad_output / input


class LogSoftmax(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        """Log softmax of a tensor."""
        # TODO: Optimize this like the cpu backend
        ctx.save_forward_context(self)
        ctx.dim = dim

        # Shift the input for numerical stability
        x_max = self.max(axis=dim, keepdims=True)
        # TODO: Handle nan values
        shifted_logits = self - x_max
        log_softmax_output = shifted_logits - cp.log(
            cp.sum(cp.exp(shifted_logits), axis=dim, keepdims=True)
        )

        return log_softmax_output

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        """Backward pass for log softmax."""
        (input,) = ctx.forward_context
        dim = ctx.dim

        # Compute softmax
        x_max = input.max(axis=dim, keepdims=True)
        shifted_logits = input - x_max
        softmax_output = cp.exp(shifted_logits) / cp.sum(
            cp.exp(shifted_logits), axis=dim, keepdims=True
        )

        # Compute gradient
        grad_input = grad_output - softmax_output * cp.sum(
            grad_output, axis=dim, keepdims=True
        )

        return grad_input


class Softmax(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        """Softmax of a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim
        max_val = cp.max(self, axis=dim, keepdims=True)
        exps = cp.exp(self - max_val)
        ctx.exps = exps
        return exps / exps.sum(axis=dim)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        dim = ctx.dim
        exps = ctx.exps
        return grad_output * exps * (1 - exps.sum(axis=dim))


class ReLU(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray) -> cp.ndarray:
        """ReLU of a tensor."""
        ctx.save_forward_context(self)
        return cp.maximum(self, 0)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        (input,) = ctx.forward_context
        return grad_output * (input > 0)

class Sigmoid(CudaFunction):

    @staticmethod
    def forward(ctx, self: cp.ndarray) -> cp.ndarray:
        """Sigmoid of a tensor."""
        ctx.save_forward_context(self)
        result =  1 / (1 + cp.exp(-self))
        ctx.result = result
        return result

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        result = ctx.result
        return grad_output  * result * (1 - result)


class Transpose(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray, order) -> cp.ndarray:
        ctx.save_forward_context(order)
        ctx.order = order
        return cp.transpose(self, order)

    @staticmethod
    def backward(ctx, x):
        # TODO: Resolve np dep
        return cp.transpose(x, tuple(np.argsort(ctx.order)))
        # return cp.transpose(x, tuple(cp.argsort(ctx.order)))


class Reshape(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray, shape: Union[int, Tuple[int]]) -> cp.ndarray:
        ctx.save_forward_context(self.shape)
        return cp.reshape(self, shape)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        input_shape = ctx.forward_context[0]
        return cp.reshape(grad_output, input_shape)

class Flatten(CudaFunction):

    @staticmethod
    def forward(ctx, self: cp.ndarray) -> cp.ndarray:
        ctx.save_forward_context(self.shape)
        return cp.reshape(self, (self.shape[0], -1))

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        input_shape = ctx.forward_context[0]
        return cp.reshape(grad_output, input_shape)


class Take(CudaFunction):
    """Take function without a dimension parameter.

    NOTE: The current implementation only works with flat indices
    """

    @staticmethod
    def forward(ctx, input: cp.ndarray, indices: cp.ndarray) -> cp.ndarray:
        """Take elements from a tensor using indices."""
        ctx.save_forward_context(input, indices)
        # Assume indices are for the first dimension
        return input[indices]

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        input, indices = ctx.forward_context
        grad_input = cp.zeros_like(input, dtype=cp.float32)

        # TODO: Should we be concerned if indices is a memoryview?
        if isinstance(indices, memoryview):
            indices = indices.tolist()

        # Iterate over each index and add the corresponding gradient
        for idx, grad in zip(indices, grad_output):
            grad_input[idx] += grad

        return grad_input

class Dropout(CudaFunction):

    @staticmethod
    def forward(ctx, self: cp.ndarray, p: float, training: bool) -> cp.ndarray:
        """Dropout function."""
        ctx.save_forward_context(p, training)
        if training:
            mask = cp.random.binomial(1, p, size=self.shape)
        else:
            mask = None

        ctx.mask = mask
        
        if training:
            return self * mask
        else:
            return self

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        _, training = ctx.forward_context

        if training:
            return grad_output * ctx.mask
        else:
            return grad_output

class Cat(CudaFunction):

    @staticmethod
    def forward(ctx, self: cp.ndarray, tensors: Tuple[cp.ndarray], dim: int = 0):
        assert isinstance(tensors, tuple), "Tensors must be a tuple"
        ctx.save_forward_context(self, tensors)
        all_tensors = [self, *tensors]
        ctx.shapes = [t.shape for t in all_tensors]
        ctx.axis = dim
        
        # NOTE: We are passing the tensor object down into the backend, I dont think we should do that
        data = []
        for t in all_tensors:
            if isinstance(t, cp.ndarray):
                data.append(t)
            else:
                data.append(t.data)
        return cp.concatenate(data, axis=ctx.axis)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        grads = cp.split(grad_output, cp.cumsum(ctx.shapes)[:-1], axis=ctx.axis)
        return tuple(grads)
    
class ArgMax(CudaFunction):

    differentiable = False

    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        return cp.argmax(self, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        raise RuntimeError("ArgMax is not differentiable")

# Factories
def eye(n: int, m: Optional[int] = None) -> cp.ndarray:
    """Create an identity matrix.

    Args:
        n (int): The number of rows.
        m (int): The number of columns.
        requires_grad (bool, optional): Whether the tensor requires a gradient. Defaults to False.

    Returns:
        Tensor: The identity matrix.
    """
    return cp.eye(n, m)

def ones(shape: Tuple[int]) -> cp.ndarray:
    """Create a tensor of ones.

    Args:
        shape (Tuple[int]): The shape of the tensor.
        requires_grad (bool, optional): Whether the tensor requires a gradient. Defaults to False.

    Returns:
        Tensor: The tensor of ones.
    """
    return cp.ones(shape)

def zeros(shape: Tuple[int]) -> cp.ndarray:
    """Create a tensor of zeros.

    Args:
        shape (Tuple[int]): The shape of the tensor.
        requires_grad (bool, optional): Whether the tensor requires a gradient. Defaults to False.

    Returns:
        Tensor: The tensor of zeros.
    """
    return cp.zeros(shape)

def attach_op(function: Type[CudaFunction]):
    return partial(function.apply, function)

funcs = {
    "init_data": init_data,
    "copy": copy,
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
    "sigmoid": attach_op(Sigmoid),
    "dropout": attach_op(Dropout),
    "take": attach_op(Take),
    "transpose": attach_op(Transpose),
    "reshape": attach_op(Reshape),
    "flatten": attach_op(Flatten),
    "cat": attach_op(Cat),

    "argmax": attach_op(ArgMax),
}

factories = {
    "eye": eye,
    "ones": ones,
    "zeros": zeros,
}

