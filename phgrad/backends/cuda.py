"""Define the operations that can be applied to tensors.

This is more or less a copy of the CPU backend but uses cupy instead of numpy.
"""

import time
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type, Union, Dict

try:
    import cupy as cp
except ImportError:
    raise ImportError("cupy not installed (pip install cupy-cuda12x)")

import numpy as np

from .. import debug, types

BackendTensor = cp.ndarray


class CudaKernelCache:
    """Cache for compiled CUDA kernels.

    For now we would always provide the forward and backward functions.
    Therefore we can usually only check for the operation itself and just
    get both kernels.
    """

    def __init__(self):
        self.cached_operations: Dict[str, Dict[str, cp.RawKernel]] = {}

    def is_loaded(self, operation: str) -> bool:
        return operation in self.cached_operations

    def add_kernel(
        self,
        operation: str,
        kernel_name: str,
        kernel: cp.RawKernel,
    ):
        if operation not in self.cached_operations:
            self.cached_operations[operation] = {}
        self.cached_operations[operation][kernel_name] = kernel

    def get_kernel(self, operation: str, kernel_name: str) -> cp.RawKernel:
        return self.cached_operations[operation][kernel_name]

    def __contains__(self, operation: str) -> bool:
        return operation in self.cached_operations


cuda_kernel_cache = CudaKernelCache()


def _load_cuda_kernels(filename: str, kernels: List[str]) -> Any:
    """Load one or more CUDA kernels from a file and compile it.
    All cuda kernels reside in the `cuda_kernels` folder at the root of the
    project.

    Args:
        filename (str): The name of the file containing the CUDA kernels.
        kernels (Tuple[str]): The names of the kernels to load.

    Returns:
        Any: The compiled CUDA kernel.

    Note:
        CuPy caches the compiled kernels, but we still do the IO here. So maybe we
        should at least apply a cache to this function.
    """
    # TODO: There is probably a better way to do this
    cuda_file = (
        Path(__file__).parent.parent.parent / "cuda_kernels" / (filename + ".cu")
    )

    if not cuda_file.exists():
        raise FileNotFoundError(f"Could not find CUDA file {cuda_file}")

    with open(cuda_file, "r") as f:
        cuda_code = f.read()

    for kernel_name in kernels:
        cuda_kernel_cache.add_kernel(
            filename,
            kernel_name,
            cp.RawKernel(cuda_code, kernel_name),
        )


def init_data(data: Any, dtype: Type) -> BackendTensor:
    backend_type = to_backend_type(dtype)

    if isinstance(data, cp.ndarray):
        if data.dtype == backend_type:
            return data
        return data.astype(backend_type)
    try:
        data = cp.array(data, dtype=backend_type)
    except Exception as error:
        raise ValueError(f"Cannot convert {type(data)} to GPU tensor (cupy). {error}")

    return data


def copy(tensor: BackendTensor) -> BackendTensor:
    return cp.copy(tensor)


def to_dtype(tensor: BackendTensor, dtype: Type) -> BackendTensor:
    return tensor.astype(dtype)


def numpy(tensor: BackendTensor) -> np.ndarray:
    return cp.asnumpy(tensor)


def to_backend_type(frontend_type: types.DType) -> cp.dtype:
    match frontend_type:
        case types.bool:
            return cp.bool
        case types.float32:
            return cp.float32
        case types.float64:
            return cp.float64
        case types.int8:
            return cp.int8
        case types.uint8:
            return cp.uint8
        case types.int16:
            return cp.int16
        case types.int32:
            return cp.int32
        case types.int64:
            return cp.int64
        case _:  # noqa
            raise ValueError(f"Unknown dtype {frontend_type}")


class CudaFunction:
    """Our GPU (CUDA) backend. Mostly based on cupy"""

    kernel_op: Optional[str] = None
    kernel_names: Optional[List[str]] = None

    __slots__ = ("prev", "forward_context")

    differentiable = True

    def __init__(self, *tensors: Tuple[cp.ndarray]):
        self.prev = tensors
        self.forward_context: List[Tuple[cp.ndarray]] = []

        if self.kernel_op is not None and self.kernel_names is not None:
            if not cuda_kernel_cache.is_loaded(self.kernel_op):
                _load_cuda_kernels(self.kernel_op, self.kernel_names)

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
            if isinstance(t, cp.ndarray):
                passing_args.append(t)
            elif hasattr(t, "data"):
                passing_args.append(t.data)
            else:
                passing_args.append(t)

        if debug.DEBUG:
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
        out (cp.ndarray): The gradient of the output tensor.
        input_shape (Tuple[int]): The shape of the input tensor.

    Returns:
        cp.ndarray: The gradient tensor reduced to the original tensor shape.
    """
    if debug.DEBUG == 1:
        start_time = time.time()
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

    if debug.DEBUG == 1:
        end_time = time.time()
        debug.backward_time["unbroadcast"] += end_time - start_time

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
        ctx.input_size = self.size
        return self.mean(axis=dim, keepdims=True)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        (input_tensor,) = ctx.forward_context
        grad_output = grad_output / ctx.input_size
        return cp.broadcast_to(grad_output, input_tensor.shape)


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
        return cp.matmul(self, tensor)

    # @staticmethod
    # def backward(ctx, grad_output: cp.ndarray):
    #     input, weight = ctx.forward_context
    #     grad_input = grad_output @ cp.swapaxes(weight, -2, -1)
    #     grad_weight = cp.swapaxes(input, -2, -1) @ grad_output
    #     return grad_input, grad_weight

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        """Backward pass for matrix multiplication.
        NOTE: We using numpy here for easy axis manipulation. This is not the most efficient
        way to do it.
        """
        input, weight = ctx.forward_context

        # Calculate grad_input
        if weight.ndim > 1:
            # For multi-dimensional weight, transpose the last two dimensions
            axes = np.arange(weight.ndim)
            # axes[-2:] = axes[-2:][::-1]
            axes[-1], axes[-2] = axes[-2], axes[-1]
            grad_input = cp.matmul(grad_output, weight.transpose(tuple(axes)))
        else:
            # For vector weight, simply use the vector as-is
            grad_input = cp.matmul(grad_output, weight.T)

        # Calculate grad_weight
        if input.ndim > 1:
            # For multi-dimensional input, transpose the last two dimensions
            axes = np.arange(input.ndim)
            axes[-1], axes[-2] = axes[-2], axes[-1]
            grad_weight = cp.matmul(input.transpose(tuple(axes)), grad_output)
        else:
            # For vector input, simply use the vector as-is
            grad_weight = cp.matmul(input.T, grad_output)

        return unbroadcast(grad_input, input.shape), unbroadcast(
            grad_weight, weight.shape
        )


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
    kernel_op = "log_softmax"
    kernel_names = [
        "log_softmax_forward",
        "log_softmax_backward",
    ]

    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        if dim not in [-1, 1]:
            raise NotImplementedError(
                "Kernel only supports dim=-1 or dim=1 for 2D tensors."
            )
        ctx.save_forward_context(self)
        rows, cols = self.shape
        output = cp.empty_like(self)
        threads_per_block = 256
        blocks_per_grid = rows
        shared_mem_size = threads_per_block * 4  # 4 bytes per float
        cuda_kernel_cache.get_kernel(ctx.kernel_op, "log_softmax_forward")(
            (blocks_per_grid,),
            (threads_per_block,),
            (output, self, rows, cols),
            shared_mem=shared_mem_size,
        )
        ctx.softmax_output = output
        return output

    @staticmethod
    def forward_cupy(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        """Log softmax of a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim

        # Shift the input for numerical stability
        x_max = self.max(axis=dim, keepdims=True)
        shifted_logits = cp.subtract(self, x_max, out=self)

        ctx.softmax_output = cp.exp(shifted_logits)
        ctx.softmax_sum = cp.sum(ctx.softmax_output, axis=dim, keepdims=True)
        log_softmax_output = shifted_logits - cp.log(ctx.softmax_sum)

        return log_softmax_output

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        rows, cols = grad_output.shape
        grad_input = cp.empty_like(grad_output)
        threads_per_block = 256
        blocks_per_grid = rows
        shared_mem_size = threads_per_block * 4  # 4 bytes per float
        cuda_kernel_cache.get_kernel(ctx.kernel_op, "log_softmax_backward")(
            (blocks_per_grid,),
            (threads_per_block,),
            (grad_input, grad_output, ctx.softmax_output, rows, cols),
            shared_mem=shared_mem_size,
        )

        return grad_input

    @staticmethod
    def backward_cupy(ctx, grad_output: cp.ndarray):
        """Backward pass for log softmax."""
        dim = ctx.dim

        softmax_output = ctx.softmax_output / ctx.softmax_sum

        # Compute gradient
        grad_input = cp.subtract(
            grad_output,
            softmax_output * cp.sum(grad_output, axis=dim, keepdims=True),
            out=grad_output,
        )

        return grad_input


class Softmax(CudaFunction):
    kernel_op = "softmax"
    kernel_names = [
        "softmax_forward",
        "softmax_backward",
    ]

    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        if dim not in [-1, 1]:
            raise NotImplementedError(
                "Kernel only supports dim=-1 or dim=1 for 2D tensors."
            )

        ctx.save_forward_context(self)
        rows, cols = self.shape
        output = cp.empty_like(self)
        threads_per_block = 256
        blocks_per_grid = rows
        shared_mem_size = threads_per_block * 4  # 4 bytes per float

        cuda_kernel_cache.get_kernel(ctx.kernel_op, "softmax_forward")(
            (blocks_per_grid,),
            (threads_per_block,),
            (output, self, rows, cols),
            shared_mem=shared_mem_size,
        )
        ctx.softmax_output = output
        return output

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        rows, cols = grad_output.shape
        grad_input = cp.empty_like(grad_output)
        threads_per_block = 256
        blocks_per_grid = rows
        shared_mem_size = threads_per_block * 4  # 4 bytes per float
        cuda_kernel_cache.get_kernel(ctx.kernel_op, "softmax_backward")(
            (blocks_per_grid,),
            (threads_per_block,),
            (grad_input, grad_output, ctx.softmax_output, rows, cols),
            shared_mem=shared_mem_size,
        )
        return grad_input

    @staticmethod
    def forward_cupy(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        """Softmax of a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim
        exps = cp.exp(self - self.max(axis=dim, keepdims=True))
        output = exps / exps.sum(axis=dim, keepdims=True)
        ctx.softmax_output = output
        return output

    @staticmethod
    def backward_cupy(ctx, grad_output: cp.ndarray):
        """Backward pass for softmax"""
        dim = ctx.dim
        softmax_output = ctx.softmax_output
        return (
            grad_output
            * softmax_output
            * (1 - softmax_output.sum(axis=dim, keepdims=True))
        )


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
        result = 1 / (1 + cp.exp(-self))
        ctx.result = result
        return result

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        result = ctx.result
        return grad_output * result * (1 - result)


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


class GetItem(CudaFunction):
    @staticmethod
    def forward(ctx, tensor: np.ndarray, *, indices) -> np.ndarray:
        """Get item using numpy-style indexing."""
        ctx.save_forward_context(tensor.shape, indices)
        return tensor[indices]

    @staticmethod
    def backward(ctx, grad_output: np.ndarray):
        indices = ctx.forward_context.pop()
        input_shape = ctx.forward_context.pop()
        grad_input = cp.zeros(input_shape, dtype=grad_output.dtype)
        cp.add.at(grad_input, indices, grad_output)
        return grad_input


class Take(CudaFunction):
    """Take function without a dimension parameter.

    NOTE: The current implementation only works with flat indices
    """

    @staticmethod
    def forward(ctx, self: cp.ndarray, indices: cp.ndarray) -> cp.ndarray:
        """Take elements from a tensor using indices."""
        assert (
            indices.dtype == cp.int64 or indices.dtype == bool
        ), f"Indices must be of type int64 or bool, got {indices.dtype}"
        ctx.save_forward_context(self)
        ctx.indices = indices
        # Assume indices are for the first dimension
        return self[indices]

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        input_tensor = ctx.forward_context.pop(0)
        grad_input = cp.zeros_like(input_tensor, dtype=cp.float32)
        cp.add.at(grad_input, ctx.indices, grad_output)
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
        all_tensors = [self]
        for t in tensors:
            if isinstance(t, cp.ndarray):
                all_tensors.append(t)
            else:
                all_tensors.append(t.data)

        return cp.concatenate(all_tensors, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        axis = ctx.axis
        split_sizes = [shape[axis] for shape in ctx.shapes]
        split_indices = cp.cumsum(split_sizes)[:-1]
        grads = cp.split(grad_output, split_indices, axis=axis)
        return tuple(grads)


class Squeeze(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int = None):
        ctx.save_forward_context(self)
        return cp.squeeze(self, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        (input,) = ctx.forward_context
        return cp.reshape(grad_output, input.shape)


class Unsqueeze(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int):
        ctx.save_forward_context(self)
        return cp.expand_dims(self, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        return cp.squeeze(grad_output)


class ArgMax(CudaFunction):
    differentiable = False

    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        return cp.argmax(self, axis=dim)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        raise RuntimeError("ArgMax is not differentiable")


class TanH(CudaFunction):
    @staticmethod
    def forward(ctx, self: cp.ndarray) -> cp.ndarray:
        ctx.save_forward_context(self)
        return cp.tanh(self)

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        (input,) = ctx.forward_context
        return grad_output * (1 - cp.tanh(input) ** 2)


def stack(tensors: Tuple[cp.ndarray], dim: int = 0) -> cp.ndarray:
    return cp.stack(tensors, axis=dim)


def cat(tensors: Tuple[cp.ndarray], dim: int = 0) -> cp.ndarray:
    return cp.concatenate(tensors, axis=dim)


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


def arange(start: int, stop: Optional[int] = None, step: int = 1) -> cp.ndarray:
    return cp.arange(start, stop, step)


def attach_op(function: Type[CudaFunction]):
    return partial(function.apply, function)


def move_to_backend(tensor: Any) -> BackendTensor:
    tensor_type = type(tensor).__module__

    if tensor_type == "numpy":
        return cp.array(tensor)

    raise ValueError(f"Cannot move tensor of type {tensor_type} to the backend")


funcs = {
    "init_data": init_data,
    "copy": copy,
    "numpy": numpy,
    "to_dtype": to_dtype,
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
    "sigmoid": attach_op(Sigmoid),
    "dropout": attach_op(Dropout),
    "getitem": attach_op(GetItem),
    "take": attach_op(Take),
    "transpose": attach_op(Transpose),
    "reshape": attach_op(Reshape),
    "flatten": attach_op(Flatten),
    # "cat": attach_op(Cat),
    "argmax": attach_op(ArgMax),
    "tanh": attach_op(TanH),
    "squeeze": attach_op(Squeeze),
    "unsqueeze": attach_op(Unsqueeze),
}

factories = {
    "eye": eye,
    "ones": ones,
    "zeros": zeros,
    "arange": arange,
    "stack": stack,
    "cat": cat,
}
