"""CuPy implementation of the CUDA backend.

We mainly keep this graveyard of cupy code for reference, as it is replaced by custom kernels.
This will probably be not a complete list of operations, as it is not feasible to implement all
operations as a kernel. But lets see.

"""

import cupy as cp

from .cuda import CudaFunction

class Softmax(CudaFunction):

    def __init__(self, *tensors: Tuple[cp.ndarray]):
        super().__init__(*tensors)
        self.forward_kernel, = _load_cuda_kernels("softmax", "not_so_efficient_softmax_forward")

    @staticmethod
    def kernel_forward(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray: #v1
        # Slightly slower than other kernel

        if dim != -1 and dim != 1:
            raise NotImplementedError("Kernel only supports dim=-1 or dim=1 for 2D tensors.")

        ctx.save_forward_context(self)
        rows, cols = self.shape
        output = cp.empty_like(self)
        threads_per_block = 256
        blocks_per_grid = (rows * cols + threads_per_block - 1) // threads_per_block
        ctx.forward_kernel((blocks_per_grid,), (threads_per_block,), (output, self, rows, cols))
        return output

    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        """Softmax of a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim
        max_val = cp.max(self, axis=dim, keepdims=True)
        exps = cp.exp(self - max_val)
        ctx.exps = exps
        return exps / exps.sum(axis=dim, keepdims=True)



class LogSoftmax(CudaFunction):

    @staticmethod
    def forward(ctx, self: cp.ndarray, dim: int = 0) -> cp.ndarray:
        """Log softmax of a tensor."""
        ctx.save_forward_context(self)
        ctx.dim = dim

        # Shift the input for numerical stability
        x_max = self.max(axis=dim, keepdims=True)
        shifted_logits = self - x_max
        
        ctx.softmax_output = cp.exp(shifted_logits)
        ctx.softmax_sum = cp.sum(ctx.softmax_output, axis=dim, keepdims=True)
        log_softmax_output = shifted_logits - cp.log(ctx.softmax_sum)

        return log_softmax_output

    @staticmethod
    def backward(ctx, grad_output: cp.ndarray):
        """Backward pass for log softmax."""
        dim = ctx.dim

        softmax_output = ctx.softmax_output / ctx.softmax_sum

        # Compute gradient
        grad_input = cp.subtract(
            grad_output,
            softmax_output * cp.sum(grad_output, axis=dim, keepdims=True),
            out=grad_output
        )

        return grad_input

