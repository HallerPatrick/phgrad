"""CuPy implementation of the CUDA backend.

We mainly keep this graveyard of cupy code for reference, as it is replaced by custom kernels.
This will probably be not a complete list of operations, as it is not feasible to implement all
operations as a kernel. But lets see.

"""

import cupy as cp

from .cuda import CudaFunction

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

