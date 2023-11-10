from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "ctx")

    def __init__(self, value: npt.NDArray, requires_grad=True):
        if type(value) in [np.float64, np.float32, np.float16, np.float128, np.int64]:
            value = np.array(value)

        assert isinstance(
            value, np.ndarray
        ), f"Value must be a numpy array, got {type(value)}"

        self.data: npt.NDArray = value
        self.grad: Optional[npt.NDArray] = None
        self.requires_grad: bool = requires_grad

        # Context contains the information needed to compute the gradient
        # It contains the operation that created this tensor and the tensors
        self.ctx = None

    def __getitem__(self, idx) -> "Tensor":
        return Tensor(self.data[idx])

    def __setitem__(self, idx, value):
        self.data[idx] = value

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the tensor."""
        return self.data.shape

    @property
    def dtype(self):
        """Return the dtype of the tensor."""
        return self.data.dtype

    def deepwalk(self):
        def _deepwalk(node, visited, nodes):
            visited.add(node)
            if node.ctx:
                [
                    _deepwalk(i, visited, nodes)
                    for i in node.ctx.prev
                    if i not in visited
                ]
                nodes.append(node)
            return nodes

        return _deepwalk(self, set(), [])

    def backward(self, allow_fill=True):
        """Compute the gradient of this tensor.

        Args:
            allow_fill (bool, optional): Whether to fill the gradient with one if it is None. Defaults to True.
            This is needed for implicit gradient creation.
        """
        if self.ctx is None:
            return

        if self.grad is None and allow_fill:
            assert (
                self.data.size == 1
            ), f"Only tensors with size 1 can have None gradient, Tensor has size {self.data.size}"
            self.grad = np.ones_like(self.data)

        assert self.grad is not None

        grads = self.ctx.backward(self.ctx, self.grad)

        if len(self.ctx.prev) == 1 or isinstance(grads, np.ndarray):
            grads = [grads]

        # Iterate over all previous tensors and set the gradient
        # print("=== All Backward pass === ")
        # for t, g in zip(self.ctx.prev, grads):
        #     print(f"Shapes: input {t.shape}, grad {g.shape}, op={self.ctx}")
        #     print("grads:", grads)
        #     # print("Values:", t, g)
        #     print("=======")

        for t, g in zip(self.ctx.prev, grads):
            if g is None:
                continue

            assert (
                g.shape == t.data.shape
            ), "Grad shape must match tensor shape, {} != {} ({})".format(
                g.shape, t.data.shape, self.ctx
            )
            t.grad = g
            t.backward(False)

    @property
    def T(self) -> "Tensor":
        """Return the transpose of the tensor."""
        return self.transpose((1, 0))

    def __str__(self) -> str:
        return f"Tensor({self.data}, grad_fn={self.ctx}, grad={self.grad})"

    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self.add(other)

    def __radd__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self.add(other)

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self.sub(other)

    def __rsub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self.sub(other)

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self.mul(other)

    def __rmul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        return self.mul(other)

    def __neg__(self):
        return self.neg()

    # TODO: Define log_softmax as a composition of exisiting differentiable ops
    def logsoftmax(self):
        raise NotImplementedError
        # m = self.max(axis=len(self.shape)-1, keepdim=True)
        # ss = m + (self-m).exp().sum(axis=len(self.shape)-1, keepdim=True).log()
        # return self - ss

    # === Some fine utils ===
    @property
    def first_item(self):
        return self.data[0]

    def torch(self, requires_grad: bool = False):
        try:
            import torch
        except ImportError:
            raise ImportError("torch not installed (pip install torch)")

        torch_tensor = torch.from_numpy(self.data)
        torch_tensor.requires_grad = requires_grad
        return torch_tensor


# We do it like Georg Hotz and build the tensors ops and at them dinamically
from . import ops
# from .utils import generate_stub_for_class; generate_stub_for_class(Tensor, "engine")
