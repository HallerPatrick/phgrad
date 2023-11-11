from typing import Optional, Tuple, Union, Type

import numpy as np
import numpy.typing as npt

from phgrad.backends import backend_from_device

TensorOrScalar = Union["Tensor", float, int]

class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "ctx", "backend", "device")

    def __init__(
        self,
        value: npt.NDArray,
        requires_grad=True,
        device: Union[str, int] = "cpu",
        _backend=None,
    ):
        if _backend is None:
            self.backend = backend_from_device(device, tensor_type=Tensor)
        else:
            self.backend = _backend

        self.device = device

        self.data = self.backend.init_data(value)
        self.grad = None
        self.requires_grad: bool = requires_grad

        # Context contains the information needed to compute the gradient
        # It contains the operation that created this tensor
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

    # === Some fine utils ===

    def copy(self):
        return Tensor(
            self.backend.copy(self.data),
            requires_grad=self.requires_grad,
            device=self.device,
            _backend=self.backend,
        )

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

    # ========= OPS =========

    # Unary ops
    def exp(self):
        return self.backend.exp(self)

    def neg(self):
        return self.backend.neg(self)

    def log(self):
        return self.backend.log(self)

    def log_softmax(self, dim: Optional[int] = None):
        return self.backend.log_softmax(self, dim)

    def softmax(self, dim: Optional[int] = None):
        return self.backend.softmax(self, dim)

    def relu(self):
        return self.backend.relu(self)

    def sigmoid(self):
        return self.backend.sigmoid(self)

    # Unary ops + reduce
    def sum(self):
        return self.backend.sum(self)

    def mean(self):
        return self.backend.mean(self)

    def max(self):
        return self.backend.max(self)

    # Binary ops
    def add(self, other: TensorOrScalar):
        return self.backend.add(self, other)

    def sub(self, other: TensorOrScalar):
        return self.backend.sub(self, other)

    def mul(self, other: TensorOrScalar):
        return self.backend.mul(self, other)

    def div(self, other: TensorOrScalar):
        return self.backend.div(self, other)

    def matmul(self, other: "Tensor"):
        return self.backend.matmul(self, other)

    def dot(self, other: "Tensor"):
        return self.backend.matmul(self, other)

    # Transformations
    def transpose(self, order):
        return self.backend.transpose(self, order)

    def reshape(self, shape: Union[int, Tuple[int]]):
        return self.backend.reshape(self, shape)

    def flatten(self):
        return self.backend.flatten(self)

    def take(self, indices: "Tensor"):
        return self.backend.take(self, indices)

    def cat(self, others: Tuple["Tensor"], dim: Optional[int] = None):
        return self.backend.cat(self, others, dim)

    # TODO: Does not really feel like a proper op
    def dropout(self, p: float, training: bool):
        return self.backend.dropout(self, p, training)

    def __add__(self, other: TensorOrScalar) -> "Tensor":
        return self.add(other)

    def __radd__(self, other: TensorOrScalar) -> "Tensor":
        return self.add(other)

    def __sub__(self, other: TensorOrScalar) -> "Tensor":
        return self.sub(other)

    def __rsub__(self, other: TensorOrScalar) -> "Tensor":
        return self.sub(other)

    def __mul__(self, other: TensorOrScalar) -> "Tensor":
        return self.mul(other)

    def __rmul__(self, other: TensorOrScalar) -> "Tensor":
        return self.mul(other)

    def __neg__(self):
        return self.neg()
    
    def argmax(self, dim: Optional[int] = None):
        return self.backend.argmax(self, dim)

    @classmethod
    def eye(
        cls: Type["Tensor"],
        shape: Union[int, Tuple[int]],
        requires_grad: bool = False,
        device: str = "cpu",
    ):
        assert isinstance(shape, int) or len(shape) <= 2, "Only support 1D and 2D tensor"
        backend = backend_from_device(device, Tensor)
        return cls(
            backend.eye(*shape),
            requires_grad=requires_grad,
            device=device,
            _backend=backend,
        )

    @classmethod
    def zeroes(
        cls: Type["Tensor"],
        shape: Tuple[int],
        requires_grad: bool = False,
        device: str = "cpu",
    ):
        backend = backend_from_device(device, Tensor)
        return cls(
            backend.zeroes(*shape),
            requires_grad=requires_grad,
            device=device,
            _backend=backend,
        )

    @classmethod
    def ones(
        cls: Type["Tensor"],
        shape: Tuple[int],
        requires_grad: bool = False,
        device: str = "cpu",
    ):
        backend = backend_from_device(device, Tensor)
        return cls(
            backend.ones(*shape),
            requires_grad=requires_grad,
            device=device,
            _backend=backend,
        )
