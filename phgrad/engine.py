import time
from typing import Any, Optional, Tuple, Type, Union

import numpy as np

from phgrad import types
from phgrad.backends import backend_from_device
from phgrad.debug import DEBUG, backward_time, tensor_creations

TensorOrScalar = Union["Tensor", float, int]


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "ctx", "backend", "device", "dtype")

    def __init__(
        self,
        value: Any,
        requires_grad=True,
        device: Union[str, int] = "cpu",
        dtype: types.DType = types.float32,
        _backend=None,
    ):
        if _backend is None:
            self.device = device
            self.backend = backend_from_device(device, tensor_type=Tensor)
        else:
            self.backend = _backend
            # NOTE: This only works with one device for now
            self.device = self.backend.name

        if DEBUG == 1:
            tensor_creations[device] += 1

        self.dtype = dtype
        self.data = self.backend.init_data(value, self.dtype)
        self.grad = None
        self.requires_grad: bool = requires_grad

        # Context contains the information needed to compute the gradient
        # It contains the operation that created this tensor
        self.ctx = None

    def __setitem__(self, idx, value):
        self.data[idx] = value

    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the tensor."""
        return self.data.shape

    @property
    def dims(self):
        """Return the number of dimensions of the tensor."""
        return len(self.shape)

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

        if DEBUG == 1:
            start_time = time.time()

        grads = self.ctx.backward(self.ctx, self.grad)

        if DEBUG == 1:
            backward_time[str(self.ctx)] += time.time() - start_time

        if len(self.ctx.prev) == 1 or (not isinstance(grads, tuple)):
            grads = [grads]

        # Iterate over all previous tensors and set the gradient
        if DEBUG == 2:
            print("=== All Backward pass === ")
            for t, g in zip(self.ctx.prev, grads):
                if not isinstance(t, tuple):
                    t = (t,)
                print(
                    f"Shapes: input {[x.shape for x in t]}, grad {g.shape}, op={self.ctx}"
                )
                print("grads:", grads)
                print("Values:", t, g)
                print("=======")

        for t, g in zip(self.ctx.prev, grads):
            if g is None:
                continue

            if isinstance(t, tuple):
                t = t[0]

            # What if the tensor value is a scalar?
            # Check if tensor is a scalar and gradient has shape (1,)
            if not (t.data.shape == () and g.shape == (1,)):
                assert (
                    g.shape == t.data.shape
                ), "Grad shape must match tensor shape, {} != {} ({})".format(
                    g.shape, t.data.shape, self.ctx
                )
            t.grad = g
            t.backward(False)

    def __str__(self) -> str:
        return f"Tensor({self.data}, grad_fn={self.ctx}, grad={self.grad})"

    def __repr__(self) -> str:
        return self.__str__()

    # === Some fine utils ===

    def detach(self) -> "Tensor":
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

    def numpy(self) -> np.ndarray:
        return self.backend.numpy(self.data)

    def to(self, device: Union[str, int], in_place=False) -> "Tensor":
        if self.device == device:
            return self

        if in_place:
            new_tensor = self.detach()
            self = Tensor(
                new_tensor.data,
                device=device,
                requires_grad=self.requires_grad,
            )
            return self

        return Tensor(
            self.backend.copy(self.data),
            requires_grad=self.requires_grad,
            device=device,
            _backend=self.backend,
        )

    def to_dtype(self, dtype: Type) -> "Tensor":
        return self.backend.to_dtype(self, dtype)

    # ========= OPS =========

    def __getitem__(self, idx) -> "Tensor":
        """Return a new tensor with the selected indices.

        NOTE: For now only support indexing with integers and integer slices
        """
        if isinstance(idx, tuple):
            for i in idx:
                if not isinstance(i, slice):
                    assert isinstance(i, int), "Only support indexing with integers"
        return self.backend.getitem(self, indices=idx)

    # Unary ops
    def exp(self) -> "Tensor":
        return self.backend.exp(self)

    def neg(self) -> "Tensor":
        return self.backend.neg(self)

    def log(self) -> "Tensor":
        return self.backend.log(self)

    def log_softmax(self, dim: Optional[int] = None) -> "Tensor":
        return self.backend.log_softmax(self, dim=dim)

    def softmax(self, dim: Optional[int] = None) -> "Tensor":
        return self.backend.softmax(self, dim=dim)

    def relu(self) -> "Tensor":
        return self.backend.relu(self)

    def sigmoid(self) -> "Tensor":
        return self.backend.sigmoid(self)

    def tanh(self) -> "Tensor":
        return self.backend.tanh(self)

    # Unary ops + reduce
    def sum(self) -> "Tensor":
        return self.backend.sum(self)

    def mean(self, dim: Optional[int] = None) -> "Tensor":
        return self.backend.mean(self, dim=dim)

    def max(self) -> "Tensor":
        return self.backend.max(self)

    # Binary ops
    def add(self, other: TensorOrScalar) -> "Tensor":
        return self.backend.add(self, other)

    def sub(self, other: TensorOrScalar) -> "Tensor":
        return self.backend.sub(self, other)

    def mul(self, other: TensorOrScalar) -> "Tensor":
        return self.backend.mul(self, other)

    def div(self, other: TensorOrScalar) -> "Tensor":
        return self.backend.div(self, other)

    def matmul(self, other: "Tensor") -> "Tensor":
        return self.backend.matmul(self, other)

    def dot(self, other: "Tensor") -> "Tensor":
        return self.backend.matmul(self, other)

    # Transformations
    def transpose(self, order) -> "Tensor":
        return self.backend.transpose(self, order=order)

    @property
    def T(self) -> "Tensor":
        dims = self.dims

        if dims == 2:
            return self.transpose((1, 0))

        return self.transpose((0, 2, 1))

    def reshape(self, shape: Union[int, Tuple[int]]) -> "Tensor":
        return self.backend.reshape(self, shape=shape)

    def flatten(self) -> "Tensor":
        return self.backend.flatten(self)

    def take(self, indices: "Tensor") -> "Tensor":
        return self.backend.take(self, indices)

    def cat(self, others: Tuple["Tensor"], dim: Optional[int] = None) -> "Tensor":
        return self.backend.cat(self, others, dim=dim)

    # TODO: Does not really feel like a proper op
    def dropout(self, p: float, training: bool) -> "Tensor":
        return self.backend.dropout(self, p=p, training=training)

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
        return self.backend.argmax(self, dim=dim)

    def squeeze(self, dim: Optional[int] = None):
        return self.backend.squeeze(self, dim=dim)

    @classmethod
    def eye(
        cls: Type["Tensor"],
        shape: Union[int, Tuple[int]],
        requires_grad: bool = False,
        device: str = "cpu",
    ):
        assert (
            isinstance(shape, int) or len(shape) <= 2
        ), "Only support 1D and 2D tensor creation"
        backend = backend_from_device(device, Tensor)
        return cls(
            backend.eye(*shape),
            requires_grad=requires_grad,
            device=device,
            _backend=backend,
        )

    def scatter_add(
        self, indices: "Tensor", values: TensorOrScalar, axis: Optional[int] = None
    ) -> "Tensor":
        """This is inplace"""
        self.backend.scatter_add(self.data, indices.data, values, axis)

    @classmethod
    def zeros(
        cls: Type["Tensor"],
        shape: Tuple[int],
        requires_grad: bool = False,
        device: str = "cpu",
    ):
        backend = backend_from_device(device, Tensor)
        return cls(
            backend.zeros(shape),
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

    @classmethod
    def arange(
        cls: Type["Tensor"],
        start: int,
        end: Optional[int] = None,
        step: int = 1,
        requires_grad: bool = False,
        device: str = "cpu",
        dtype: types.DType = types.int64,
    ):
        backend = backend_from_device(device, Tensor)
        return cls(
            backend.arange(start, end, step),
            requires_grad=requires_grad,
            device=device,
            dtype=dtype,
            _backend=backend,
        )

    @classmethod
    def stack(
        cls: Type["Tensor"],
        tensors: Tuple["Tensor"],
        dim: int = 0,
    ):
        backend = tensors[0].backend
        return cls(
            backend.stack([t.data for t in tensors], dim=dim),
            requires_grad=tensors[0].requires_grad,
            device=tensors[0].device,
            _backend=backend,
        )
