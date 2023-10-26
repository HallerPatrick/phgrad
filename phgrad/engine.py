import math
import operator
import random
from typing import Any, Tuple


def autoconvert(func):
    def wrapper(*args, **kwargs):
        if len(args) < 2:
            return func(*args, **kwargs)
        other = args[1]
        if not isinstance(other, Scalar):
            other = Scalar(other)
        args = (args[0], other, *args[2:])
        return func(*args, **kwargs)

    return wrapper


class Scalar:
    def __init__(self, value, children=(), op=None):
        self.value = value
        self.grad = 0
        self._children = children
        self._backward = lambda: None
        self.op = op

    @autoconvert
    def __add__(self, other: "Scalar"):
        new_value = operator.add(self.value, other.value)
        new = Scalar(new_value, children=(self, other), op=operator.add)

        def _backward():
            self.grad = self.grad + new.grad
            other.grad = other.grad + new.grad

        new._backward = _backward

        return new

    @autoconvert
    def __mul__(self, other: "Scalar"):
        new_value = operator.mul(self.value, other.value)
        new = Scalar(new_value, children=(self, other), op=operator.mul)

        def _backward():
            self.grad = self.grad + new.grad * other.value
            other.grad = other.grad + new.grad * self.value

        new._backward = _backward

        return new

    @autoconvert
    def __radd__(self, other: "Scalar"):
        return self + other

    @autoconvert
    def __pow__(self, other: "Scalar"):
        new_value = operator.pow(self.value, other.value)
        new = Scalar(new_value, children=(self, other), op=operator.pow)

        def _backward():
            self.grad += (other.value * self.value ** (other.value - 1)) * new.grad

        new._backward = _backward

        return new

    @autoconvert
    def __rpow__(self, other: "Scalar"):
        return other**self

    @autoconvert
    def __rmul__(self, other: "Scalar"):
        return self * other

    @autoconvert
    def __sub__(self, other: "Scalar"):
        return self + (-other)

    @autoconvert
    def __rsub__(self, other: "Scalar"):
        return other + (-self)

    @autoconvert
    def __neg__(self):
        return self * -1

    @autoconvert
    def __truediv__(self, other: "Scalar"):
        return self * (other**-1)

    @autoconvert
    def __rtruediv__(self, other: "Scalar"):
        return other * (self**-1)

    def backward(self):
        self.grad = 1
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for v in reversed(topo):
            v._backward()

    def __str__(self) -> str:
        return f"Scalar({self.value}, grad={self.grad})"

    def __repr__(self) -> str:
        return self.__str__()

    def relu(self):
        new_value = max(self.value, 0)
        new = Scalar(new_value, children=(self,), op=max)

        def _backward():
            self.grad = self.grad + new.grad * (self.value > 0)

        new._backward = _backward

        return new


class Pensor:
    """Pensor is for now only a collection of scalar values and does not behave like a tensor.
    So every operation is done element-wise.
    """

    def __init__(self, values):
        self.values = []
        _current_dim_len = None
        for value in values:
            if isinstance(value, list):
                if _current_dim_len is None:
                    _current_dim_len = len(value)
                else:
                    assert _current_dim_len == len(value), "All dimensions must be of same length"

                self.values.append([self._value_to_scalar(val) for val in value])
            else:
                self.values.append(self._value_to_scalar(value))

    def __getitem__(self, idx):
        # NOTE: Here we are getting into the problem of returning a view or a copy
        return self.values[idx]

    def __setitem__(self, idx, value):
        self.values[idx] = self._value_to_scalar(value)

    def __str__(self) -> str:
        return f"Pensor({self.values})"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __add__(self, other):
        return self._elementwise(other, operator.add)

    def __radd__(self, other):
        return self._elementwise(other, operator.add)

    def __sub__(self, other):
        return self._elementwise(other, operator.sub)

    def __rsub__(self, other):
        return self._elementwise(other, operator.sub)

    def __mul__(self, other):
        return self._elementwise(other, operator.mul)

    def __rmul__(self, other):
        return self._elementwise(other, operator.mul)

    def __truediv__(self, other):
        return self._elementwise(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._elementwise(other, operator.truediv)

    def __pow__(self, other):
        return self._elementwise(other, operator.pow)

    def __rpow__(self, other):
        return self._elementwise(other, operator.pow)

    @property
    def shape(self):
        # TODO: For now only 1D and 2D pensors are supported
        dims = []
        if isinstance(self.values[0], list):
            dims.append(len(self))
            dims.append(len(self.values[0]))
        else:
            dims.append(len(self))
        return tuple(dims)

    def _elementwise(self, other, op):
        if isinstance(other, Pensor):
            assert len(self) == len(other)
            return Pensor(
                [op(self.values[i], other.values[i]) for i in range(len(self))]
            )
        return Pensor([op(self.values[i], other) for i in range(len(self))])
        
    def relu(self):
        return Pensor([val.relu() for val in self.values])

    @staticmethod
    def _value_to_scalar(value):
        if isinstance(value, Scalar):
            return value
        else:
            return Scalar(value)
    
    @classmethod
    def ones(cls, dims: Tuple):
        """Create a pensors of ones."""
        assert len(dims) <= 2, "Only 1D and 2D pensors are supported"
        if len(dims) == 1:
            return cls([1 for _ in range(dims[0])])
        else:
            return cls([[1 for _ in range(dims[1])] for _ in range(dims[0])])
        
    @classmethod
    def zeros(cls, dims: Tuple):
        """Create a pensors of zeros."""
        assert len(dims) <= 2, "Only 1D and 2D pensors are supported"
        if len(dims) == 1:
            return cls([0 for _ in range(dims[0])])
        else:
            return cls([[0 for _ in range(dims[1])] for _ in range(dims[0])])
        
    @classmethod
    def randn(cls, dims: Tuple):
        """Create a pensors of random values from a normal distribution."""
        assert len(dims) <= 2, "Only 1D and 2D pensors are supported"
        if len(dims) == 1:
            return cls([random.gauss(0, 1) for _ in range(dims[0])])
        else:
            return cls([[random.gauss(0, 1) for _ in range(dims[1])] for _ in range(dims[0])])
    
    @classmethod
    def rand(cls, dims: Tuple):
        """Create a pensors of random values from a uniform distribution."""
        assert len(dims) <= 2, "Only 1D and 2D pensors are supported"
        if len(dims) == 1:
            return cls([random.uniform(0, 1) for _ in range(dims[0])])
        else:
            return cls([[random.uniform(0, 1) for _ in range(dims[1])] for _ in range(dims[0])])
        
    @classmethod
    def arange(cls, start: int, end: int, step: int = 1):
        """Create a pensors of values from start to end with a step."""
        return cls([i for i in range(start, end, step)])
    
    @classmethod
    def eye(cls, n: int):
        """Create a pensors of identity matrix."""
        return cls([[1 if i == j else 0 for i in range(n)] for j in range(n)])



class Module:
    def __init__(self):
        self._parameters = {}

    def parameters(self):
        for name, param in self._parameters.items():
            yield param

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0


class Linear(Module):
    def __init__(self, inp_dim: int, output_dim: int):
        super().__init__()
        self.weights = [
            [Scalar(random.uniform(-1, 1)) for _ in range(inp_dim)]
            for _ in range(output_dim)
        ]
        self.biases = [Scalar(0) for _ in range(output_dim)]

    def __call__(self, input):
        return self._forward(input)

    def _forward(self, input):
        output = [Scalar(0) for _ in range(len(self.weights))]
        for i, weight in enumerate(self.weights):
            for j, inp in enumerate(input):
                output[i] += weight[j] * inp
            output[i] += self.biases[i]
        return output


class MNISTClassifier(Module):
    def __init__(self, inp_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.linear1 = Linear(inp_dim, hidden)
        self.linear2 = Linear(hidden, out_dim)

    def __call__(self, x) -> Any:
        x = self.linear1(x)
        for x_i in x:
            x_i = x_i.relu()
        return self.linear2(x)
