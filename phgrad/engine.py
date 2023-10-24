import math
import operator
import random
from typing import Any

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
            self.grad += (other.value * self.value**(other.value-1)) * new.grad

        new._backward = _backward

        return new
    
    @autoconvert
    def __rpow__(self, other: "Scalar"):
        return other ** self

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
        return self * (other ** -1)
    
    @autoconvert
    def __rtruediv__(self, other: "Scalar"):
        return other * (self ** -1)
    
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
    

    def relu(self):
        new_value = max(self.value, 0)
        new = Scalar(new_value, children=(self,), op=max)

        def _backward():
            self.grad = self.grad + new.grad * (self.value > 0)

        new._backward = _backward

        return new

def softmax_scalars(inputs):
    exp_values = [pow(2.718281828459045, scalar.value) for scalar in inputs]
    total = sum(exp_values)
    softmax_outputs = [val / total for val in exp_values]
    results = []

    for idx, softmax_output in enumerate(softmax_outputs):
        scalar = Scalar(softmax_output)

        def create_backward(idx):
            def backward():
                s = softmax_outputs[idx]
                for j, inp in enumerate(inputs):
                    if idx == j:
                        grad_val = s * (1 - s)
                    else:
                        grad_val = -s * softmax_outputs[j]
                    inp.grad += grad_val * scalar.grad
            return backward

        scalar._backward = create_backward(idx)
        results.append(scalar)

    return results


class Module:
    
    def __init__(self):
        self._parameters = {}

    def parameters(self):
        for name, param in self._parameters.items():
            yield param

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

class Neuron(Module):

    def __init__(self, inp_dim: int):
        super().__init__()
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in range(inp_dim)]
        self.bias = Scalar(0)

    def __call__(self, input):
        return sum(w * i for w, i in zip(self.weights, input)) + self.bias



class Linear(Module):

    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        self.neurons = [Neuron(inp_dim) for _ in range(out_dim)]

    def __call__(self, input):
        return [n(input) for n in self.neurons]

    def backward(self):
        pass

class MLP(Module):

    def __init__(self, inp_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = Linear(inp_dim, hid_dim)
        self.linear2 = Linear(hid_dim, out_dim)

    def __call__(self, input):
        x = self.linear1(input)
        for x_i in x:
            x_i = x_i.relu()
        return self.linear2(x)

    def backward(self):
        pass

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
