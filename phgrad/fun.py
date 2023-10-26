from typing import List, Tuple, Union

from .engine import Scalar, Pensor

def argmax(xs: Union[List[Scalar], Pensor], dim: int) -> Tuple[Scalar, int]:

    if isinstance(xs, Pensor):
        assert len(xs.shape) == 2, "Only 2D pensors are supported"
        results = []
        for i in range(len(xs)):
            max_idx = 0
            max_val = xs[i][0].value
            for j in range(len(xs[i])):
                if xs[i][j].value > max_val:
                    max_val = xs[i][j].value
                    max_idx = j
            results.append(max_idx)
        return Pensor(results), max_idx

    assert len(xs) > 0, "Empty list"
    max_idx = 0
    max_val = xs[0].value
    for i in range(len(xs)):
        if xs[i].value > max_val:
            max_val = xs[i].value
            max_idx = i
    return xs[max_idx], max_idx

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

def softmax(p: Pensor):
    """Compute softmax values for each sets of scores in x."""
    assert len(p.shape) == 2, "Only 2D pensors are supported"
    results = []
    for i in range(len(p)):
        exp_values = [pow(2.718281828459045, scalar.value) for scalar in p[i]]
        total = sum(exp_values)
        softmax_outputs = [val / total for val in exp_values]
        results.append(softmax_outputs)

    return Pensor(results)
