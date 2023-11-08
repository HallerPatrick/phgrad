import unittest
from copy import deepcopy

import numpy as np

from phgrad.engine import Tensor
from phgrad.optim import SGD


class TestSGD(unittest.TestCase):
    def test_linear_function_optimization(self):
        """Test SGD on a simple linear function optimization."""
        x = Tensor(np.array([10.0]), requires_grad=True)
        a = 2.0

        # Simulate gradient computation
        x.grad = a

        optimizer = SGD([x], lr=0.01)
        old_x = deepcopy(x.data)
        optimizer.step()

        # Check if x is decreasing (since a is positive)
        self.assertTrue(x.data < old_x)

    def test_zero_gradient(self):
        """Test SGD does not update parameter with zero gradient."""
        p = Tensor(np.array([5.0]), requires_grad=True)
        p.grad = 0.0  # Zero gradient

        optimizer = SGD([p], lr=0.01)
        old_p = deepcopy(p.data)
        optimizer.step()

        # Check if p remains unchanged
        self.assertEqual(p.data, old_p)
