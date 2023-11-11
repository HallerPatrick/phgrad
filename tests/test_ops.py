import unittest

import numpy as np

from utils import requires_torch

from phgrad.engine import Tensor as Tensor
from phgrad.ops import LogSoftmax


class TestReshape(unittest.TestCase):
    def test_reshape(self):
        tensor = Tensor(np.array([1, 2, 3, 4, 5, 6]))
        tensor = tensor.reshape((2, 3))
        assert isinstance(tensor, Tensor)
        assert isinstance(tensor.data, np.ndarray)
        assert tensor.data.shape == (2, 3)


class TestOps(unittest.TestCase):
    def test_add(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.eye(3))
        t3 = t1 + t2
        t3.sum().backward()

        assert isinstance(t3, Tensor)
        assert isinstance(t3.data, np.ndarray)
        assert t3.data.shape == (3, 3)
        # assert np.all(t3.data == np.array([5, 7, 9]))

    def test_add_with_broadcasting(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.array([4, 5, 6]))
        t3 = t1 + t2
        t3.sum().backward()

        assert isinstance(t3, Tensor)
        assert isinstance(t3.data, np.ndarray)
        assert t3.data.shape == (3, 3)

    def test_relu(self):
        t1 = Tensor(np.array([1, 2, -3]))
        t2 = t1.relu()

        assert isinstance(t2, Tensor)
        assert isinstance(t2.data, np.ndarray)
        assert t2.data.shape == (3,)
        assert np.all(t2.data == np.array([1, 2, 0]))

    def test_sigmoid(self):
        t1 = Tensor(np.array([1, 2, -3]))
        t2 = t1.sigmoid()
        assert isinstance(t2, Tensor)
        assert isinstance(t2.data, np.ndarray)
        assert t2.data.shape == (3,)
        assert np.allclose(t2.data, np.array([0.73105858, 0.88079708, 0.04742587]))

    @requires_torch
    def test_softmax(self):
        import torch

        t1 = Tensor(np.array([1, 2, -3]))
        t2 = t1.softmax()

        tt1 = torch.tensor([1, 2, -3], dtype=torch.float32)
        tt2 = torch.nn.functional.softmax(tt1, dim=0)

        assert isinstance(t2, Tensor)
        assert isinstance(t2.data, np.ndarray)
        assert t2.data.shape == (3,)
        assert np.allclose(t2.data, tt2.detach().numpy())

    @requires_torch
    @unittest.skip("Not implemented")
    def test_softmax_with_axis(self):
        import torch
        t1 = Tensor(np.array([[1, 2, -3], [4, 5, 6]]))
        t2 = t1.softmax()

        tt1 = torch.tensor([[1, 2, -3], [4, 5, 6]], dtype=torch.float32)
        tt2 = torch.nn.functional.softmax(tt1, dim=1)

        assert isinstance(t2, Tensor)
        assert isinstance(t2.data, np.ndarray)
        assert t2.data.shape == (2, 3)
        assert np.allclose(t2.data, tt2.detach().numpy())

    @requires_torch
    def test_log_softmax(self):
        import torch

        t1 = Tensor(np.array([1, 2, -3]))
        t2 = t1.log_softmax()

        tt1 = torch.tensor([1, 2, -3], dtype=torch.float32)
        tt2 = torch.nn.functional.log_softmax(tt1, dim=0)

        assert isinstance(t2, Tensor)
        assert isinstance(t2.data, np.ndarray)
        assert t2.data.shape == (3,)
        assert np.allclose(t2.data, tt2.detach().numpy())


@requires_torch
class TestLogSoftmax(unittest.TestCase):
    def test_forward_backward(self):
        """Test the forward pass with a simple input."""
        import torch

        input_np = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

        ctx = LogSoftmax()
        log_softmax_np = LogSoftmax.forward(ctx, input_np, dim=1)
        log_softmax_torch = torch.nn.functional.log_softmax(input_torch, dim=1)

        np.testing.assert_almost_equal(
            log_softmax_np, log_softmax_torch.detach().numpy(), decimal=5
        )

        grad_output = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        log_softmax_torch.backward(grad_output)
        grad_np = LogSoftmax.backward(ctx, grad_output.numpy())

        np.testing.assert_almost_equal(grad_np, input_torch.grad.numpy(), decimal=5)

class TestCat(unittest.TestCase):

    def test_cat(self):
        t1, t2 = Tensor(np.array([0.1, 0.2])), Tensor(np.array([0.3, 0.4]))
        t3 = t1.cat((t2, ))
        np.testing.assert_equal(t3.data, np.array([0.1, 0.2, 0.3, 0.4]))

    def test_cat_dim1(self):
        t1, t2 = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]])), Tensor(np.array([[0.5, 0.6], [0.7, 0.8]]))
        t3 = t1.cat((t2,), dim=1)
        np.testing.assert_equal(t3.data, np.array([[0.1, 0.2, 0.5, 0.6], [0.3, 0.4, 0.7, 0.8]]))


