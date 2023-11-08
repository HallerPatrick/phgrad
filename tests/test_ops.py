import unittest

import numpy as np
import torch

from phgrad.engine import Tensor as Tensor
from phgrad.ops import unbroadcast


class TestReshape(unittest.TestCase):

    def test_reshape(self):

        tensor = Tensor(np.array([1, 2, 3, 4, 5, 6]))

        tensor = tensor.reshape((2, 3))
        assert isinstance(tensor, Tensor)
        assert isinstance(tensor.data, np.ndarray)
        assert tensor.data.shape == (2, 3)


class TestOps(unittest.TestCase):

    # def test_unbroadcasting(self):

    #     t1 = pensor(np.array([1, 2, 3]))
    #     t2 = pensor(np.array([4, 5, 6]))
    #     t3 = t1 + t2
    #     t3.sum().backward()

    #     assert isinstance(t3, pensor)
    #     assert isinstance(t3.data, np.ndarray)
    #     assert t3.data.shape == (3,)
    #     assert np.all(t3.data == np.array([5, 7, 9]))


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

    def test_softmax(self):

        t1 = Tensor(np.array([1, 2, -3]))
        t2 = t1.softmax()

        tt1 = torch.tensor([1, 2, -3], dtype=torch.float32)
        tt2 = torch.nn.functional.softmax(tt1, dim=0)

        assert isinstance(t2, Tensor)
        assert isinstance(t2.data, np.ndarray)
        assert t2.data.shape == (3,)
        assert np.allclose(t2.data, tt2.detach().numpy())

    def test_softmax_with_axis(self):
            
        t1 = Tensor(np.array([[1, 2, -3], [4, 5, 6]]))
        t2 = t1.softmax()

        tt1 = torch.tensor([[1, 2, -3], [4, 5, 6]], dtype=torch.float32)
        tt2 = torch.nn.functional.softmax(tt1, dim=1)

        assert isinstance(t2, Tensor)
        assert isinstance(t2.data, np.ndarray)
        assert t2.data.shape == (2, 3)
        assert np.allclose(t2.data, tt2.detach().numpy())

    def test_log_softmax(self):

        t1 = Tensor(np.array([1, 2, -3]))
        t2 = t1.log_softmax()

        tt1 = torch.tensor([1, 2, -3], dtype=torch.float32)
        tt2 = torch.nn.functional.log_softmax(tt1, dim=0)

        assert isinstance(t2, Tensor)
        assert isinstance(t2.data, np.ndarray)
        assert t2.data.shape == (3,)
        assert np.allclose(t2.data, tt2.detach().numpy())

import unittest
import numpy as np
import torch

from phgrad.ops import LogSoftmax

# Assuming your LogSoftmax class is defined here
# from your_module import LogSoftmax

class TestLogSoftmax(unittest.TestCase):

    def test_forward_backward(self):
        """Test the forward pass with a simple input."""
        input_np = np.array([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6]])
        input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

        ctx = LogSoftmax()
        # Your implementation
        log_softmax_np = LogSoftmax.forward(ctx, input_np, dim=1)

        # PyTorch implementation
        log_softmax_torch = torch.nn.functional.log_softmax(input_torch, dim=1)
        
        # Check if the forward outputs are close
        np.testing.assert_almost_equal(log_softmax_np, log_softmax_torch.detach().numpy(), decimal=5)

        grad_output = torch.tensor([[1.0, 0.0, 0.0], 
                                    [0.0, 1.0, 0.0]], dtype=torch.float32)
        # Backward pass
        log_softmax_torch.backward(grad_output)
        grad_np = LogSoftmax.backward(ctx, grad_output.numpy())

        # Check if the gradients are close
        np.testing.assert_almost_equal(grad_np, input_torch.grad.numpy(), decimal=5)
