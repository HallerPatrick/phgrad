import unittest

from functools import partial

import numpy as np

try:
    import cupy as cp
    from phgrad.backends.cuda import LogSoftmax, Softmax
except ImportError:
    cp = None

from utils import requires_torch, requires_cupy

from phgrad.engine import Tensor as Tensor


def tensor_with_cuda(*args, **kwargs):
    return partial(Tensor, *args, **kwargs, device="cuda")


TensorType = Tensor
Tensor = tensor_with_cuda()


@requires_cupy
class TestReshape(unittest.TestCase):
    def test_reshape(self):
        tensor = Tensor(np.array([1, 2, 3, 4, 5, 6]))
        tensor = tensor.reshape((2, 3))
        assert isinstance(tensor, TensorType)
        assert isinstance(tensor.data, cp.ndarray)
        assert tensor.data.shape == (2, 3)


@requires_cupy
class TestOps(unittest.TestCase):
    def test_add(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.eye(3))
        t3 = t1 + t2
        t3.sum().backward()

        assert isinstance(t3, TensorType)
        assert isinstance(t3.data, cp.ndarray)
        assert t3.data.shape == (3, 3)
        # assert np.all(t3.data == np.array([5, 7, 9]))

    def test_add_with_broadcasting(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.array([4, 5, 6]))
        t3 = t1 + t2
        t3.sum().backward()

        assert isinstance(t3.data, cp.ndarray)
        assert t3.data.shape == (3, 3)

    def test_relu(self):
        t1 = Tensor(np.array([1, 2, -3]))
        t2 = t1.relu()

        assert isinstance(t2.data, cp.ndarray)
        assert t2.data.shape == (3,)
        assert np.all(t2.data == cp.array([1, 2, 0]))

    def test_sigmoid(self):
        t1 = Tensor(np.array([1, 2, -3]))
        t2 = t1.sigmoid()
        assert isinstance(t2.data, cp.ndarray)
        assert t2.data.shape == (3,)
        assert np.allclose(t2.data, cp.array([0.73105858, 0.88079708, 0.04742587]))

    @requires_torch
    def test_softmax(self):
        import torch

        t1 = Tensor(np.array([[1, 2, -3]]))
        t2 = t1.softmax(dim=-1)

        tt1 = torch.tensor([[1, 2, -3]], dtype=torch.float32, requires_grad=True)
        tt2 = torch.nn.functional.softmax(tt1, dim=-1)

        assert isinstance(t2.data, cp.ndarray)
        assert t2.data.shape == (1, 3)
        assert np.allclose(t2.data, tt2.detach().numpy())
        # assert np.allclose(t1.grad, tt1.grad.detach().numpy())

    @requires_torch
    @unittest.skip("Not implemented")
    def test_softmax_with_axis(self):
        import torch

        t1 = Tensor(np.array([[1, 2, -3], [4, 5, 6]]))
        t2 = t1.softmax()

        tt1 = torch.tensor([[1, 2, -3], [4, 5, 6]], dtype=torch.float32)
        tt2 = torch.nn.functional.softmax(tt1, dim=1)

        assert isinstance(t2.data, cp.ndarray)
        assert t2.data.shape == (2, 3)
        assert np.allclose(t2.data, tt2.detach().numpy())

    def test_add_backward(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.eye(3))

        t3 = t1.add(t2).sum()

        t3.backward()

        assert t1.grad.shape == (3, 3)
        assert t2.grad.shape == (3, 3)
        assert np.all(t1.grad == cp.ones((3, 3)))
        assert np.all(t2.grad == cp.ones((3, 3)))

    def test_mul_backward(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        t3 = t1.mul(t2).sum()

        t3.backward()

        assert t1.grad.shape == (3, 3)
        assert t2.grad.shape == (3, 3)
        assert np.all(t1.grad == cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        assert np.all(t2.grad == cp.eye(3))
        assert np.all(t3.data == cp.array([15]))

    def test_sub_backward(self):
        t1 = Tensor(np.array([1, 2, 3]))
        t2 = Tensor(np.array([4, 5, 6]))
        t3 = t1.sub(t2).sum()
        t3.backward()
        assert np.all(t1.grad == cp.array([1, 1, 1]))
        assert np.all(t2.grad == cp.array([-1, -1, -1]))

    def test_div_backward(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = Tensor(np.array([4.0, 5.0, 6.0]))
        t3 = t1.div(t2).sum()
        t3.backward()
        assert np.allclose(t1.grad, cp.array([0.25, 0.2, 0.16666667]))
        assert np.allclose(t2.grad, cp.array([-0.0625, -0.08, -0.08333333]))

    @unittest.skip("Not implemented")
    def test_pow_backward(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = t1.pow(2).sum()
        t2.backward()
        assert np.allclose(t1.grad, cp.array([2.0, 4.0, 6.0]))

    def test_dot_backward(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.array([[2.0, 0.0, -2.0]]))
        t3 = t2.dot(t1).sum()
        t3.backward()

        assert np.allclose(
            t1.grad, cp.array([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]])
        )
        assert np.allclose(t2.grad, cp.eye(1))

    def test_transpose(self):
        t1 = Tensor(np.array([[1.0, 2.0, 3.0]]))

        t2 = t1.transpose((1, 0))

        assert np.allclose(t2.data, cp.array([[1.0], [2.0], [3.0]]))
        assert t2.shape == (3, 1)

    def test_dot(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.array([[2.0, 0.0, -2.0]]))

        t3 = t2.dot(t1)

        assert np.allclose(t3.data, cp.array([[2.0, 0.0, -2.0]]))
        assert t3.shape == (1, 3)


@requires_torch
@requires_cupy
class TestLogSoftmax(unittest.TestCase):
    def test_forward_backward(self):
        """Test the forward pass with a simple input."""
        import torch

        input_np = cp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=cp.float32)
        input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

        ctx = LogSoftmax()
        log_softmax_np = LogSoftmax.forward(ctx, input_np, dim=1)
        # log_softmax_np = LogSoftmax._forward(ctx, input_np, dim=1)
        log_softmax_torch = torch.nn.functional.log_softmax(input_torch, dim=1)

        np.testing.assert_almost_equal(
            log_softmax_np.get(), log_softmax_torch.detach().numpy(), decimal=3
        )

        grad_output = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        log_softmax_torch.backward(grad_output)
        grad_np = LogSoftmax.backward(
            ctx, cp.array(grad_output.numpy(), dtype=cp.float32)
        )

        np.testing.assert_almost_equal(
            grad_np.get(), input_torch.grad.numpy(), decimal=3
        )


@requires_torch
@requires_cupy
class TestSoftmax(unittest.TestCase):
    def test_forward_backward(self):
        import torch

        input_np = cp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=cp.float32)
        input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

        ctx = Softmax()
        softmax_np = Softmax.forward(ctx, input_np, dim=-1)
        softmax_torch = torch.nn.functional.softmax(input_torch, dim=-1)

        np.testing.assert_almost_equal(
            softmax_np.get(), softmax_torch.detach().numpy(), decimal=3
        )

        grad_output = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )

        softmax_torch.backward(grad_output)
        # TODO: Check for grad
        # Check for grad
        # grad_np = Softmax.backward(ctx, cp.array(grad_output.numpy(), dtype=cp.float32))
        # np.testing.assert_almost_equal(
        #     grad_np.get(), input_torch.grad.numpy(), decimal=3
        # )


@requires_cupy
class TestCat(unittest.TestCase):
    def test_cat(self):
        t1, t2 = Tensor(np.array([0.1, 0.2])), Tensor(np.array([0.3, 0.4]))
        t3 = t1.cat((t2,))
        print(t3.shape)
        np.testing.assert_equal(
            t3.data.get(), cp.array([0.1, 0.2, 0.3, 0.4], dtype=cp.float32).get()
        )

    def test_cat_dim1(self):
        t1, t2 = (
            Tensor(np.array([[0.1, 0.2], [0.3, 0.4]])),
            Tensor(np.array([[0.5, 0.6], [0.7, 0.8]])),
        )
        t3 = t1.cat((t2,), dim=1)
        np.testing.assert_equal(
            t3.data.get(),
            cp.array(
                [[0.1, 0.2, 0.5, 0.6], [0.3, 0.4, 0.7, 0.8]], dtype=cp.float32
            ).get(),
        )
