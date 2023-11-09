import unittest
import numpy as np
import torch

from phgrad.engine import Tensor as Tensor


class TestTensor(unittest.TestCase):
    def test_init(self):
        t = Tensor(np.array([1, 2, 3]))

        assert isinstance(t, Tensor)
        assert isinstance(t.data, np.ndarray)
        assert t.data.shape == (3,)

    def test_add(self):
        t1 = Tensor(np.array([1, 2, 3]))
        t2 = Tensor(np.array([4, 5, 6]))

        t3 = t1.add(t2)

        assert isinstance(t3, Tensor)
        assert isinstance(t3.data, np.ndarray)
        assert t3.data.shape == (3,)
        assert np.all(t3.data == np.array([5, 7, 9]))

    def test_add_backward(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.eye(3))

        t3 = t1.add(t2).sum()

        t3.backward()

        assert isinstance(t1.grad, np.ndarray)
        assert isinstance(t2.grad, np.ndarray)
        assert t1.grad.shape == (3, 3)
        assert t2.grad.shape == (3, 3)
        assert np.all(t1.grad == np.ones((3, 3)))
        assert np.all(t2.grad == np.ones((3, 3)))

    def test_mul_backward(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

        t3 = t1.mul(t2).sum()

        t3.backward()

        assert isinstance(t1.grad, np.ndarray)
        assert isinstance(t2.grad, np.ndarray)
        assert t1.grad.shape == (3, 3)
        assert t2.grad.shape == (3, 3)
        assert np.all(t1.grad == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        assert np.all(t2.grad == np.eye(3))
        assert np.all(t3.data == np.array([15]))

    def test_sub_backward(self):
        t1 = Tensor(np.array([1, 2, 3]))
        t2 = Tensor(np.array([4, 5, 6]))
        t3 = t1.sub(t2).sum()
        t3.backward()
        assert np.all(t1.grad == np.array([1, 1, 1]))
        assert np.all(t2.grad == np.array([-1, -1, -1]))

    def test_div_backward(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = Tensor(np.array([4.0, 5.0, 6.0]))
        t3 = t1.div(t2).sum()
        t3.backward()
        assert np.allclose(t1.grad, np.array([0.25, 0.2, 0.16666667]))
        assert np.allclose(t2.grad, np.array([-0.0625, -0.08, -0.08333333]))

    @unittest.skip("Not implemented")
    def test_pow_backward(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = t1.pow(2).sum()
        t2.backward()
        assert np.allclose(t1.grad, np.array([2.0, 4.0, 6.0]))

    def test_dot_backward(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.array([[2.0, 0.0, -2.0]]))
        t3 = t2.dot(t1).sum()
        t3.backward()

        assert np.allclose(
            t1.grad, np.array([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]])
        )
        assert np.allclose(t2.grad, np.eye(1))

    def test_log_softmax(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = t1.log_softmax()
        assert np.allclose(t2.data, np.array([-2.40760596, -1.40760596, -0.40760596]))

    def test_log_softmax_backward(self):
        t1 = Tensor(np.array([1.0, 2.0, 3.0]))
        t2 = t1.log_softmax().sum()
        t2.backward()
        # print(t1.grad)
        # assert np.allclose(t1.grad, np.array([0.09003057, 0.24472847, 0.66524096]))

    def test_transpose(self):
        t1 = Tensor(np.array([[1.0, 2.0, 3.0]]))

        t2 = t1.transpose((1, 0))

        assert np.allclose(t2.data, np.array([[1.0], [2.0], [3.0]]))
        assert t2.shape == (3, 1)

    def test_dot(self):
        t1 = Tensor(np.eye(3))
        t2 = Tensor(np.array([[2.0, 0.0, -2.0]]))

        t3 = t2.dot(t1)

        assert np.allclose(t3.data, np.array([[2.0, 0.0, -2.0]]))
        assert t3.shape == (1, 3)
