import unittest

import numpy as np
import torch

from phgrad.engine import Tensor as Tensor
from phgrad.ops import unbroadcast


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

    def test_log_softmax(self):

        t1 = Tensor(np.array([1, 2, -3]))
        t2 = t1.log_softmax()

        tt1 = torch.tensor([1, 2, -3], dtype=torch.float32)
        tt2 = torch.nn.functional.log_softmax(tt1, dim=0)

        assert isinstance(t2, Tensor)
        assert isinstance(t2.data, np.ndarray)
        assert t2.data.shape == (3,)
        assert np.allclose(t2.data, tt2.detach().numpy())