import unittest

import numpy as np

from phgrad.loss import nllloss, cross_entropy
from phgrad.engine import PensorTensor as pensor


class TestLoss(unittest.TestCase):


    @unittest.skip("Not implemented")
    def test_cross_entropy(self):
            
        t1 = pensor(np.array([[1, 2, 3]]))
        t2 = pensor(np.array([2]))
        t3 = cross_entropy(t1, t2)

        assert isinstance(t3, pensor)
        assert isinstance(t3.data, np.ndarray)
        assert t3.data.shape == (3,)
        assert np.all(t3.data == np.array([3, 3, 3]))


    @unittest.skip("Torch Debugging")
    def test_torch_nllloss(self):

        from torch import tensor
        from torch.nn import NLLLoss
        import torch.nn.functional as F

        torch_nllloss = NLLLoss()
            
        t1 = tensor(np.array([[1., 2., 3.]]), requires_grad=True)
        t2 = tensor(np.array([2]), requires_grad=False)
        t3 = torch_nllloss(t1, t2)

        t3.backward()
        assert t3.data == -3.0

    def test_nllloss(self):

        t1 = pensor(np.array([[1, 2, 3]]))
        t2 = pensor(np.array([[2]]), requires_grad=False)
        t3 = nllloss(t1, t2)

        t3.backward()

        assert t3.data == -3.0
        assert t1.grad.shape == (1, 3)
        assert np.all(t1.grad == np.array([[0, 0, -1]]))
    
