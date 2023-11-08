import unittest

import numpy as np

from phgrad.loss import nllloss, cross_entropy
from phgrad.engine import Tensor as Tensor


class TestLoss(unittest.TestCase):

    def test_single_class_correct_prediction(self):
        """Test NLL loss with a correct prediction for a single class."""
        inputs = Tensor(np.array([[np.log(0.1), np.log(0.9)]]))  # Log probabilities; correct class is class 1
        targets = Tensor(np.array([1]))
        loss = nllloss(inputs, targets)
        self.assertAlmostEqual(loss.first_item, -np.log(0.9), places=5)  # Loss should be less than -log(0.9)

    def test_single_class_incorrect_prediction(self):
        """Test NLL loss with an incorrect prediction for a single class."""
        inputs = Tensor(np.array([[np.log(0.9), np.log(0.1)]]))  # Log probabilities; correct class is class 1
        targets = Tensor(np.array([1]))
        loss = nllloss(inputs, targets)
        self.assertGreater(loss.first_item, -np.log(0.9))  # Loss should be greater than -log(0.9)

    def test_multiple_class_predictions(self):
        """Test NLL loss with multiple class predictions."""
        inputs = Tensor(np.array([
            [np.log(0.1), np.log(0.9)],
            [np.log(0.8), np.log(0.2)]
        ], dtype=np.float32)) # Log probabilities; correct classes are 1 and 0
        targets = Tensor(np.array([1, 0]))

        # Mean reduction
        mean_loss = nllloss(inputs, targets, reduce="mean")
        expected_mean_loss = (-np.log(0.9) - np.log(0.8)) / 2
        self.assertAlmostEqual(mean_loss.first_item, expected_mean_loss, places=5)

        # # Sum reduction
        sum_loss = nllloss(inputs, targets, reduce="sum")
        expected_sum_loss = -np.log(0.9) - np.log(0.8)
        self.assertAlmostEqual(sum_loss.first_item, expected_sum_loss, places=5)

    @unittest.skip("Not implemented")
    def test_cross_entropy(self):
            
        t1 = Tensor(np.array([[1, 2, 3]]))
        t2 = Tensor(np.array([2]))
        t3 = cross_entropy(t1, t2)

        assert isinstance(t3, Tensor)
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

        t1 = Tensor(np.array([[1, 2, 3]]))
        t2 = Tensor(np.array([[2]]), requires_grad=False)
        t3 = nllloss(t1, t2)

        t3.backward()

        assert t3.data == -3.0
        assert t1.grad.shape == (1, 3)
        assert np.all(t1.grad == np.array([[0, 0, -1]]))

    def test_gradient_propagation(self):
        inputs = Tensor(np.array([[np.log(0.1), np.log(0.9)]]), requires_grad=True)  # Log probabilities; correct class is 1
        targets = Tensor(np.array([1]))
        
        loss = nllloss(inputs, targets)
        loss.backward()  # Compute gradients

        # Expected gradient: -1 for the true class, 0 for others
        expected_grad = np.array([[0, -1]])
        np.testing.assert_array_almost_equal(inputs.grad, expected_grad)

