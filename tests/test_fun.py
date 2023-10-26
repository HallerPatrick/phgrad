import unittest

import torch
import numpy as np

from phgrad.loss import nllloss
from phgrad.engine import Scalar, Pensor
from phgrad.fun import softmax, argmax, softmax_scalars


class TestFuncs(unittest.TestCase):

    def test_softmax(self):

        p = Pensor.ones((4, 4))

        result = softmax(p)

        self.assertEqual(result.shape, (4, 4))
        self.assertEqual(sum(result[0]).value, 1.0)

    def test_argmax(self):
        
        p = Pensor.eye(4)

        result, idx = argmax(p, dim=1)

        self.assertEqual(result.shape, (4,))
        self.assertEqual(idx, 3)



    def test_softmax(self):

        a = Scalar(-4.0)
        b = Scalar(2.0)

        c = softmax_scalars([a, b])
        c[0].grad = 1.0
        c[1].grad = 1.0
        c[0].backward()
        c[1].backward()

        amg, bmg = a, b

        a = torch.tensor([-4.0, 2.0], dtype=torch.float64, requires_grad=True)
        c = torch.softmax(a, dim=0)

        c[0].backward(retain_graph=True)
        c[1].backward()

        apt, bpt = a, b

        tol = 1e-6
        # forward pass went well
        self.assertTrue(abs(amg.value - apt.data[0].item()) < tol)
        self.assertTrue(abs(bmg.value - apt.data[1].item()) < tol)
        # backward pass went well
        self.assertTrue(abs(amg.grad - apt.grad[0].item()) < tol)
        self.assertTrue(abs(bmg.grad - apt.grad[1].item()) < tol)

    def test_nllloss(self):

        def softmax(x):
            return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
        
        labels = [0, 2, 1, 3]
        logits = np.array([
            [3.5, -3.45, 0.23, 1.25],
            [-2.14, 0.54, 2.67, -5.23],
            [-1.34, 5.01, -1.54, -1.17],
            [ -2.98, -1.37, 1.54,5.23]
        ])
        print(logits.shape)
        probs = softmax(logits)
        log_probs = np.log(softmax(logits))
        nll = -(log_probs[range(len(labels)), labels])
        print(nll)

        print(log_probs)

    # def test_nllloss_backward(self):

    #     x = torch.randn(10, 10, requires_grad=True)
    #     target = torch.randint(0, 10, (10,))

    #     loss = torch.nn.functional.nll_loss(x, target)
    #     loss.backward()

    #     x2 = torch.randn(10, 10, requires_grad=True)
    #     target2 = torch.randint(0, 10, (10,))

    #     loss2 = torch.nn.functional.nll_loss(x2, target2)
    #     loss2.backward()

    #     self.assertEqual(x.grad, x2.grad)
