import unittest

import torch

from phgrad.engine import Scalar, softmax_scalars

class TestScalar(unittest.TestCase):

    def test_sanity_check(self):

        x = Scalar(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xmg, ymg = x, y

        x = torch.Tensor([-4.0]).double()
        x.requires_grad = True
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xpt, ypt = x, y

        # forward pass went well
        self.assertEqual(ymg.value, ypt.data.item())
        # backward pass went well
        self.assertEqual(xmg.grad, xpt.grad.item())

    def test_more_ops(self):

        a = Scalar(-4.0)
        b = Scalar(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        g.backward()
        amg, bmg, gmg = a, b, g

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([2.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        tol = 1e-6
        # forward pass went well
        self.assertTrue(abs(gmg.value - gpt.data.item()) < tol)
        self.assertTrue(abs(amg.grad - apt.grad.item()) < tol)
        self.assertTrue(abs(bmg.grad - bpt.grad.item()) < tol)

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

