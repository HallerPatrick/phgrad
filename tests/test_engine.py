import unittest

import torch

from phgrad.engine import Scalar, Linear, Pensor

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


    @unittest.skip("Not implemented yet")
    def test_linear(self):

        linear = Linear(2, 3)

        input = [Scalar(1.0), Scalar(2.0)]
        print(input)
        output = linear(input)
        output[0].grad = 1.0
        output[1].grad = 1.0
        output[2].grad = 1.0
        output[0].backward()
        output[1].backward()
        output[2].backward()


        linear_pt = torch.nn.Linear(2, 3)
        weights = [[scalar.value for scalar in row] for row in linear.weights]

        biases = [scalar.value for scalar in linear.biases]
        linear_pt.weight.data = torch.tensor(weights, dtype=torch.float64)
        linear_pt.bias.data = torch.tensor(biases, dtype=torch.float64)
        input_pt = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        output_pt = linear_pt(input_pt.unsqueeze(0))
        output_pt.backward(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64).unsqueeze(0), retain_graph=True)
        # output_pt[0].backward(retain_graph=True)
        # output_pt[1].backward(retain_graph=True)
        # output_pt[2].backward()

        tol = 1e-6
        # forward pass went well
        self.assertTrue(abs(output[0].value - output_pt.data[0].item()) < tol)
        self.assertTrue(abs(output[1].value - output_pt.data[1].item()) < tol)
        self.assertTrue(abs(output[2].value - output_pt.data[2].item()) < tol)
        # backward pass went well
        self.assertTrue(abs(input[0].grad - input_pt.grad[0].item()) < tol)
        self.assertTrue(abs(input[1].grad - input_pt.grad[1].item()) < tol)
        self.assertTrue(abs(linear.weights[0][0].grad - linear_pt.weight.grad[0][0].item()) < tol)
        self.assertTrue(abs(linear.weights[0][1].grad - linear_pt.weight.grad[0][1].item()) < tol)
        self.assertTrue(abs(linear.weights[1][0].grad - linear_pt.weight.grad[1][0].item()) < tol)
        self.assertTrue(abs(linear.weights[1][1].grad - linear_pt.weight.grad[1][1].item()) < tol)
        self.assertTrue(abs(linear.weights[2][0].grad - linear_pt.weight.grad[2][0].item()) < tol)
        self.assertTrue(abs(linear.weights[2][1].grad - linear_pt.weight.grad[2][1].item()) < tol)
        self.assertTrue(abs(linear.biases[0].grad - linear_pt.bias.grad[0].item()) < tol)
        self.assertTrue(abs(linear.biases[1].grad - linear_pt.bias.grad[1].item()) < tol)
        self.assertTrue(abs(linear.biases[2].grad - linear_pt.bias.grad[2].item()) < tol)


class TestPensor(unittest.TestCase):

    def test_pensor(self):
        
        p = Pensor([2, 3])
        self.assertEqual(p.shape, (2,))

        p = Pensor([[1, 2], [3, 4]])
        self.assertEqual(p.shape, (2, 2))

    def test_pensor_wrong_dimensions_length(self):
        # Check for that all dimensions are of same length
        with self.assertRaises(AssertionError):
            Pensor([[1, 2], [3, 4, 5]])

    def test_pensor_ones(self):
        p = Pensor.ones((2,))
        self.assertEqual(p.shape, (2, ))

        p = Pensor.ones((2, 3))
        self.assertEqual(p.shape, (2, 3))

        for i in range(2):
            for j in range(3):
                self.assertEqual(p[i][j].value, 1)

    def test_pensor_zeros(self):
        p = Pensor.zeros((2,))
        self.assertEqual(p.shape, (2, ))

        p = Pensor.zeros((2, 3))
        self.assertEqual(p.shape, (2, 3))

        for i in range(2):
            for j in range(3):
                self.assertEqual(p[i][j].value, 0)




