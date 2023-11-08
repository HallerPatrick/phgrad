import unittest

import numpy as np
import torch

torch.manual_seed(69)

from torch.nn import Linear as TorchLinear


from phgrad.nn import Linear, MLP
from phgrad.engine import Tensor
from phgrad.optim import SGD


class TestLinearLayer(unittest.TestCase):
    def test_linear_layer(self):
        tlinear = TorchLinear(2, 1, bias=False)

        result = tlinear(torch.tensor([[1, 2]], dtype=torch.float32))

        linear = Linear(2, 1, bias=False)
        linear.weights.data = tlinear.weight.detach().numpy()

        result2 = linear(Tensor(np.array([[1, 2]], dtype=np.float32)))

        result.backward()
        result2.backward()

        self.assertEqual(result.shape, result2.shape)
        np.testing.assert_allclose(result.detach().numpy(), result2.data)
        np.testing.assert_allclose(tlinear.weight.grad, linear.weights.grad)

    def test_linear_layer_bias(self):
        tlinear = TorchLinear(2, 1, bias=True)

        result = tlinear(torch.tensor([[1, 2]], dtype=torch.float32))

        linear = Linear(2, 1, bias=True)
        linear.weights.data = tlinear.weight.detach().numpy()
        linear.biases.data = tlinear.bias.detach().numpy()

        result2 = linear(Tensor(np.array([[1, 2]], dtype=np.float32)))

        result.backward()
        result2.backward()

        self.assertEqual(result.shape, result2.shape)
        np.testing.assert_allclose(result.detach().numpy(), result2.data)
        np.testing.assert_allclose(tlinear.weight.grad, linear.weights.grad)
        np.testing.assert_allclose(tlinear.bias.grad, linear.biases.grad)

    def test_mlp(self):
        class TorchMLP(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.l1 = torch.nn.Linear(2, 2, bias=False)
                self.l2 = torch.nn.Linear(2, 1, bias=False)

            def forward(self, x):
                x = self.l1(x)
                x = torch.relu(x)
                x = self.l2(x)
                return x

        torch_mlp = TorchMLP()
        torch_optimizer = torch.optim.SGD(torch_mlp.parameters(), lr=0.01)
        mlp = MLP(2, 2, 1, bias=False)
        optimizer = SGD(mlp.parameters(), lr=0.01)
        mlp.l1.weights.data = torch_mlp.l1.weight.detach().numpy()
        mlp.l2.weights.data = torch_mlp.l2.weight.detach().numpy()

        result = torch_mlp(torch.tensor([[1, 2]], dtype=torch.float32))
        result2 = mlp(Tensor(np.array([[1, 2]], dtype=np.float32)))

        result.backward()
        result2.backward()

        self.assertEqual(result.shape, result2.shape)
        np.testing.assert_allclose(result.detach().numpy(), result2.data)
        np.testing.assert_allclose(torch_mlp.l1.weight.grad, mlp.l1.weights.grad)

        for _ in range(10):
            optimizer.zero_grad()
            torch_optimizer.zero_grad()

            result = torch_mlp(torch.tensor([[1, 2]], dtype=torch.float32))
            result2 = mlp(Tensor(np.array([[1, 2]], dtype=np.float32)))

            result.backward()
            result2.backward()

            optimizer.step()
            torch_optimizer.step()

            np.testing.assert_allclose(
                torch_mlp.l1.weight.detach().numpy(), mlp.l1.weights.data
            )
            np.testing.assert_allclose(
                torch_mlp.l2.weight.detach().numpy(), mlp.l2.weights.data
            )

    def test_classifier(self):
        class TorchClassifier(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.l1 = torch.nn.Linear(784, 10, bias=False)
                self.l2 = torch.nn.Linear(10, 10, bias=False)

            def forward(self, x):
                x = self.l1(x)
                x = torch.relu(x)
                x = self.l2(x)
                return x
                # return torch.nn.functional.log_softmax(x, dim=1)

        torch_classifier = TorchClassifier()
        classifier = MLP(784, 10, 10, bias=False)

        optimizer = SGD(list(classifier.parameters()), lr=0.01)
        torch_optimizer = torch.optim.SGD(torch_classifier.parameters(), lr=0.01)

        classifier.l1.weights.data = torch_classifier.l1.weight.detach().numpy()
        classifier.l2.weights.data = torch_classifier.l2.weight.detach().numpy()

        for _ in range(100):
            optimizer.zero_grad()
            torch_optimizer.zero_grad()

            # Batch of 32 random images
            random_input = np.random.randn(32, 784)

            result = torch_classifier(torch.tensor(random_input, dtype=torch.float32))
            result = torch.nn.functional.log_softmax(result, dim=1)
            result2 = classifier(Tensor(np.array(random_input, dtype=np.float32)))
            result2 = result2.log_softmax(dim=1)

            result = result.mean()
            result2 = result2.mean()

            result.backward()
            result2.backward()

            optimizer.step()
            torch_optimizer.step()

            np.testing.assert_allclose(
                result.detach().numpy(), result2.data, rtol=1e-3, atol=1e-3
            )

            np.testing.assert_allclose(
                torch_classifier.l1.weight.detach().numpy(), classifier.l1.weights.data
            )
            np.testing.assert_allclose(
                torch_classifier.l2.weight.detach().numpy(), classifier.l2.weights.data
            )

    def test_classifier_sparse_input(self):
        class TorchClassifier(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.l1 = torch.nn.Linear(784, 10, bias=False)
                self.l2 = torch.nn.Linear(10, 10, bias=False)

            def forward(self, x):
                x = self.l1(x)
                x = torch.relu(x)
                x = self.l2(x)
                return x
                # return torch.nn.functional.log_softmax(x, dim=1)

        torch_classifier = TorchClassifier()
        classifier = MLP(784, 10, 10, bias=False)

        optimizer = SGD(list(classifier.parameters()), lr=0.01)
        torch_optimizer = torch.optim.SGD(torch_classifier.parameters(), lr=0.01)

        classifier.l1.weights.data = torch_classifier.l1.weight.detach().numpy()
        classifier.l2.weights.data = torch_classifier.l2.weight.detach().numpy()

        for _ in range(100):
            optimizer.zero_grad()
            torch_optimizer.zero_grad()

            # Batch of 32 random images
            random_input = np.random.randn(32, 784)
            # About half of the input is zero
            random_input[random_input < 0] = 0

            result = torch_classifier(torch.tensor(random_input, dtype=torch.float32))
            result = torch.nn.functional.log_softmax(result, dim=1)
            result2 = classifier(Tensor(np.array(random_input, dtype=np.float32)))
            result2 = result2.log_softmax(dim=1)

            result = result.mean()
            result2 = result2.mean()

            result.backward()
            result2.backward()

            optimizer.step()
            torch_optimizer.step()

            np.testing.assert_allclose(
                result.detach().numpy(), result2.data, rtol=1e-3, atol=1e-3
            )

            np.testing.assert_allclose(
                torch_classifier.l1.weight.detach().numpy(), classifier.l1.weights.data
            )
            np.testing.assert_allclose(
                torch_classifier.l2.weight.detach().numpy(), classifier.l2.weights.data
            )
