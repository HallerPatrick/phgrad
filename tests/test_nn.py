import unittest

import numpy as np
import torch

torch.manual_seed(69)

from torch.nn import Linear as TorchLinear


from phgrad.nn import Linear
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

        class MLP:
            def __init__(self) -> None:
                self.l1 = Linear(2, 2, bias=False)
                self.l2 = Linear(2, 1, bias=False)

            def __call__(self, x):
                x = self.l1(x)
                x = x.relu()
                x = self.l2(x)
                return x

            def parameters(self):
                return [self.l1.weights, self.l2.weights]

        torch_mlp = TorchMLP()
        torch_optimizer = torch.optim.SGD(torch_mlp.parameters(), lr=0.01)
        mlp = MLP()
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
                return torch.nn.functional.log_softmax(x, dim=1)

        class Classifier:
            def __init__(self) -> None:
                self.l1 = Linear(784, 10, bias=False)
                self.l2 = Linear(10, 10, bias=False)

            def __call__(self, x):
                x = self.l1(x)
                x = x.relu()
                x = self.l2(x)
                return x.log_softmax(dim=1)

            def parameters(self):
                return [self.l1.weights, self.l2.weights]

        torch_classifier = TorchClassifier()
        classifier = Classifier()

        optimizer = SGD(list(classifier.parameters()), lr=0.01)
        torch_optimizer = torch.optim.SGD(torch_classifier.parameters(), lr=0.01)

        classifier.l1.weights.data = torch_classifier.l1.weight.detach().numpy()
        classifier.l2.weights.data = torch_classifier.l2.weight.detach().numpy()


        for _ in range(100):
            optimizer.zero_grad()
            torch_optimizer.zero_grad()

            random_input = np.random.randn(1, 784)

            result = torch_classifier(torch.tensor(random_input, dtype=torch.float32))
            result2 = classifier(Tensor(np.array(random_input, dtype=np.float32)))
            print(torch.nn.functional.softmax(result))
            result = result.mean()
            print(result)
            result2 = result2.mean()

            result.backward()
            result2.backward()

            optimizer.step()
            torch_optimizer.step()

            np.testing.assert_allclose(result.detach().numpy(), result2.data, rtol=1e-3, atol=1e-3)

            # np.testing.assert_allclose(
            #     torch_classifier.l1.weight.detach().numpy(), classifier.l1.weights.data
            # )
            # np.testing.assert_allclose(
            #     torch_classifier.l2.weight.detach().numpy(), classifier.l2.weights.data
            # )
