import unittest
import pytest
import numpy as np

from utils import requires_torch, load_mnist

from phgrad.engine import Tensor
from phgrad.nn import MLP
from phgrad.optim import SGD
from phgrad.loss import nllloss


@pytest.mark.slow
class TestMNIST(unittest.TestCase):
    def setUp(self) -> None:
        X_train, Y_train, X_test, Y_test = load_mnist()
        self.X_train = X_train.reshape(-1, 28 * 28) / 255.0
        self.X_test = X_test.reshape(-1, 28 * 28) / 255.0
        self.Y_train = np.eye(10)[Y_train.reshape(-1)]
        self.Y_test = np.eye(10)[Y_test.reshape(-1)]

    # @unittest.skip("Not implemented")
    def test_mlp(self):
        mlp = MLP(784, 64, 10, bias=False)
        optimizer = SGD(mlp.parameters(), lr=0.01)

        current_accuracy = 0
        for epoch in range(5):
            for i in range(0, len(self.X_train), 32):
                optimizer.zero_grad()
                x = Tensor(self.X_train[i : i + 32])
                y = Tensor(np.argmax(self.Y_train[i : i + 32], axis=1))
                y_pred = mlp(x)
                y_pred = y_pred.log_softmax(dim=1)
                loss = nllloss(y_pred, y, reduce="mean")
                loss.backward()
                optimizer.step()

            y_pred = mlp(Tensor(self.X_test))
            y_pred = np.argmax(y_pred.data, axis=1)

            accuracy = (y_pred == self.Y_test.argmax(axis=1)).mean()
            print(f"Epoch {epoch}: {accuracy:.2f}")

            # Stupid test, but good for regression
            self.assertGreater(accuracy, current_accuracy)
            current_accuracy = accuracy

    @requires_torch
    @unittest.skip("Not implemented")
    def test_mlp_torch(self):
        import torch
        from torch import tensor

        torch_mlp = torch.nn.Sequential(
            torch.nn.Linear(784, 64, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10, bias=False),
            torch.nn.LogSoftmax(dim=1),
        )

        torch_optimizer = torch.optim.SGD(torch_mlp.parameters(), lr=0.01)

        for epoch in range(10):
            for i in range(0, len(self.X_train), 32):
                torch_mlp.zero_grad()
                x = tensor(self.X_train[i : i + 32], dtype=torch.float32)
                y = tensor(
                    np.argmax(self.Y_train[i : i + 32], axis=1), dtype=torch.long
                )
                y_pred = torch_mlp(x)
                loss = torch.nn.functional.nll_loss(y_pred, y, reduction="mean")
                loss.backward()
                torch_optimizer.step()

            print(f"Epoch {epoch}: {loss.item()}")

        y_pred = torch_mlp(tensor(self.X_test, dtype=torch.float32))
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_pred = y_pred.argmax(dim=1)

        accuracy = (
            (y_pred == tensor(self.Y_test.argmax(axis=1), dtype=torch.long))
            .float()
            .mean()
        )
        print(f"Accuracy: {accuracy}")
