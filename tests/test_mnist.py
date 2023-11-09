import unittest
import os
import hashlib
import requests
import gzip

import numpy as np
import torch


from phgrad.engine import Tensor
from phgrad.nn import Linear, MLP
from phgrad.optim import SGD
from phgrad.loss import nllloss

def fetch(url):
    fp = os.path.join("/tmp", hashlib.md5(url.encode("utf-8")).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

class TestMNIST(unittest.TestCase):

    def setUp(self) -> None:
        X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[
            0x10:
        ].reshape((-1, 28, 28))
        Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
        X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[
            0x10:
        ].reshape((-1, 28, 28))
        Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

        self.X_train = X_train.reshape(-1, 28 * 28) / 255.0
        self.X_test = X_test.reshape(-1, 28 * 28) / 255.0
        self.Y_train = np.eye(10)[Y_train.reshape(-1)]
        self.Y_test = np.eye(10)[Y_test.reshape(-1)]

    def setupModels(self):

        class TorchClassifier(torch.nn.Module):
            """A simple MLP classifier.

            We getting with and without a bias term a accuracy of 93%.

            """
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.l1 = torch.nn.Linear(784, 64, bias=False)
                self.l2 = torch.nn.Linear(64, 10, bias=False)

            def forward(self, x):
                x = self.l1(x)
                # print("TORCH: After l1", x.shape, x)
                x = torch.relu(x)
                # print("TORCH: After l2", x.shape, x)
                x = self.l2(x)
                # print("TORCH: After l3", x.shape, x)
                return x

        self.torch_mlp = TorchClassifier()
        self.mlp = MLP(784, 64, 10, bias=False)

        self.mlp.l1.weights.data = self.torch_mlp.l1.weight.data.numpy()
        self.mlp.l2.weights.data = self.torch_mlp.l2.weight.data.numpy()
    
    # @unittest.skip("Not implemented")
    def test_mlp(self):
        self.setupModels()

        optimizer = SGD(self.mlp.parameters(), lr=0.01)
        torch_optimizer = torch.optim.SGD(self.torch_mlp.parameters(), lr=0.01)

        batch_size = 2

        for epoch in range(10):
            for i in range(0, len(self.X_train), batch_size):
                optimizer.zero_grad()
                torch_optimizer.zero_grad()
                x = Tensor(self.X_train[i:i + batch_size])
                y = Tensor(np.argmax(self.Y_train[i:i + batch_size], axis=1))
                y_pred = self.mlp(x)
                y_pred = y_pred.log_softmax()
                loss = nllloss(y_pred, y, reduce="mean")
                loss.backward()
                optimizer.step()

            print(f"PhGrad -> Epoch {epoch}: {loss.first_item}")

        y_pred = self.mlp(Tensor(self.X_test))
        y_pred = np.argmax(y_pred.data, axis=1)

        accuracy = (y_pred == self.Y_test.argmax(axis=1)).mean()
        print(f"Accuracy: {accuracy}")


    
    @unittest.skip("Not implemented")
    def test_mlp_torch(self):
        self.setupModels()
        from torch import tensor

        torch_optimizer = torch.optim.SGD(self.torch_mlp.parameters(), lr=0.01)

        batch_size = 2

        for epoch in range(10):
            for i in range(0, len(self.X_train), batch_size):
                self.torch_mlp.zero_grad()
                x = tensor(self.X_train[i:i + batch_size], dtype=torch.float32)
                y = tensor(np.argmax(self.Y_train[i:i + batch_size], axis=1), dtype=torch.long)
                y_pred = self.torch_mlp(x)
                loss = torch.nn.functional.nll_loss(y_pred, y, reduction="mean")
                loss.backward()
                torch_optimizer.step()

            print(f"Torch -> Epoch {epoch}: {loss.item()}")

        y_pred = self.torch_mlp(tensor(self.X_test, dtype=torch.float32))
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_pred = y_pred.argmax(dim=1)

        accuracy = (y_pred == tensor(self.Y_test.argmax(axis=1), dtype=torch.long)).float().mean()
        print(f"Accuracy: {accuracy}")

    def test_combinedd(self):
        from torch import tensor
        self.setupModels()
        optimizer = SGD(self.mlp.parameters(), lr=0.01)
        torch_optimizer = torch.optim.SGD(self.torch_mlp.parameters(), lr=0.01)

        batch_size = 2

        for epoch in range(10):
            for i in range(0, len(self.X_train), batch_size):
                optimizer.zero_grad()
                x = Tensor(self.X_train[i:i + batch_size])
                y = Tensor(np.argmax(self.Y_train[i:i + batch_size], axis=1))

                x_torch = tensor(self.X_train[i:i + batch_size], dtype=torch.float32)
                y_torch = tensor(np.argmax(self.Y_train[i:i + batch_size], axis=1), dtype=torch.long)

                self.assertEqual(x.shape, x_torch.shape)
                self.assertEqual(y.shape, y_torch.shape)
                # np.testing.assert_equal(x.data, x_torch.data.numpy())
                # np.testing.assert_equal(y.data, y_torch.data.numpy())

                y_pred = self.mlp(x)
                y_pred_torch = self.torch_mlp(x_torch)
                # np.testing.assert_allclose(y_pred, y_pred_torch.data.numpy())

                y_pred = y_pred.log_softmax(dim=1)

                y_pred_torch = torch.nn.functional.log_softmax(y_pred_torch, dim=1)
                # np.testing.assert_allclose(y_pred, y_pred_torch.data.numpy())

                loss = nllloss(y_pred, y, reduce="mean")
                loss_torch = torch.nn.functional.nll_loss(y_pred_torch, y_torch, reduction="mean")

                np.testing.assert_allclose(loss.first_item, loss_torch.data.numpy())

                loss.backward()
                optimizer.step()

                loss_torch.backward()
                torch_optimizer.step()

            print(f"PhGrad -> Epoch {epoch}: {loss.first_item}")

        y_pred = self.mlp(Tensor(self.X_test))
        y_pred = np.argmax(y_pred.data, axis=1)

        accuracy = (y_pred == self.Y_test.argmax(axis=1)).mean()
        print(f"Accuracy: {accuracy}")
