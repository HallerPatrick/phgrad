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

    # @unittest.skip("Not implemented")
    def test_mlp(self):
        mlp = MLP(784, 64, 10, bias=False)
        optimizer = SGD(mlp.parameters(), lr=0.01)


        for epoch in range(10):
            for i in range(0, len(self.X_train), 32):
                optimizer.zero_grad()
                x = Tensor(self.X_train[i:i + 32])
                y = Tensor(np.argmax(self.Y_train[i:i + 32], axis=1))
                y_pred = mlp(x)
                # print(y_pred.shape, y.shape)
                loss = nllloss(y_pred, y, reduce="mean")
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}: {loss.first_item}")

        y_pred = mlp(Tensor(self.X_test))
        y_pred = np.argmax(y_pred.data, axis=1)

        accuracy = (y_pred == self.Y_test.argmax(axis=1)).mean()
        print(f"Accuracy: {accuracy}")

    @unittest.skip("Not implemented")
    def test_mlp_torch(self):
        from torch import tensor
        from torch.nn import NLLLoss

        torch_mlp = torch.nn.Sequential(
            torch.nn.Linear(784, 64, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10, bias=False),
            torch.nn.LogSoftmax(dim=1)
        )

        torch_optimizer = torch.optim.SGD(torch_mlp.parameters(), lr=0.01)

        for epoch in range(10):
            for i in range(0, len(self.X_train), 32):
                torch_mlp.zero_grad()
                x = tensor(self.X_train[i:i + 32], dtype=torch.float32)
                y = tensor(np.argmax(self.Y_train[i:i + 32], axis=1), dtype=torch.long)
                y_pred = torch_mlp(x)
                loss = torch.nn.functional.nll_loss(y_pred, y, reduction="mean")
                loss.backward()
                torch_optimizer.step()

            print(f"Epoch {epoch}: {loss.item()}")

        y_pred = torch_mlp(tensor(self.X_test, dtype=torch.float32))
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        y_pred = y_pred.argmax(dim=1)

        accuracy = (y_pred == tensor(self.Y_test.argmax(axis=1), dtype=torch.long)).float().mean()
        print(f"Accuracy: {accuracy}")
