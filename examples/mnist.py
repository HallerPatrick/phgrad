import gzip
import hashlib
import os
import sys

import numpy as np
import requests
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phgrad.engine import Tensor
from phgrad.fun import argmax
from phgrad.loss import nllloss
from phgrad.nn import MLP, Linear
from phgrad.optim import SGD


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


X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[
    0x10:
].reshape((-1, 28, 28))
Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[
    0x10:
].reshape((-1, 28, 28))
Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

print(X_train.shape, Y_train.shape)

X_train = X_train.reshape(-1, 28 * 28) / 255.0
X_test = X_test.reshape(-1, 28 * 28) / 255.0
Y_train = np.eye(10)[Y_train.reshape(-1)]
Y_test = np.eye(10)[Y_test.reshape(-1)]


class Classifier:
    """A simple MLP classifier.

    We getting with and without a bias term a accuracy of 85%. We also converge way slower
    than torch.

    Loss also seem to spike a lot.

    """

    def __init__(self) -> None:
        self.l1 = Linear(784, 64)
        self.l2 = Linear(64, 10)

    def __call__(self, x: Tensor):
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        return x.log_softmax(dim=1)

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()




class TorchClassifier(torch.nn.Module):
    """A simple MLP classifier.

    We getting with and without a bias term a accuracy of 93%.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.Linear(784, 64, bias=True)
        self.l2 = torch.nn.Linear(64, 10, bias=True)

    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


classifier = MLP(784, 64, 10)
torch_classifier = TorchClassifier()

optimizer = SGD(classifier.parameters(), lr=0.01)
torch_optimizer = torch.optim.SGD(torch_classifier.parameters(), lr=0.01)

total_correct = 0  # initialize total number of correct predictions
total_samples = 0

TORCH = False

for epoch in range(20):
    for i in range(0, len(X_train), 32):
        optimizer.zero_grad()

        x = Tensor(X_train[i : i + 32])
        y = Tensor(np.argmax(Y_train[i : i + 32], axis=1))
        y_pred = classifier(x)
        y_pred = y_pred.log_softmax(dim=1)
        loss = nllloss(y_pred, y, reduce="mean")
        loss.backward()
        optimizer.step()
        total_samples += 1
        # pbar.set_description(f"Loss: {loss.data[0]:5.5f}, Accuracy: {accuracy:.2f}")

    y_pred = classifier(Tensor(X_test))
    y_pred = np.argmax(y_pred.data, axis=1)

    accuracy = (y_pred == Y_test.argmax(axis=1)).mean()
    print(f"Epoch {epoch}: {loss.first_item:5.5f}, Accuracy: {accuracy:.2f}")
