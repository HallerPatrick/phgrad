import sys

import requests, gzip, os, hashlib
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phgrad.engine import Tensor
from phgrad.nn import Linear 
from phgrad.fun import argmax
from phgrad.loss import nllloss
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
        return x.log_softmax()
    
    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

import torch

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

max_steps = 60000

classifier = Classifier()
torch_classifier = TorchClassifier()

optimizer = SGD(list(classifier.parameters()), lr=0.01)
torch_optimizer = torch.optim.SGD(torch_classifier.parameters(), lr=0.01)

total_correct = 0  # initialize total number of correct predictions
total_samples = 0

TORCH = False

pbar = tqdm(enumerate(zip(X_train, Y_train)), total=max_steps)
for step, (sample, target) in pbar:

    if TORCH:
        torch_classifier.zero_grad()

        torch_result = torch_classifier(torch.tensor(sample, dtype=torch.float32).unsqueeze(0))
        loss = torch.nn.functional.nll_loss(torch_result, torch.tensor([np.argmax(target)]))

        pred_idx = argmax(torch_result, dim=1)
        target_idx = np.argmax(target, axis=0)
        total_correct += int(pred_idx == target_idx)
        loss.backward()
        torch_optimizer.step()

    else:
        optimizer.zero_grad()

        sample = Tensor(np.float32(np.expand_dims(sample, 0)))
        result = classifier(sample)
        logits = result.softmax()
        pred_idx = argmax(logits, dim=1)
        target = target.tolist()
        target = list(map(int, target))
        target_idx = np.argmax(target, axis=0)
        target_vec = Tensor(np.array([[target_idx]]), requires_grad=True)
        loss = nllloss(result, target_vec)
        loss.backward()
        optimizer.step()
        total_correct += int(pred_idx == target_idx)

    total_samples += 1

    if step == max_steps:
        break

    accuracy = total_correct / total_samples  # calculate overall accuracy
    if TORCH:
        pbar.set_description(f"Loss: {loss.data.item():5.5f}, Accuracy: {accuracy:.2f}")
    else:
        pbar.set_description(f"Loss: {loss.data[0]:5.5f}, Accuracy: {accuracy:.2f}")

accuracy = total_correct / total_samples  # calculate overall accuracy
print(f"Final accuracy: {accuracy}")
