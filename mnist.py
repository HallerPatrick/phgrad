import requests, gzip, os, hashlib
import numpy as np

from phgrad.engine import Linear, softmax_scalars
from phgrad.fun import argmax
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

    def __init__(self) -> None:
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)

    def __call__(self, x):
        # x: (1, 28, 28)
        x = self.l1(x)
        for x_i in x:
            x_i.relu()
        x = self.l2(x)
        x = softmax_scalars(x)
        return x 
    
classifier = Classifier()

for sample, target in zip(X_train, Y_train):
    result = classifier(sample)
    pred, pred_idx = argmax(result)
    target = target.tolist()
    target = list(map(int, target))
    loss = nllloss(result, target)
    print(loss)
    exit()