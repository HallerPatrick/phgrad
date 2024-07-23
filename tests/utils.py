import gzip
import hashlib
import importlib
import os

import numpy as np
import pytest
import requests

requires_torch = pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires torch",
)

requires_cupy = pytest.mark.skipif(
    not importlib.util.find_spec("cupy"),
    reason="requires cupy",
)


def fetch(url):
    fp = os.path.join("/tmp", hashlib.md5(url.encode("utf-8")).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)

    breakpoint()
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


def _load_mnist():
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[
        0x10:
    ].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[
        0x10:
    ].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test


def load_mnist():
    from datasets import load_dataset

    ds = load_dataset("ylecun/mnist")

    def to_numpy(x):
        x["image"] = np.array(x["image"])

        return x

    ds = ds.map(to_numpy)

    X_train = np.stack([x["image"] for x in ds["train"]])
    Y_train = np.array([x["label"] for x in ds["train"]])
    X_test = np.stack([x["image"] for x in ds["test"]])
    Y_test = np.array([x["label"] for x in ds["test"]])
    return X_train, Y_train, X_test, Y_test
