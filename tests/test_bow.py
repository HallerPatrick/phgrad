from typing import List
import unittest

import pytest
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

from phgrad.nn import MLP
from phgrad.loss import nllloss
from phgrad.optim import SGD
from phgrad.engine import Tensor

from utils import requires_torch


def load_imdb_dataset(split="train"):
    # We want a even disbtibution of positive and negative reviews
    dataset = load_dataset("imdb")
    df = pd.DataFrame(dataset[split])

    pos_reviews = df[df["label"] == 1]
    neg_reviews = df[df["label"] == 0]

    N = 10000
    pos_samples = pos_reviews.sample(N)
    neg_samples = neg_reviews.sample(N)

    balanced_dataset = pd.concat([pos_samples, neg_samples])
    balanced_dataset = balanced_dataset.sample(frac=1).reset_index(drop=True)
    dataset_dict = {
        col: balanced_dataset[col].tolist() for col in balanced_dataset.columns
    }
    balanced_dataset = Dataset.from_dict(dataset_dict)
    return balanced_dataset


def make_bow_vector(sentences: List[str], word_to_ix: dict) -> Tensor:
    vec = np.zeros((len(sentences), len(word_to_ix)))
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            vec[i][word_to_ix[word]] += 1
    return Tensor(vec, requires_grad=False)


def make_target(label: List[int]) -> Tensor:
    return Tensor(np.array(label), requires_grad=False)


@pytest.mark.slow
@requires_torch
class TestBoW(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = load_imdb_dataset()
        self.test_dataset = load_imdb_dataset("test")

    def test_bow(self):
        word_to_ix = {}
        for sent in self.dataset["text"]:
            for word in sent.split():
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

        for sent in self.test_dataset["text"]:
            for word in sent.split():
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

        V = len(word_to_ix)
        num_classes = 2

        model = MLP(V, 64, num_classes, bias=True)
        criterion = nllloss
        optimizer = SGD(model.parameters(), lr=0.01)

        train_data_len = len(self.dataset)

        eval_step = train_data_len // 10

        for _ in range(10):
            for i in tqdm(range(0, train_data_len, 32)):
                optimizer.zero_grad()
                bow_vec = make_bow_vector(self.dataset["text"][i : i + 32], word_to_ix)
                target = make_target(self.dataset["label"][i : i + 32])
                y_pred = model(bow_vec)
                y_pred = y_pred.log_softmax(dim=1)
                loss = criterion(y_pred, target)
                loss.backward()
                optimizer.step()

                # Perform evaluation at regular intervals
                if (i // 32) % eval_step == 0 and i != 0:
                    # Evaluation
                    bow_vec = make_bow_vector(self.test_dataset["text"], word_to_ix)
                    target = make_target(self.test_dataset["label"])
                    y_pred = model(bow_vec)
                    y_pred = y_pred.log_softmax(dim=1)
                    loss = criterion(y_pred, target)
                    accuracy = (np.argmax(y_pred.data, axis=1) == target.data).mean()
                    print(f"Evaluation at step {i // 32}, Accuracy: {accuracy:.2f}")

            # Final evaluation after epoch
            bow_vec = make_bow_vector(self.test_dataset["text"], word_to_ix)
            target = make_target(self.test_dataset["label"])
            y_pred = model(bow_vec)
            y_pred = y_pred.log_softmax(dim=1)
            loss = criterion(y_pred, target)
            accuracy = (np.argmax(y_pred.data, axis=1) == target.data).mean()
            print(f"Final Accuracy after epoch: {accuracy:.2f}")
