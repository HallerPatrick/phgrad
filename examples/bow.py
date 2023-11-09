"""Bag of Words example.


Notes (8.11.23): We compare our and a torch implementation. We can observe following 
behavior.

* Params: Batch Size: 2, LR: 0.01, Epochs: 10
* The torch model directly starts to learn (increase of acc immediatly) until an accuracy of 95
* Our model learns nothing in the first epoch and then only increase per Epoch about 1 or 2 accuracy points, 
after 10 Epochs ~64 (74?) Acc

Why is that?


"""
import os
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from torchviz import make_dot
from graphviz import Source

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phgrad.engine import Tensor
from phgrad.nn import Linear, Module
from phgrad.loss import nllloss
from phgrad.optim import SGD, l1_regularization
from phgrad.fun import argmax


class BOWClassifier(Module):
    def __init__(self, vocab_size: int, num_classes: int) -> None:
        super().__init__()
        self.l1 = Linear(vocab_size, 64, bias=False)
        self.l2 = Linear(64, num_classes, bias=False)

    def forward(self, bow_vec: Tensor) -> Tensor:
        h = self.l1(bow_vec)
        h = h.relu()
        h = self.l2(h)
        return h.log_softmax(dim=1)

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()


class TorchBOWClassifier(torch.nn.Module):
    def __init__(self, vocab_size: int, num_classes: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(vocab_size, 64, bias=False)
        self.l2 = torch.nn.Linear(64, num_classes, bias=False)

    def forward(self, bow_vec):
        h = self.l1(bow_vec)
        h = torch.relu(h)
        h = self.l2(h)
        return torch.nn.functional.log_softmax(h, dim=1)


def make_bow_vector(sentence: str, word_to_ix: dict) -> Tensor:
    vec = np.zeros(len(word_to_ix))
    for word in sentence.split():
        vec[word_to_ix[word]] += 1
    return Tensor(np.expand_dims(vec, 0), requires_grad=False)


def make_target(label: int, label_to_ix: dict) -> Tensor:
    return Tensor(np.array([[label]]), requires_grad=False)


def load_imdb_dataset():
    # We want a even disbtibution of positive and negative reviews
    dataset = load_dataset("imdb")
    df = pd.DataFrame(dataset["train"])

    pos_reviews = df[df["label"] == 1]
    neg_reviews = df[df["label"] == 0]

    N = 100
    pos_samples = pos_reviews.sample(N)
    neg_samples = neg_reviews.sample(N)

    balanced_dataset = pd.concat([pos_samples, neg_samples])
    balanced_dataset = balanced_dataset.sample(frac=1).reset_index(drop=True)
    dataset_dict = {
        col: balanced_dataset[col].tolist() for col in balanced_dataset.columns
    }
    balanced_dataset = Dataset.from_dict(dataset_dict)
    return balanced_dataset


def main():
    # 1000 Samples
    # train_data = load_dataset("imdb", split="train[1000:1200]").shuffle()
    train_data = load_imdb_dataset()

    print(train_data["label"])

    word_to_ix = {}
    label_to_ix = {"neg": 0, "pos": 1}

    for sample in train_data["text"]:
        for word in sample.split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    print(f"Number of words in vocab: {len(word_to_ix)}")

    torch_classifier = TorchBOWClassifier(len(word_to_ix), 2)

    model = BOWClassifier(len(word_to_ix), 2)

    # Copy all params
    model.l1.weights.data = torch_classifier.l1.weight.detach().numpy()
    model.l2.weights.data = torch_classifier.l2.weight.detach().numpy()

    loss_function = nllloss

    optimizer = SGD(model.parameters(), lr=1)
    torch_optimizer = torch.optim.SGD(torch_classifier.parameters(), lr=0.01)

    batch_size = 2
    TORCH = True

    for epoch in range(10):
        print(f"Epoch {epoch}")
        pbar = tqdm(enumerate(train_data), total=len(train_data))

        total_correct = 0
        total_samples = 0
        currrent_batch_source = []
        currrent_batch_target = []
        for i, sample in pbar:
            # Batching is no problem!
            if i % batch_size != 0 and i != 0:
                currrent_batch_source.append(
                    make_bow_vector(sample["text"], word_to_ix)
                )
                currrent_batch_target.append(make_target(sample["label"], label_to_ix))

            if i % batch_size == 0 and i != 0:
                currrent_batch_source.append(
                    make_bow_vector(sample["text"], word_to_ix)
                )
                currrent_batch_target.append(make_target(sample["label"], label_to_ix))
                currrent_batch_source = [t.data for t in currrent_batch_source]
                currrent_batch_target = [t.data for t in currrent_batch_target]
                if TORCH:
                    bow_vec = torch.tensor(
                        np.concatenate(currrent_batch_source, axis=0),
                        dtype=torch.float32,
                    )
                    target = torch.tensor(
                        np.concatenate(currrent_batch_target, axis=0), dtype=torch.long
                    ).squeeze(1)
                    torch_optimizer.zero_grad()

                    log_probs = torch_classifier(bow_vec)
                    loss = torch.nn.functional.nll_loss(log_probs, target)
                    # exit()
                    # torch_classifier.backward(loss)
                    loss.backward()
                    torch_optimizer.step()

                    logits = log_probs.softmax(dim=1)
                    # print(logits.shape)
                    # print(logits)
                    pred_idxs = argmax(logits, dim=1)
                    # print(pred_idxs.shape)
                    # print(pred_idxs)

                    target_idxs = target.data
                    batch_correct = np.sum((pred_idxs.numpy() == target_idxs.numpy()))
                    total_correct += batch_correct
                    total_samples += batch_size

                    accuracy = total_correct / total_samples
                    accuracy = accuracy.item()
                    loss = loss.item()
                    # print(loss)

                bow_vec = Tensor(np.concatenate(currrent_batch_source, axis=0))
                target = Tensor(np.concatenate(currrent_batch_target, axis=0))
                optimizer.zero_grad()

                log_probs = model(bow_vec)
                loss = loss_function(log_probs, target)

                # loss.data += l1_regularization(model.parameters(), 0.01)

                loss.backward()
                optimizer.step()

                logits = log_probs.softmax()
                pred_idxs = argmax(logits, dim=1)

                target_idxs = np.squeeze(target.data, axis=1)
                batch_correct = np.sum((pred_idxs == target_idxs))
                total_correct += batch_correct
                total_samples += batch_size

                accuracy = total_correct / total_samples
                loss = loss.data[0]

                pbar.set_description(f"Loss: {loss:5.5f}, Accuracy: {accuracy:.2f}")

                currrent_batch_source = []
                currrent_batch_target = []


if __name__ == "__main__":
    main()
