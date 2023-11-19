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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phgrad.engine import Tensor
from phgrad.nn import Linear, Module, MLP
from phgrad.loss import nllloss
from phgrad.optim import SGD, l1_regularization


def make_bow_vector(sentence: str, word_to_ix: dict) -> Tensor:
    vec = np.zeros(len(word_to_ix))
    for word in sentence.split():
        vec[word_to_ix[word]] += 1
    return Tensor(np.expand_dims(vec, 0), requires_grad=False)


def make_target(label: int, label_to_ix: dict) -> Tensor:
    # Make a 2-hot vector for the label
    vec = np.zeros(2)
    vec[label] = 1
    return Tensor(np.expand_dims(vec, 0), requires_grad=False)


def load_imdb_dataset():
    # We want a even disbtibution of positive and negative reviews
    dataset = load_dataset("imdb")
    df = pd.DataFrame(dataset["train"])

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


def main():
    # 1000 Samples
    # train_data = load_dataset("imdb", split="train[1000:1200]").shuffle()

    device = "cuda"
    train_data = load_imdb_dataset()

    word_to_ix = {}
    label_to_ix = {"neg": 0, "pos": 1}

    for sample in train_data["text"]:
        for word in sample.split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    print(f"Number of words in vocab: {len(word_to_ix)}")

    model = MLP(len(word_to_ix), 256, 2)

    loss_function = nllloss

    optimizer = SGD(model.parameters(), lr=0.1)

    batch_size = 100

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

                bow_vec = Tensor(np.concatenate(currrent_batch_source, axis=0))
                target = Tensor(np.concatenate(currrent_batch_target, axis=0))
                optimizer.zero_grad()

                logits = model(bow_vec)
                log_probs = logits.log_softmax(dim=1)
                target = target.argmax(dim=1)
                loss = loss_function(log_probs, target)

                loss.backward()
                optimizer.step()

                logits = logits.softmax(dim=-1).numpy()
                pred_idxs = np.argmax(logits, axis=1)

                target_idxs = target.numpy()
                batch_correct = np.sum((pred_idxs == target_idxs))
                total_correct += batch_correct
                total_samples += batch_size

                accuracy = total_correct / total_samples

                pbar.set_description(
                    f"Loss: {loss.first_item:5.5f}, Accuracy: {accuracy.item():2.2f}, Current Correct: {batch_correct}/{batch_size}, Total Correct: {total_correct}/{total_samples}"
                )

                currrent_batch_source = []
                currrent_batch_target = []


if __name__ == "__main__":
    main()
