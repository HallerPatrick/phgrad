"""Bag of Words example."""
import os
import sys

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phgrad.engine import Tensor
from phgrad.nn import Linear, Module
from phgrad.loss import nllloss
from phgrad.optim import SGD
from phgrad.fun import argmax

class BOWClassifier(Module):

    def __init__(self, vocab_size: int, num_classes: int) -> None:
        super().__init__()
        self.l1 = Linear(vocab_size, 64)
        self.l2 = Linear(64, num_classes)

    def forward(self, bow_vec: Tensor) -> Tensor:
        h = self.l1(bow_vec)
        h = h.relu()
        h = self.l2(h)
        return h.log_softmax()

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()    
    
def make_bow_vector(sentence: str, word_to_ix: dict) -> Tensor:
    vec = np.zeros(len(word_to_ix))
    for word in sentence.split():
        vec[word_to_ix[word]] += 1
    return Tensor(np.expand_dims(vec, 0), requires_grad=False)

def make_target(label: int, label_to_ix: dict) -> Tensor:
    return Tensor(np.array([[label]]), requires_grad=False)

def main():
    dataset = load_dataset("imdb")
    # dataset = dataset.shuffle()

    train_data = dataset["train"]

    word_to_ix = {}
    label_to_ix = {"neg": 0, "pos": 1}

    for sample in train_data:
        for word in sample["text"].split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    print(word_to_ix)

    model = BOWClassifier(len(word_to_ix), 2)
    loss_function = nllloss
    optimizer = SGD(model.parameters(), lr=0.01)

    total_correct = 0  # initialize total number of correct predictions
    total_samples = 0

    batch_size = 16

    for epoch in range(10):
        print(f"Epoch {epoch}")
        pbar = tqdm(enumerate(train_data), total=len(train_data))

        currrent_batch_source = []
        currrent_batch_target = []
        for i, sample in pbar:
            
            # Batching is no problem!
            if i % batch_size != 0 and i != 0:
                currrent_batch_source.append(make_bow_vector(sample["text"], word_to_ix))
                currrent_batch_target.append(make_target(sample["label"], label_to_ix))

            if i % batch_size == 0 and i != 0:
                currrent_batch_source = [t.data for t in currrent_batch_source]
                currrent_batch_target = [t.data for t in currrent_batch_target]
                bow_vec = Tensor(np.concatenate(currrent_batch_source, axis=0))
                target = Tensor(np.concatenate(currrent_batch_target, axis=0))
                optimizer.zero_grad()

                log_probs = model(bow_vec)
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()

                logits = log_probs.softmax()
                pred_idxs = argmax(logits, dim=1)

                target_idxs = target.data
                batch_correct = np.sum((pred_idxs == np.squeeze(target_idxs, axis=1)))
                total_correct += batch_correct
                total_samples += batch_size

                accuracy = total_correct / total_samples  # calculate overall accuracy
                pbar.set_description(f"Loss: {loss.data[0]:5.2f}, Accuracy: {accuracy:.2f}")

                currrent_batch_source = []
                currrent_batch_target = []
            

if __name__ == "__main__":
    main()