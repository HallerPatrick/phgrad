from typing import Tuple

import torch
from torch import nn
from torch import Tensor
from torch.optim import SGD
from torch.nn import functional as F

import numpy as np
from datasets import load_dataset

from tqdm import tqdm


class LM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.embedding(x)
        x, hidden_state = self.rnn(x.squeeze(0), hidden_state)
        x = self.decoder(x)
        return x, hidden_state


def main():
    text = "".join(load_dataset("PatrickHaller/hurt")["train"]["text"])
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    epochs = 1
    seq_length = 50
    model = LM(len(vocab), 256, 256)
    optimizer = SGD(model.parameters(), lr=0.1)
    hidden_state = Tensor(np.zeros((1, 256), dtype=np.float32))

    for _ in range(epochs):
        pbar = tqdm(range(0, len(text_as_int) - seq_length, seq_length))
        for i in pbar:
            optimizer.zero_grad()
            sequence = torch.tensor([text_as_int[i : i + seq_length]])
            target_sequence = torch.tensor([text_as_int[i + 1 : i + seq_length + 1]])
            res, hidden_state = model(sequence, hidden_state)
            res = res.log_softmax(dim=1)
            loss = F.nll_loss(res, target_sequence.squeeze(dim=0), reduction="mean")
            loss.backward()
            optimizer.step()
            hidden_state = hidden_state.detach()
            pbar.set_description(f"loss: {loss.item():.4f}")

    hidden_state = Tensor(np.zeros((1, 256), dtype=np.float32))
    for _ in range(100):
        sequence = torch.tensor([text_as_int[:seq_length]])
        res, hidden_state = model(sequence, hidden_state)
        res = res.log_softmax(dim=1)
        res = res.detach().numpy().squeeze()

        print(">?")
        print("".join(idx2char[text_as_int[:seq_length]]))
        print(">1")
        print("".join(idx2char[res.argmax(axis=1)]))


if __name__ == "__main__":
    main()
