import os
import sys
from typing import Tuple

from tqdm import tqdm
from datasets import load_dataset
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phgrad import Tensor
from phgrad.nn import Embedding, Module, Linear
from phgrad.nn.rnn import RNN
from phgrad import types as phtypes
from phgrad.loss import nllloss
from phgrad.optim import SGD

class LM(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, device="cpu"):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, device=device)
        self.rnn = RNN(embedding_dim, hidden_size, device=device)
        self.decoder = Linear(hidden_size, vocab_size, device=device)

    def forward(self, x: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.embedding(x)
        x, hidden_state = self.rnn(x, hidden_state)
        x = self.decoder(x)
        return x, hidden_state



def main():
    text = "".join(load_dataset('PatrickHaller/hurt')["train"]["text"])

    vocab = sorted(set(text))

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    seq_length = 100

    examples_per_epoch = len(text) // (seq_length + 1)

    # It is a RNN, so we have to do backpropagation through time if we want to forward batches

    model = LM(len(vocab), 256, 256)
    optimizer = SGD(model.parameters())

    hidden_state = Tensor(np.zeros((256), dtype=np.float32))

    pbar = tqdm(range(0, len(text_as_int) - seq_length, seq_length))
    # pbar = range(0, len(text_as_int) - seq_length, seq_length)
    for i in pbar:
        optimizer.zero_grad()
        sequence = Tensor(text_as_int[i : i + seq_length], dtype=phtypes.int64)
        res, hidden_state = model(sequence, hidden_state)
        res = res.log_softmax(dim=1)
        loss = nllloss(res, sequence, reduce="mean")
        loss.backward()
        optimizer.step()
        hidden_state = hidden_state.detach()

        pbar.set_description(f"Loss: {loss}")

    # Generate some text
    start_string = "I "

    hidden_state = Tensor(np.zeros((256), dtype=np.float32))

    sequence = Tensor([char2idx[c] for c in start_string], dtype=phtypes.int64)
    print("Input:", start_string, end="")

    for i in range(1000):
        res, hidden_state = model(sequence, hidden_state)
        res = res.softmax(dim=1)
        res = res.detach().numpy()[0]
        idx = np.random.choice(len(vocab), p=res)
        sequence = Tensor([idx], dtype=phtypes.int64)
        print(idx2char[idx], end="")











if __name__ == "__main__":
    main()