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
from phgrad.utils import has_cuda_support


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
    
    epochs = 1
    seq_length = 200

    model = LM(len(vocab), 256, 256)
    optimizer = SGD(model.parameters(), lr=0.5)

    hidden_state = Tensor(np.zeros((1, 256), dtype=np.float32))
    
    for _ in range(epochs):
        pbar = tqdm(range(0, len(text_as_int) - 600000 - seq_length, seq_length))
        for i in pbar:
            optimizer.zero_grad()
            sequence = Tensor([text_as_int[i : i + seq_length]], dtype=phtypes.int64)
            target_sequence = Tensor([text_as_int[i + 1 : i + seq_length + 1]], dtype=phtypes.int64)
            res, hidden_state = model(sequence, hidden_state)
            res = res.log_softmax(dim=1)
            loss = nllloss(res.squeeze(dim=0), target_sequence.squeeze(dim=0), reduce="mean")
            loss.backward()
            optimizer.step()
            hidden_state = hidden_state.detach()
            pbar.set_description(f"Loss: {loss.first_item:.3f}")

    # Generate some text
    input_ids = text_as_int[0 : 0 + seq_length]
    current_text = "".join(idx2char[input_ids])
    # current_text = "I hurt myself"

    hidden_state = Tensor(np.zeros((1, 256), dtype=np.float32))

    print("Input:", current_text, end="")
    print("=====")

    for i in range(10):
        # sequence = Tensor([text_as_int[i : i + seq_length + 1]], dtype=phtypes.int64)
        sequence = Tensor([[char2idx[c] for c in current_text]], dtype=phtypes.int64)
        res, hidden_state = model(sequence, hidden_state)
        res = res.squeeze(dim=0).softmax(dim=-1)
        res = res[-1, :]
        # Take the last character
        idx = res.argmax().detach().numpy()
        current_text += idx2char[int(idx)]
        print(idx2char[int(idx)], end="")

if __name__ == "__main__":
    main()

