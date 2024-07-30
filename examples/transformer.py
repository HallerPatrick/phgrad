import os
import sys
from typing import Tuple

from tqdm import tqdm
from datasets import load_dataset
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phgrad import Tensor
from phgrad.nn import Embedding, Module, Linear
from phgrad.nn.rnn import RNN, LSTM
from phgrad import types as phtypes
from phgrad.loss import nllloss
from phgrad.optim import SGD
from phgrad.utils import has_cuda_support
from phgrad.debug import print_summary


class LM(Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn = RNN(embedding_dim, hidden_size)l
        # self.rnn = LSTM(embedding_dim, hidden_size)
        self.decoder = Linear(hidden_size, vocab_size)

    def forward(self, x: Tensor, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.embedding(x)
        x, hidden_state = self.rnn(x, hidden_state)
        x = self.decoder(x)
        return x, hidden_state


def main(device: str):
    text = "".join(load_dataset("PatrickHaller/hurt")["train"]["text"])
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])


    seq_length = 200
    dims = 4096
    
    # Seq len: 200, 256 hidden size/embedding dim
    # CPU: 16 sec
    # GPU: 41 sec

    # Seq len: 200, 4096 hidden size/embedding dim
    # CPU: >~13 min
    # GPU: 1 min
    model = LM(len(vocab), dims, dims).to(device)

    is_lstm = isinstance(model.rnn, LSTM)
    epochs = 10 if is_lstm else 1

    optimizer = SGD(model.parameters(), lr=15 if is_lstm else 0.5)

    # hidden_state = Tensor(np.zeros((1, 256), dtype=np.float32), device=device)
    hidden_state = model.rnn.init_hidden(1, device)
    
    for _ in range(epochs):
        pbar = tqdm(range(0, len(text_as_int) - 600000 - seq_length, seq_length))
        for i in pbar:
            optimizer.zero_grad()
            sequence = Tensor(
                [text_as_int[i : i + seq_length]], dtype=phtypes.int64, device=device
            )
            target_sequence = Tensor(
                [text_as_int[i + 1 : i + seq_length + 1]],
                dtype=phtypes.int64,
                device=device,
            )
            res, hidden_state = model(sequence, hidden_state)
            res = res.log_softmax(dim=-1)
            loss = nllloss(
                res.squeeze(dim=0), target_sequence.squeeze(dim=0), reduce="mean"
            )
            loss.backward()
            optimizer.step()
            if isinstance(hidden_state, tuple):
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            else:
                hidden_state = hidden_state.detach()
            pbar.set_description(f"Loss: {loss.first_item:.3f}")

    # Generate some text
    input_ids = text_as_int[:seq_length]
    current_text = "".join(idx2char[input_ids])

    # hidden_state = Tensor(np.zeros((1, 256), dtype=np.float32))
    hidden_state = model.rnn.init_hidden(1, device)

    print("Generate text based (continuation is bold and green):", current_text, end="")

    def print_green_bold(text):
        green_bold_code = "\033[1;32m"  # 1 for bold, 32 for green
        reset_code = "\033[0m"  # Reset to default text style
        print(f"{green_bold_code}{text}{reset_code}", end="")

    for i in range(200):
        sequence = Tensor([[char2idx[c] for c in current_text]], dtype=phtypes.int64)
        res, hidden_state = model(sequence, hidden_state)
        # Take the last character
        res = res.squeeze(dim=0)[-1].softmax(dim=-1)
        idx = res.argmax().detach().numpy()
        current_text += idx2char[int(idx)]
        print_green_bold(idx2char[int(idx)])


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        print("Usage: python examples/mnist.py <cpu|cuda>")
        sys.exit(0)

    device = args[0]
    if device == "cuda" and not has_cuda_support():
        print("CUDA is not available on this system. Using CPU instead.")
        device = "cpu"

    main(device)
