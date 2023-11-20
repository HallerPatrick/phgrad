
from phgrad.engine import Tensor
from phgrad.nn import Linear, Module

class RNN(Module):
    """Recurrent Neural Network.

    TODOs:
    * Stacked RNN
    * Init hidden_state if None
    * Impl. other activation functions
    * Biderctional ?

    Ref: https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#RNN
    """

    def __init__(self, inp_dim: int, hidden_size: int, device: str = "cpu") -> None:
        super().__init__()
        self.in2hidden = Linear(inp_dim, hidden_size, device=device)
        self.hid2hid = Linear(hidden_size, hidden_size, device=device)
        self.hid2out = Linear(hidden_size, hidden_size, device=device)

    def forward(self, inp: Tensor, hidden_state: Tensor):
        combined = self.in2hidden(inp).add(self.hid2hid(hidden_state))
        next_hidden = combined.tanh()
        print(combined.shape)
        print(next_hidden.shape)
        logits = self.hid2out(next_hidden)
        return logits, next_hidden

