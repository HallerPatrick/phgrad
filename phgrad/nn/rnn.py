
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
        self.hidden2hidden = Linear(hidden_size, hidden_size, device=device)
        self.hid2out = Linear(hidden_size, inp_dim, device=device)

    def forward(self, inp: Tensor, hidden_state: Tensor):
        inp = self.in2hidden(inp)
        hidden = self.hidden2hidden(hidden_state)
        comb = inp + hidden_state
        # NOTE: Is sigmoid the right activation function to use here? Torch uses Tanh or ReLu
        comb = self.hid2out(comb)
        comb = comb.tanh()
        return comb, hidden

