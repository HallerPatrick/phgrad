
from phgrad.engine import Tensor
from phgrad.nn import Linear, Module

class RNN(Module):
    """Recurrent Neural Network.

    TODOs:
    * Stacked RNN
    * Init hidden_state if None
    * Impl. other activation functions
    * Biderctional ?
    """

    def __init__(self, inp_dim: int, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.in2hidden = Linear(inp_dim + hidden_size, hidden_size)
        self.in2out = Linear(inp_dim + hidden_size, hidden_size)

    def forward(self, inp: Tensor, hidden_state: Tensor):
        comb = inp.cat((hidden_state,), 1)
        hidden = self.in2hidden(combined)
        # NOTE: Is sigmoid the right activation function to use here? Torch uses Tanh or ReLu
        hidden = hidden.sigmoid()
        output = self.in2out(comb)
        return output, hidden

