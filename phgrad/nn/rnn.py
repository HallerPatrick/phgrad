from typing import Optional

from phgrad.engine import Tensor
from phgrad.nn import Linear, Module
from phgrad.debug import print_summary


class RNNCell(Module):
    def __init__(self, inp_dim: int, hidden_size: int, device: str = "cpu") -> None:
        super().__init__()
        self.in2hidden = Linear(inp_dim, hidden_size, device=device)
        self.hid2hid = Linear(hidden_size, hidden_size, device=device)

    def forward(self, inp: Tensor, hidden_state: Tensor):
        # inp: [batch_size, num_features]
        print_summary()
        breakpoint()
        combined = self.in2hidden(inp) + self.hid2hid(hidden_state)
        hidden_state = combined.tanh()
        print_summary()
        breakpoint()
        return hidden_state


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
        self.inp_dim = inp_dim
        self.hidden_size = hidden_size
        self.device = device
        self.cell = RNNCell(inp_dim, hidden_size, device=device)

    def forward(self, inp: Tensor, hidden_state: Optional[Tensor] = None):
        """Forward pass.

        NOTE: Lets only support batch_first=True for now.

        input: [batch_size, seq_len, hidden_size]

        """
        batch_size, seq_len, _ = inp.shape

        if hidden_state is None:
            hidden_state = Tensor.zeros(
                (batch_size, self.hidden_size), device=self.device
            )

        outputs = []
        for i in range(seq_len):
            hidden_state = self.cell(inp[:, i, :], hidden_state)
            outputs.append(hidden_state)

        # return ph.stack(tuple(outputs), dim=1), hidden_state
        return Tensor.stack(tuple(outputs), dim=1), hidden_state
