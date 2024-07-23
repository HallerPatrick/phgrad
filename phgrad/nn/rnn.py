from typing import Optional, Tuple

from phgrad.engine import Tensor
from phgrad.nn import Linear, Module


class RNNCell(Module):
    def __init__(self, inp_dim: int, hidden_size: int):
        super().__init__()
        self.in2hidden = Linear(inp_dim, hidden_size)
        self.hid2hid = Linear(hidden_size, hidden_size)

    def forward(self, inp: Tensor, hidden_state: Tensor):
        # inp: [batch_size, num_features]
        combined = self.in2hidden(inp) + self.hid2hid(hidden_state)
        hidden_state = combined.tanh()
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

    def __init__(self, inp_dim: int, hidden_size: int):
        super().__init__()
        self.inp_dim = inp_dim
        self.hidden_size = hidden_size
        self.cell = RNNCell(inp_dim, hidden_size)

    def forward(self, inp: Tensor, hidden_state: Optional[Tensor] = None):
        """Forward pass.

        NOTE: Lets only support batch_first=True for now.

        input: [batch_size, seq_len, hidden_size]

        """
        batch_size, seq_len, _ = inp.shape

        outputs = []
        for i in range(seq_len):
            hidden_state = self.cell(inp[:, i, :], hidden_state)
            outputs.append(hidden_state)

        # return ph.stack(tuple(outputs), dim=1), hidden_state
        return Tensor.stack(tuple(outputs), dim=1), hidden_state

    def init_hidden(self, batch_size: int, device: str):
        return Tensor.zeros((batch_size, self.hidden_size), device=device)


class LSTMCell(Module):
    def __init__(self, inp_dim: int, hidden_size: int):
        super().__init__()
        self.forget_gate = Linear(inp_dim + hidden_size, hidden_size)
        self.input_gate =  Linear(inp_dim + hidden_size, hidden_size)
        self.output_gate = Linear(inp_dim + hidden_size, hidden_size)
        self.cell_gate =   Linear(inp_dim + hidden_size, hidden_size)

    def forward(self, inp: Tensor, hidden_state: Tuple[Tensor]):
        h_prev, c_prev = hidden_state

        combined = Tensor.cat((inp, h_prev), dim=1)
        # inp: [batch_size, num_features]
        f_t = self.forget_gate(combined).sigmoid()
        i_t = self.input_gate(combined).sigmoid()
        o_t = self.output_gate(combined).sigmoid()
        c_t = self.cell_gate(combined).tanh()

        next_cell_state = f_t * c_prev + i_t * c_t
        next_hidden_state = o_t * next_cell_state.tanh()

        return next_hidden_state, next_cell_state


class LSTM(Module):

    def __init__(self, inp_dim: int, hidden_size: int):
        super().__init__()
        self.inp_dim = inp_dim
        self.hidden_size = hidden_size
        self.cell = LSTMCell(inp_dim, hidden_size)

    def forward(self, inp: Tensor, hidden_state: Optional[Tuple[Tensor]] = None):
        """Forward pass.
        NOTE: Lets only support batch_first=True for now.
        input: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = inp.shape

        outputs = []

        for i in range(seq_len):
            hidden_state = self.cell(inp[:, i, :], hidden_state)
            outputs.append(hidden_state[0])

        return Tensor.stack(tuple(outputs), dim=1), hidden_state

    def init_hidden(self, batch_size: int, device: str):
        return (
            Tensor.zeros((batch_size, self.hidden_size), device=device),
            Tensor.zeros((batch_size, self.hidden_size), device=device),
        )



