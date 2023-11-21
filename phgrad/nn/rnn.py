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
        # inp: [batch_size, seq_len, inp_dim]
        # hidden_state: [batch_size, hidden_size]

        outputs = []
        for t in range(inp.shape[1]):
            inp_t = inp[:, t, :]
            combined = self.in2hidden(inp_t) + self.hid2hid(hidden_state)
            hidden_state = combined.tanh()
            logits = self.hid2out(hidden_state)
            outputs.append(logits)

        outputs = Tensor.stack(outputs, dim=1)

        return outputs, hidden_state
