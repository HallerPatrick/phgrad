from phgrad import Tensor
from phgrad import nn


class MultiHeadAttention:
    pass


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.MLP(d_model, d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float
    ):
        super().__init__()
        self.layers = [
            TransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
