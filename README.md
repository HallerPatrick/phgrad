# phgrad 

### Cause there are not enough unusable autograd libraries in python

![Logo](logo.png)

A tiny [autograd](https://en.wikipedia.org/wiki/Automatic_differentiation) library for learning purposes, inspired by [micrograd](https://github.com/karpathy/micrograd/tree/master)
and [tinygrad](https://github.com/tinygrad/tinygrad).

Everything at this point is experimental and not optimized for performance. So be aware that if you
touch this library, you will probably break something.
The goal is to optimize the library for performance and add more features. I will mostly adjust this library to 
be usable in NLP tasks.

I will try to keep the API as close as possible to [PyTorch](https://pytorch.org/). A goal is to
provide CUDA support, while keeping the dependency list as small as possible. (Currently only numpy, and now cupy).

The [example](./examples) folder will contain some examples of how to use the library.


### Example

```python
from phgrad.engine import Tensor
from phgrad.nn import Linear, Module

# We now have cuda support!
device = "cuda"

class MLP(Module):
    """A simple Multi Layer Perceptron."""

    def __init__(
        self,
        inp_dim: int,
        hidden_size: int,
        output_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.l1 = Linear(inp_dim, hidden_size, bias=bias)
        self.l2 = Linear(hidden_size, output_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        return x

model = MLP(784, 256, 10).to(device)
x = Tensor(np.random.randn(32, 784), device=device)
y = model(x).mean()
y.backward()
```


### Resources
1. https://github.com/torch/nn/blob/master/doc/transfer.md
2. https://github.com/karpathy/micrograd/tree/master
3. https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb
4. https://github.com/ICEORY/softmax_loss_gradient
5. https://notesbylex.com/negative-log-likelihood#:~:text=Negative%20log%2Dlikelihood%20is%20a,all%20items%20in%20the%20batch.
6. https://github.com/huggingface/candle/tree/main

