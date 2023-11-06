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
provide CUDA support, while keeping the dependency list as small as possible. (Currently only numpy).

The [example](./examples) folder will contain some examples of how to use the library.


### Example

```python

from phgrad.engine import PensorTensor as pensor
from phgrad.nn import Linear

class Classifier:
    def __init__(self):
        self.l1 = Linear(784, 64, bias=False)
        self.l2 = Linear(64, 1, bias=False)

    def __call__(self, x: pensor):
        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)
        return x.log_softmax()

model = Classifier()
x = pensor(np.random.randn(1, 784))
y = model(x)
y.backward()

```


### Resources
1. https://github.com/torch/nn/blob/master/doc/transfer.md
2. https://github.com/karpathy/micrograd/tree/master
3. https://github.com/geohot/ai-notebooks/blob/master/mnist_from_scratch.ipynb
4. https://github.com/ICEORY/softmax_loss_gradient
5. https://notesbylex.com/negative-log-likelihood#:~:text=Negative%20log%2Dlikelihood%20is%20a,all%20items%20in%20the%20batch.
6. https://github.com/huggingface/candle/tree/main
