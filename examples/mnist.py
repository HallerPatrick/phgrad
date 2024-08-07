import os
import sys
import time
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from phgrad.debug import print_summary
from phgrad.engine import Tensor
from phgrad.nn import MLP
from phgrad.optim import SGD, Adam
from phgrad.loss import nllloss
from phgrad.utils import has_cuda_support
from tests.utils import load_mnist
from phgrad import types



class MNIST:
    def setUp(self) -> None:
        X_train, Y_train, X_test, Y_test = load_mnist()
        self.X_train = X_train.reshape(-1, 28 * 28) / 255.0
        self.X_test = X_test.reshape(-1, 28 * 28) / 255.0
        self.Y_train = np.eye(10)[Y_train.reshape(-1)]
        self.Y_test = np.eye(10)[Y_test.reshape(-1)]

    def run(self, device=None):
        if device is None:
            if has_cuda_support():
                device = "cuda"
            else:
                device = "cpu"

        # At around 256, the GPU is faster than CPU
        # Hidden size 256,    BS: 4: CPU: 22.86,  GPU: 21.86 sec per epoch
        # Hidden size 1024,   BS: 4: CPU: 108.99, GPU: 21.61 sec per epoch
        # Hidden size 1024, BS: 256: CPU: 2.61,   GPU: 0.42 sec per epoch

        hidden_size = 1024

        mlp = MLP(784, hidden_size, 10, bias=True).to(device)
        optimizer = SGD(mlp.parameters(), lr=0.005)
        # optimizer = Adam(mlp.parameters())

        losses = []
        accuracies = []

        dataset_loader = []

        start_time = time.time()
        batch_size = 256
        for i in range(0, len(self.X_train), batch_size):
            dataset_loader.append(
                (
                    Tensor(
                        self.X_train[i : i + batch_size].astype(np.float32),
                        device=device,
                    ),
                    Tensor(
                        np.argmax(
                            self.Y_train[i : i + batch_size].astype(np.float32), axis=1
                        ),
                        device=device,
                        dtype=types.int64,
                    ),
                )
            )
        end_time = time.time()
        print(f"Preprocess Time: {end_time - start_time:.2f} seconds")

        start_time = time.time()
        for epoch in range(1):
            for batch in tqdm(dataset_loader):
                optimizer.zero_grad()
                x, y = batch
                y_pred = mlp(x)
                y_pred_log_sm = y_pred.log_softmax(dim=1)
                loss = nllloss(y_pred_log_sm, y, reduce="mean")
                loss.backward()
                y_pred.softmax(dim=-1)
                optimizer.step()
                losses.append(loss.data)

            y_pred = mlp(Tensor(self.X_test, device=device))
            y_pred = y_pred.softmax(dim=-1)
            y_pred = np.argmax(y_pred.numpy(), axis=1)

            accuracy = (y_pred == self.Y_test.argmax(axis=1)).mean()

            accuracies.append(accuracy)

            print(f"Epoch {epoch}: {accuracy:.2f}")

        end_time = time.time()

        print(f"Training time: {end_time - start_time:.2f} seconds")

        if False:
            import matplotlib.pyplot as plt

            plt.plot(losses)
            plt.title("Loss")
            plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Usage: python examples/mnist.py <cpu|cuda>")
        sys.exit(0)
    device = args[0]
    mnist = MNIST()
    mnist.setUp()
    mnist.run(device)
