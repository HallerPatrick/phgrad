import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from phgrad.debug import print_summary
from phgrad.engine import Tensor
from phgrad.nn import MLP
from phgrad.optim import SGD, Adam
from phgrad.loss import nllloss
from phgrad.utils import has_cuda_support
from tests.utils import load_mnist

class MNIST():
    def setUp(self) -> None:
        X_train, Y_train, X_test, Y_test = load_mnist()
        self.X_train = X_train.reshape(-1, 28 * 28) / 255.0
        self.X_test = X_test.reshape(-1, 28 * 28) / 255.0
        self.Y_train = np.eye(10)[Y_train.reshape(-1)]
        self.Y_test = np.eye(10)[Y_test.reshape(-1)]

    def run(self):
        if has_cuda_support():
            device = "cuda"
        else:
            device = "cpu"
        
        # At around 256, the GPU is faster than CPU
        hidden_size = 256

        mlp = MLP(784, hidden_size, 10, bias=True, device=device)
        optimizer = SGD(mlp.parameters(), lr=0.005)
        # optimizer = Adam(mlp.parameters())

        losses = []
        accuracies = []

        dataset_loader = []
        
        start_time = time.time()
        batch_size = 128
        for i in range(0, len(self.X_train), batch_size):
            dataset_loader.append((Tensor(self.X_train[i : i + batch_size].astype(np.float32), device=device), Tensor(np.argmax(self.Y_train[i : i + batch_size].astype(np.float32), axis=1), device=device)))
        end_time = time.time()
        print(f"Preprocess Time: {end_time - start_time:.2f} seconds")


        start_time = time.time()
        for epoch in range(10):
            for batch in dataset_loader:
                optimizer.zero_grad()
                x, y = batch
                y_pred = mlp(x)
                y_pred_log_sm = y_pred.log_softmax(dim=1)
                loss = nllloss(y_pred_log_sm, y, reduce="mean")
                loss.backward()
                # Benchmark
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
        print_summary()

        print(f"Training time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    mnist = MNIST()
    mnist.setUp()
    mnist.run()
