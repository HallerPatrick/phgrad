import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phgrad.engine import Tensor
from phgrad.nn import MLP
from phgrad.optim import SGD, Adam
from phgrad.loss import nllloss
from tests.utils import load_mnist

class MNIST():
    def setUp(self) -> None:
        X_train, Y_train, X_test, Y_test = load_mnist()
        self.X_train = X_train.reshape(-1, 28 * 28) / 255.0
        self.X_test = X_test.reshape(-1, 28 * 28) / 255.0
        self.Y_train = np.eye(10)[Y_train.reshape(-1)]
        self.Y_test = np.eye(10)[Y_test.reshape(-1)]

    def run(self):
        mlp = MLP(784, 64, 10, bias=False)
        # optimizer = SGD(mlp.parameters(), lr=0.005)
        optimizer = Adam(mlp.parameters())

        losses = []
        accuracies = []

        start_time = time.time()

        for epoch in range(10):
            for i in range(0, len(self.X_train), 32):
                optimizer.zero_grad()
                x = Tensor(self.X_train[i : i + 32])
                y = Tensor(np.argmax(self.Y_train[i : i + 32], axis=1))
                y_pred = mlp(x)
                y_pred = y_pred.log_softmax(dim=1)
                loss = nllloss(y_pred, y, reduce="mean")
                loss.backward()
                optimizer.step()
                losses.append(loss.data)

            y_pred = mlp(Tensor(self.X_test))
            y_pred = np.argmax(y_pred.data, axis=1)

            accuracy = (y_pred == self.Y_test.argmax(axis=1)).mean()
            accuracies.append(accuracy)

            print(f"Epoch {epoch}: {accuracy:.2f}")

        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")

        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(losses, label='Loss')
        # plt.title('Training Loss')
        # plt.xlabel('Step')
        # plt.ylabel('Loss')
        # plt.legend()
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(accuracies, label='Accuracy')
        # plt.title('Training Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.legend()
        #
        # plt.show()


if __name__ == "__main__":
    mnist = MNIST()
    mnist.setUp()
    mnist.run()
