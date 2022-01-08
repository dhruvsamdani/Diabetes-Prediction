import matplotlib.pyplot as plt
import numpy as np


class LogisticRegression:
    def __init__(self, data):
        self.importantValues = {"weights": [], "bias": 0}
        self.importantValues["weights"] = np.zeros(data.shape[1] - 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradiant(self, x, y, iterations=100, alpha=0.01):
        weight = self.importantValues["weights"]
        bias = self.importantValues["bias"]

        weight, bias = self.batchGD(
            x, y, iterations=100000, alpha=0.0002, regularization=20
        )
        # weight, bias = self.stochasticGD(x, y, iterations=100000, alpha=0.001)

        self.importantValues["weights"] = weight
        self.importantValues["bias"] = bias

    def stochasticGD(self, x, y, iterations=100, alpha=0.001):
        weight = [0] * x.shape[1]
        bias = 0
        m, c = x.shape
        cost = []
        j = np.random.choice(range(len(x)))
        i = np.random.choice(range(m))
        for _ in range(iterations):
            y_hat = self.sigmoid(np.dot(x[i], weight) + bias)
            partial_bias = y_hat - y[i]
            bias -= alpha * partial_bias
            for j in range(c):
                partial_W = (y_hat - y[i]) * x[i, j]
                weight[j] -= alpha * partial_W
                loss = -(y[i] * np.log(y_hat)) + (1 - y[i]) * np.log(1 - y_hat)
                cost.append(loss)  # append cost function values to list

        plt.figure(figsize=(10, 8))
        plt.title("Cost Function Slope")
        plt.plot(cost)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Error Values")
        plt.show()
        return weight, bias

    def batchGD(self, x, y, iterations=100, alpha=0.001, regularization=0):
        weight = [0] * x.shape[1]
        bias = 0
        m, c = x.shape
        cost = []
        j = np.random.choice(range(len(x)))
        for _ in range(iterations):
            y_hat = self.sigmoid(np.dot(x, weight) + bias)

            for j in range(c):

                partial_W = (y_hat - y) * x[:, j]
                weight[j] -= (
                    alpha / m * (np.sum(partial_W) + regularization * weight[j])
                )

            partial_bias = y_hat - y
            bias -= alpha / m * np.sum(partial_bias)
            loss = -1 / m * np.sum((y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat))
            cost.append(loss)  # append cost function values to list

        plt.figure(figsize=(10, 8))
        plt.title("Cost Function Slope")
        plt.plot(cost)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Error Values")
        plt.show()
        return weight, bias

