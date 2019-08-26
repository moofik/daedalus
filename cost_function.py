from sigmoid import sigmoid
import numpy as np


def cost_function(y, X, weights):
    m = len(y)
    h = sigmoid(X.dot(weights))

    cost = 1 / m * -y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))
    grad = 1 / m * (h - y).T.dot(X)

    return cost, grad
