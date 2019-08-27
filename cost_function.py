import numpy as np

from sigmoid import sigmoid


def cost_function(y, X, weights):
    m = len(y)
    h = sigmoid(X.dot(weights))

    cost = 1.0 / m * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
    grad = 1.0 / m * X.T.dot(h - y)
    return cost, grad.to_numpy()
