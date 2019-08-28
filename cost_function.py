from numpy import log, square, errstate
from sigmoid import sigmoid


def cost_function(weights, X, y):
    m = len(y)
    h = sigmoid(X.dot(weights))

    cost = 1.0 / m * (-y.T.dot(log(h)) - (1 - y).T.dot(log(1 - h)))
    grad = 1.0 / m * X.T.dot(h - y)

    return cost, grad


def cost_function_regularized(weights, X, y, l):
    m = len(y)
    h = sigmoid(X.dot(weights))

    cost_regularization_term = l / (2 * m) * sum(square(weights)[1:])
    cost = 1.0 / m * (-y.T.dot(log(h)) - (1 - y).T.dot(log(1 - h))) + cost_regularization_term

    regularized_weights = list(weights)
    regularized_weights[0] = 0
    grad_regularization_term = l / m * weights
    grad = 1.0 / m * X.T.dot(h - y) + grad_regularization_term

    return cost, grad
