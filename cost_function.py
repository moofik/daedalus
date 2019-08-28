from numpy import log
from sigmoid import sigmoid


def cost_function_with_grad(weights, X, y):
    m = len(y)
    h = sigmoid(X.dot(weights))

    cost = 1.0 / m * (-y.T.dot(log(h)) - (1 - y).T.dot(log(1 - h)))
    grad = 1.0 / m * X.T.dot(h - y)

    return cost, grad


def cost_function(weights, X, y):
    m = len(y)
    h = sigmoid(X.dot(weights))

    return 1.0 / m * (-y.T.dot(log(h)) - (1 - y).T.dot(log(1 - h)))


def grad(weights, X, y):
    m = len(y)
    h = sigmoid(X.dot(weights))

    return 1.0 / m * X.T.dot(h - y)


def cost_function_regularized():
    pass


def grad_regularized():
    pass
