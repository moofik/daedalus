from cost_function import cost_function_with_grad


def gradient_descent(weights, X, y, iterations, alpha):
    for i in range(iterations):
        cost, grad = cost_function_with_grad(weights, X, y)
        weights = weights - alpha * grad

    return weights


def gradient_descent_regularized():
    pass