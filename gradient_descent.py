from cost_function import cost_function_regularized
from terminal import print_progress_bar


def gradient_descent(weights, X, y, iterations, alpha, l):
    total = iterations / 100

    for i in range(iterations):
        cost, grad = cost_function_regularized(weights, X, y, l)
        weights = weights - alpha * grad

        if i % total == 0 and i > total:
            print_progress_bar(i/total + 1, 100, 'Calculating: ')

    return weights
