from cost_function import cost_function


def gradient_descent(y, X, weights, iterations, alpha, progress):
    delimeter = iterations / 100
    for i in range(iterations):
        cost, grad = cost_function(y, X, weights)
        weights = weights - alpha * grad

        if progress and i % delimeter == 0:
            print('Progress: ', i / delimeter, '% ...')

    return weights
