import numpy as np


def map_feature(X1, X2, degree=6):
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)


def calculate_precision(y, X, weights):
    result = round(X.dot(weights)).to_numpy()
    correct = 0

    for i in range(len(y)):
        if y[i][0] == result[i][0]:
            correct = correct + 1

    precision = correct / len(y)
    print('Precision is: ', precision * 100, '%')
