import numpy as np
from sigmoid import sigmoid


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


def calculate_precision(weights, X, y):
    p = predict(weights, X)
    print('Precision is: ', np.mean(p == y) * 100, '%')


#  here was the mistake
def predict_old(weights, X):
    return np.round(X.dot(weights)).astype(int)


def predict(theta, X):
    p = sigmoid(X.dot(theta)) >= 0.5
    return p.astype(int)
