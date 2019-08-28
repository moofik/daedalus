import sys
import termios
import tty

import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
from numpy import append, ones, zeros

import utils as u
from gradient_descent import gradient_descent
from cost_function import cost_function_with_grad


def draw():
    admitted = data[y == 1]
    not_admitted = data[y == 0]

    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    plt.legend()
    plt.show()


def console_log(arg):
    print('-------------------------')
    print(arg)
    print('type is ', type(arg))
    print('-------------------------')


if __name__ == "__main__":
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)

    data = pd.read_csv("data/ex2data1.txt", ",")
    #  load data
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    #  prepare data for processing
    X = append(ones((len(y), 1)), X, 1)
    y = y.to_numpy()
    weights = zeros(X.shape[1])

    print('Menu:\n1) Press G for gradient descent.\n2) Press O for optimized algorithm.\n3) Press Q for quit.')
    key = sys.stdin.read(1)[0]

    if key == 'g' or key == 'G':
        print("Run gradient descent")
        weights = gradient_descent(weights, X, y, 500000, 0.00101)
        u.calculate_precision(weights, X, y)
    elif key == 'o' or key == 'O':
        print("Run optimized algorithm")
        weights, nfeval, rc = opt.fmin_tnc(func=cost_function_with_grad, x0=weights, args=(X, y), messages=0)
        u.calculate_precision(weights, X, y)
    elif key == 'q' or key == 'Q':
        exit()
