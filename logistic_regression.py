import sys
import termios
import tty

import utils
import scipy.optimize as opt
import pandas as pd

from numpy import zeros
from gradient_descent import gradient_descent
from cost_function import cost_function_regularized


if __name__ == "__main__":
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)

    #  load and init data
    data = pd.read_csv("data/ex2data1.txt", ",")
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    X = utils.map_feature(X[:, 0], X[:, 1], 2)
    weights = zeros(X.shape[1])

    print('Menu:\n1) Press G for gradient descent.\n2) Press O for optimized algorithm.\n3) Press Q for quit.')
    key = sys.stdin.read(1)[0]

    if key == 'g' or key == 'G':
        print("Run gradient descent")
        weights = gradient_descent(weights, X, y, 500000, 0.00101, 0.003)
        utils.calculate_precision(weights, X, y)
    elif key == 'o' or key == 'O':
        print("Run optimized algorithm")
        weights, _, _ = opt.fmin_tnc(func=cost_function_regularized, x0=weights, args=(X, y, 0.1), messages=0)
        utils.calculate_precision(weights, X, y)
    elif key == 'q' or key == 'Q':
        exit()
