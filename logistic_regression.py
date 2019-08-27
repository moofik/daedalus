import sys
import termios
import tty
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils as u
from gradient_descent import gradient_descent

if __name__ == "__main__":
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)

    data = pd.read_csv("data/ex2data1.txt", ",")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y[:, np.newaxis]
    weights = np.zeros((X.shape[1], 1))

    #  uncomment it if you need to plot
    # admitted = data[y == 1]
    # not_admitted = data[y == 0]
    #
    # plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    # plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    # plt.legend()
    # plt.show()

    print('Press G for gradient descent. If you press any other key optimized algorithm will run by default.')
    key = sys.stdin.read(1)[0]

    if key == 'g':
        print("Run gradient descent")
        weights = gradient_descent(y, X, weights, 100000, 0.00101, True)
        u.calculate_precision(y, X, weights)
    else:
        print("Run optimized algorithm")
