import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cost_function import cost_function

if __name__ == "__main__":
    data = pd.read_csv("data/ex2data1.txt", ",")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y[:, np.newaxis]
    weights = np.zeros((X.shape[1], 1))

    # uncomment it if you need to plot
    # admitted = data[y == 1]
    # not_admitted = data[y == 0]
    #
    # plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    # plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    # plt.legend()
    # plt.show()

    cost_function(y, X, weights)