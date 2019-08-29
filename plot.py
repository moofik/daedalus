import matplotlib.pyplot as plt


def draw_training_data(data, y, label_success, label_fail):
    success = data[y == 1]
    fail = data[y == 0]

    plt.scatter(success.iloc[:, 0], success.iloc[:, 1], s=10, label=label_success)
    plt.scatter(fail.iloc[:, 0], fail.iloc[:, 1], s=10, label=label_fail)
    plt.legend()
    plt.show()


def draw_decision_boundary():
    pass
