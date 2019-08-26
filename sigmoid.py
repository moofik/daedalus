import numpy as np


def sigmoid(z):
    """
    Compute sigmoid function.
    Parameters
    ----------
    z : array-like
        Variable for sigmoid function.
    Returns
    -------
    ndarray
        The sigmoid of each value of z.
    """
    return 1 / (1 + np.exp(-z))


