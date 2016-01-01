import numpy as np

def is_power_of_two(num):
    """
    Checks if num is power of two
    """
    return ((num & (num - 1)) == 0) and num > 0

def rmse(x, y):
    """
    Root mean square error.

    :param x: numpy array
    :param y: numpy array
    :return: RMSE(x,y)
    """

    return np.sqrt(((x - y) ** 2).mean())
