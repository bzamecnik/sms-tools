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

def to_db_magnitudes(amplitudes):
    abs_amplitudes = abs(amplitudes)
    # ensure non-zero values for logarithm
    eps = np.finfo(float).eps
    abs_amplitudes[abs_amplitudes < eps] = eps
    # magnitude spectrum of positive frequencies in dB
    return 20 * np.log10(abs_amplitudes)

def from_db_magnitudes(magnitudes_db):
    return 10 ** (magnitudes_db * 0.05)
