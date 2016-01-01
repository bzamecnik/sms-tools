import numpy as np


def blackman_harris_lobe(x):
    """
    Generates the main lobe of a Blackman-Harris window.

    :param x: bin positions to compute (real values)
    :returns: y: main lobe os spectrum of a Blackman-Harris window
    """

    N = 512  # size of fft to use
    f = x * np.pi * 2 / N  # frequency sampling
    df = 2 * np.pi / N
    y = np.zeros(x.size)  # initialize window
    consts = [0.35875, 0.48829, 0.14128, 0.01168]  # window constants
    for m in range(0, 4):  # iterate over the four sincs to sum
        y += consts[m] / 2 * (sinc(f - df * m, N) + sinc(f + df * m, N))  # sum of scaled sinc functions
    y = y / N / consts[0]  # normalize
    return y


def sinc(x, N):
    """
    Generates the main lobe of a sinc function (Dirichlet kernel).

    :param x: array of indexes to compute
    :param N: size of FFT to simulate
    :returns: y: samples of the main lobe of a sinc function
    """

    y = np.sin(N * x / 2) / np.sin(x / 2)  # compute the sinc function
    y[np.isnan(y)] = N  # avoid NaN if x == 0
    return y
