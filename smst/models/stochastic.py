"""
Functions that implement analysis and synthesis of sounds using the Stochastic
Model.

In this model the sound is modeled using a magnitude envelope of the signal's
spectrum. The magnitude envelope is decimated. On reconstruction the phases
are generated randomly.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hanning, resample
from scipy.fftpack import fft, ifft

from . import stft
from ..utils.math import is_power_of_two


def from_audio(x, H, N, stocf):
    """
    Analyzes a sound using the stochastic model.

    :param x: input array sound
    :param H: hop size
    :param N: FFT size
    :param stocf: decimation factor of mag spectrum for stochastic analysis, bigger than 0, maximum of 1
    :returns: stocEnv: stochastic envelope
    """

    hN = N / 2 + 1  # positive size of fft
    No2 = N / 2  # half of N
    if hN * stocf < 3:  # raise exception if decimation factor too small
        raise ValueError("Stochastic decimation factor too small")

    if stocf > 1:  # raise exception if decimation factor too big
        raise ValueError("Stochastic decimation factor above 1")

    if H <= 0:  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    if not (is_power_of_two(N)):  # raise error if N not a power of two
        raise ValueError("FFT size (N) is not a power of 2")

    w = hanning(N)  # analysis window
    x = np.append(np.zeros(No2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(No2))  # add zeros at the end to analyze last sample
    for frame_index, x1 in enumerate(stft.iterate_analysis_frames(x, H, No2, No2)):
        xw = x1 * w  # window the input sound
        X = fft(xw)  # compute FFT
        mX = 20 * np.log10(abs(X[:hN]))  # magnitude spectrum of positive frequencies
        mY = resample(np.maximum(-200, mX), stocf * hN)  # decimate the mag spectrum
        if frame_index == 0:  # first frame
            stocEnv = np.array([mY])
        else:  # rest of frames
            stocEnv = np.vstack((stocEnv, np.array([mY])))
    return stocEnv


def to_audio(stocEnv, H, N):
    """
    Synthesizes sound from a stochastic model.

    :param stocEnv: stochastic envelope
    :param H: hop size
    :param N: fft size
    :returns: y: output sound
    """

    if not (is_power_of_two(N)):  # raise error if N not a power of two
        raise ValueError("N is not a power of two")

    hN = N / 2 + 1  # positive size of fft
    No2 = N / 2  # half of N
    L = stocEnv.shape[0]  # number of frames
    ysize = H * (L + 3)  # output sound size
    y = np.zeros(ysize)  # initialize output array
    ws = 2 * hanning(N)  # synthesis window
    pout = 0  # output sound pointer
    for l in range(L):
        mY = resample(stocEnv[l, :], hN)  # interpolate to original size
        pY = 2 * np.pi * np.random.rand(hN)  # generate phase random values
        Y = np.zeros(N, dtype=complex)  # initialize synthesis spectrum
        Y[:hN] = 10 ** (mY / 20) * np.exp(1j * pY)  # generate positive freq.
        Y[hN:] = 10 ** (mY[-2:0:-1] / 20) * np.exp(-1j * pY[-2:0:-1])  # generate negative freq.
        fftbuffer = np.real(ifft(Y))  # inverse FFT
        y[pout:pout + N] += ws * fftbuffer  # overlap-add
        pout += H
    y = np.delete(y, range(No2))  # delete half of first window
    y = np.delete(y, range(y.size - No2, y.size))  # delete half of the last window
    return y

# functions that implement transformations using the stochastic

def scale_time(stocEnv, timeScaling):
    """
    Scales the stochastic model of a sound in time.

    :param stocEnv: stochastic envelope
    :param timeScaling: scaling factors, in time-value pairs
    :returns: ystocEnv: stochastic envelope
    """
    if timeScaling.size % 2 != 0:  # raise exception if array not even length
        raise ValueError("Time scaling array does not have an even size")

    L = stocEnv.shape[0]  # number of input frames
    outL = int(L * timeScaling[-1] / timeScaling[-2])  # number of synthesis frames
    # create interpolation object with the time scaling values
    timeScalingEnv = interp1d(timeScaling[::2] / timeScaling[-2], timeScaling[1::2] / timeScaling[-1])
    indexes = (L - 1) * timeScalingEnv(np.arange(outL) / float(outL))  # generate output time indexes
    ystocEnv = stocEnv[0, :]  # first output frame is same than input
    for l in indexes[1:]:  # step through the output frames
        ystocEnv = np.vstack((ystocEnv, stocEnv[round(l), :]))  # get the closest input frame
    return ystocEnv
