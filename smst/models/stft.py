# functions that implement analysis and synthesis of sounds using the Short-Time Fourier Transform
# (for example usage check stft_function.py in the models_interface directory)

import math

import numpy as np
from scipy.signal import resample

from . import dft


def from_audio(x, w, N, H):
    """
    Analysis of a sound using the short-time Fourier transform
    x: input array sound, w: analysis window, N: FFT size, H: hop size
    returns xmX, xpX: magnitude and phase spectra
    """
    if (H <= 0):  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    M = w.size  # size of analysis window
    hM1, hM2 = dft.half_window_sizes(M)
    x_padded = pad_signal(x, hM2)
    w = w / sum(w)  # normalize analysis window
    for frame_index, x1 in enumerate(iterate_analysis_frames(x_padded, H, hM1, hM2)):
        mX, pX = dft.from_audio(x1, w, N)  # compute dft
        if frame_index == 0:  # if first frame create output arrays
            xmX = np.array([mX])
            xpX = np.array([pX])
        else:  # append output to existing array
            xmX = np.vstack((xmX, np.array([mX])))
            xpX = np.vstack((xpX, np.array([pX])))
    return xmX, xpX


def to_audio(mY, pY, M, H):
    """
    Synthesis of a sound using the short-time Fourier transform
    mY: magnitude spectra, pY: phase spectra, M: window size, H: hop-size
    returns y: output sound
    """
    hM1, hM2 = dft.half_window_sizes(M)
    nFrames = mY[:, 0].size  # number of frames
    y = np.zeros(nFrames * H + hM1 + hM2)  # initialize output array
    pin = hM1
    for i in range(nFrames):  # iterate over all frames
        y1 = dft.to_audio(mY[i, :], pY[i, :], M)  # compute idft
        y[pin - hM1:pin + hM2] += H * y1  # overlap-add to generate output sound
        pin += H  # advance sound pointer
    y = np.delete(y, range(hM2))  # delete half of first window
    y = np.delete(y, range(y.size - hM1, y.size))  # delete the end of the sound
    return y

# functions that implement transformations using the stft

# NOTE: the following transformations work directly with audio input and output

def filter(x, fs, w, N, H, filter):
    """
    Apply a filter to a sound by using the STFT
    x: input sound, w: analysis window, N: FFT size, H: hop size
    filter: magnitude response of filter with frequency-magnitude pairs (in dB)
    returns y: output sound
    """

    M = w.size  # size of analysis window
    hM1, hM2 = dft.half_window_sizes(M)
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM1))  # add zeros at the end to analyze last sample
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    y = np.zeros(x.size)  # initialize output array
    while pin < pend:  # while sound pointer is smaller than last sample
        # -----analysis-----
        x1 = x[pin - hM1:pin + hM2]  # select one frame of input sound
        mX, pX = dft.from_audio(x1, w, N)  # compute DFT
        # ------transformation-----
        mY = mX + filter  # filter input magnitude spectrum
        # -----synthesis-----
        y1 = dft.to_audio(mY, pX, M)  # compute IDFT
        y[pin - hM1:pin + hM2] += H * y1  # overlap-add to generate output sound
        pin += H  # advance sound pointer
    y = np.delete(y, range(hM2))  # delete half of first window
    y = np.delete(y, range(y.size - hM1, y.size))  # add zeros at the end to analyze last sample
    return y


def morph(x1, x2, fs, w1, N1, w2, N2, H1, smoothf, balancef):
    """
    Morph of two sounds using the STFT
    x1, x2: input sounds, fs: sampling rate
    w1, w2: analysis windows, N1, N2: FFT sizes, H1: hop size
    smoothf: smooth factor of sound 2, bigger than 0 to max of 1, where 1 is no smoothing,
    balancef: balance between the 2 sounds, from 0 to 1, where 0 is sound 1 and 1 is sound 2
    returns y: output sound
    """

    if N2 / 2 * smoothf < 3:  # raise exception if decimation factor too small
        raise ValueError("Smooth factor too small")

    if smoothf > 1:  # raise exception if decimation factor too big
        raise ValueError("Smooth factor above 1")

    if balancef > 1 or balancef < 0:  # raise exception if balancef outside 0-1
        raise ValueError("Balance factor outside range")

    if H1 <= 0:  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H1) smaller or equal to 0")

    M1 = w1.size  # size of analysis window
    hM1_1 = int(math.floor((M1 + 1) / 2))  # half analysis window size by rounding
    hM1_2 = int(math.floor(M1 / 2))  # half analysis window size by floor
    L = int(x1.size / H1)  # number of frames for x1
    x1 = np.append(np.zeros(hM1_2), x1)  # add zeros at beginning to center first window at sample 0
    x1 = np.append(x1, np.zeros(hM1_1))  # add zeros at the end to analyze last sample
    pin1 = hM1_1  # initialize sound pointer in middle of analysis window
    w1 = w1 / sum(w1)  # normalize analysis window
    M2 = w2.size  # size of analysis window
    hM2_1 = int(math.floor((M2 + 1) / 2))  # half analysis window size by rounding
    hM2_2 = int(math.floor(M2 / 2))  # half analysis window size by floor2
    H2 = int(x2.size / L)  # hop size for second sound
    x2 = np.append(np.zeros(hM2_2), x2)  # add zeros at beginning to center first window at sample 0
    x2 = np.append(x2, np.zeros(hM2_1))  # add zeros at the end to analyze last sample
    pin2 = hM2_1  # initialize sound pointer in middle of analysis window
    y = np.zeros(x1.size)  # initialize output array
    for l in range(L):
        # -----analysis-----
        mX1, pX1 = dft.from_audio(x1[pin1 - hM1_1:pin1 + hM1_2], w1, N1)  # compute dft
        mX2, pX2 = dft.from_audio(x2[pin2 - hM2_1:pin2 + hM2_2], w2, N2)  # compute dft
        # -----transformation-----
        mX2smooth = resample(np.maximum(-200, mX2), mX2.size * smoothf)  # smooth spectrum of second sound
        mX2 = resample(mX2smooth, mX1.size)  # generate back the same size spectrum
        mY = balancef * mX2 + (1 - balancef) * mX1  # generate output spectrum
        # -----synthesis-----
        y[pin1 - hM1_1:pin1 + hM1_2] += H1 * dft.to_audio(mY, pX1, M1)  # overlap-add to generate output sound
        pin1 += H1  # advance sound pointer
        pin2 += H2  # advance sound pointer
    y = np.delete(y, range(hM1_2))  # delete half of first window
    y = np.delete(y, range(y.size - hM1_1, y.size))  # add zeros at the end to analyze last sample
    return y

def pad_signal(x, hM2):
    x_padded = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x_padded = np.append(x_padded, np.zeros(hM2))  # add zeros at the end to analyze last sample
    return x_padded

def iterate_analysis_frames(x, H, hM1, hM2):
    """
    Iterate over frames of input signal for analysis.

    :param x: input signal
    :param H: hop size
    :param hM1: half analysis window size by rounding
    :param hM2: half analysis window size by floor
    :return: generator over frames of input signal
    """
    pin = hM1  # initialize sound pointer in middle of analysis window
    pend = x.size - hM1  # last sample to start a frame
    while pin < pend:  # while sound pointer is smaller than last sample
        frame_start = pin - hM1
        frame_end = pin + hM2
        yield x[frame_start:frame_end]  # select one frame of input sound
        pin += H  # advance sound pointer
