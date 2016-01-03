import numpy as np
from scipy.fftpack import fft, ifft, fftshift
from scipy.signal import resample, blackmanharris, triang

from .utilFunctions_C import spec_synth
from .math import to_db_magnitudes

def subtract_sinusoids(x, N, H, sfreq, smag, sphase, fs):
    """
    Subtracts sinusoids from a sound.

    :param x: input sound
    :param N: FFT size
    :param H: hop size
    :param sfreq: sinusoidal frequencies
    :param smag: sinusoidal magnitudes
    :param sphase: sinusoidal phases
    :returns: xr: residual sound
    """

    hN = N / 2  # half of fft size
    x = np.append(np.zeros(hN), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hN))  # add zeros at the end to analyze last sample
    bh = blackmanharris(N)  # blackman harris window
    w = bh / sum(bh)  # normalize window
    sw = np.zeros(N)  # initialize synthesis window
    sw[hN - H:hN + H] = triang(2 * H) / w[hN - H:hN + H]  # synthesis window
    L = sfreq.shape[0]  # number of frames, this works if no sines
    xr = np.zeros(x.size)  # initialize output array
    pin = 0
    for l in range(L):
        xw = x[pin:pin + N] * w  # window the input sound
        X = fft(fftshift(xw))  # compute FFT
        Yh = spec_synth.genSpecSines(N * sfreq[l, :] / fs, smag[l, :], sphase[l, :], N)  # generate spec sines
        Xr = X - Yh  # subtract sines from original spectrum
        xrw = np.real(fftshift(ifft(Xr)))  # inverse FFT
        xr[pin:pin + N] += xrw * sw  # overlap-add
        pin += H  # advance sound pointer
    xr = np.delete(xr, range(hN))  # delete half of first window
    xr = np.delete(xr, range(xr.size - hN, xr.size))  # delete half of last window
    return xr


# TODO: unused code
def subtract_sinusoids_with_stochastic_residual(x, N, H, sfreq, smag, sphase, fs, stocf):
    """
    Subtracts sinusoids from a sound and approximate the residual with an envelope.

    :param x: input sound
    :param N: FFT size
    :param H: hop size
    :param sfreq: sinusoidal frequencies
    :param smag: sinusoidal magnitudes
    :param sphase: sinusoidal phases
    :param fs: sampling rate
    :param stocf: stochastic factor, used in the approximation
    :returns: stocEnv: stochastic approximation of residual
    """

    hN = N / 2  # half of fft size
    x = np.append(np.zeros(hN), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hN))  # add zeros at the end to analyze last sample
    bh = blackmanharris(N)  # synthesis window
    w = bh / sum(bh)  # normalize synthesis window
    L = sfreq.shape[0]  # number of frames, this works if no sines
    pin = 0
    for l in range(L):
        xw = x[pin:pin + N] * w  # window the input sound
        X = fft(fftshift(xw))  # compute FFT
        Yh = spec_synth.genSpecSines(N * sfreq[l, :] / fs, smag[l, :], sphase[l, :], N)  # generate spec sines
        Xr = X - Yh  # subtract sines from original spectrum
        mXr = to_db_magnitudes(Xr[:hN])  # magnitude spectrum of residual
        mXrenv = resample(np.maximum(-200, mXr), mXr.size * stocf)  # decimate the mag spectrum
        if l == 0:  # if first frame
            stocEnv = np.array([mXrenv])
        else:  # rest of frames
            stocEnv = np.vstack((stocEnv, np.array([mXrenv])))
        pin += H  # advance sound pointer
    return stocEnv
