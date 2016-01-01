"""
Functions that implement analysis and synthesis of sounds using the Sinusoidal plus Residual Model.
"""

import math

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import fft, ifft

from . import dft, sine
from ..utils import peaks, residual, synth


def from_audio(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope):
    """
    Analyzes a sound using the sinusoidal plus residual model.

    :param x: input sound
    :param fs: sampling rate
    :param w: analysis window
    :param N: FFT size
    :param t: threshold in negative dB
    :param minSineDur: minimum duration of sinusoidal tracks
    :param maxnSines: maximum number of parallel sinusoids
    :param freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
    :param freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
    :returns: hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; xr: residual signal
    """

    # perform sinusoidal analysis
    tfreq, tmag, tphase = sine.from_audio(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
    Ns = 512
    xr = residual.subtract_sinusoids(x, Ns, H, tfreq, tmag, tphase, fs)  # subtract sinusoids from original sound
    return tfreq, tmag, tphase, xr


def to_audio(tfreq, tmag, tphase, xr, N, H, fs):
    """
    Synthesizes a sound using the sinusoidal plus residual model.

    :param tfreq: sinusoidal frequencies
    :param tmag: sinusoidal amplitudes
    :param tphase: sinusoidal phases
    :param stocEnv: stochastic envelope
    :param N: synthesis FFT size
    :param H: hop size
    :param fs: sampling rate
    :returns:
      - y: output sound
      - ys: sinusoidal component
    """

    ys = sine.to_audio(tfreq, tmag, tphase, N, H, fs)  # synthesize sinusoids
    y = ys[:min(ys.size, xr.size)] + xr[:min(ys.size, xr.size)]  # sum sinusoids and residual components
    return y, ys
