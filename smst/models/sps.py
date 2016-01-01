# functions that implement analysis and synthesis of sounds using the Sinusoidal plus Stochastic Model
# (for example usage check the models_interface directory)

import math

import numpy as np
from scipy.signal import resample, blackmanharris, triang, hanning
from scipy.fftpack import fft, ifft

from . import dft, sine, stochastic
from ..utils import peaks, residual, synth


def from_audio(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope, stocf):
    """
    Analysis of a sound using the sinusoidal plus stochastic model
    x: input sound, fs: sampling rate, w: analysis window; N: FFT size, t: threshold in negative dB,
    minSineDur: minimum duration of sinusoidal tracks
    maxnSines: maximum number of parallel sinusoids
    freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
    freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
    stocf: decimation factor used for the stochastic approximation
    returns hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; stocEnv: stochastic residual
    """

    # perform sinusoidal analysis
    tfreq, tmag, tphase = sine.from_audio(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
    Ns = 512
    xr = residual.subtract_sinusoids(x, Ns, H, tfreq, tmag, tphase, fs)  # subtract sinusoids from original sound
    stocEnv = stochastic.from_audio(xr, H, H * 2, stocf)  # compute stochastic model of residual
    return tfreq, tmag, tphase, stocEnv


def to_audio(tfreq, tmag, tphase, stocEnv, N, H, fs):
    """
    Synthesis of a sound using the sinusoidal plus stochastic model
    tfreq, tmag, tphase: sinusoidal frequencies, amplitudes and phases; stocEnv: stochastic envelope
    N: synthesis FFT size; H: hop size, fs: sampling rate
    returns y: output sound, ys: sinusoidal component, yst: stochastic component
    """

    ys = sine.to_audio(tfreq, tmag, tphase, N, H, fs)  # synthesize sinusoids
    yst = stochastic.to_audio(stocEnv, H, H * 2)  # synthesize stochastic residual
    y = ys[:min(ys.size, yst.size)] + yst[:min(ys.size, yst.size)]  # sum sinusoids and stochastic components
    return y, ys, yst
