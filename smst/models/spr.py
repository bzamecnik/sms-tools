# functions that implement analysis and synthesis of sounds using the Sinusoidal plus Residual Model
# (for example usage check the examples models_interface)

import math

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import fft, ifft

from . import dft, sine
from ..utils import peaks, residual, synth


def from_audio(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope):
    """
    Analysis of a sound using the sinusoidal plus residual model
    x: input sound, fs: sampling rate, w: analysis window; N: FFT size, t: threshold in negative dB,
    minSineDur: minimum duration of sinusoidal tracks
    maxnSines: maximum number of parallel sinusoids
    freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
    freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
    returns hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; xr: residual signal
    """

    # perform sinusoidal analysis
    tfreq, tmag, tphase = sine.from_audio(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
    Ns = 512
    xr = residual.subtract_sinusoids(x, Ns, H, tfreq, tmag, tphase, fs)  # subtract sinusoids from original sound
    return tfreq, tmag, tphase, xr


def to_audio(tfreq, tmag, tphase, xr, N, H, fs):
    """
    Synthesis of a sound using the sinusoidal plus residual model
    tfreq, tmag, tphase: sinusoidal frequencies, amplitudes and phases; stocEnv: stochastic envelope
    N: synthesis FFT size; H: hop size, fs: sampling rate
    returns y: output sound, y: sinusoidal component
    """

    ys = sine.to_audio(tfreq, tmag, tphase, N, H, fs)  # synthesize sinusoids
    y = ys[:min(ys.size, xr.size)] + xr[:min(ys.size, xr.size)]  # sum sinusoids and residual components
    return y, ys
