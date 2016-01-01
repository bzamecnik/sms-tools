# functions that implement analysis and synthesis of sounds using the Harmonic plus Residual Model
# (for example usage check the models_interface directory)

import math

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import fft, ifft

from . import dft, harmonic, sine
from ..utils import peaks, residual, synth


def from_audio(x, fs, w, N, H, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope):
    """Analysis of a sound using the harmonic plus residual model
    x: input sound, fs: sampling rate, w: analysis window; N: FFT size, t: threshold in negative dB,
    minSineDur: minimum duration of sinusoidal tracks
    nH: maximum number of harmonics; minf0: minimum fundamental frequency in sound
    maxf0: maximum fundamental frequency in sound; f0et: maximum error accepted in f0 detection algorithm
    harmDevSlope: allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
    returns hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; xr: residual signal
    """

    # perform harmonic analysis
    hfreq, hmag, hphase = harmonic.from_audio(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
    Ns = 512
    xr = residual.subtract_sinusoids(x, Ns, H, hfreq, hmag, hphase, fs)  # subtract sinusoids from original sound
    return hfreq, hmag, hphase, xr


def to_audio(hfreq, hmag, hphase, xr, N, H, fs):
    """
    Synthesis of a sound using the sinusoidal plus residual model
    tfreq, tmag, tphase: sinusoidal frequencies, amplitudes and phases; stocEnv: stochastic envelope
    N: synthesis FFT size; H: hop size, fs: sampling rate
    returns y: output sound, yh: harmonic component
    """

    yh = sine.to_audio(hfreq, hmag, hphase, N, H, fs)  # synthesize sinusoids
    y = yh[:min(yh.size, xr.size)] + xr[:min(yh.size, xr.size)]  # sum sinusoids and residual components
    return y, yh
