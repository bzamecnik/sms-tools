"""
Functions that implement analysis and synthesis of sounds using the Harmonic plus Residual Model.
"""

from . import harmonic, sine
from ..utils import residual


def from_audio(x, fs, w, N, H, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope):
    """
    Analyzes a sound using the harmonic plus residual model.

    :param x: input sound
    :param fs: sampling rate
    :param w: analysis window
    :param N: FFT size
    :param t: threshold in negative dB
    :param minSineDur: minimum duration of sinusoidal tracks
    :param nH: maximum number of harmonics
    :param minf0: minimum fundamental frequency in sound
    :param maxf0: maximum fundamental frequency in sound
    :param f0et: maximum error accepted in f0 detection algorithm
    :param harmDevSlope: allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
    :returns:
      - hfreq, hmag, hphase: harmonic frequencies, magnitude and phases
      - xr: residual signal
    """

    # perform harmonic analysis
    hfreq, hmag, hphase = harmonic.from_audio(
        x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)

    # subtract sinusoids from original sound
    Ns = 512
    xr = residual.subtract_sinusoids(x, Ns, H, hfreq, hmag, hphase, fs)

    return hfreq, hmag, hphase, xr


def to_audio(hfreq, hmag, hphase, xr, N, H, fs):
    """
    Synthesizes a sound using the sinusoidal plus residual model.

    :param tfreq: sinusoidal frequencies
    :param tmag: sinusoidal amplitudes
    :param tphase: sinusoidal phases
    :param stocEnv: stochastic envelope
    :param N: synthesis FFT size
    :param H: hop size
    :param fs: sampling rate
    :returns: y: output sound, yh: harmonic component
    """

    # synthesize sinusoids
    yh = sine.to_audio(hfreq, hmag, hphase, N, H, fs)

    # sum sinusoids and residual components
    end = min(yh.size, xr.size)
    y = yh[:end] + xr[:end]

    return y, yh
