"""
Functions that implement analysis and synthesis of sounds using the Harmonic
plus Stochastic Model.

In this model the signal is first modeled using the harmonic model. Then the
residual is modeled using the stochastic model.
"""

import numpy as np
from scipy.interpolate import interp1d

from . import harmonic, sine, stochastic
from ..utils import residual


def from_audio(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf):
    """
    Analyzes a sound using the harmonic plus stochastic model.

    :param x: input sound
    :param fs: sampling rate
    :param w: analysis window
    :param N: FFT size
    :param t: threshold in negative dB,
    :param nH: maximum number of harmonics
    :param minf0: minimum f0 frequency in Hz,
    :param maxf0: maximum f0 frequency in Hz
    :param f0et: error threshold in the f0 detection (ex: 5),
    :param harmDevSlope: slope of harmonic deviation
    :param minSineDur: minimum length of harmonics
    :returns:
      - hfreq, hmag, hphase: harmonic frequencies, magnitude and phases
      - stocEnv: stochastic residual
    """

    # perform harmonic analysis
    hfreq, hmag, hphase = harmonic.from_audio(
        x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
    # subtract sinusoids from original sound
    xr = residual.subtract_sinusoids(x, Ns, H, hfreq, hmag, hphase, fs)
    # perform stochastic analysis of residual
    stocEnv = stochastic.from_audio(xr, H, H * 2, stocf)
    return hfreq, hmag, hphase, stocEnv


def to_audio(hfreq, hmag, hphase, stocEnv, N, H, fs):
    """
    Synthesizes a sound using the harmonic plus stochastic model.

    :param hfreq: harmonic frequencies
    :param hmag: harmonic amplitudes
    :param stocEnv: stochastic envelope
    :param Ns: synthesis FFT size
    :param H: hop size
    :param fs: sampling rate
    :returns:
      - y: output sound
      - yh: harmonic component
      - yst: stochastic component
    """

    # synthesize harmonics
    yh = sine.to_audio(hfreq, hmag, hphase, N, H, fs)
    # synthesize stochastic residual
    yst = stochastic.to_audio(stocEnv, H, H * 2)
    # sum harmonic and stochastic components
    end = min(yh.size, yst.size)
    y = yh[:end] + yst[:end]
    return y, yh, yst

# functions that implement transformations using the hpsModel

def scale_time(hfreq, hmag, stocEnv, timeScaling):
    """
    Scales the harmonic plus stochastic model of a sound in time.

    :param hfreq: harmonic frequencies
    :param hmag: harmonic magnitudes
    :param stocEnv: residual envelope
    :param timeScaling: scaling factors, in time-value pairs
    :returns: yhfreq, yhmag, ystocEnv: hps output representation
    """

    if timeScaling.size % 2 != 0:  # raise exception if array not even length
        raise ValueError("Time scaling array does not have an even size")

    L = hfreq.shape[0]  # number of input frames
    inputScaling = timeScaling[::2]
    outputScaling = timeScaling[1::2]
    maxInTime = max(inputScaling)  # maximum value used as input times
    maxOutTime = max(outputScaling)  # maximum value used in output times
    outL = int(L * maxOutTime / maxInTime)  # number of output frames
    inFrames = (L - 1) * inputScaling / maxInTime  # input time values in frames
    outFrames = outL * outputScaling / maxOutTime  # output time values in frames
    timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)  # interpolation function

    # generate frame indexes for the output
    # round to get the closest input frame
    indexes = [int(round(l)) for l in timeScalingEnv(np.arange(outL))]

    yhfreq = hfreq[indexes]
    yhmag = hmag[indexes]
    ystocEnv = stocEnv[indexes]

    return yhfreq, yhmag, ystocEnv


def morph(hfreq1, hmag1, stocEnv1, hfreq2, hmag2, stocEnv2, hfreqIntp, hmagIntp, stocIntp):
    """
    Morphs between two sounds using the harmonic plus stochastic model.

    :param hfreq1, hmag1, stocEnv1: hps representation of sound 1
    :param hfreq2, hmag2, stocEnv2: hps representation of sound 2
    :param hfreqIntp: interpolation factor between the harmonic frequencies of the two sounds, 0 is sound 1 and 1 is sound 2 (time,value pairs)
    :param hmagIntp: interpolation factor between the harmonic magnitudes of the two sounds, 0 is sound 1 and 1 is sound 2  (time,value pairs)
    :param stocIntp: interpolation factor between the stochastic representation of the two sounds, 0 is sound 1 and 1 is sound 2  (time,value pairs)
    :returns: yhfreq, yhmag, ystocEnv: hps output representation
    """

    if hfreqIntp.size % 2 != 0:  # raise exception if array not even length
        raise ValueError("Harmonic frequencies interpolation array does not have an even size")

    if hmagIntp.size % 2 != 0:  # raise exception if array not even length
        raise ValueError("Harmonic magnitudes interpolation does not have an even size")

    if stocIntp.size % 2 != 0:  # raise exception if array not even length
        raise ValueError("Stochastic component array does not have an even size")

    L1 = hfreq1.shape[0]  # number of frames of sound 1
    L2 = hfreq2.shape[0]  # number of frames of sound 2

    intps = (hfreqIntp, hmagIntp, stocIntp)
    # normalize input values
    for intp in intps:
        intp[::2] = (L1 - 1) * intp[::2] / intp[-2]
    l1_samples = np.arange(L1)
    # generate frame indexes for the output via an interpolation function
    hfreqIndexes, hmagIndexes, stocIndexes = [
        interp1d(intp[0::2], intp[1::2], fill_value=0)(l1_samples)
        for intp in intps]
    # create empty output matrices
    yhfreq, yhmag, ystocEnv = [np.zeros_like(src) for src in (hfreq1, hmag1, stocEnv1)]

    def interpolate(x, y, a):
        return (1 - a) * x + a * y

    def find_first_nonzero(values):
        return np.array(np.nonzero(values), dtype=np.int)[0]

    # generate morphed frames
    # l1, l2 - frame indexes of source models
    for l1 in range(L1):
        l2 = round(L2 * l1 / float(L1))
        # identify harmonics that are present in both frames
        harmonics = np.intersect1d(
            find_first_nonzero(hfreq1[l1, :]),
            find_first_nonzero(hfreq2[l2, :]))
        # interpolate the components of both frames
        yhfreq[l1, harmonics] = interpolate(hfreq1[l1, harmonics], hfreq2[l2, harmonics], hfreqIndexes[l1])
        yhmag[l1, harmonics] = interpolate(hmag1[l1, harmonics], hmag2[l2, harmonics], hmagIndexes[l1])
        ystocEnv[l1, :] = interpolate(stocEnv1[l1, :], stocEnv2[l2, :], stocIndexes[l1])
    return yhfreq, yhmag, ystocEnv
