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
    hfreq, hmag, hphase = harmonic.from_audio(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
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

    yh = sine.to_audio(hfreq, hmag, hphase, N, H, fs)  # synthesize harmonics
    yst = stochastic.to_audio(stocEnv, H, H * 2)  # synthesize stochastic residual
    y = yh[:min(yh.size, yst.size)] + yst[:min(yh.size, yst.size)]  # sum harmonic and stochastic components
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
    maxInTime = max(timeScaling[::2])  # maximum value used as input times
    maxOutTime = max(timeScaling[1::2])  # maximum value used in output times
    outL = int(L * maxOutTime / maxInTime)  # number of output frames
    inFrames = (L - 1) * timeScaling[::2] / maxInTime  # input time values in frames
    outFrames = outL * timeScaling[1::2] / maxOutTime  # output time values in frames
    timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)  # interpolation function
    indexes = timeScalingEnv(np.arange(outL))  # generate frame indexes for the output
    yhfreq = hfreq[round(indexes[0]), :]  # first output frame
    yhmag = hmag[round(indexes[0]), :]  # first output frame
    ystocEnv = stocEnv[round(indexes[0]), :]  # first output frame
    for l in indexes[1:]:  # iterate over all output frame indexes
        yhfreq = np.vstack((yhfreq, hfreq[round(l), :]))  # get the closest input frame
        yhmag = np.vstack((yhmag, hmag[round(l), :]))  # get the closest input frame
        ystocEnv = np.vstack((ystocEnv, stocEnv[round(l), :]))  # get the closest input frame
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
    hfreqIntp[::2] = (L1 - 1) * hfreqIntp[::2] / hfreqIntp[-2]  # normalize input values
    hmagIntp[::2] = (L1 - 1) * hmagIntp[::2] / hmagIntp[-2]  # normalize input values
    stocIntp[::2] = (L1 - 1) * stocIntp[::2] / stocIntp[-2]  # normalize input values
    hfreqIntpEnv = interp1d(hfreqIntp[0::2], hfreqIntp[1::2], fill_value=0)  # interpolation function
    hfreqIndexes = hfreqIntpEnv(np.arange(L1))  # generate frame indexes for the output
    hmagIntpEnv = interp1d(hmagIntp[0::2], hmagIntp[1::2], fill_value=0)  # interpolation function
    hmagIndexes = hmagIntpEnv(np.arange(L1))  # generate frame indexes for the output
    stocIntpEnv = interp1d(stocIntp[0::2], stocIntp[1::2], fill_value=0)  # interpolation function
    stocIndexes = stocIntpEnv(np.arange(L1))  # generate frame indexes for the output
    yhfreq = np.zeros_like(hfreq1)  # create empty output matrix
    yhmag = np.zeros_like(hmag1)  # create empty output matrix
    ystocEnv = np.zeros_like(stocEnv1)  # create empty output matrix

    for l in range(L1):  # generate morphed frames
        # identify harmonics that are present in both frames
        harmonics = np.intersect1d(np.array(np.nonzero(hfreq1[l, :]), dtype=np.int)[0],
                                   np.array(np.nonzero(hfreq2[round(L2 * l / float(L1)), :]), dtype=np.int)[0])
        # interpolate the frequencies of the existing harmonics
        yhfreq[l, harmonics] = (1 - hfreqIndexes[l]) * hfreq1[l, harmonics] + hfreqIndexes[l] * hfreq2[
            round(L2 * l / float(L1)), harmonics]
        # interpolate the magnitudes of the existing harmonics
        yhmag[l, harmonics] = (1 - hmagIndexes[l]) * hmag1[l, harmonics] + hmagIndexes[l] * hmag2[
            round(L2 * l / float(L1)), harmonics]
        # interpolate the stochastic envelopes of both frames
        ystocEnv[l, :] = (1 - stocIndexes[l]) * stocEnv1[l, :] + stocIndexes[l] * stocEnv2[round(L2 * l / float(L1)), :]
    return yhfreq, yhmag, ystocEnv
