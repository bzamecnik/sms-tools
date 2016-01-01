"""
Functions that implement analysis and synthesis of sounds using the Harmonic Model.
"""

import math

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft

from . import dft, sine, stft
from ..utils import peaks, synth


def from_audio(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope=0.01, minSineDur=.02):
    """
    Analyzes a sound using the sinusoidal harmonic model.

    :param x: input sound
    :param fs: sampling rate
    :param w: analysis window
    :param N: FFT size (minimum 512)
    :param t: threshold in negative dB
    :param nH: maximum number of harmonics
    :param minf0: minimum f0 frequency in Hz
    :param maxf0: maximim f0 frequency in Hz
    :param f0et: error threshold in the f0 detection (ex: 5)
    :param harmDevSlope: slope of harmonic deviation
    :param minSineDur: minimum length of harmonics
    :returns: xhfreq, xhmag, xhphase: harmonic frequencies, magnitudes and phases
    """

    if minSineDur < 0:  # raise exception if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hM1, hM2 = dft.half_window_sizes(w.size)
    x_padded = stft.pad_signal(x, hM2)
    w = w / sum(w)  # normalize analysis window
    hfreqp = []  # initialize harmonic frequencies of previous frame
    f0stable = 0  # initialize f0 stable
    for frame_index, x1 in enumerate(stft.iterate_analysis_frames(x_padded, H, hM1, hM2)):
        mX, pX = dft.from_audio(x1, w, N)  # compute dft
        ploc = peaks.find_peaks(mX, t)  # detect peak locations
        iploc, ipmag, ipphase = peaks.interpolate_peaks(mX, pX, ploc)  # refine peak values
        ipfreq = fs * iploc / N  # convert locations to Hz
        f0t = peaks.find_fundamental_twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
        if ((f0stable == 0) & (f0t > 0)) \
                or ((f0stable > 0) & (np.abs(f0stable - f0t) < f0stable / 5.0)):
            f0stable = f0t  # consider a stable f0 if it is close to the previous one
        else:
            f0stable = 0
        hfreq, hmag, hphase = find_harmonics(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs, harmDevSlope)  # find harmonics
        hfreqp = hfreq
        if frame_index == 0:  # first frame
            xhfreq = np.array([hfreq])
            xhmag = np.array([hmag])
            xhphase = np.array([hphase])
        else:  # next frames
            xhfreq = np.vstack((xhfreq, np.array([hfreq])))
            xhmag = np.vstack((xhmag, np.array([hmag])))
            xhphase = np.vstack((xhphase, np.array([hphase])))
    xhfreq = sine.clean_sinusoid_tracks(xhfreq, round(fs * minSineDur / H))  # delete tracks shorter than minSineDur
    return xhfreq, xhmag, xhphase

# transformations applied to the harmonics of a sound

def scale_frequencies(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs):
    """
    Scales the frequencies of the harmonics of a sound.

    :param hfreq: frequencies of input harmonics
    :param hmag: magnitudes of input harmonics
    :param freqScaling: scaling factors, in time-value pairs (value of 1 no scaling)
    :param freqStretching: stretching factors, in time-value pairs (value of 1 no stretching)
    :param timbrePreservation: 0  no timbre preservation, 1 timbre preservation
    :param fs: sampling rate of input sound
    :returns: yhfreq, yhmag: frequencies and magnitudes of output harmonics
    """
    if freqScaling.size % 2 != 0:  # raise exception if array not even length
        raise ValueError("Frequency scaling array does not have an even size")

    if freqStretching.size % 2 != 0:  # raise exception if array not even length
        raise ValueError("Frequency stretching array does not have an even size")

    L = hfreq.shape[0]  # number of frames
    # create interpolation object with the scaling values
    freqScalingEnv = np.interp(np.arange(L), L * freqScaling[::2] / freqScaling[-2], freqScaling[1::2])
    # create interpolation object with the stretching values
    freqStretchingEnv = np.interp(np.arange(L), L * freqStretching[::2] / freqStretching[-2], freqStretching[1::2])
    yhfreq = np.zeros_like(hfreq)  # create empty output matrix
    yhmag = np.zeros_like(hmag)  # create empty output matrix
    for l in range(L):  # go through all frames
        ind_valid = np.where(hfreq[l, :] != 0)[0]  # check if there are frequency values
        if ind_valid.size == 0:  # if no values go to next frame
            continue
        if (timbrePreservation == 1) & (ind_valid.size > 1):  # create spectral envelope
            # values of harmonic locations to be considered for interpolation
            x_vals = np.append(np.append(0, hfreq[l, ind_valid]), fs / 2)
            # values of harmonic magnitudes to be considered for interpolation
            y_vals = np.append(np.append(hmag[l, 0], hmag[l, ind_valid]), hmag[l, -1])
            specEnvelope = interp1d(x_vals, y_vals, kind='linear', bounds_error=False, fill_value=-100)
        yhfreq[l, ind_valid] = hfreq[l, ind_valid] * freqScalingEnv[l]  # scale frequencies
        yhfreq[l, ind_valid] = yhfreq[l, ind_valid] * (freqStretchingEnv[l] ** ind_valid)  # stretch frequencies
        if (timbrePreservation == 1) & (ind_valid.size > 1):  # if timbre preservation
            yhmag[l, ind_valid] = specEnvelope(yhfreq[l, ind_valid])  # change amplitudes to maintain timbre
        else:
            yhmag[l, ind_valid] = hmag[l, ind_valid]  # use same amplitudes as input
    return yhfreq, yhmag

# -- supporting function --

def find_fundamental_freq(x, fs, w, N, H, t, minf0, maxf0, f0et):
    """
    Finds fundamental frequencies of a sound using the TWM (Two-Way Mismatch) algorithm.

    :param x: input sound
    :param fs: sampling rate
    :param w: analysis window
    :param N: FFT size
    :param t: threshold in negative dB
    :param minf0: minimum f0 frequency in Hz
    :param maxf0: maximim f0 frequency in Hz
    :param f0et: error threshold in the f0 detection (ex: 5)
    :returns: f0: fundamental frequency
    """
    if minf0 < 0:  # raise exception if minf0 is smaller than 0
        raise ValueError("Minumum fundamental frequency (minf0) smaller than 0")

    # TODO: use fs/2 instead a constant
    if maxf0 >= 10000:  # raise exception if maxf0 is bigger than fs/2
        raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")

    if H <= 0:  # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    hM1, hM2 = dft.half_window_sizes(w.size)
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM1))  # add zeros at the end to analyze last sample
    w = w / sum(w)  # normalize analysis window
    f0 = []  # initialize f0 output
    f0stable = 0  # initialize f0 stable
    for x1 in stft.iterate_analysis_frames(x, H, hM1, hM2):
        mX, pX = dft.from_audio(x1, w, N)  # compute dft
        ploc = peaks.find_peaks(mX, t)  # detect peak locations
        iploc, ipmag, ipphase = peaks.interpolate_peaks(mX, pX, ploc)  # refine peak values
        ipfreq = fs * iploc / N  # convert locations to Hez
        f0t = peaks.find_fundamental_twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
        if ((f0stable == 0) & (f0t > 0)) \
                or ((f0stable > 0) & (np.abs(f0stable - f0t) < f0stable / 5.0)):
            f0stable = f0t  # consider a stable f0 if it is close to the previous one
        else:
            f0stable = 0
        f0 = np.append(f0, f0t)  # add f0 to output array
    return f0


def find_harmonics(pfreq, pmag, pphase, f0, nH, hfreqp, fs, harmDevSlope=0.01):
    """
    Finds harmonics of a frame from a set of spectral peaks using f0
    to the ideal harmonic series built on top of a fundamental frequency.

    :param pfreq: peak frequencies
    :param pmag: peak magnitudes
    :param pphase: peak phases
    :param f0: fundamental frequency
    :param nH: number of harmonics
    :param hfreqp: harmonic frequencies of previous frame
    :param fs: sampling rate
    :param harmDevSlope: slope of change of the deviation allowed to perfect harmonic
    :returns: hfreq, hmag, hphase: harmonic frequencies, magnitudes, phases
    """

    if f0 <= 0:  # if no f0 return no harmonics
        return np.zeros(nH), np.zeros(nH), np.zeros(nH)
    hfreq = np.zeros(nH)  # initialize harmonic frequencies
    hmag = np.zeros(nH) - 100  # initialize harmonic magnitudes
    hphase = np.zeros(nH)  # initialize harmonic phases
    hf = f0 * np.arange(1, nH + 1)  # initialize harmonic frequencies
    hi = 0  # initialize harmonic index
    if hfreqp == []:  # if no incoming harmonic tracks initialize to harmonic series
        hfreqp = hf
    while (f0 > 0) and (hi < nH) and (hf[hi] < fs / 2):  # find harmonic peaks
        pei = np.argmin(abs(pfreq - hf[hi]))  # closest peak
        dev1 = abs(pfreq[pei] - hf[hi])  # deviation from perfect harmonic
        dev2 = (abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi] > 0 else fs)  # deviation from previous frame
        threshold = f0 / 3 + harmDevSlope * pfreq[pei]
        if (dev1 < threshold) or (dev2 < threshold):  # accept peak if deviation is small
            hfreq[hi] = pfreq[pei]  # harmonic frequencies
            hmag[hi] = pmag[pei]  # harmonic magnitudes
            hphase[hi] = pphase[pei]  # harmonic phases
        hi += 1  # increase harmonic index
    return hfreq, hmag, hphase
