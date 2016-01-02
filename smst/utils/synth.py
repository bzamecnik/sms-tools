import numpy as np

from .window import blackman_harris_lobe
from .utilFunctions_C import spec_synth as UF_C
from .math import from_db_magnitudes

def spectrum_for_sinusoids(ipfreq, ipmag, ipphase, N, fs):
    """
    Generates a spectrum from a series of sine values, calling a C function.

    :param ipfreq: sine peaks frequencies
    :param ipmag: sine peaks magnitudes
    :param ipphase: sine peaks phases
    :param N: size of the complex spectrum to generate
    :param fs: sampling frequency
    :returns: Y: generated complex spectrum of sines
    """

    Y = UF_C.genSpecSines(N * ipfreq / float(fs), ipmag, ipphase, N)
    return Y


def spectrum_for_sinusoids_py(ipfreq, ipmag, ipphase, N, fs):
    """
    Generates a spectrum from a series of sine values. Python implementation.

    :param ipfreq: sine peaks frequencies
    :param ipmag: sine peaks magnitudes
    :param ipphase: sine peaks phases
    :param N: size of the complex spectrum to generate
    :param fs: sampling frequency
    :returns: Y: generated complex spectrum of sines
    """

    Y = np.zeros(N, dtype=complex)  # initialize output complex spectrum
    hN = N / 2  # size of positive freq. spectrum
    for i in range(0, ipfreq.size):  # generate all sine spectral lobes
        loc = N * ipfreq[i] / fs  # it should be in range ]0,hN-1[
        if loc == 0 or loc > hN - 1:
            continue
        binremainder = round(loc) - loc
        lb = np.arange(binremainder - 4, binremainder + 5)  # main lobe (real value) bins to read
        lmag = blackman_harris_lobe(lb) * from_db_magnitudes(ipmag[i])  # lobe magnitudes of the complex exponential
        b = np.arange(round(loc) - 4, round(loc) + 5)
        for m in range(0, 9):
            if b[m] < 0:  # peak lobe crosses DC bin
                Y[-b[m]] += lmag[m] * np.exp(-1j * ipphase[i])
            elif b[m] > hN:  # peak lobe croses Nyquist bin
                Y[b[m]] += lmag[m] * np.exp(-1j * ipphase[i])
            elif b[m] == 0 or b[m] == hN:  # peak lobe in the limits of the spectrum
                Y[b[m]] += lmag[m] * np.exp(1j * ipphase[i]) + lmag[m] * np.exp(-1j * ipphase[i])
            else:  # peak lobe in positive freq. range
                Y[b[m]] += lmag[m] * np.exp(1j * ipphase[i])
        Y[hN + 1:] = Y[hN - 1:0:-1].conjugate()  # fill the negative part of the spectrum
    return Y


def synthesize_sinusoid(freqs, amp, H, fs):
    """
    Synthesizes one sinusoid with time-varying frequency.

    :param freqs: frequencies of sinusoids
    :param amps: amplitudes of sinusoids
    :param H: hop size
    :param fs: sampling rate
    :returns: y: output sound
    """

    t = np.arange(H) / float(fs)  # time array
    lastphase = 0  # initialize synthesis phase
    lastfreq = freqs[0]  # initialize synthesis frequency
    y = np.array([])  # initialize output array
    for l in range(freqs.size):  # iterate over all frames
        if (lastfreq == 0) & (freqs[l] == 0):  # if 0 freq add zeros
            A = np.zeros(H)
            freq = np.zeros(H)
        elif (lastfreq == 0) & (freqs[l] > 0):  # if starting freq ramp up the amplitude
            A = np.arange(0, amp, amp / H)
            freq = np.ones(H) * freqs[l]
        elif (lastfreq > 0) & (freqs[l] > 0):  # if freqs in boundaries use both
            A = np.ones(H) * amp
            if lastfreq == freqs[l]:
                freq = np.ones(H) * lastfreq
            else:
                freq = np.arange(lastfreq, freqs[l], (freqs[l] - lastfreq) / H)
        elif (lastfreq > 0) & (freqs[l] == 0):  # if ending freq ramp down the amplitude
            A = np.arange(amp, 0, -amp / H)
            freq = np.ones(H) * lastfreq
        phase = 2 * np.pi * freq * t + lastphase  # generate phase values
        yh = A * np.cos(phase)  # compute sine for one frame
        lastfreq = freqs[l]  # save frequency for phase propagation
        lastphase = np.remainder(phase[H - 1], 2 * np.pi)  # save phase to be use for next frame
        y = np.append(y, yh)  # append frame to previous one
    return y
