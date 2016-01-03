import numpy as np

from .utilFunctions_C.twm import twm as native_twm
from .math import from_db_magnitudes

def find_peaks(mX, t):
    """
    Detects spectral peak locations.

    :param mX: magnitude spectrum
    :param t: threshold
    :returns: ploc: peak locations
    """

    thresh = np.where(mX[1:-1] > t, mX[1:-1], 0)  # locations above threshold
    next_minor = np.where(mX[1:-1] > mX[2:], mX[1:-1], 0)  # locations higher than the next one
    prev_minor = np.where(mX[1:-1] > mX[:-2], mX[1:-1], 0)  # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor  # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1  # add 1 to compensate for previous steps
    return ploc


def interpolate_peaks(mX, pX, ploc):
    """
    Interpolates peak values using parabolic interpolation.

    :param mX: magnitude spectrum
    :param pX: phase spectrum
    :param ploc: locations of peaks
    :returns: iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
    """

    val = mX[ploc]  # magnitude of peak bin
    lval = mX[ploc - 1]  # magnitude of bin at left
    rval = mX[ploc + 1]  # magnitude of bin at right
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)  # center of parabola
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)  # magnitude of peaks
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)  # phase of peaks by linear interpolation
    return iploc, ipmag, ipphase


def find_fundamental_twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):
    """
    Function that wraps the f0 detection function TWM, selecting the possible f0 candidates
    and calling the function TWM with them.

    :param pfreq: peak frequencies
    :param pmag: peak magnitudes
    :param ef0max: maximum error allowed
    :param minf0: minimum allowed f0
    :param maxf0: maximum allowed f0
    :param f0t: f0 of previous frame if stable
    :returns: f0: fundamental frequency in Hz
    """
    if minf0 < 0:  # raise exception if minf0 is smaller than 0
        raise ValueError("Minumum fundamental frequency (minf0) smaller than 0")

    if maxf0 >= 10000:  # raise exception if maxf0 is bigger than 10000Hz
        raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")

    if (pfreq.size < 3) & (f0t == 0):  # return 0 if less than 3 peaks and not previous f0
        return 0

    f0c = np.argwhere((pfreq > minf0) & (pfreq < maxf0))[:, 0]  # use only peaks within given range
    if f0c.size == 0:  # return 0 if no peaks within range
        return 0
    f0cf = pfreq[f0c]  # frequencies of peak candidates
    f0cm = pmag[f0c]  # magnitude of peak candidates

    if f0t > 0:  # if stable f0 in previous frame
        shortlist = np.argwhere(np.abs(f0cf - f0t) < f0t / 2.0)[:, 0]  # use only peaks close to it
        maxc = np.argmax(f0cm)
        maxcfd = f0cf[maxc] % f0t
        if maxcfd > f0t / 2:
            maxcfd = f0t - maxcfd
        if (maxc not in shortlist) and (maxcfd > (f0t / 4)):  # or the maximum magnitude peak is not a harmonic
            shortlist = np.append(maxc, shortlist)
        f0cf = f0cf[shortlist]  # frequencies of candidates

    if (f0cf.size == 0):  # return 0 if no peak candidates
        return 0

    f0, f0error = native_twm(pfreq, pmag, f0cf)  # call the TWM function with peak candidates

    if (f0 > 0) and (f0error < ef0max):  # accept and return f0 if below max error allowed
        return f0
    else:
        return 0


def find_fundamental_twm_py(pfreq, pmag, f0c):
    """
    Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher).
    Better to use the C version of this function: native_twm().

    :param pfreq: peak frequencies in Hz
    :param pmag: peak magnitudes
    :param f0c: frequencies of f0 candidates
    :returns: f0, f0Error: fundamental frequency detected and its error
    """

    p = 0.5  # weighting by frequency value
    q = 1.4  # weighting related to magnitude of peaks
    r = 0.5  # scaling related to magnitude of peaks
    rho = 0.33  # weighting of MP error
    Amax = max(pmag)  # maximum peak magnitude
    maxnpeaks = 10  # maximum number of peaks used
    harmonic = np.matrix(f0c)
    ErrorPM = np.zeros(harmonic.size)  # initialize PM errors
    MaxNPM = min(maxnpeaks, pfreq.size)
    for i in range(0, MaxNPM):  # predicted to measured mismatch error
        difmatrixPM = harmonic.T * np.ones(pfreq.size)
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1)) * pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)  # minimum along rows
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T) ** (-p))
        PeakMag = pmag[peakloc]
        MagFactor = from_db_magnitudes(PeakMag - Amax)
        ErrorPM = ErrorPM + (Ponddif + MagFactor * (q * Ponddif - r)).T
        harmonic += f0c

    ErrorMP = np.zeros(harmonic.size)  # initialize MP errors
    MaxNMP = min(maxnpeaks, pfreq.size)
    for i in range(0, f0c.size):  # measured to predicted mismatch error
        nharm = np.round(pfreq[:MaxNMP] / f0c[i])
        nharm = (nharm >= 1) * nharm + (nharm < 1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm * f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP] ** (-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = from_db_magnitudes(PeakMag - Amax)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor * (q * Ponddif - r)))

    Error = (ErrorPM[0] / MaxNPM) + (rho * ErrorMP / MaxNMP)  # total error
    f0index = np.argmin(Error)  # get the smallest error
    f0 = f0c[f0index]  # f0 with the smallest error

    return f0, Error[f0index]


def clean_sinusoid_track(track, minTrackLength=3):
    """
    Deletes fragments of one single track smaller than minTrackLength.

    :param track: array of values
    :param minTrackLength: minimum duration of tracks in number of frames
    :returns: cleanTrack: array of clean values
    """

    nFrames = track.size  # number of frames
    cleanTrack = np.copy(track)  # copy arrat
    trackBegs = np.nonzero((track[:nFrames - 1] <= 0)  # begining of track contours
                           & (track[1:] > 0))[0] + 1
    if track[0] > 0:
        trackBegs = np.insert(trackBegs, 0, 0)
    trackEnds = np.nonzero((track[:nFrames - 1] > 0) & (track[1:] <= 0))[0] + 1
    if track[nFrames - 1] > 0:
        trackEnds = np.append(trackEnds, nFrames - 1)
    trackLengths = 1 + trackEnds - trackBegs  # lengths of trach contours
    for i, j in zip(trackBegs, trackLengths):  # delete short track contours
        if j <= minTrackLength:
            cleanTrack[i:i + j] = 0
    return cleanTrack
