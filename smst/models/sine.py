# functions that implement analysis and synthesis of sounds using the Sinusoidal Model
# (for example usage check the examples models_interface)

import math

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft, fftshift

from . import dft, stft
from ..utils import peaks, synth


def from_audio(x, fs, w, N, H, t, maxnSines=100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01):
    """
    Analysis of a sound using the sinusoidal model with sine tracking
    x: input array sound, w: analysis window, N: size of complex spectrum, H: hop-size, t: threshold in negative dB
    maxnSines: maximum number of sines per frame, minSineDur: minimum duration of sines in seconds
    freqDevOffset: minimum frequency deviation at 0Hz, freqDevSlope: slope increase of minimum frequency deviation
    returns xtfreq, xtmag, xtphase: frequencies, magnitudes and phases of sinusoidal tracks
    """

    if (minSineDur < 0):  # raise error if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hM1, hM2 = dft.half_window_sizes(w.size)
    x_padded = stft.pad_signal(x, hM2)
    w = w / sum(w)  # normalize analysis window
    tfreq = np.array([])
    for frame_index, x1 in enumerate(stft.iterate_analysis_frames(x_padded, H, hM1, hM2)):
        mX, pX = dft.from_audio(x1, w, N)  # compute dft
        ploc = peaks.find_peaks(mX, t)  # detect locations of peaks
        iploc, ipmag, ipphase = peaks.interpolate_peaks(mX, pX, ploc)  # refine peak values by interpolation
        ipfreq = fs * iploc / float(N)  # convert peak locations to Hertz
        # perform sinusoidal tracking by adding peaks to trajectories
        tfreq, tmag, tphase = track_sinusoids(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
        tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))  # limit number of tracks to maxnSines
        tmag = np.resize(tmag, min(maxnSines, tmag.size))  # limit number of tracks to maxnSines
        tphase = np.resize(tphase, min(maxnSines, tphase.size))  # limit number of tracks to maxnSines
        jtfreq = np.zeros(maxnSines)  # temporary output array
        jtmag = np.zeros(maxnSines)  # temporary output array
        jtphase = np.zeros(maxnSines)  # temporary output array
        jtfreq[:tfreq.size] = tfreq  # save track frequencies to temporary array
        jtmag[:tmag.size] = tmag  # save track magnitudes to temporary array
        jtphase[:tphase.size] = tphase  # save track magnitudes to temporary array
        if frame_index == 0:  # if first frame initialize output sine tracks
            xtfreq = jtfreq
            xtmag = jtmag
            xtphase = jtphase
        else:  # rest of frames append values to sine tracks
            xtfreq = np.vstack((xtfreq, jtfreq))
            xtmag = np.vstack((xtmag, jtmag))
            xtphase = np.vstack((xtphase, jtphase))
    # delete sine tracks shorter than minSineDur
    xtfreq = clean_sinusoid_tracks(xtfreq, round(fs * minSineDur / H))
    return xtfreq, xtmag, xtphase


def to_audio(tfreq, tmag, tphase, N, H, fs):
    """
    Synthesis of a sound using the sinusoidal model
    tfreq,tmag,tphase: frequencies, magnitudes and phases of sinusoids
    N: synthesis FFT size, H: hop size, fs: sampling rate
    returns y: output array sound
    """

    hN = N / 2  # half of FFT size for synthesis
    L = tfreq.shape[0]  # number of frames
    pout = 0  # initialize output sound pointer
    ysize = H * (L + 3)  # output sound size
    y = np.zeros(ysize)  # initialize output array
    sw = np.zeros(N)  # initialize synthesis window
    ow = triang(2 * H)  # triangular window
    sw[hN - H:hN + H] = ow  # add triangular window
    bh = blackmanharris(N)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hN - H:hN + H] = sw[hN - H:hN + H] / bh[hN - H:hN + H]  # normalized synthesis window
    lastytfreq = tfreq[0, :]  # initialize synthesis frequencies
    ytphase = 2 * np.pi * np.random.rand(tfreq[0, :].size)  # initialize synthesis phases
    for l in range(L):  # iterate over all frames
        if tphase.size > 0:  # if no phases generate them
            ytphase = tphase[l, :]
        else:
            ytphase += (np.pi * (lastytfreq + tfreq[l, :]) / fs) * H  # propagate phases
        Y = synth.spectrum_for_sinusoids(tfreq[l, :], tmag[l, :], ytphase, N, fs)  # generate sines in the spectrum
        lastytfreq = tfreq[l, :]  # save frequency for phase propagation
        ytphase %= 2 * np.pi  # make phase inside 2*pi
        yw = np.real(fftshift(ifft(Y)))  # compute inverse FFT
        y[pout:pout + N] += sw * yw  # overlap-add and apply a synthesis window
        pout += H  # advance sound pointer
    y = np.delete(y, range(hN))  # delete half of first window
    y = np.delete(y, range(y.size - hN, y.size))  # delete half of the last window
    return y

# functions that implement transformations using the sineModel

def scale_time(sfreq, smag, timeScaling):
    """
    Time scaling of sinusoidal tracks
    sfreq, smag: frequencies and magnitudes of input sinusoidal tracks
    timeScaling: scaling factors, in time-value pairs
    returns ysfreq, ysmag: frequencies and magnitudes of output sinusoidal tracks
    """
    if timeScaling.size % 2 != 0:  # raise exception if array not even length
        raise ValueError("Time scaling array does not have an even size")

    L = sfreq.shape[0]  # number of input frames
    maxInTime = max(timeScaling[::2])  # maximum value used as input times
    maxOutTime = max(timeScaling[1::2])  # maximum value used in output times
    outL = int(L * maxOutTime / maxInTime)  # number of output frames
    inFrames = (L - 1) * timeScaling[::2] / maxInTime  # input time values in frames
    outFrames = outL * timeScaling[1::2] / maxOutTime  # output time values in frames
    timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)  # interpolation function
    indexes = timeScalingEnv(np.arange(outL))  # generate frame indexes for the output
    ysfreq = sfreq[round(indexes[0]), :]  # first output frame
    ysmag = smag[round(indexes[0]), :]  # first output frame
    for l in indexes[1:]:  # generate frames for output sine tracks
        ysfreq = np.vstack((ysfreq, sfreq[round(l), :]))  # get closest frame to scaling value
        ysmag = np.vstack((ysmag, smag[round(l), :]))  # get closest frame to scaling value
    return ysfreq, ysmag


def scale_frequencies(sfreq, freqScaling):
    """
    Frequency scaling of sinusoidal tracks
    sfreq: frequencies of input sinusoidal tracks
    freqScaling: scaling factors, in time-value pairs (value of 1 is no scaling)
    returns ysfreq: frequencies of output sinusoidal tracks
    """
    if (freqScaling.size % 2 != 0):  # raise exception if array not even length
        raise ValueError("Frequency scaling array does not have an even size")

    L = sfreq.shape[0]  # number of input frames
    # create interpolation object from the scaling values
    freqScalingEnv = np.interp(np.arange(L), L * freqScaling[::2] / freqScaling[-2], freqScaling[1::2])
    ysfreq = np.zeros_like(sfreq)  # create empty output matrix
    for l in range(L):  # go through all frames
        ind_valid = np.where(sfreq[l, :] != 0)[0]  # check if there are frequency values
        if ind_valid.size == 0:  # if no values go to next frame
            continue
        ysfreq[l, ind_valid] = sfreq[l, ind_valid] * freqScalingEnv[l]  # scale of frequencies
    return ysfreq


# -- support functions --

def track_sinusoids(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
    """
    Tracking sinusoids from one frame to the next
    pfreq, pmag, pphase: frequencies and magnitude of current frame
    tfreq: frequencies of incoming tracks from previous frame
    freqDevOffset: minimum frequency deviation at 0Hz
    freqDevSlope: slope increase of minimum frequency deviation
    returns tfreqn, tmagn, tphasen: frequency, magnitude and phase of tracks
    """

    tfreqn = np.zeros(tfreq.size)  # initialize array for output frequencies
    tmagn = np.zeros(tfreq.size)  # initialize array for output magnitudes
    tphasen = np.zeros(tfreq.size)  # initialize array for output phases
    pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[0]  # indexes of current peaks
    incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0]  # indexes of incoming tracks
    newTracks = np.zeros(tfreq.size, dtype=np.int) - 1  # initialize to -1 new tracks
    magOrder = np.argsort(-pmag[pindexes])  # order current peaks by magnitude
    pfreqt = np.copy(pfreq)  # copy current peaks to temporary array
    pmagt = np.copy(pmag)  # copy current peaks to temporary array
    pphaset = np.copy(pphase)  # copy current peaks to temporary array

    # continue incoming tracks
    if incomingTracks.size > 0:  # if incoming tracks exist
        for i in magOrder:  # iterate over current peaks
            if incomingTracks.size == 0:  # break when no more incoming tracks
                break
            track = np.argmin(abs(pfreqt[i] - tfreq[incomingTracks]))  # closest incoming track to peak
            freqDistance = abs(pfreq[i] - tfreq[incomingTracks[track]])  # measure freq distance
            if freqDistance < (freqDevOffset + freqDevSlope * pfreq[i]):  # choose track if distance is small
                newTracks[incomingTracks[track]] = i  # assign peak index to track index
                incomingTracks = np.delete(incomingTracks, track)  # delete index of track in incoming tracks
    indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[0]  # indexes of assigned tracks
    if indext.size > 0:
        indexp = newTracks[indext]  # indexes of assigned peaks
        tfreqn[indext] = pfreqt[indexp]  # output freq tracks
        tmagn[indext] = pmagt[indexp]  # output mag tracks
        tphasen[indext] = pphaset[indexp]  # output phase tracks
        pfreqt = np.delete(pfreqt, indexp)  # delete used peaks
        pmagt = np.delete(pmagt, indexp)  # delete used peaks
        pphaset = np.delete(pphaset, indexp)  # delete used peaks

    # create new tracks from non used peaks
    emptyt = np.array(np.nonzero(tfreq == 0), dtype=np.int)[0]  # indexes of empty incoming tracks
    peaksleft = np.argsort(-pmagt)  # sort left peaks by magnitude
    if ((peaksleft.size > 0) & (emptyt.size >= peaksleft.size)):  # fill empty tracks
        tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
        tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
        tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
    elif ((peaksleft.size > 0) & (emptyt.size < peaksleft.size)):  # add more tracks if necessary
        tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
        tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
        tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
        tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
        tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
        tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
    return tfreqn, tmagn, tphasen


def clean_sinusoid_tracks(tfreq, minTrackLength=3):
    """
    Delete short fragments of a collection of sinusoidal tracks
    tfreq: frequency of tracks
    minTrackLength: minimum duration of tracks in number of frames
    returns tfreqn: output frequency of tracks
    """

    if tfreq.shape[1] == 0:  # if no tracks return input
        return tfreq
    nFrames = tfreq[:, 0].size  # number of frames
    nTracks = tfreq[0, :].size  # number of tracks in a frame
    for t in range(nTracks):  # iterate over all tracks
        trackFreqs = tfreq[:, t]  # frequencies of one track
        trackBegs = np.nonzero((trackFreqs[:nFrames - 1] <= 0)  # beginning of track contours
                               & (trackFreqs[1:] > 0))[0] + 1
        if trackFreqs[0] > 0:
            trackBegs = np.insert(trackBegs, 0, 0)
        trackEnds = np.nonzero((trackFreqs[:nFrames - 1] > 0)  # end of track contours
                               & (trackFreqs[1:] <= 0))[0] + 1
        if trackFreqs[nFrames - 1] > 0:
            trackEnds = np.append(trackEnds, nFrames - 1)
        trackLengths = 1 + trackEnds - trackBegs  # lengths of track contours
        for i, j in zip(trackBegs, trackLengths):  # delete short track contours
            if j <= minTrackLength:
                trackFreqs[i:i + j] = 0
    return tfreq
