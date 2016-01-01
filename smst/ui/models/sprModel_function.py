# function to call the main analysis/synthesis functions in software/models/sprModel.py

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window

from smst.utils import audio, files
from smst.models import spr, stft
from .. import demo_sound_path
from smst.utils.files import strip_file


def main(inputFile=demo_sound_path('bendir.wav'), window='hamming', M=2001, N=2048, t=-80,
         minSineDur=0.02, maxnSines=150, freqDevOffset=10, freqDevSlope=0.001,
         interactive=True, plotFile=False):
    """
    inputFile: input sound file (monophonic with sampling rate of 44100)
    window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
    M: analysis window size
    N: fft size (power of two, bigger or equal than M)
    t: magnitude threshold of spectral peaks
    minSineDur: minimum duration of sinusoidal tracks
    maxnSines: maximum number of parallel sinusoids
    freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
    freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
    """

    # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = 128

    # read input sound
    (fs, x) = audio.read_wav(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # perform sinusoidal plus residual analysis
    tfreq, tmag, tphase, xr = spr.from_audio(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope)

    # compute spectrogram of residual
    mXr, pXr = stft.from_audio(xr, w, N, H)

    # sum sinusoids and residual
    y, ys = spr.to_audio(tfreq, tmag, tphase, xr, Ns, H, fs)

    # output sound file (monophonic with sampling rate of 44100)
    baseFileName = strip_file(inputFile)
    outputFileSines, outputFileResidual, outputFile = [
        'output_sounds/%s_sprModel%s.wav' % (baseFileName, i)
        for i in ('_sines', '_residual', '')
    ]

    # write sounds files for sinusoidal, residual, and the sum
    audio.write_wav(ys, fs, outputFileSines)
    audio.write_wav(xr, fs, outputFileResidual)
    audio.write_wav(y, fs, outputFile)

    # create figure to show plots
    plt.figure(figsize=(12, 9))

    # frequency range to plot
    maxplotfreq = 5000.0

    # plot the input sound
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(x.size) / float(fs), x)
    plt.axis([0, x.size / float(fs), min(x), max(x)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')

    # plot the magnitude spectrogram of residual
    plt.subplot(3, 1, 2)
    maxplotbin = int(N * maxplotfreq / fs)
    numFrames = int(mXr.shape[0])
    frmTime = H * np.arange(numFrames) / float(fs)
    binFreq = np.arange(maxplotbin + 1) * float(fs) / N
    plt.pcolormesh(frmTime, binFreq, np.transpose(mXr[:, :maxplotbin + 1]))
    plt.autoscale(tight=True)

    # plot the sinusoidal frequencies on top of the residual spectrogram
    if (tfreq.shape[1] > 0):
        tracks = tfreq * np.less(tfreq, maxplotfreq)
        tracks[tracks <= 0] = np.nan
        plt.plot(frmTime, tracks, color='k')
        plt.title('sinusoidal tracks + residual spectrogram')
        plt.autoscale(tight=True)

    # plot the output sound
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(y.size) / float(fs), y)
    plt.axis([0, y.size / float(fs), min(y), max(y)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('output sound: y')

    plt.tight_layout()

    if interactive:
        plt.show()
    if plotFile:
        plt.savefig('output_plots/%s_spr_model.png' % files.strip_file(inputFile))


if __name__ == "__main__":
    main()
