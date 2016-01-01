# function call to the transformation functions of relevance for the hpsModel

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window

from smst.utils import audio, files
from smst.models import harmonic, hps
from .. import demo_sound_path
from smst.utils.files import strip_file


def analysis(inputFile=demo_sound_path('sax-phrase-short.wav'), window='blackman', M=601, N=1024, t=-100,
             minSineDur=0.1, nH=100, minf0=350, maxf0=700, f0et=5, harmDevSlope=0.01, stocf=0.1,
             interactive=True, plotFile=False):
    """
    Analyze a sound with the harmonic plus stochastic model
    inputFile: input sound file (monophonic with sampling rate of 44100)
    window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
    M: analysis window size
    N: fft size (power of two, bigger or equal than M)
    t: magnitude threshold of spectral peaks
    minSineDur: minimum duration of sinusoidal tracks
    nH: maximum number of harmonics
    minf0: minimum fundamental frequency in sound
    maxf0: maximum fundamental frequency in sound
    f0et: maximum error accepted in f0 detection algorithm
    harmDevSlope: allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
    stocf: decimation factor used for the stochastic approximation
    returns inputFile: input file name; fs: sampling rate of input file,
            hfreq, hmag: harmonic frequencies, magnitude; mYst: stochastic residual
    """

    # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = 128

    # read input sound
    (fs, x) = audio.read_wav(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # compute the harmonic plus stochastic model of the whole sound
    hfreq, hmag, hphase, mYst = hps.from_audio(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns,
                                              stocf)

    # synthesize the harmonic plus stochastic model without original phases
    y, yh, yst = hps.to_audio(hfreq, hmag, np.array([]), mYst, Ns, H, fs)

    # write output sound
    outputFile = 'output_sounds/' + strip_file(inputFile) + '_hpsModel.wav'
    audio.write_wav(y, fs, outputFile)

    # create figure to plot
    plt.figure(figsize=(12, 9))

    # frequency range to plot
    maxplotfreq = 15000.0

    # plot the input sound
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(x.size) / float(fs), x)
    plt.axis([0, x.size / float(fs), min(x), max(x)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')

    # plot spectrogram stochastic compoment
    plt.subplot(3, 1, 2)
    numFrames = int(mYst.shape[0])
    sizeEnv = int(mYst.shape[1])
    frmTime = H * np.arange(numFrames) / float(fs)
    binFreq = (.5 * fs) * np.arange(sizeEnv * maxplotfreq / (.5 * fs)) / sizeEnv
    plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:, :sizeEnv * maxplotfreq / (.5 * fs) + 1]))
    plt.autoscale(tight=True)

    # plot harmonic on top of stochastic spectrogram
    if (hfreq.shape[1] > 0):
        harms = hfreq * np.less(hfreq, maxplotfreq)
        harms[harms == 0] = np.nan
        numFrames = int(harms.shape[0])
        frmTime = H * np.arange(numFrames) / float(fs)
        plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
        plt.xlabel('time (sec)')
        plt.ylabel('frequency (Hz)')
        plt.autoscale(tight=True)
        plt.title('harmonics + stochastic spectrogram')

    # plot the output sound
    plt.subplot(3, 1, 3)
    plt.plot(np.arange(y.size) / float(fs), y)
    plt.axis([0, y.size / float(fs), min(y), max(y)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('output sound: y')

    plt.tight_layout()

    if interactive:
        plt.show(block=False)
    if plotFile:
        plt.savefig('output_plots/%s_hps_transformation_analysis.png' % files.strip_file(inputFile))

    return inputFile, fs, hfreq, hmag, mYst


def transformation_synthesis(inputFile, fs, hfreq, hmag, mYst,
                             freqScaling=np.array([0, 1.2, 2.01, 1.2, 2.679, .7, 3.146, .7]),
                             freqStretching=np.array([0, 1, 2.01, 1, 2.679, 1.5, 3.146, 1.5]), timbrePreservation=1,
                             timeScaling=np.array([0, 0, 2.138, 2.138 - 1.0, 3.146, 3.146]),
                             interactive=True, plotFile=False):
    """
    transform the analysis values returned by the analysis function and synthesize the sound
    inputFile: name of input file
    fs: sampling rate of input file
    hfreq, hmag: harmonic frequencies and magnitudes
    mYst: stochastic residual
    freqScaling: frequency scaling factors, in time-value pairs (value of 1 no scaling)
    freqStretching: frequency stretching factors, in time-value pairs (value of 1 no stretching)
    timbrePreservation: 1 preserves original timbre, 0 it does not
    timeScaling: time scaling factors, in time-value pairs
    """

    # size of fft used in synthesis
    Ns = 512

    # hop size (has to be 1/4 of Ns)
    H = 128

    # frequency scaling of the harmonics
    hfreqt, hmagt = harmonic.scale_frequencies(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)

    # time scaling the sound
    yhfreq, yhmag, ystocEnv = hps.scale_time(hfreqt, hmagt, mYst, timeScaling)

    # synthesis from the trasformed hps representation
    y, yh, yst = hps.to_audio(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)

    # write output sound
    outputFile = 'output_sounds/' + strip_file(inputFile) + '_hpsModelTransformation.wav'
    audio.write_wav(y, fs, outputFile)

    # create figure to plot
    plt.figure(figsize=(12, 6))

    # frequency range to plot
    maxplotfreq = 15000.0

    # plot spectrogram of transformed stochastic compoment
    plt.subplot(2, 1, 1)
    numFrames = int(ystocEnv.shape[0])
    sizeEnv = int(ystocEnv.shape[1])
    frmTime = H * np.arange(numFrames) / float(fs)
    binFreq = (.5 * fs) * np.arange(sizeEnv * maxplotfreq / (.5 * fs)) / sizeEnv
    plt.pcolormesh(frmTime, binFreq, np.transpose(ystocEnv[:, :sizeEnv * maxplotfreq / (.5 * fs) + 1]))
    plt.autoscale(tight=True)

    # plot transformed harmonic on top of stochastic spectrogram
    if (yhfreq.shape[1] > 0):
        harms = yhfreq * np.less(yhfreq, maxplotfreq)
        harms[harms == 0] = np.nan
        numFrames = int(harms.shape[0])
        frmTime = H * np.arange(numFrames) / float(fs)
        plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
        plt.xlabel('time (sec)')
        plt.ylabel('frequency (Hz)')
        plt.autoscale(tight=True)
        plt.title('harmonics + stochastic spectrogram')

    # plot the output sound
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(y.size) / float(fs), y)
    plt.axis([0, y.size / float(fs), min(y), max(y)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('output sound: y')

    plt.tight_layout()

    if interactive:
        plt.show()
    if plotFile:
        plt.savefig('output_plots/%s_hps_transformation_synthesis.png' % files.strip_file(inputFile))


def main(interactive=True, plotFile=False):
    # analysis
    inputFile, fs, hfreq, hmag, mYst = analysis(interactive=interactive, plotFile=plotFile)

    # transformation and synthesis
    transformation_synthesis(inputFile, fs, hfreq, hmag, mYst, interactive=interactive, plotFile=plotFile)


if __name__ == "__main__":
    main()
