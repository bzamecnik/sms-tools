# function for doing a morph between two sounds using the stft

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window

from smst.utils import audio, files
from smst.models import stft
from .. import demo_sound_path


def main(inputFile1=demo_sound_path('ocean.wav'), inputFile2=demo_sound_path('speech-male.wav'), window1='hamming',
         window2='hamming',
         M1=1024, M2=1024, N1=1024, N2=1024, H1=256, smoothf=.5, balancef=0.2,
         interactive=True, plotFile=False):
    """
    Function to perform a morph between two sounds
    inputFile1: name of input sound file to be used as source
    inputFile2: name of input sound file to be used as filter
    window1 and window2: windows for both files
    M1 and M2: window sizes for both files
    N1 and N2: fft sizes for both sounds
    H1: hop size for sound 1 (the one for sound 2 is computed automatically)
    smoothf: smoothing factor to be applyed to magnitude spectrum of sound 2 before morphing
    balancef: balance factor between booth sounds, 0 is sound 1 and 1 is sound 2
    """

    # read input sounds
    (fs, x1) = audio.read_wav(inputFile1)
    (fs, x2) = audio.read_wav(inputFile2)

    # compute analysis windows
    w1 = get_window(window1, M1)
    w2 = get_window(window2, M2)

    # perform morphing
    y = stft.morph(x1, x2, fs, w1, N1, w2, N2, H1, smoothf, balancef)

    # compute the magnitude and phase spectrogram of input sound (for plotting)
    mX1, pX1 = stft.from_audio(x1, w1, N1, H1)

    # compute the magnitude and phase spectrogram of output sound (for plotting)
    mY, pY = stft.from_audio(y, w1, N1, H1)

    # write output sound
    outputFile = 'output_sounds/' + os.path.basename(inputFile1)[:-4] + '_stftMorph.wav'
    audio.write_wav(y, fs, outputFile)

    # create figure to plot
    plt.figure(figsize=(12, 9))

    # frequency range to plot
    maxplotfreq = 10000.0

    # plot sound 1
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(x1.size) / float(fs), x1)
    plt.axis([0, x1.size / float(fs), min(x1), max(x1)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')

    # plot magnitude spectrogram of sound 1
    plt.subplot(4, 1, 2)
    numFrames = int(mX1.shape[0])
    frmTime = H1 * np.arange(numFrames) / float(fs)
    binFreq = fs * np.arange(N1 * maxplotfreq / fs) / N1
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX1[:, :N1 * maxplotfreq / fs + 1]))
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.title('magnitude spectrogram of x')
    plt.autoscale(tight=True)

    # plot magnitude spectrogram of morphed sound
    plt.subplot(4, 1, 3)
    numFrames = int(mY.shape[0])
    frmTime = H1 * np.arange(numFrames) / float(fs)
    binFreq = fs * np.arange(N1 * maxplotfreq / fs) / N1
    plt.pcolormesh(frmTime, binFreq, np.transpose(mY[:, :N1 * maxplotfreq / fs + 1]))
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.title('magnitude spectrogram of y')
    plt.autoscale(tight=True)

    # plot the morphed sound
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(y.size) / float(fs), y)
    plt.axis([0, y.size / float(fs), min(y), max(y)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('output sound: y')

    plt.tight_layout()

    if interactive:
        plt.show()
    if plotFile:
        plt.savefig('output_plots/%s_%s_stft_morph.png' % (
            files.strip_file(inputFile1),
            files.strip_file(inputFile2)
        ))


if __name__ == '__main__':
    main()
