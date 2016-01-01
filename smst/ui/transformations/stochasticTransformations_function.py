# function call to the transformation function of relevance to the stochasticModel

import os

import matplotlib.pyplot as plt
import numpy as np

from smst.utils import audio, files
from smst.models import stochastic
from .. import demo_sound_path
from smst.utils.files import strip_file


def main(inputFile=demo_sound_path('rain.wav'), stocf=0.1, timeScaling=np.array([0, 0, 1, 2]),
         interactive=True, plotFile=False):
    """
    function to perform a time scaling using the stochastic model
    inputFile: name of input sound file
    stocf: decimation factor used for the stochastic approximation
    timeScaling: time scaling factors, in time-value pairs
    """

    # hop size
    H = 128

    # read input sound
    (fs, x) = audio.read_wav(inputFile)

    # perform stochastic analysis
    mYst = stochastic.from_audio(x, H, H * 2, stocf)

    # perform time scaling of stochastic representation
    ystocEnv = stochastic.scale_time(mYst, timeScaling)

    # synthesize output sound
    y = stochastic.to_audio(ystocEnv, H, H * 2)

    # write output sound
    outputFile = 'output_sounds/' + strip_file(inputFile) + '_stochasticModelTransformation.wav'
    audio.write_wav(y, fs, outputFile)

    # create figure to plot
    plt.figure(figsize=(12, 9))

    # plot the input sound
    plt.subplot(4, 1, 1)
    plt.plot(np.arange(x.size) / float(fs), x)
    plt.axis([0, x.size / float(fs), min(x), max(x)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')

    # plot stochastic representation
    plt.subplot(4, 1, 2)
    numFrames = int(mYst.shape[0])
    frmTime = H * np.arange(numFrames) / float(fs)
    binFreq = np.arange(stocf * H) * float(fs) / (stocf * 2 * H)
    plt.pcolormesh(frmTime, binFreq, np.transpose(mYst))
    plt.autoscale(tight=True)
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.title('stochastic approximation')

    # plot modified stochastic representation
    plt.subplot(4, 1, 3)
    numFrames = int(ystocEnv.shape[0])
    frmTime = H * np.arange(numFrames) / float(fs)
    binFreq = np.arange(stocf * H) * float(fs) / (stocf * 2 * H)
    plt.pcolormesh(frmTime, binFreq, np.transpose(ystocEnv))
    plt.autoscale(tight=True)
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.title('modified stochastic approximation')

    # plot the output sound
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(y.size) / float(fs), y)
    plt.axis([0, y.size / float(fs), min(y), max(y)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')

    plt.tight_layout()

    if interactive:
        plt.show()
    if plotFile:
        plt.savefig('output_plots/%s_stochastic_transformation.png' % files.strip_file(inputFile))


if __name__ == '__main__':
    main()
