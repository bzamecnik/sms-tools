import essentia.standard as ess
# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming

from smst.utils import audio, synth
from smst.models import stft


def f0Yin(x, N, H, minf0, maxf0):
    # fundamental frequency detection using the Yin algorithm
    # x: input sound, N: window size,
    # minf0: minimum f0 frequency in Hz, maxf0: maximum f0 frequency in Hz,
    # returns f0

    spectrum = ess.Spectrum(size=N)
    window = ess.Windowing(size=N, type='hann')
    pitchYin = ess.PitchYinFFT(minFrequency=minf0, maxFrequency=maxf0)
    pin = 0
    pend = x.size - N
    f0 = []

    while pin < pend:
        mX = spectrum(window(x[pin:pin + N]))
        f0t = pitchYin(mX)
        f0 = np.append(f0, f0t[0])
        pin += H
    return f0


if __name__ == '__main__':
    (fs, x) = audio.read_wav('../../../sounds/bendir.wav')

    plt.figure(1, figsize=(9, 7))
    N = 2048
    H = 256
    w = hamming(2048)
    mX, pX = stft.from_audio(x, w, N, H)
    maxplotfreq = 2000.0
    frmTime = H * np.arange(mX.shape[0]) / float(fs)
    binFreq = fs * np.arange(N * maxplotfreq / fs) / N
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:, :N * maxplotfreq / fs + 1]))

    N = 2048
    minf0 = 130
    maxf0 = 300
    H = 256
    f0 = f0Yin(x, N, H, minf0, maxf0)
    yf0 = synth.synthesize_sinusoid(f0, .8, H, fs)
    frmTime = H * np.arange(f0.size) / float(fs)
    plt.plot(frmTime, f0, linewidth=2, color='k')
    plt.autoscale(tight=True)
    plt.title('mX + f0 (vignesh.wav), YIN: N=2048, H = 256 ')

    plt.tight_layout()
    plt.savefig('f0Yin.png')
    audio.write_wav(yf0, fs, 'f0Yin.wav')
