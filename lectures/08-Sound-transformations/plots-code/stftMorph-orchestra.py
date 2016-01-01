# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from smst.utils import audio
from smst.models import stft, stochastic

(fs, x1) = audio.read_wav('../../../sounds/orchestra.wav')
(fs, x2) = audio.read_wav('../../../sounds/speech-male.wav')
w1 = np.hamming(1024)
N1 = 1024
H1 = 256
w2 = np.hamming(1024)
N2 = 1024
smoothf = .2
balancef = 0.5
y = stft.morph(x1, x2, fs, w1, N1, w2, N2, H1, smoothf, balancef)
mX2 = stochastic.from_audio(x2, H1, H1 * 2, smoothf)
mX, pX = stft.from_audio(x1, w1, N1, H1)
mY, pY = stft.from_audio(y, w1, N1, H1)
maxplotfreq = 10000.0

plt.figure(1, figsize=(12, 9))
plt.subplot(311)
numFrames = int(mX.shape[0])
frmTime = H1 * np.arange(numFrames) / float(fs)
binFreq = fs * np.arange(N1 * maxplotfreq / fs) / N1
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:, :N1 * maxplotfreq / fs + 1]))
plt.title('mX (orchestra.wav)')
plt.autoscale(tight=True)

plt.subplot(312)
numFrames = int(mX2.shape[0])
frmTime = H1 * np.arange(numFrames) / float(fs)

N = 2 * mX2.shape[1]
binFreq = fs * np.arange(N * maxplotfreq / fs) / N
plt.pcolormesh(frmTime, binFreq, np.transpose(mX2[:, :N * maxplotfreq / fs + 1]))
plt.title('mX2 (speech-male.wav)')
plt.autoscale(tight=True)

plt.subplot(313)
numFrames = int(mY.shape[0])
frmTime = H1 * np.arange(numFrames) / float(fs)
binFreq = fs * np.arange(N1 * maxplotfreq / fs) / N1
plt.pcolormesh(frmTime, binFreq, np.transpose(mY[:, :N1 * maxplotfreq / fs + 1]))
plt.title('mY')
plt.autoscale(tight=True)

plt.tight_layout()
audio.write_wav(y, fs, 'orchestra-speech-stftMorph.wav')
plt.savefig('stftMorph-orchestra.png')
