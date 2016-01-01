# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from smst.utils import audio
from smst.models import stft

(fs, x) = audio.read_wav('../../../sounds/orchestra.wav')
w = np.hamming(2048)
N = 2048
H = 512
# design a band stop filter using a hanning window
startBin = int(N * 500.0 / fs)
nBins = int(N * 2000.0 / fs)
bandpass = (np.hanning(nBins) * 65.0) - 60
filt = np.zeros(N / 2 + 1) - 60
filt[startBin:startBin + nBins] = bandpass
y = stft.filter(x, fs, w, N, H, filt)
mX, pX = stft.from_audio(x, w, N, H)
mY, pY = stft.from_audio(y, w, N, H)

plt.figure(1, figsize=(12, 9))
plt.subplot(311)
numFrames = int(mX.shape[0])
frmTime = H * np.arange(numFrames) / float(fs)
binFreq = np.arange(mX.shape[1]) * float(fs) / N
plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
plt.title('mX (orchestra.wav)')
plt.autoscale(tight=True)

plt.subplot(312)
plt.plot(fs * np.arange(mX.shape[1]) / float(N), filt, 'k', lw=1.3)
plt.axis([0, fs / 2, -60, 7])
plt.title('filter shape')

plt.subplot(313)
numFrames = int(mY.shape[0])
frmTime = H * np.arange(numFrames) / float(fs)
binFreq = np.arange(mY.shape[1]) * float(fs) / N
plt.pcolormesh(frmTime, binFreq, np.transpose(mY))
plt.title('mY')
plt.autoscale(tight=True)

plt.tight_layout()
audio.write_wav(y, fs, 'orchestra-stft-filtering.wav')
plt.savefig('stftFiltering-orchestra.png')
