import numpy as np
# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from smst.utils import audio
from smst.models import stft

(fs, x) = audio.read_wav('../../../sounds/piano.wav')
w = np.hamming(1001)
N = 1024
H = 256
mX, pX = stft.from_audio(x, w, N, H)

plt.figure(1, figsize=(9.5, 6))

plt.subplot(211)
numFrames = int(mX.shape[0])
frmTime = H * np.arange(numFrames) / float(fs)
binFreq = np.arange(N / 2 + 1) * float(fs) / N
plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
plt.title('mX (piano.wav), M=1001, N=1024, H=256')
plt.autoscale(tight=True)

plt.subplot(212)
numFrames = int(pX.shape[0])
frmTime = H * np.arange(numFrames) / float(fs)
binFreq = np.arange(N / 2 + 1) * float(fs) / N
plt.pcolormesh(frmTime, binFreq, np.diff(np.transpose(pX), axis=0))
plt.title('pX derivative (piano.wav), M=1001, N=1024, H=256')
plt.autoscale(tight=True)

plt.tight_layout()
plt.savefig('spectrogram.png')
