# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from smst.utils import audio
from smst.models import stft

(fs, x) = audio.read_wav('../../../sounds/piano.wav')

plt.figure(1, figsize=(9.5, 6))

w = np.hamming(256)
N = 256
H = 128
mX1, pX1 = stft.from_audio(x, w, N, H)
plt.subplot(211)
numFrames = int(mX1.shape[0])
frmTime = H * np.arange(numFrames) / float(fs)
binFreq = np.arange(mX1.shape[1]) * float(fs) / N
plt.pcolormesh(frmTime, binFreq, np.transpose(mX1))
plt.title('mX (piano.wav), M=256, N=256, H=128')
plt.autoscale(tight=True)

w = np.hamming(1024)
N = 1024
H = 128
mX2, pX2 = stft.from_audio(x, w, N, H)

plt.subplot(212)
numFrames = int(mX2.shape[0])
frmTime = H * np.arange(numFrames) / float(fs)
binFreq = np.arange(mX2.shape[1]) * float(fs) / N
plt.pcolormesh(frmTime, binFreq, np.transpose(mX2))
plt.title('mX (piano.wav), M=1024, N=1024, H=128')
plt.autoscale(tight=True)

plt.tight_layout()
plt.savefig('time-freq-compromise.png')
