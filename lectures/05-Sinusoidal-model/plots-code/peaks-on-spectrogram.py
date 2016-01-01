# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from smst.utils import audio
from smst.models import sine, stft

(fs, x) = audio.read_wav('../../../sounds/speech-male.wav')
start = 1.25
end = 1.79
x1 = x[start * fs:end * fs]
w = np.hamming(801)
N = 2048
H = 200
t = -70
minSineDur = 0
maxnSines = 150
freqDevOffset = 10
freqDevSlope = 0.001
mX, pX = stft.from_audio(x1, w, N, H)
tfreq, tmag, tphase = sine.from_audio(x1, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

plt.figure(1, figsize=(9.5, 7))
maxplotfreq = 800.0
maxplotbin = int(N * maxplotfreq / fs)
numFrames = int(mX.shape[0])
frmTime = H * np.arange(numFrames) / float(fs)
binFreq = np.arange(maxplotbin + 1) * float(fs) / N
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:, :maxplotbin + 1]))
plt.autoscale(tight=True)

tracks = tfreq * np.less(tfreq, maxplotfreq)
tracks[tracks <= 0] = np.nan
plt.plot(frmTime, tracks, 'x', color='k', markeredgewidth=1.5)
plt.autoscale(tight=True)
plt.title('mX + spectral peaks (speech-male.wav)')

plt.tight_layout()
plt.savefig('peaks-on-spectrogram.png')
