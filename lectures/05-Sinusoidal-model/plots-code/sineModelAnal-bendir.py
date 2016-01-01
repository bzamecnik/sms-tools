# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from smst.utils import audio
from smst.models import sine, stft

(fs, x) = audio.read_wav('../../../sounds/bendir.wav')
w = np.hamming(2001)
N = 2048
H = 200
t = -80
minSineDur = .02
maxnSines = 150
freqDevOffset = 10
freqDevSlope = 0.001
mX, pX = stft.from_audio(x, w, N, H)
tfreq, tmag, tphase = sine.from_audio(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

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
plt.plot(frmTime, tracks, color='k', lw=1.5)
plt.autoscale(tight=True)
plt.title('mX + sinusoidal tracks (bendir.wav)')

plt.tight_layout()
plt.savefig('sineModelAnal-bendir.png')
