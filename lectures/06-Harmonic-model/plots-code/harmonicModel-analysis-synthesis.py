# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from smst.utils import audio
from smst.models import harmonic, sine

(fs, x) = audio.read_wav('../../../sounds/vignesh.wav')
w = np.blackman(1201)
N = 2048
t = -90
nH = 100
minf0 = 130
maxf0 = 300
f0et = 7
Ns = 512
H = Ns / 4
minSineDur = .1
harmDevSlope = 0.01
hfreq, hmag, hphase = harmonic.from_audio(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
y = sine.to_audio(hfreq, hmag, hphase, Ns, H, fs)

numFrames = int(hfreq.shape[0])
frmTime = H * np.arange(numFrames) / float(fs)

plt.figure(1, figsize=(9, 7))

plt.subplot(3, 1, 1)
plt.plot(np.arange(x.size) / float(fs), x, 'b')
plt.axis([0, x.size / float(fs), min(x), max(x)])
plt.title('x (vignesh.wav)')

plt.subplot(3, 1, 2)
yhfreq = hfreq
yhfreq[hfreq == 0] = np.nan
plt.plot(frmTime, hfreq, lw=1.2)
plt.axis([0, y.size / float(fs), 0, 8000])
plt.title('f_h, harmonic frequencies')

plt.subplot(3, 1, 3)
plt.plot(np.arange(y.size) / float(fs), y, 'b')
plt.axis([0, y.size / float(fs), min(y), max(y)])
plt.title('yh')

plt.tight_layout()
audio.write_wav(y, fs, 'vignesh-harmonic-synthesis.wav')
plt.savefig('harmonicModel-analysis-synthesis.png')
