import essentia.standard as ess
# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from smst.utils import audio

(fs, x) = audio.read_wav('../../../sounds/oboe-A4.wav')

M = 500
start = .8 * fs
xp = x[start:start + M] / float(max(x[start:start + M]))
r = ess.AutoCorrelation(normalization='standard')(xp)
r = r / max(r)
peaks = ess.PeakDetection(threshold=.2, interpolate=False, minPosition=.01)(r)

plt.figure(1, figsize=(9, 7))
plt.subplot(211)
plt.plot(np.arange(M) / float(fs), xp, lw=1.5)
plt.axis([0, (M - 1) / float(fs), min(xp), max(xp)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.title('x (oboe-A4.wav)')

plt.subplot(212)
plt.plot(np.arange(M) / float(fs), r, 'r', lw=1.5)
plt.plot(peaks[0] * (M - 1) / float(fs), peaks[1], 'x', color='k', markeredgewidth=1.5)
plt.axis([0, (M - 1) / float(fs), min(r), max(r)])
plt.title('autocorrelation function + peaks')
plt.xlabel('lag time (sec)')
plt.ylabel('correlation')

plt.tight_layout()
plt.savefig('oboe-autocorrelation.png')
