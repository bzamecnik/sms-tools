# matplotlib without any blocking GUI
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

M = 10
N = 100
hN = N / 2
hM = M / 2
k = np.arange(-hM, hM, M / float(N))
W = 20 * np.log10(abs(np.sin(np.pi * k) / np.sin(np.pi * k / M)))

plt.figure(1, figsize=(7.5, 3.5))
plt.plot(np.arange(-hN, hN), W - max(W), 'r', lw=1.5)
plt.axis([-hN, hN, -40, 0])

plt.tight_layout()
plt.savefig('rectangular-3.png')
