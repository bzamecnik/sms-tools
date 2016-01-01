import math
import numpy as np
from scipy.signal import get_window

from smst.utils.math import rmse
from smst.utils import audio
from smst.models import sine, harmonic

# TODO: the test needs fixing after the model is fixed

def test_reconstruct_sound():
    fs, x = audio.read_wav("../sounds/sax-phrase-short.wav")

    window_size, fft_size, hop_size = 4001, 4096, 2048
    window = get_window('hamming', window_size)

    xtfreq, xtmag, xtphase = harmonic.from_audio(
        x, fs, window, fft_size, hop_size,
        t=-80, nH=20, minf0=100, maxf0=2000, f0et=5, harmDevSlope=0.01, minSineDur=.02)
    x_reconstructed = sine.to_audio(xtfreq, xtmag, xtphase, fft_size, hop_size, fs)

    assert 138746 == len(x)

    expected_frame_count = int(math.ceil(float(len(x)) / hop_size))
    assert expected_frame_count == len(xtfreq)
    assert expected_frame_count == len(xtmag)
    assert expected_frame_count == len(xtphase)

    assert xtfreq.shape[1] <= 100

    # statistics of the model for regression testing without explicitly storing the whole data
    assert np.allclose(1738.618043903208, xtfreq.mean())
    assert np.allclose(-64.939768348945279, xtmag.mean())
    assert np.allclose(1.6687005886001871, xtphase.mean())

    # TODO: this is completely off, it should be equal to len(x)!
    assert 69 * 2048 == len(x_reconstructed)

    assert np.allclose(0.036941947007791701, rmse(x, x_reconstructed[:len(x)]))
