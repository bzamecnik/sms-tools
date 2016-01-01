import math

import numpy as np
from scipy.signal import get_window

from smst.utils.math import rmse
from smst.utils import audio
from smst.models import hpr
from .common import sound_path

# TODO: the test needs fixing after the model is fixed

def test_reconstruct_sound():
    fs, x = audio.read_wav(sound_path("sax-phrase-short.wav"))

    window_size, fft_size, hop_size = 2001, 2048, 128
    window = get_window('hamming', window_size)

    # fix the random seed for reproducibility
    np.random.seed(42)

    xtfreq, xtmag, xtphase, x_residual = hpr.from_audio(
        x, fs, window, fft_size, hop_size,
        t=-80, minSineDur=.02, nH=20, minf0=100, maxf0=2000, f0et=5, harmDevSlope=0.01)
    x_reconstructed, x_sine = hpr.to_audio(xtfreq, xtmag, xtphase, x_residual, 512, hop_size, fs)

    assert 138746 == len(x)

    assert len(x) == len(x_residual)

    expected_frame_count = int(math.ceil(float(len(x)) / hop_size))
    assert expected_frame_count == len(xtfreq)
    assert expected_frame_count == len(xtmag)
    assert expected_frame_count == len(xtphase)

    assert xtfreq.shape[1] <= 100

    # statistics of the model for regression testing without explicitly storing the whole data
    assert np.allclose(1731.8324721982437, xtfreq.mean())
    assert np.allclose(-69.877742948220671, xtmag.mean())
    assert np.allclose(1.8019294703328628, xtphase.mean())

    # TODO: this is completely off, it should be equal to len(x)!
    assert 1083 * 128 == len(x_reconstructed)
    assert 1083 * 128 == len(x_sine)

    assert np.allclose(2.1079553110776107e-17, rmse(x[:len(x_reconstructed)], x_reconstructed))
    assert np.allclose(0.025543282494159769, rmse(x[:len(x_reconstructed)], x_sine))
