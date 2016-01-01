import math
import numpy as np
from scipy.signal import get_window
from smst.ui import demo_sound_path

from smst.utils.math import rmse
from smst.utils import audio
from smst.models import hps


# TODO: the test needs fixing after the model is fixed

def test_reconstruct_sound():
    fs, x = audio.read_wav(demo_sound_path("sax-phrase-short.wav"))

    window_size, fft_size, hop_size = 2001, 2048, 128
    window = get_window('hamming', window_size)

    # fix the random seed for reproducibility
    np.random.seed(42)

    xtfreq, xtmag, xtphase, stocEnv = hps.from_audio(
        x, fs, window, fft_size, hop_size,
        t=-80, minSineDur=.02, nH=20, minf0=100, maxf0=2000, f0et=5, harmDevSlope=0.01, Ns=512, stocf=0.5)
    x_reconstructed, x_sine, x_stochastic = hps.to_audio(xtfreq, xtmag, xtphase, stocEnv, 512, hop_size, fs)

    assert 138746 == len(x)

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
    # TODO: this is insane
    assert 1085 * 128 == len(x_stochastic)

    assert np.allclose(0.038065851967889502, rmse(x[:len(x_reconstructed)], x_reconstructed))
    assert np.allclose(0.025543282494159769, rmse(x[:len(x_reconstructed)], x_sine))
    assert np.allclose(0.097999320671614418, rmse(x[:len(x_reconstructed)], x_stochastic[:len(x_reconstructed)]))
    assert np.allclose(0.0, rmse(x_sine + x_stochastic[:len(x_reconstructed)], x_reconstructed))
