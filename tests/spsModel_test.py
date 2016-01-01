import math
import numpy as np
from scipy.signal import get_window

from smst.utils.math import rmse
from smst.utils import audio
from smst.models import sps


# TODO: the test needs fixing after the model is fixed

def test_reconstruct_sound():
    fs, x = audio.read_wav("../sounds/sax-phrase-short.wav")

    window_size, fft_size, hop_size = 2001, 2048, 128
    window = get_window('hamming', window_size)

    # fix the random seed for reproducibility
    np.random.seed(42)

    xtfreq, xtmag, xtphase, stocEnv = sps.from_audio(
        x, fs, window, fft_size, hop_size,
        t=-80, maxnSines=100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01, stocf=0.5)
    x_reconstructed, x_sine, x_stochastic = sps.to_audio(xtfreq, xtmag, xtphase, stocEnv, 512, hop_size, fs)

    assert 138746 == len(x)

    expected_frame_count = int(math.ceil(float(len(x)) / hop_size))
    assert expected_frame_count == len(xtfreq)
    assert expected_frame_count == len(xtmag)
    assert expected_frame_count == len(xtphase)

    assert xtfreq.shape[1] <= 100

    # statistics of the model for regression testing without explicitly storing the whole data
    assert np.allclose(799.3384358567838, xtfreq.mean())
    assert np.allclose(-24.080251067421795, xtmag.mean())
    assert np.allclose(1.0900513921895467, xtphase.mean())

    # TODO: this is completely off, it should be equal to len(x)!
    assert 1083 * 128 == len(x_reconstructed)
    assert 1083 * 128 == len(x_sine)
    # TODO: this is insane
    assert 1085 * 128 == len(x_stochastic)

    assert np.allclose(0.0061891379818097133, rmse(x[:len(x_reconstructed)], x_reconstructed))
    assert np.allclose(0.0043912712540510645, rmse(x[:len(x_reconstructed)], x_sine))
    assert np.allclose(0.093780097561056638, rmse(x[:len(x_reconstructed)], x_stochastic[:len(x_reconstructed)]))
    assert np.allclose(0.0, rmse(x_sine + x_stochastic[:len(x_reconstructed)], x_reconstructed))
