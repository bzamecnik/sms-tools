import math
import numpy as np
from scipy.signal import get_window

from smst.utils.math import rmse
from smst.utils import audio
from smst.models import spr

# TODO: the test needs fixing after the model is fixed

def test_reconstruct_sound():
    fs, x = audio.read_wav("../sounds/sax-phrase-short.wav")

    window_size, fft_size, hop_size = 2001, 2048, 128
    window = get_window('hamming', window_size)

    # fix the random seed for reproducibility
    np.random.seed(42)

    xtfreq, xtmag, xtphase, x_residual = spr.from_audio(
        x, fs, window, fft_size, hop_size,
        t=-80, maxnSines=100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01)
    x_reconstructed, x_sine = spr.to_audio(xtfreq, xtmag, xtphase, x_residual, 512, hop_size, fs)

    assert 138746 == len(x)

    assert len(x) == len(x_residual)

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

    assert np.allclose(2.1079553110776107e-17, rmse(x[:len(x_reconstructed)], x_reconstructed))
    assert np.allclose(0.0043912712540510645, rmse(x[:len(x_reconstructed)], x_sine))

    x_dir_reconstructed, x_dir_sine, x_dir_residual = spr.reconstruct(x, fs, window, fft_size, t=-80)

    assert len(x) == len(x_dir_reconstructed)
    assert len(x) == len(x_dir_sine)
    assert len(x) == len(x_dir_residual)

    # the reconstruction via from_audio()/to_audio() and reconstruct() should be equal
    assert np.allclose(0.014037759480944659, rmse(x, x_dir_reconstructed[:len(x)]))
    # should be 0.0 and have equal length
    assert np.allclose(0.014043933504228221, rmse(x_reconstructed[:138624], x_dir_reconstructed[:138624]))

    print(len(x_sine), len(x_dir_sine), len(x_dir_reconstructed))
    assert np.allclose(0.014064093953064703, rmse(x_sine[:138624], x_dir_sine[:138624]))
    assert np.allclose(0.0015165722766401456, rmse(x_residual[:138624], x_dir_residual[:138624]))
