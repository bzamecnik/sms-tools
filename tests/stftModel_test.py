import math

import numpy as np
from scipy.signal import get_window

from smst.utils.math import rmse
from smst.utils import audio
from smst.models import stft
from .common import sound_path

# TODO: the test needs fixing after the model is fixed

def test_reconstruct_sound():
    fs, x = audio.read_wav(sound_path("sax-phrase-short.wav"))

    window_size, fft_size, hop_size = 4001, 4096, 2048
    window = get_window('hamming', window_size)

    mag_spectrogram, phase_spectrogram = stft.from_audio(x, window, fft_size, hop_size)
    x_reconstructed = stft.to_audio(mag_spectrogram, phase_spectrogram, window_size, hop_size)

    assert 138746 == len(x)

    expected_frame_count = int(math.ceil(float(len(x)) / hop_size))
    assert expected_frame_count == len(mag_spectrogram)
    assert expected_frame_count == len(phase_spectrogram)

    # statistics of the spectrogram for regression testing without explicitly storing the whole data
    assert np.allclose(-102.86187076588583, np.mean(mag_spectrogram))
    assert np.allclose(11.368333745102881, np.mean(phase_spectrogram))

    # TODO: should be the same as len(x)
    assert expected_frame_count * hop_size == len(x_reconstructed)

    assert np.allclose(0.0014030089623073237, rmse(x, x_reconstructed[:len(x)]))
