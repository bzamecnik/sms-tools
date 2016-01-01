"""
Functions that implement analysis and synthesis of sounds using the Discrete Fourier Transform.

For example usage check the `smst.ui.models.dftModel_function` module.
"""

import math

import numpy as np
from scipy.fftpack import fft, ifft

from ..utils.math import is_power_of_two


def from_audio(samples, window, fft_size):
    """
    Analyzes time-domain samples of a real signal using the
    Discrete Fourier Transform (DFT) into magnitude and phase spectrum
    of positive frequencies.

    :param samples: samples of the input signal
    :param window: samples of the analysis window
    :param fft_size: size of the spectrum (power of two)

    :returns:
        - magnitude_db_spectrum: magnitude spectrum (in decibels) of positive frequencies
        - phase_spectrum: unwrapped phase spectrum of positive frequencies
    """

    if not (is_power_of_two(fft_size)):
        raise ValueError("FFT size must be a power of 2")

    if window.size > fft_size:
        raise ValueError("Window size must not be greater than FFT size")

    fft_buffer = apply_zero_phase_window(samples, window, fft_size)

    spectrum = fft(fft_buffer)

    pos_spectrum = select_positive_spectrum(spectrum)
    magnitude_db_spectrum = select_magnitude_db_spectrum(pos_spectrum)
    phase_spectrum = select_phase_spectrum(pos_spectrum)

    return magnitude_db_spectrum, phase_spectrum


def to_audio(magnitude_db_spectrum, phase_spectrum, window_size):
    """
    Synthesizes samples of windowed time-domain signal from the
    positive magnitude and phase spectrum using the Inverse
    Discrete Fourier Transform (IDFT).

    :param magnitude_db_spectrum: positive magnitude spectrum in decibels
    :param phase_spectrum: positive phase spectrum
    :param window_size: window size (also size of the output signal)

    :returns: samples: reconstructed samples of the windowed signal
    """

    fft_size = (magnitude_db_spectrum.size - 1) * 2  # FFT size
    if not is_power_of_two(fft_size):
        raise ValueError("Full spectrum size must be power of two")

    spectrum = spectrum_from_phase_and_magnitude(magnitude_db_spectrum, phase_spectrum, fft_size)
    fft_buffer = np.real(ifft(spectrum))  # compute inverse FFT
    samples = unapply_zero_phase_window(fft_buffer, window_size)
    return samples

# -- support functions --


def select_positive_spectrum(spectrum):
    """Selects positive frequencies from a full spectrum."""
    fft_size = len(spectrum)
    # size of positive spectrum, it includes sample 0
    size = (fft_size / 2) + 1
    return spectrum[:size]


def select_phase_spectrum(spectrum, phase_eps=1e-14):
    """
    Computes unwrapped phase spectrum out of complex spectrum.

    :param spectrum: complex-valued spectrum
    :param tol: threshold used to compute phase
    """
    # for phase calculation set to 0 the small values
    spectrum.real[np.abs(spectrum.real) < phase_eps] = 0.0
    spectrum.imag[np.abs(spectrum.imag) < phase_eps] = 0.0
    # unwrapped phase spectrum of
    return np.unwrap(np.angle(spectrum))


def select_magnitude_db_spectrum(spectrum):
    """Computes magnitude spectrum in decibels from complex-valued spectrum."""
    abs_spectrum = abs(spectrum)
    # if zeros add epsilon to handle log
    abs_spectrum[abs_spectrum < np.finfo(float).eps] = np.finfo(float).eps
    # magnitude spectrum of positive frequencies in dB
    magnitude_db_spectrum = 20 * np.log10(abs_spectrum)
    return magnitude_db_spectrum


def spectrum_from_phase_and_magnitude(pos_magnitude_db_spectrum, pos_phase_spectrum, fft_size):
    # size of positive spectrum, it includes sample 0
    half_fft_size = pos_magnitude_db_spectrum.size
    spectrum = np.zeros(fft_size, dtype=complex)
    pos_magnitude_spectrum = 10 ** (pos_magnitude_db_spectrum / 20)
    # generate positive frequencies
    spectrum[:half_fft_size] = pos_magnitude_spectrum * np.exp(1j * pos_phase_spectrum)
    # generate negative frequencies
    spectrum[half_fft_size:] = pos_magnitude_spectrum[-2:0:-1] * np.exp(-1j * pos_phase_spectrum[-2:0:-1])
    return spectrum


def apply_zero_phase_window(samples, window, fft_size):
    windowed_samples = apply_normalized_window(samples, window)
    half_win_round, half_win_floor = half_window_sizes(window.size)
    fft_buffer = np.zeros(fft_size)  # initialize buffer for FFT
    # zero-phase window in fftbuffer
    fft_buffer[:half_win_round] = windowed_samples[half_win_floor:]
    fft_buffer[-half_win_floor:] = windowed_samples[:half_win_floor]
    return fft_buffer


def apply_normalized_window(samples, window):
    normalized_window = window / sum(window)
    windowed_samples = samples * normalized_window
    return windowed_samples


def unapply_zero_phase_window(fft_buffer, window_size):
    samples = np.zeros(window_size)
    half_win_round, half_win_floor = half_window_sizes(window_size)
    samples[:half_win_floor] = fft_buffer[-half_win_floor:]
    samples[half_win_floor:] = fft_buffer[:half_win_round]
    return samples


def half_window_sizes(window_size):
    half_win_round = int(math.floor((window_size + 1) / 2))  # half analysis window size by rounding
    half_win_floor = int(math.floor(window_size / 2))  # half analysis window size by floor
    return half_win_round, half_win_floor
