from scipy.signal import butter, lfilter, lfilter_zi


def high_pass_filter(data, cutoff, fs, order=5):
    """
    Apply a high-pass filter to the data (batch mode).

    :param data: The input data to filter.
    :param cutoff: The cutoff frequency of the filter in Hz.
    :param fs: The sampling rate in Hz.
    :param order: The order of the filter (default: 5).
    :return: The filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)


def get_high_pass_filter_coeffs(cutoff, fs, order=5):
    """
    Get high-pass filter coefficients and initial state for real-time filtering.

    :param cutoff: The cutoff frequency of the filter in Hz.
    :param fs: The sampling rate in Hz.
    :param order: The order of the filter (default: 5).
    :return: b, a, zi (filter coefficients and initial state)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    zi = lfilter_zi(b, a)
    return b, a, zi


def high_pass_filter_sample(sample, b, a, zi):
    """
    Filter a single sample with stateful processing.

    :param sample: The new sample to filter.
    :param b: Filter numerator coefficients.
    :param a: Filter denominator coefficients.
    :param zi: Filter state.
    :return: filtered_sample, updated_zi
    """
    filtered, zi = lfilter(b, a, [sample], zi=zi)
    return filtered[0], zi


# ------------------- Band-pass filter functions -------------------
def band_pass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a band-pass filter to the data (batch mode).

    :param data: The input data to filter.
    :param lowcut: The lower cutoff frequency in Hz.
    :param highcut: The upper cutoff frequency in Hz.
    :param fs: The sampling rate in Hz.
    :param order: The order of the filter (default: 4).
    :return: The filtered data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


def get_band_pass_filter_coeffs(lowcut, highcut, fs, order=4):
    """
    Get band-pass filter coefficients and initial state for real-time filtering.

    :param lowcut: The lower cutoff frequency in Hz.
    :param highcut: The upper cutoff frequency in Hz.
    :param fs: The sampling rate in Hz.
    :param order: The order of the filter (default: 4).
    :return: b, a, zi (filter coefficients and initial state)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    zi = lfilter_zi(b, a)
    return b, a, zi


def band_pass_filter_sample(sample, b, a, zi):
    """
    Filter a single sample with stateful processing (band-pass).

    :param sample: The new sample to filter.
    :param b: Filter numerator coefficients.
    :param a: Filter denominator coefficients.
    :param zi: Filter state.
    :return: filtered_sample, updated_zi
    """
    filtered, zi = lfilter(b, a, [sample], zi=zi)
    return filtered[0], zi


# ------------------- Spike removal function -------------------
import numpy as np
from scipy.signal import medfilt

def remove_spikes(signal, kernel_size=5, threshold=3):
    """
    Remove spikes using a median filter and thresholding.
    :param signal: 1D numpy array of the signal
    :param kernel_size: Size of the median filter window (odd integer)
    :param threshold: Multiplier for standard deviation to detect spikes
    :return: Signal with spikes replaced by median-filtered values
    """
    filtered = medfilt(signal, kernel_size)
    diff = np.abs(signal - filtered)
    spikes = diff > threshold * np.std(signal)
    signal_out = np.copy(signal)
    signal_out[spikes] = filtered[spikes]
    return signal_out
