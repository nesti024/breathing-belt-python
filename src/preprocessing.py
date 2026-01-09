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
