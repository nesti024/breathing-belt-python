import time
from scipy.signal import butter, lfilter, lfilter_zi
import numpy as np
from scipy.signal import medfilt


# ------------------- MaxMinTracker with 1-minute reset -------------------
class MaxMinTracker:
    """
    Tracks max and min values, resetting every one minute.
    Call update(value) for each new value. Use get_max_min() to get current max/min.
    """
    def __init__(self, reset_interval_sec=60):
        self.reset_interval = reset_interval_sec
        self.last_reset = time.time()
        self.max_val = None
        self.min_val = None

    def update(self, value: float) -> None:
        now = time.time()
        if (now - self.last_reset) > self.reset_interval:
            self.max_val = value
            self.min_val = value
            self.last_reset = now
        else:
            if self.max_val is None or value > self.max_val:
                self.max_val = value
            if self.min_val is None or value < self.min_val:
                self.min_val = value

    def get_max_min(self) -> tuple:
        return self.max_val, self.min_val
    

# ------------------- Normalization function -------------------
def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to the range [0, 1] given min and max.
    If min_val == max_val, returns 0.5 (centered).
    If min_val > max_val, returns np.nan.
    """
    if min_val == max_val:
        return 0.5
    if min_val > max_val:
        return np.nan
    return (value - min_val) / (max_val - min_val)


def interpolate_artifacts(signal: np.ndarray) -> np.ndarray:
    """
    Interpolate over NaN values (artifacts) in the signal using linear interpolation.
    Handles case where all values are NaN by returning zeros.
    :param signal: 1D numpy array with NaNs marking artifacts
    :return: Signal with NaNs replaced by interpolated values
    """
    signal = np.asarray(signal)
    nans = np.isnan(signal)
    if np.all(nans):
        return np.zeros_like(signal)
    if np.any(nans):
        not_nans = ~nans
        signal[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), signal[not_nans])
    return signal



def detect_motion_artifacts(signal: np.ndarray, window: int = 10, threshold: float = 5) -> np.ndarray:
    """
    Detect motion artifacts by looking for large changes in the signal.
    Returns a boolean mask where True indicates a motion artifact.
    Uses a rolling standard deviation for efficiency.
    :param signal: 1D numpy array of the signal
    :param window: Number of samples for local std calculation
    :param threshold: Multiplier for std to detect artifact
    :return: Boolean numpy array (True = artifact)
    """
    signal = np.asarray(signal)
    diff = np.abs(np.diff(signal, prepend=signal[0]))
    # Efficient rolling std using pandas
    try:
        import pandas as pd
        local_std = pd.Series(signal).rolling(window, min_periods=1).std().to_numpy()
    except ImportError:
        # Fallback to slower method if pandas not available
        if len(signal) < window:
            local_std = np.std(signal)
        else:
            local_std = np.array([
                np.std(signal[max(0, i-window):i+1]) if i > 0 else np.std(signal[:1])
                for i in range(len(signal))
            ])
    artifact_mask = diff > threshold * local_std
    return artifact_mask

def high_pass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
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


def get_high_pass_filter_coeffs(cutoff: float, fs: float, order: int = 5, initial_value: float = None):
    """
    Get high-pass filter coefficients and initial state for real-time filtering.

    :param cutoff: The cutoff frequency of the filter in Hz.
    :param fs: The sampling rate in Hz.
    :param order: The order of the filter (default: 5).
    :param initial_value: Optional first sample value to scale zi and avoid transient artifacts.
    :return: b, a, zi (filter coefficients and initial state)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    zi = lfilter_zi(b, a)
    if initial_value is not None:
        zi = zi * initial_value
    return b, a, zi


def high_pass_filter_sample(sample: float, b, a, zi):
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
def band_pass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
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


def get_band_pass_filter_coeffs(lowcut: float, highcut: float, fs: float, order: int = 4, initial_value: float = None):
    """
    Get band-pass filter coefficients and initial state for real-time filtering.

    :param lowcut: The lower cutoff frequency in Hz.
    :param highcut: The upper cutoff frequency in Hz.
    :param fs: The sampling rate in Hz.
    :param order: The order of the filter (default: 4).
    :param initial_value: Optional first sample value to scale zi and avoid transient artifacts.
    :return: b, a, zi (filter coefficients and initial state)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    zi = lfilter_zi(b, a)
    if initial_value is not None:
        zi = zi * initial_value
    return b, a, zi


def band_pass_filter_sample(sample: float, b, a, zi):
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


def remove_spikes(signal: np.ndarray, kernel_size: int = 5, threshold: float = 3) -> np.ndarray:
    """
    Remove spikes using a median filter and thresholding.
    Checks signal length to avoid unwanted padding.
    :param signal: 1D numpy array of the signal
    :param kernel_size: Size of the median filter window (odd integer)
    :param threshold: Multiplier for standard deviation to detect spikes
    :return: Signal with spikes replaced by median-filtered values
    """
    if len(signal) < kernel_size:
        # Return original signal if too short for median filter
        return signal
    filtered = medfilt(signal, kernel_size)
    diff = np.abs(signal - filtered)
    spikes = diff > threshold * np.std(signal)
    signal_out = np.copy(signal)
    signal_out[spikes] = filtered[spikes]
    return signal_out
