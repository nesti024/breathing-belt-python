import time
from scipy.signal import butter, lfilter, lfilter_zi, sosfilt, sosfilt_zi
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
        self.last_value = None
        self.activity_threshold = 0.01  # Minimum change to consider as activity

    def update(self, value: float) -> None:
        now = time.time()
        # Initialize on first call with a small range to avoid division by zero
        if self.max_val is None:
            self.max_val = value + 0.1  # Add small offset to create initial range
            self.min_val = value - 0.1
            self.last_value = value
            return
        
        # Check if there's significant signal change (breathing activity)
        signal_change = abs(value - self.last_value) if self.last_value is not None else 0
        is_active = signal_change > self.activity_threshold
        
        if (now - self.last_reset) > self.reset_interval and is_active:
            # Soft reset: only decay when there's active breathing
            # Move max down toward the center by 10%
            self.max_val = max(value, self.max_val - abs(self.max_val) * 0.1)
            # Move min up toward the center by 10%
            self.min_val = min(value, self.min_val + abs(self.min_val) * 0.1)
            self.last_reset = now
        else:
            # Always update max/min if new extremes are reached
            if value > self.max_val:
                self.max_val = value
            if value < self.min_val:
                self.min_val = value
        
        self.last_value = value
        
        # Safety: ensure min < max with at least a small margin
        if self.max_val - self.min_val < 0.01:
            center = (self.max_val + self.min_val) / 2
            self.max_val = center + 0.01
            self.min_val = center - 0.01

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
    signal = np.asarray(signal).copy()  # Make a copy to avoid modifying input
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
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.zeros(0, dtype=bool)

    diff = np.abs(np.diff(signal, prepend=signal[0]))
    # Efficient rolling std using pandas
    try:
        import pandas as pd
        local_std = pd.Series(signal).rolling(window, min_periods=2).std(ddof=0).to_numpy()
    except ImportError:
        # Fallback to slower method if pandas not available
        if len(signal) < window:
            # Return array filled with global std for consistency
            local_std = np.full(len(signal), np.std(signal) if len(signal) > 1 else 1.0)
        else:
            local_std = np.array([
                np.std(signal[max(0, i-window):i+1]) if i > 0 else np.std(signal[:1])
                for i in range(len(signal))
            ])

    # Prevent very small local std at plateaus from flagging normal inhale/exhale starts.
    global_std = np.std(signal) if len(signal) > 1 else 0.0
    std_floor = max(global_std * 0.25, 1e-4)
    local_std = np.nan_to_num(local_std, nan=global_std)
    local_std = np.maximum(local_std, std_floor)

    artifact_mask = diff > threshold * local_std
    return artifact_mask


def smooth_signal(signal: np.ndarray, window: int = 31) -> np.ndarray:
    """
    Apply light moving-average smoothing with edge padding.
    Intended for preprocessing before artifact or peak decisions.

    :param signal: 1D numpy array of the signal
    :param window: Smoothing window in samples (odd integer preferred)
    :return: Smoothed signal with same length as input
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size < 3:
        return signal.copy()

    win = int(window)
    if win <= 1:
        return signal.copy()
    if win > signal.size:
        win = signal.size
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return signal.copy()

    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    padded = np.pad(signal, (pad, pad), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def high_pass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Apply a high-pass filter to the data (batch mode).
    Uses second-order sections (SOS) for improved numerical stability.

    :param data: The input data to filter.
    :param cutoff: The cutoff frequency of the filter in Hz.
    :param fs: The sampling rate in Hz.
    :param order: The order of the filter (default: 5).
    :return: The filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0 or normal_cutoff <= 0:
        raise ValueError(f"Cutoff {cutoff} Hz invalid for sampling rate {fs} Hz")
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sosfilt(sos, data)


def get_high_pass_filter_coeffs(cutoff: float, fs: float, order: int = 5, initial_value: float = None):
    """
    Get high-pass filter coefficients and initial state for real-time filtering.
    Uses second-order sections (SOS) for improved numerical stability with low cutoff frequencies.

    :param cutoff: The cutoff frequency of the filter in Hz.
    :param fs: The sampling rate in Hz.
    :param order: The order of the filter (default: 5).
    :param initial_value: Optional first sample value to scale zi and avoid transient artifacts.
    :return: sos, zi (second-order sections and initial state)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0 or normal_cutoff <= 0:
        raise ValueError(f"Cutoff {cutoff} Hz invalid for sampling rate {fs} Hz")
    sos = butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    zi = sosfilt_zi(sos)
    if initial_value is not None:
        zi = zi * initial_value
    return sos, zi


def high_pass_filter_sample(sample: float, sos, zi):
    """
    Filter a single sample with stateful processing.
    Uses second-order sections (SOS) for improved numerical stability.

    :param sample: The new sample to filter.
    :param sos: Second-order sections filter representation.
    :param zi: Filter state.
    :return: filtered_sample, updated_zi
    """
    filtered, zi = sosfilt(sos, [sample], zi=zi)
    return filtered[0], zi


# ------------------- Band-pass filter functions -------------------
def band_pass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Apply a band-pass filter to the data (batch mode).
    Uses second-order sections (SOS) for improved numerical stability.

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
    if low <= 0 or low >= 1.0 or high <= 0 or high >= 1.0 or low >= high:
        raise ValueError(f"Invalid band-pass cutoffs: {lowcut}-{highcut} Hz for sampling rate {fs} Hz")
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, data)


def get_band_pass_filter_coeffs(lowcut: float, highcut: float, fs: float, order: int = 4, initial_value: float = None):
    """
    Get band-pass filter coefficients and initial state for real-time filtering.
    Uses second-order sections (SOS) for improved numerical stability.

    :param lowcut: The lower cutoff frequency in Hz.
    :param highcut: The upper cutoff frequency in Hz.
    :param fs: The sampling rate in Hz.
    :param order: The order of the filter (default: 4).
    :param initial_value: Optional first sample value to scale zi and avoid transient artifacts.
    :return: sos, zi (second-order sections and initial state)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or low >= 1.0 or high <= 0 or high >= 1.0 or low >= high:
        raise ValueError(f"Invalid band-pass cutoffs: {lowcut}-{highcut} Hz for sampling rate {fs} Hz")
    sos = butter(order, [low, high], btype='band', output='sos')
    zi = sosfilt_zi(sos)
    if initial_value is not None:
        zi = zi * initial_value
    return sos, zi


def band_pass_filter_sample(sample: float, sos, zi):
    """
    Filter a single sample with stateful processing (band-pass).
    Uses second-order sections (SOS) for improved numerical stability.

    :param sample: The new sample to filter.
    :param sos: Second-order sections filter representation.
    :param zi: Filter state.
    :return: filtered_sample, updated_zi
    """
    filtered, zi = sosfilt(sos, [sample], zi=zi)
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
