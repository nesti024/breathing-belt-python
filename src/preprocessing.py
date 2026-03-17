"""Signal-processing helpers for respiration-belt data.

The utilities in this module support several stages of the breathing-belt
pipeline:

- slow extrema tracking for legacy normalization workflows,
- artifact interpolation and detection,
- light smoothing,
- causal IIR filtering for batch and sample-wise operation, and
- median-based spike replacement.

The current live pipeline primarily uses the stateful high-pass and low-pass
helpers, but the remaining functions are retained for offline analysis and
regression tests.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.signal import butter, medfilt, sosfilt, sosfilt_zi


class MaxMinTracker:
    """Track running extrema with a slow activity-gated reset.

    This helper is designed for normalization schemes that adapt to a signal's
    recent operating range. The reset mechanism is intentionally conservative:
    the stored extrema decay only when there is measurable signal activity, so
    quiet periods or breath holds do not collapse the range immediately.
    """

    def __init__(self, reset_interval_sec: float = 60) -> None:
        self.reset_interval = reset_interval_sec
        self.last_reset = time.time()
        self.max_val = None
        self.min_val = None
        self.last_value = None
        self.activity_threshold = 0.01

    def update(self, value: float) -> None:
        """Update extrema with one new sample."""

        now = time.time()
        if self.max_val is None:
            # Seed a finite range to avoid zero-width normalization at startup.
            self.max_val = value + 0.1
            self.min_val = value - 0.1
            self.last_value = value
            return

        signal_change = abs(value - self.last_value) if self.last_value is not None else 0
        is_active = signal_change > self.activity_threshold

        if (now - self.last_reset) > self.reset_interval and is_active:
            # The reset is intentionally soft so one update does not erase the
            # recently observed range.
            self.max_val = max(value, self.max_val - abs(self.max_val) * 0.1)
            self.min_val = min(value, self.min_val + abs(self.min_val) * 0.1)
            self.last_reset = now
        else:
            if value > self.max_val:
                self.max_val = value
            if value < self.min_val:
                self.min_val = value

        self.last_value = value

        if self.max_val - self.min_val < 0.01:
            center = (self.max_val + self.min_val) / 2
            self.max_val = center + 0.01
            self.min_val = center - 0.01

    def get_max_min(self) -> tuple[float | None, float | None]:
        """Return the current extrema estimates."""

        return self.max_val, self.min_val


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Map a scalar linearly into ``[0, 1]``.

    Returns ``0.5`` when the range collapses to a point, and ``np.nan`` when an
    invalid inverted range is supplied.
    """

    if min_val == max_val:
        return 0.5
    if min_val > max_val:
        return np.nan
    return (value - min_val) / (max_val - min_val)


def interpolate_artifacts(signal: np.ndarray) -> np.ndarray:
    """Replace ``NaN`` samples by linear interpolation.

    When all samples are missing, the function returns zeros so downstream code
    can continue with a finite-valued signal.
    """

    signal = np.asarray(signal).copy()
    nans = np.isnan(signal)
    if np.all(nans):
        return np.zeros_like(signal)
    if np.any(nans):
        not_nans = ~nans
        signal[nans] = np.interp(
            np.flatnonzero(nans),
            np.flatnonzero(not_nans),
            signal[not_nans],
        )
    return signal


def detect_motion_artifacts(
    signal: np.ndarray,
    window: int = 10,
    threshold: float = 5,
) -> np.ndarray:
    """Detect motion artifacts from large local sample-to-sample changes.

    The decision rule compares the absolute first difference to a local
    variability estimate. A floor based on the global standard deviation keeps
    flat plateaus from being incorrectly labeled as artifacts because of a
    near-zero local variance estimate.
    """

    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.zeros(0, dtype=bool)

    diff = np.abs(np.diff(signal, prepend=signal[0]))
    try:
        import pandas as pd

        local_std = (
            pd.Series(signal).rolling(window, min_periods=2).std(ddof=0).to_numpy()
        )
    except ImportError:
        if len(signal) < window:
            local_std = np.full(
                len(signal),
                np.std(signal) if len(signal) > 1 else 1.0,
            )
        else:
            local_std = np.array(
                [
                    np.std(signal[max(0, i - window) : i + 1])
                    if i > 0
                    else np.std(signal[:1])
                    for i in range(len(signal))
                ]
            )

    global_std = np.std(signal) if len(signal) > 1 else 0.0
    std_floor = max(global_std * 0.25, 1e-4)
    local_std = np.nan_to_num(local_std, nan=global_std)
    local_std = np.maximum(local_std, std_floor)

    artifact_mask = diff > threshold * local_std
    return artifact_mask


def smooth_signal(signal: np.ndarray, window: int = 31) -> np.ndarray:
    """Apply moving-average smoothing with edge padding."""

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
    padded = np.pad(signal, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def high_pass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a causal Butterworth high-pass filter in batch mode.

    Second-order sections are used for numerical stability at low cutoff
    frequencies, which are common for respiration-baseline suppression.
    """

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0 or normal_cutoff <= 0:
        raise ValueError(f"Cutoff {cutoff} Hz invalid for sampling rate {fs} Hz")
    sos = butter(order, normal_cutoff, btype="high", analog=False, output="sos")
    return sosfilt(sos, data)


def get_high_pass_filter_coeffs(
    cutoff: float,
    fs: float,
    order: int = 5,
    initial_value: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create high-pass filter coefficients and initial state for live filtering.

    ``initial_value`` scales the steady-state filter state to reduce startup
    transients when the first observed sample is already representative of the
    current baseline.
    """

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0 or normal_cutoff <= 0:
        raise ValueError(f"Cutoff {cutoff} Hz invalid for sampling rate {fs} Hz")
    sos = butter(order, normal_cutoff, btype="high", analog=False, output="sos")
    zi = sosfilt_zi(sos)
    if initial_value is not None:
        zi = zi * initial_value
    return sos, zi


def high_pass_filter_sample(
    sample: float,
    sos: np.ndarray,
    zi: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Filter one sample with a stateful high-pass filter."""

    filtered, zi = sosfilt(sos, [sample], zi=zi)
    return filtered[0], zi


def low_pass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 2,
) -> np.ndarray:
    """Apply a causal Butterworth low-pass filter in batch mode."""

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0 or normal_cutoff <= 0:
        raise ValueError(f"Cutoff {cutoff} Hz invalid for sampling rate {fs} Hz")
    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    return sosfilt(sos, data)


def get_low_pass_filter_coeffs(
    cutoff: float,
    fs: float,
    order: int = 2,
    initial_value: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create low-pass filter coefficients and initial state for live filtering."""

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0 or normal_cutoff <= 0:
        raise ValueError(f"Cutoff {cutoff} Hz invalid for sampling rate {fs} Hz")
    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    zi = sosfilt_zi(sos)
    if initial_value is not None:
        zi = zi * initial_value
    return sos, zi


def low_pass_filter_sample(
    sample: float,
    sos: np.ndarray,
    zi: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Filter one sample with a stateful low-pass filter."""

    filtered, zi = sosfilt(sos, [sample], zi=zi)
    return filtered[0], zi


def band_pass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a causal Butterworth band-pass filter in batch mode."""

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or low >= 1.0 or high <= 0 or high >= 1.0 or low >= high:
        raise ValueError(
            f"Invalid band-pass cutoffs: {lowcut}-{highcut} Hz for sampling rate {fs} Hz"
        )
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, data)


def get_band_pass_filter_coeffs(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
    initial_value: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create band-pass filter coefficients and initial state for live filtering."""

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or low >= 1.0 or high <= 0 or high >= 1.0 or low >= high:
        raise ValueError(
            f"Invalid band-pass cutoffs: {lowcut}-{highcut} Hz for sampling rate {fs} Hz"
        )
    sos = butter(order, [low, high], btype="band", output="sos")
    zi = sosfilt_zi(sos)
    if initial_value is not None:
        zi = zi * initial_value
    return sos, zi


def band_pass_filter_sample(
    sample: float,
    sos: np.ndarray,
    zi: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Filter one sample with a stateful band-pass filter."""

    filtered, zi = sosfilt(sos, [sample], zi=zi)
    return filtered[0], zi


def remove_spikes(
    signal: np.ndarray,
    kernel_size: int = 5,
    threshold: float = 3,
) -> np.ndarray:
    """Replace large isolated deviations with a local median estimate."""

    if len(signal) < kernel_size:
        return signal

    filtered = medfilt(signal, kernel_size)
    diff = np.abs(signal - filtered)
    spikes = diff > threshold * np.std(signal)
    signal_out = np.copy(signal)
    signal_out[spikes] = filtered[spikes]
    return signal_out
