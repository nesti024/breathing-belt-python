"""Range calibration utilities for respiration-belt control mapping.

This module estimates a robust signal center and amplitude from a short
calibration window. It intentionally uses percentile bounds instead of raw
minimum/maximum so occasional spikes do not dominate the normalization range.
It also checks ADC rail saturation, which is a practical sign of clipping
(e.g., belt too tight or gain/range mismatch).

The resulting ``center`` and ``amplitude`` are intended for mapping incoming
samples to a normalized VR control signal in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for robust range calibration.

    Percentiles are used to estimate a stable operating range while rejecting
    outliers from transient motion or sensor glitches.
    """

    duration_s: float = 15.0
    fs_hz: float = 100.0
    percentile_lo: float = 5.0
    percentile_hi: float = 95.0
    saturation_lo: float = 0.0
    saturation_hi: float = 1023.0
    amplitude_scale: float = 1.0
    amplitude_floor: float = 1e-6
    padding_ratio: float = 0.2


@dataclass(frozen=True)
class CalibrationResult:
    """Outputs of robust range calibration.

    ``center`` and ``amplitude`` define the affine mapping to normalized
    breathing control. ``saturated`` flags whether any calibration samples
    clipped near ADC rails.
    """

    global_min: float
    global_max: float
    center: float
    amplitude: float
    y_min: float
    y_max: float
    saturated: bool
    n_samples: int
    saturated_count: int
    lo_idx: int
    hi_idx: int


def run_range_calibration(
    samples: list[float] | np.ndarray,
    cfg: CalibrationConfig,
) -> CalibrationResult:
    """Estimate robust center/amplitude from pre-collected samples.

    Method:
    - Sort the calibration window.
    - Use low/high percentile indices as robust min/max estimates.
    - Derive center/amplitude from those bounds.
    - Apply an amplitude safety floor to avoid division-by-zero mapping.
    - Count rail hits to detect likely clipping/saturation.

    Percentile clipping improves robustness to occasional outliers, while
    saturation checks highlight calibration windows where the signal likely
    exceeded usable ADC range (commonly from belt tightness or clipping).
    """

    arr = np.asarray(samples, dtype=float).reshape(-1)
    n = int(arr.size)
    if n == 0:
        raise ValueError("Calibration requires at least one sample.")

    sorted_samples = np.sort(arr)

    lo_idx = int(n * cfg.percentile_lo / 100.0)
    hi_idx = int(n * cfg.percentile_hi / 100.0) - 1

    lo_idx = max(0, min(lo_idx, n - 1))
    hi_idx = max(0, min(hi_idx, n - 1))

    if hi_idx < lo_idx:
        lo_idx = 0
        hi_idx = n - 1

    global_min = float(sorted_samples[lo_idx])
    global_max = float(sorted_samples[hi_idx])
    center = 0.5 * (global_max + global_min)

    raw_amplitude = 0.5 * (global_max - global_min)
    amplitude = max(raw_amplitude * cfg.amplitude_scale, cfg.amplitude_floor)

    padding = (global_max - global_min) * cfg.padding_ratio
    y_min = global_min - padding
    y_max = global_max + padding

    saturated_mask = (arr <= cfg.saturation_lo) | (arr >= cfg.saturation_hi)
    saturated_count = int(np.count_nonzero(saturated_mask))
    saturated = saturated_count > 0

    return CalibrationResult(
        global_min=global_min,
        global_max=global_max,
        center=center,
        amplitude=amplitude,
        y_min=y_min,
        y_max=y_max,
        saturated=saturated,
        n_samples=n,
        saturated_count=saturated_count,
        lo_idx=lo_idx,
        hi_idx=hi_idx,
    )


def normalize_sample(
    x: float,
    center: float,
    amplitude: float,
    clamp: bool = True,
) -> float:
    """Normalize one sample into breathing control space.

    The normalized control is:
    ``y = 0.5 + (x - center) / (2 * amplitude)``

    This maps values near ``center`` to 0.5 and typical inhale/exhale range to
    approximately ``[0, 1]``. Clamping keeps the control bounded for VR input.
    """

    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive.")

    y = 0.5 + (x - center) / (2.0 * amplitude)
    if clamp:
        return float(min(1.0, max(0.0, y)))
    return float(y)


def collect_samples(
    get_latest_fn: Callable[[], float],
    cfg: CalibrationConfig,
) -> Callable[[], tuple[bool, np.ndarray]]:
    """Create a non-blocking calibration collector.

    This helper returns a ``step`` callback that can be called from a main loop.
    Each call performs at most one sample read (if its scheduled time has
    arrived), so the loop remains responsive.
    """

    if cfg.fs_hz <= 0.0:
        raise ValueError("fs_hz must be positive.")

    target_samples = max(1, int(round(cfg.duration_s * cfg.fs_hz)))
    dt = 1.0 / cfg.fs_hz
    next_t = time.perf_counter()
    collected: list[float] = []

    def step() -> tuple[bool, np.ndarray]:
        nonlocal next_t
        now = time.perf_counter()
        if len(collected) < target_samples and now >= next_t:
            collected.append(float(get_latest_fn()))
            next_t += dt
        done = len(collected) >= target_samples
        return done, np.asarray(collected, dtype=float)

    return step


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    fs_hz = 100.0
    duration_s = 15.0
    t = np.arange(0.0, duration_s, 1.0 / fs_hz)

    breathing = (
        512.0
        + 120.0 * np.sin(2.0 * np.pi * 0.22 * t)
        + rng.normal(0.0, 8.0, size=t.size)
    )

    # Inject a few outliers to demonstrate robust percentile calibration.
    breathing[100] = 0.0
    breathing[420] = 1023.0
    breathing[900] = 1023.0

    config = CalibrationConfig(
        duration_s=duration_s,
        fs_hz=fs_hz,
        saturation_lo=10.0,
        saturation_hi=1013.0,
        percentile_lo=5.0,
        percentile_hi=95.0,
        amplitude_floor=1e-3,
    )

    result = run_range_calibration(breathing, config)
    print("Calibration result:")
    print(result)

    print("\nFirst 8 normalized samples:")
    for value in breathing[:8]:
        print(normalize_sample(float(value), result.center, result.amplitude))
