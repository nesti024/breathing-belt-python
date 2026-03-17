"""Regression tests for robust percentile-based calibration."""

from __future__ import annotations

import numpy as np

from src.calibration import CalibrationConfig, normalize_sample, run_range_calibration


def test_percentile_clipping_rejects_outliers() -> None:
    """Extreme outliers should not dominate the calibrated operating range."""

    rng = np.random.default_rng(123)
    main = rng.normal(loc=100.0, scale=2.0, size=1000)
    samples = np.concatenate([main, np.array([-1000.0, 2500.0, -900.0, 2600.0])])

    cfg = CalibrationConfig(
        fs_hz=100.0,
        percentile_lo=5.0,
        percentile_hi=95.0,
        saturation_lo=-1e9,
        saturation_hi=1e9,
    )
    result = run_range_calibration(samples, cfg)

    assert result.global_min > 94.0
    assert result.global_max < 106.0
    assert result.global_min > -100.0
    assert result.global_max < 500.0


def test_saturation_detection_counts_rail_hits() -> None:
    """Samples at or beyond the configured rails should be counted as saturated."""

    samples = np.array([0.0, 1.0, 10.0, 500.0, 1022.0, 1023.0], dtype=float)
    cfg = CalibrationConfig(
        fs_hz=100.0,
        saturation_lo=1.0,
        saturation_hi=1022.0,
    )
    result = run_range_calibration(samples, cfg)

    assert result.saturated is True
    assert result.saturated_count == 4


def test_index_clamping_tiny_sample_counts() -> None:
    """Percentile index logic should remain valid for very short inputs."""

    one = np.array([42.0], dtype=float)
    cfg_one = CalibrationConfig(fs_hz=100.0, percentile_lo=5.0, percentile_hi=95.0)
    r_one = run_range_calibration(one, cfg_one)
    assert r_one.lo_idx == 0
    assert r_one.hi_idx == 0

    two = np.array([2.0, 1.0], dtype=float)
    cfg_two = CalibrationConfig(fs_hz=100.0, percentile_lo=-50.0, percentile_hi=250.0)
    r_two = run_range_calibration(two, cfg_two)
    assert r_two.lo_idx == 0
    assert r_two.hi_idx == 1

    ten = np.arange(10.0, dtype=float)
    cfg_ten = CalibrationConfig(fs_hz=100.0, percentile_lo=80.0, percentile_hi=20.0)
    r_ten = run_range_calibration(ten, cfg_ten)
    assert r_ten.lo_idx == 0
    assert r_ten.hi_idx == 9


def test_amplitude_floor_for_constant_signal() -> None:
    """A constant calibration trace should fall back to the amplitude floor."""

    samples = np.full(128, 7.5, dtype=float)
    cfg = CalibrationConfig(fs_hz=100.0, amplitude_floor=0.25)
    result = run_range_calibration(samples, cfg)

    assert result.global_min == 7.5
    assert result.global_max == 7.5
    assert result.center == 7.5
    assert result.amplitude == 0.25
    assert normalize_sample(7.5, result.center, result.amplitude) == 0.5


def test_normalize_sample_clamps_outside_calibrated_range() -> None:
    """Samples beyond the calibrated range should clamp to the control rails."""

    assert normalize_sample(-10.0, center=0.0, amplitude=1.0, clamp=True) == 0.0
    assert normalize_sample(10.0, center=0.0, amplitude=1.0, clamp=True) == 1.0


def test_fixed_calibration_stream_normalization_is_bounded() -> None:
    """Runtime samples must remain bounded even when they exceed calibration amplitude."""

    fs_hz = 100.0
    t = np.arange(0.0, 12.0, 1.0 / fs_hz)
    calibration_samples = np.sin(2.0 * np.pi * 0.22 * t)
    cfg = CalibrationConfig(fs_hz=fs_hz, percentile_lo=5.0, percentile_hi=95.0)
    result = run_range_calibration(calibration_samples, cfg)

    runtime_samples = 1.8 * np.sin(2.0 * np.pi * 0.22 * t)
    normalized = np.array(
        [
            normalize_sample(float(x), result.center, result.amplitude, clamp=True)
            for x in runtime_samples
        ]
    )

    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    assert np.any(normalized == 0.0)
    assert np.any(normalized == 1.0)
