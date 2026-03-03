import numpy as np

from src.calibration import CalibrationConfig, normalize_sample, run_range_calibration


def test_percentile_clipping_rejects_outliers() -> None:
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
    samples = np.full(128, 7.5, dtype=float)
    cfg = CalibrationConfig(fs_hz=100.0, amplitude_floor=0.25)
    result = run_range_calibration(samples, cfg)

    assert result.global_min == 7.5
    assert result.global_max == 7.5
    assert result.center == 7.5
    assert result.amplitude == 0.25
    assert normalize_sample(7.5, result.center, result.amplitude) == 0.5
