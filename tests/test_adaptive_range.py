"""Regression tests for adaptive normalization during live operation."""

from __future__ import annotations

import numpy as np

from src.calibration import (
    AdaptiveRangeConfig,
    AdaptiveRangeState,
    CalibrationConfig,
    initialize_adaptive_range,
    run_range_calibration,
    update_adaptive_range,
)


def _make_calibrated_state(
    fs_hz: float = 100.0,
    duration_s: float = 15.0,
) -> tuple[np.ndarray, AdaptiveRangeConfig, AdaptiveRangeState]:
    """Construct a calibrated adaptive state from a synthetic breathing trace."""

    t = np.arange(0.0, duration_s, 1.0 / fs_hz)
    calibration_samples = 0.8 * np.sin(2.0 * np.pi * 0.22 * t)
    cal_cfg = CalibrationConfig(
        fs_hz=fs_hz,
        percentile_lo=5.0,
        percentile_hi=95.0,
        saturation_lo=-1e9,
        saturation_hi=1e9,
        amplitude_floor=1e-3,
    )
    cal_result = run_range_calibration(calibration_samples, cal_cfg)
    adapt_cfg = AdaptiveRangeConfig(
        fs_hz=fs_hz,
        center_tau_s=180.0,
        amplitude_tau_s=300.0,
        amplitude_floor=1e-3,
    )
    state = initialize_adaptive_range(calibration_samples, cal_result, adapt_cfg)
    return calibration_samples, adapt_cfg, state


def _stream_normalize(
    stream: np.ndarray,
    cfg: AdaptiveRangeConfig,
    state: AdaptiveRangeState,
    allow_updates: np.ndarray | None = None,
) -> tuple[np.ndarray, AdaptiveRangeState]:
    """Normalize a stream sample-by-sample using the runtime update function."""

    normalized = np.zeros(stream.size, dtype=float)
    if allow_updates is None:
        allow_updates = np.ones(stream.size, dtype=bool)

    for idx, value in enumerate(stream):
        y, state = update_adaptive_range(
            x=float(value),
            state=state,
            cfg=cfg,
            allow_update=bool(allow_updates[idx]),
        )
        normalized[idx] = y
    return normalized, state


def test_initialize_adaptive_range_seeds_from_calibration() -> None:
    """Adaptive initialization should inherit the calibration operating point."""

    calibration_samples, adapt_cfg, state = _make_calibrated_state()
    cal_result = run_range_calibration(
        calibration_samples,
        CalibrationConfig(
            fs_hz=adapt_cfg.fs_hz,
            percentile_lo=5.0,
            percentile_hi=95.0,
            saturation_lo=-1e9,
            saturation_hi=1e9,
            amplitude_floor=1e-3,
        ),
    )

    assert np.isclose(state.center, cal_result.center)
    assert np.isclose(state.amplitude, cal_result.amplitude)
    assert np.isfinite(state.abs_dev_to_amplitude_scale)
    assert state.abs_dev_to_amplitude_scale > 0.0


def test_artifact_gating_prevents_state_updates() -> None:
    """Disabled updates must freeze both center and amplitude estimates."""

    _, adapt_cfg, state = _make_calibrated_state()
    start_center = state.center
    start_amplitude = state.amplitude

    for _ in range(5000):
        _, state = update_adaptive_range(
            x=5.0,
            state=state,
            cfg=adapt_cfg,
            allow_update=False,
        )

    assert np.isclose(state.center, start_center)
    assert np.isclose(state.amplitude, start_amplitude)


def test_slow_drift_keeps_control_centered_without_rail_pinning() -> None:
    """Slow baseline drift should not push the normalized output onto the rails."""

    _, adapt_cfg, state = _make_calibrated_state()
    fs_hz = adapt_cfg.fs_hz
    duration_s = 6.0 * 60.0
    t = np.arange(0.0, duration_s, 1.0 / fs_hz)
    breathing = 0.7 * np.sin(2.0 * np.pi * 0.22 * t)
    drift = np.linspace(-0.5, 0.5, t.size)
    stream = breathing + drift

    normalized, _ = _stream_normalize(stream=stream, cfg=adapt_cfg, state=state)

    median_value = float(np.median(normalized))
    low_fraction = float(np.mean(normalized <= 0.02))
    high_fraction = float(np.mean(normalized >= 0.98))

    assert 0.35 <= median_value <= 0.65
    assert low_fraction < 0.20
    assert high_fraction < 0.20


def test_spike_transients_with_gating_do_not_destabilize_state() -> None:
    """Masked spikes should not materially bias the adaptive normalization state."""

    _, adapt_cfg, initial_state = _make_calibrated_state()
    fs_hz = adapt_cfg.fs_hz
    duration_s = 120.0
    t = np.arange(0.0, duration_s, 1.0 / fs_hz)
    clean_stream = 0.8 * np.sin(2.0 * np.pi * 0.22 * t)

    base_normalized, base_state = _stream_normalize(
        stream=clean_stream,
        cfg=adapt_cfg,
        state=initial_state,
    )
    assert base_normalized.size == clean_stream.size

    spiky_stream = clean_stream.copy()
    artifact_mask = np.zeros_like(spiky_stream, dtype=bool)
    spike_indices = np.arange(500, spiky_stream.size, 1300)
    spiky_stream[spike_indices] += 12.0
    artifact_mask[spike_indices] = True

    _, spiky_state = _stream_normalize(
        stream=spiky_stream,
        cfg=adapt_cfg,
        state=initial_state,
        allow_updates=~artifact_mask,
    )

    assert abs(spiky_state.center - base_state.center) < 0.05
    assert abs(spiky_state.amplitude - base_state.amplitude) < 0.05


def test_adaptive_normalization_output_is_bounded() -> None:
    """Normalized output must remain inside the control interval [0, 1]."""

    _, adapt_cfg, state = _make_calibrated_state()
    fs_hz = adapt_cfg.fs_hz
    duration_s = 240.0
    t = np.arange(0.0, duration_s, 1.0 / fs_hz)
    amp_mod = 0.4 + 0.35 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.005 * t))
    breathing = amp_mod * np.sin(2.0 * np.pi * 0.22 * t)
    drift = 0.35 * np.sin(2.0 * np.pi * 0.0015 * t)
    stream = breathing + drift

    normalized, _ = _stream_normalize(stream=stream, cfg=adapt_cfg, state=state)

    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)


def test_update_allows_disabling_center_updates_only() -> None:
    """Amplitude adaptation should work when center adaptation is disabled."""

    _, adapt_cfg, state = _make_calibrated_state()
    start_center = state.center
    start_amplitude = state.amplitude

    _, updated_state = update_adaptive_range(
        x=2.0,
        state=state,
        cfg=adapt_cfg,
        allow_center_update=False,
        allow_amplitude_update=True,
    )

    assert np.isclose(updated_state.center, start_center)
    assert updated_state.amplitude != start_amplitude


def test_update_allows_disabling_amplitude_updates_only() -> None:
    """Center adaptation should work when amplitude adaptation is disabled."""

    _, adapt_cfg, state = _make_calibrated_state()
    start_center = state.center
    start_amplitude = state.amplitude

    _, updated_state = update_adaptive_range(
        x=2.0,
        state=state,
        cfg=adapt_cfg,
        allow_center_update=True,
        allow_amplitude_update=False,
    )

    assert updated_state.center != start_center
    assert np.isclose(updated_state.amplitude, start_amplitude)
