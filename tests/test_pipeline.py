"""End-to-end replay tests for the live breathing-belt pipeline."""

from __future__ import annotations

import numpy as np

from src.pipeline import PipelineConfig, create_pipeline_state, process_device_row
from src.quality import raw_qc_summary
from src.settings import (
    AdaptationSettings,
    ArtifactConfig,
    CalibrationSettings,
    ExtremaConfig,
    FilterConfig,
    HoldConfig,
    MovementConfig,
    OutputSmoothingConfig,
    RawQCConfig,
)


FS_HZ = 100


def _make_pipeline_config(
    *,
    processed_sensor_column: int = 5,
    calibration_duration_s: float = 0.2,
    invert_signal: bool = False,
    processing_mode: str = "control",
    adaptation: AdaptationSettings | None = None,
    hold: HoldConfig | None = None,
    movement: MovementConfig | None = None,
    output_smoothing: OutputSmoothingConfig | None = None,
    extrema: ExtremaConfig | None = None,
    raw_qc: RawQCConfig | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        sampling_rate_hz=FS_HZ,
        processed_sensor_column=processed_sensor_column,
        invert_signal=invert_signal,
        filter=FilterConfig(hp_cutoff_hz=0.005, hp_order=1, lp_cutoff_hz=1.5, lp_order=2),
        artifact=ArtifactConfig(spike_threshold=2.5, artifact_window=10),
        calibration=CalibrationSettings(
            duration_s=calibration_duration_s,
            percentile_lo=5.0,
            percentile_hi=95.0,
            amplitude_floor=1e-3,
            padding_ratio=0.20,
        ),
        adaptation=adaptation
        or AdaptationSettings(
            center_enabled=True,
            amplitude_enabled=True,
            center_tau_s=20.0,
            amplitude_tau_s=0.5,
            startup_duration_s=0.2,
            startup_center_tau_s=0.2,
            startup_amplitude_tau_s=0.2,
        ),
        hold=hold
        or HoldConfig(
            enabled=True,
            activity_window_ms=100,
            ratio_per_sec_enter=0.2,
            ratio_per_sec_exit=0.4,
            floor_per_sec=0.01,
            edge_margin_ratio=0.20,
        ),
        output_smoothing=output_smoothing
        or OutputSmoothingConfig(
            enabled=True,
            activity_window_ms=500,
            tau_active_s=0.25,
            tau_extreme_s=0.75,
            tau_hold_s=5.0,
            activity_low_ratio_per_sec=0.10,
            activity_high_ratio_per_sec=0.50,
            activity_floor_per_sec=0.01,
            edge_margin_ratio=0.10,
        ),
        extrema=extrema or ExtremaConfig(min_interval_ms=800, prominence_ratio=0.1),
        raw_qc=raw_qc or RawQCConfig(),
        processing_mode=processing_mode,
        movement=movement or MovementConfig(),
    )


def _make_row(
    selected_value: float,
    *,
    alternate_value: float = 0.0,
    processed_sensor_column: int = 5,
) -> np.ndarray:
    row = np.zeros(7, dtype=float)
    row[5] = selected_value if processed_sensor_column != 6 else alternate_value
    row[6] = selected_value if processed_sensor_column == 6 else alternate_value
    return row


def _make_breathing_values(length: int, amplitude: float, offset: float = 512.0) -> np.ndarray:
    t = np.arange(length, dtype=float) / FS_HZ
    return offset + amplitude * np.sin(2.0 * np.pi * 0.22 * t)


def _replay(values: np.ndarray, cfg: PipelineConfig) -> tuple[list, object]:
    state = create_pipeline_state(cfg)
    samples = []
    for value in values:
        sample, state = process_device_row(
            _make_row(float(value), processed_sensor_column=cfg.processed_sensor_column),
            state,
            cfg,
        )
        samples.append(sample)
    return samples, state


def _mapped_candidate_levels(samples: list, state) -> np.ndarray:
    if state.calibration_result is None:
        raise AssertionError("Calibration result must exist for runtime candidate mapping.")

    control_min = float(state.calibration_result.y_min)
    control_max = float(state.calibration_result.y_max)
    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    mapped = [
        (sample.cleaned_value - control_min) / (control_max - control_min)
        for sample in runtime_samples
    ]
    return np.clip(np.asarray(mapped, dtype=float), 0.0, 1.0)


def test_pipeline_transitions_from_calibration_to_runtime() -> None:
    cfg = _make_pipeline_config()
    values = _make_breathing_values(40, amplitude=20.0)

    samples, state = _replay(values, cfg)

    assert all(sample.stage == "calibration" for sample in samples[: cfg.calibration_target_samples])
    first_runtime = samples[cfg.calibration_target_samples]
    assert first_runtime.stage == "runtime"
    assert first_runtime.sample_index == 0
    assert first_runtime.normalized_value is not None
    assert state.calibration_result is not None
    assert state.adaptive_state is not None


def test_pipeline_uses_fixed_calibration_after_amplitude_change() -> None:
    cfg = _make_pipeline_config()
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0)
    runtime_low = _make_breathing_values(120, amplitude=20.0)
    runtime_high = _make_breathing_values(240, amplitude=80.0)
    samples, state = _replay(
        np.concatenate([calibration_values, runtime_low, runtime_high]),
        cfg,
    )

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    amplitudes = np.array(
        [sample.adaptive_amplitude for sample in runtime_samples if sample.adaptive_amplitude is not None],
        dtype=float,
    )

    assert state.adaptive_state is not None
    assert amplitudes.size > 0
    assert np.allclose(amplitudes, amplitudes[0])
    assert np.isclose(state.adaptive_state.amplitude, amplitudes[0])


def test_pipeline_hold_mode_freezes_output_level() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        output_smoothing=OutputSmoothingConfig(
            enabled=False,
            activity_window_ms=100,
            tau_active_s=0.25,
            tau_hold_s=5.0,
            activity_low_ratio_per_sec=0.10,
            activity_high_ratio_per_sec=0.50,
            activity_floor_per_sec=0.01,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=25.0)
    hold_values = np.full(250, 540.0, dtype=float)
    samples, state = _replay(np.concatenate([calibration_values, hold_values]), cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    hold_start = next(idx for idx, sample in enumerate(runtime_samples) if sample.hold_mode_active)
    frozen_value = runtime_samples[hold_start].normalized_value
    frozen_tail = np.array(
        [sample.normalized_value for sample in runtime_samples[hold_start:] if sample.normalized_value is not None],
        dtype=float,
    )

    assert frozen_value is not None
    assert runtime_samples[hold_start].hold_mode_active is True
    assert np.allclose(frozen_tail, float(frozen_value))
    assert state.hold_mode_active is True


def test_pipeline_midrange_plateau_and_ramp_do_not_enter_hold() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        output_smoothing=OutputSmoothingConfig(
            enabled=False,
            activity_window_ms=100,
            tau_active_s=0.25,
            tau_hold_s=5.0,
            activity_low_ratio_per_sec=0.10,
            activity_high_ratio_per_sec=0.50,
            activity_floor_per_sec=0.01,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=25.0)
    midrange_values = np.concatenate(
        [
            np.full(250, 512.0, dtype=float),
            np.linspace(500.0, 520.0, 250, dtype=float),
        ]
    )
    samples, _ = _replay(np.concatenate([calibration_values, midrange_values]), cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    normalized = np.array(
        [sample.normalized_value for sample in runtime_samples if sample.normalized_value is not None],
        dtype=float,
    )

    assert all(not sample.hold_mode_active for sample in runtime_samples)
    assert np.max(np.abs(np.diff(normalized))) < 0.03


def test_pipeline_hold_mode_releases_immediately_when_motion_resumes() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        output_smoothing=OutputSmoothingConfig(
            enabled=False,
            activity_window_ms=100,
            tau_active_s=0.25,
            tau_hold_s=5.0,
            activity_low_ratio_per_sec=0.10,
            activity_high_ratio_per_sec=0.50,
            activity_floor_per_sec=0.01,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=25.0)
    plateau_values = np.full(250, 540.0, dtype=float)
    release_values = np.linspace(480.0, 450.0, 80, dtype=float)
    samples, _ = _replay(
        np.concatenate([calibration_values, plateau_values, release_values]),
        cfg,
    )

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    ramp_start_index = len(plateau_values)
    assert any(sample.hold_mode_active for sample in runtime_samples[:ramp_start_index])

    release_index = next(
        idx
        for idx in range(ramp_start_index, len(runtime_samples))
        if runtime_samples[idx - 1].hold_mode_active and not runtime_samples[idx].hold_mode_active
    )
    release_jump = abs(
        float(runtime_samples[release_index].normalized_value)
        - float(runtime_samples[release_index - 1].normalized_value)
    )

    assert release_index <= ramp_start_index + 2
    assert release_jump < 0.1
    assert not any(sample.hold_mode_active for sample in runtime_samples[release_index : release_index + 10])


def test_pipeline_hold_detection_can_be_disabled() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=HoldConfig(
            enabled=False,
            activity_window_ms=100,
            ratio_per_sec_enter=0.2,
            ratio_per_sec_exit=0.4,
            floor_per_sec=0.01,
            edge_margin_ratio=0.20,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=25.0)
    hold_like_values = np.concatenate(
        [
            np.full(250, 540.0, dtype=float),
            np.linspace(540.0, 300.0, 250, dtype=float),
        ]
    )
    samples, _ = _replay(np.concatenate([calibration_values, hold_like_values]), cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    normalized = np.array(
        [sample.normalized_value for sample in runtime_samples if sample.normalized_value is not None],
        dtype=float,
    )

    assert all(not sample.hold_mode_active for sample in runtime_samples)
    assert np.max(np.abs(np.diff(normalized))) < 0.03


def test_pipeline_output_smoothing_reduces_low_activity_drift_without_steps() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=HoldConfig(
            enabled=False,
            activity_window_ms=100,
            ratio_per_sec_enter=0.2,
            ratio_per_sec_exit=0.4,
            floor_per_sec=0.01,
            edge_margin_ratio=0.20,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=25.0)
    runtime_values = np.concatenate(
        [
            np.linspace(512.0, 520.0, 200, dtype=float),
            np.linspace(520.0, 540.0, 500, dtype=float),
        ]
    )
    samples, state = _replay(np.concatenate([calibration_values, runtime_values]), cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    normalized = np.array(
        [sample.normalized_value for sample in runtime_samples if sample.normalized_value is not None],
        dtype=float,
    )
    candidate = _mapped_candidate_levels(samples, state)
    drift_slice = slice(200, None)

    candidate_drift = float(candidate[drift_slice][-1] - candidate[drift_slice][0])
    normalized_drift = float(normalized[drift_slice][-1] - normalized[drift_slice][0])

    assert all(not sample.hold_mode_active for sample in runtime_samples)
    assert normalized_drift < candidate_drift * 0.70
    assert np.max(np.abs(np.diff(normalized[drift_slice]))) < 0.01


def test_pipeline_output_smoothing_reaches_lower_edge_during_sustained_trough() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=HoldConfig(
            enabled=False,
            activity_window_ms=100,
            ratio_per_sec_enter=0.2,
            ratio_per_sec_exit=0.4,
            floor_per_sec=0.01,
            edge_margin_ratio=0.20,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0)
    runtime_values = np.concatenate(
        [
            np.linspace(512.0, 300.0, 200, dtype=float),
            np.full(100, 300.0, dtype=float),
        ]
    )
    samples, state = _replay(np.concatenate([calibration_values, runtime_values]), cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    normalized = np.array(
        [sample.normalized_value for sample in runtime_samples if sample.normalized_value is not None],
        dtype=float,
    )
    candidate = _mapped_candidate_levels(samples, state)

    assert np.allclose(candidate[-100:], 0.0)
    assert float(normalized[-1]) < 0.01


def test_pipeline_output_smoothing_reaches_upper_edge_during_sustained_peak() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=HoldConfig(
            enabled=False,
            activity_window_ms=100,
            ratio_per_sec_enter=0.2,
            ratio_per_sec_exit=0.4,
            floor_per_sec=0.01,
            edge_margin_ratio=0.20,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0)
    runtime_values = np.concatenate(
        [
            np.linspace(512.0, 700.0, 200, dtype=float),
            np.full(100, 700.0, dtype=float),
        ]
    )
    samples, state = _replay(np.concatenate([calibration_values, runtime_values]), cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    normalized = np.array(
        [sample.normalized_value for sample in runtime_samples if sample.normalized_value is not None],
        dtype=float,
    )
    candidate = _mapped_candidate_levels(samples, state)

    assert np.allclose(candidate[-100:], 1.0)
    assert float(normalized[-1]) > 0.99


def test_pipeline_output_smoothing_preserves_active_breathing_responsiveness() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=HoldConfig(
            enabled=False,
            activity_window_ms=100,
            ratio_per_sec_enter=0.2,
            ratio_per_sec_exit=0.4,
            floor_per_sec=0.01,
            edge_margin_ratio=0.20,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0)
    runtime_values = _make_breathing_values(800, amplitude=25.0)
    samples, state = _replay(np.concatenate([calibration_values, runtime_values]), cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    normalized = np.array(
        [sample.normalized_value for sample in runtime_samples if sample.normalized_value is not None],
        dtype=float,
    )
    candidate = _mapped_candidate_levels(samples, state)
    compare_slice = slice(100, None)

    assert np.max(np.abs(normalized[compare_slice] - candidate[compare_slice])) < 0.15
    assert (np.max(normalized) - np.min(normalized)) > 0.8 * (
        np.max(candidate) - np.min(candidate)
    )


def test_pipeline_edge_aware_smoothing_is_inactive_in_midrange() -> None:
    base_hold = HoldConfig(
        enabled=False,
        activity_window_ms=100,
        ratio_per_sec_enter=0.2,
        ratio_per_sec_exit=0.4,
        floor_per_sec=0.01,
        edge_margin_ratio=0.20,
    )
    cfg_edge_aware = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=base_hold,
        output_smoothing=OutputSmoothingConfig(
            enabled=True,
            activity_window_ms=500,
            tau_active_s=0.25,
            tau_extreme_s=0.75,
            tau_hold_s=5.0,
            activity_low_ratio_per_sec=0.10,
            activity_high_ratio_per_sec=0.50,
            activity_floor_per_sec=0.01,
            edge_margin_ratio=0.10,
        ),
    )
    cfg_edge_disabled = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=base_hold,
        output_smoothing=OutputSmoothingConfig(
            enabled=True,
            activity_window_ms=500,
            tau_active_s=0.25,
            tau_extreme_s=5.0,
            tau_hold_s=5.0,
            activity_low_ratio_per_sec=0.10,
            activity_high_ratio_per_sec=0.50,
            activity_floor_per_sec=0.01,
            edge_margin_ratio=0.10,
        ),
    )
    calibration_values = _make_breathing_values(cfg_edge_aware.calibration_target_samples, amplitude=25.0)
    runtime_values = _make_breathing_values(800, amplitude=8.0)

    edge_samples, edge_state = _replay(
        np.concatenate([calibration_values, runtime_values]),
        cfg_edge_aware,
    )
    disabled_samples, _ = _replay(
        np.concatenate([calibration_values, runtime_values]),
        cfg_edge_disabled,
    )

    edge_normalized = np.array(
        [sample.normalized_value for sample in edge_samples if sample.stage == "runtime"],
        dtype=float,
    )
    disabled_normalized = np.array(
        [sample.normalized_value for sample in disabled_samples if sample.stage == "runtime"],
        dtype=float,
    )
    candidate = _mapped_candidate_levels(edge_samples, edge_state)

    assert np.min(candidate) > cfg_edge_aware.output_smoothing.edge_margin_ratio
    assert np.max(candidate) < 1.0 - cfg_edge_aware.output_smoothing.edge_margin_ratio
    assert np.allclose(edge_normalized, disabled_normalized)


def test_pipeline_output_smoothing_can_be_disabled() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=HoldConfig(
            enabled=False,
            activity_window_ms=100,
            ratio_per_sec_enter=0.2,
            ratio_per_sec_exit=0.4,
            floor_per_sec=0.01,
            edge_margin_ratio=0.20,
        ),
        output_smoothing=OutputSmoothingConfig(
            enabled=False,
            activity_window_ms=100,
            tau_active_s=0.25,
            tau_hold_s=5.0,
            activity_low_ratio_per_sec=0.10,
            activity_high_ratio_per_sec=0.50,
            activity_floor_per_sec=0.01,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0)
    runtime_values = _make_breathing_values(600, amplitude=25.0)
    samples, state = _replay(np.concatenate([calibration_values, runtime_values]), cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    normalized = np.array(
        [sample.normalized_value for sample in runtime_samples if sample.normalized_value is not None],
        dtype=float,
    )
    candidate = _mapped_candidate_levels(samples, state)

    assert np.allclose(normalized, candidate)


def test_pipeline_uses_padding_headroom_before_clipping() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        output_smoothing=OutputSmoothingConfig(
            enabled=False,
            activity_window_ms=100,
            tau_active_s=0.25,
            tau_hold_s=5.0,
            activity_low_ratio_per_sec=0.10,
            activity_high_ratio_per_sec=0.50,
            activity_floor_per_sec=0.01,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0)
    runtime_values = _make_breathing_values(800, amplitude=26.0)
    samples, state = _replay(np.concatenate([calibration_values, runtime_values]), cfg)

    assert state.calibration_result is not None
    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    in_headroom = [
        sample
        for sample in runtime_samples
        if state.calibration_result.global_max < sample.filtered_value < state.calibration_result.y_max
    ]

    assert in_headroom
    assert all(sample.normalized_value is not None for sample in in_headroom)
    assert all(float(sample.normalized_value) < 1.0 for sample in in_headroom)
    assert max(float(sample.normalized_value) for sample in in_headroom) > (1.0 - cfg.hold.edge_margin_ratio)


def test_pipeline_output_remains_bounded() -> None:
    cfg = _make_pipeline_config()
    values = np.concatenate(
        [
            _make_breathing_values(cfg.calibration_target_samples, amplitude=15.0),
            _make_breathing_values(300, amplitude=120.0),
        ]
    )
    samples, _ = _replay(values, cfg)

    normalized = np.array(
        [sample.normalized_value for sample in samples if sample.normalized_value is not None],
        dtype=float,
    )
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)


def test_pipeline_selected_sensor_column_is_processed() -> None:
    cfg = _make_pipeline_config(processed_sensor_column=6)
    state = create_pipeline_state(cfg)

    sample, _ = process_device_row(
        _make_row(750.0, alternate_value=100.0, processed_sensor_column=6),
        state,
        cfg,
    )

    assert np.isclose(sample.selected_sensor_raw, 750.0)


def test_pipeline_invert_signal_flips_control_direction() -> None:
    normal_cfg = _make_pipeline_config(invert_signal=False)
    inverted_cfg = _make_pipeline_config(invert_signal=True)
    values = np.concatenate(
        [
            _make_breathing_values(normal_cfg.calibration_target_samples, amplitude=20.0),
            _make_breathing_values(120, amplitude=20.0),
        ]
    )

    normal_samples, _ = _replay(values, normal_cfg)
    inverted_samples, _ = _replay(values, inverted_cfg)
    normal_runtime = next(sample for sample in normal_samples if sample.stage == "runtime")
    inverted_runtime = next(sample for sample in inverted_samples if sample.stage == "runtime")

    assert normal_runtime.normalized_value is not None
    assert inverted_runtime.normalized_value is not None
    assert np.isclose(
        float(normal_runtime.normalized_value) + float(inverted_runtime.normalized_value),
        1.0,
        atol=0.05,
    )


def test_pipeline_movement_mode_outputs_centered_unclamped_signal() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        processing_mode="movement",
    )
    values = np.concatenate(
        [
            _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0),
            _make_breathing_values(500, amplitude=25.0),
        ]
    )

    samples, state = _replay(values, cfg)

    calibration_filtered = np.array(
        [sample.cleaned_value for sample in samples[: cfg.calibration_target_samples]],
        dtype=float,
    )
    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    movement_values = np.array(
        [sample.movement_value for sample in runtime_samples if sample.movement_value is not None],
        dtype=float,
    )

    assert state.calibration_result is not None
    assert np.isclose(
        state.calibration_result.center,
        float(np.median(calibration_filtered)),
        atol=0.5,
    )
    assert runtime_samples[0].normalized_value is None
    assert runtime_samples[0].movement_value is not None
    assert all(not sample.hold_mode_active for sample in runtime_samples)
    assert float(np.max(movement_values)) > 1.0
    assert float(np.min(movement_values)) < -1.0


def test_pipeline_movement_mode_invert_signal_flips_movement_direction() -> None:
    normal_cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        processing_mode="movement",
        invert_signal=False,
    )
    inverted_cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        processing_mode="movement",
        invert_signal=True,
    )
    values = np.concatenate(
        [
            _make_breathing_values(normal_cfg.calibration_target_samples, amplitude=20.0),
            _make_breathing_values(400, amplitude=25.0),
        ]
    )

    normal_samples, _ = _replay(values, normal_cfg)
    inverted_samples, _ = _replay(values, inverted_cfg)
    normal_movement = np.array(
        [sample.movement_value for sample in normal_samples if sample.movement_value is not None],
        dtype=float,
    )
    inverted_movement = np.array(
        [sample.movement_value for sample in inverted_samples if sample.movement_value is not None],
        dtype=float,
    )

    compare_slice = slice(50, None)
    assert np.allclose(
        normal_movement[compare_slice],
        -inverted_movement[compare_slice],
        atol=0.75,
    )


def test_pipeline_adaptive_mode_outputs_bounded_control_and_centered_movement() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        processing_mode="adaptive",
        adaptation=AdaptationSettings(
            center_enabled=True,
            amplitude_enabled=True,
            center_tau_s=8.0,
            amplitude_tau_s=1.0,
            startup_duration_s=0.5,
            startup_center_tau_s=0.2,
            startup_amplitude_tau_s=0.2,
        ),
    )
    values = np.concatenate(
        [
            _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0),
            _make_breathing_values(500, amplitude=25.0),
        ]
    )

    samples, state = _replay(values, cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    normalized = np.array(
        [sample.normalized_value for sample in runtime_samples if sample.normalized_value is not None],
        dtype=float,
    )
    movement = np.array(
        [sample.movement_value for sample in runtime_samples if sample.movement_value is not None],
        dtype=float,
    )

    assert state.calibration_result is not None
    assert state.adaptive_state is not None
    assert runtime_samples[0].normalized_value is not None
    assert runtime_samples[0].movement_value is not None
    assert all(not sample.hold_mode_active for sample in runtime_samples)
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    assert float(np.min(movement)) < 0.0
    assert float(np.max(movement)) > 0.0


def test_pipeline_adaptive_mode_updates_amplitude_for_changed_breathing_range() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        processing_mode="adaptive",
        adaptation=AdaptationSettings(
            center_enabled=True,
            amplitude_enabled=True,
            center_tau_s=10.0,
            amplitude_tau_s=0.8,
            startup_duration_s=0.5,
            startup_center_tau_s=0.2,
            startup_amplitude_tau_s=0.2,
        ),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=18.0)
    runtime_low = _make_breathing_values(120, amplitude=18.0)
    runtime_high = _make_breathing_values(320, amplitude=60.0)
    samples, state = _replay(
        np.concatenate([calibration_values, runtime_low, runtime_high]),
        cfg,
    )

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    amplitudes = np.array(
        [sample.adaptive_amplitude for sample in runtime_samples if sample.adaptive_amplitude is not None],
        dtype=float,
    )

    assert state.adaptive_state is not None
    assert amplitudes.size > 0
    assert float(np.median(amplitudes[-60:])) > float(np.median(amplitudes[:60])) * 1.3


def test_pipeline_adaptive_mode_invert_signal_flips_centered_movement_direction() -> None:
    normal_cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        processing_mode="adaptive",
        invert_signal=False,
    )
    inverted_cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        processing_mode="adaptive",
        invert_signal=True,
    )
    values = np.concatenate(
        [
            _make_breathing_values(normal_cfg.calibration_target_samples, amplitude=20.0),
            _make_breathing_values(400, amplitude=25.0),
        ]
    )

    normal_samples, _ = _replay(values, normal_cfg)
    inverted_samples, _ = _replay(values, inverted_cfg)
    normal_movement = np.array(
        [sample.movement_value for sample in normal_samples if sample.movement_value is not None],
        dtype=float,
    )
    inverted_movement = np.array(
        [sample.movement_value for sample in inverted_samples if sample.movement_value is not None],
        dtype=float,
    )

    compare_slice = slice(50, None)
    assert np.allclose(
        normal_movement[compare_slice],
        -inverted_movement[compare_slice],
        atol=0.75,
    )


def test_pipeline_emits_inhale_and_exhale_events_for_breath_cycles() -> None:
    cfg = _make_pipeline_config(extrema=ExtremaConfig(min_interval_ms=600, prominence_ratio=0.05))
    values = np.concatenate(
        [
            _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0),
            _make_breathing_values(800, amplitude=25.0),
        ]
    )
    samples, _ = _replay(values, cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    labels = [sample.extrema_event_label for sample in runtime_samples if sample.extrema_event_label is not None]

    assert "inhale_peak" in labels
    assert "exhale_trough" in labels
    assert abs(labels.count("inhale_peak") - labels.count("exhale_trough")) <= 1


def test_pipeline_movement_mode_emits_inhale_and_exhale_events_for_breath_cycles() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        processing_mode="movement",
        extrema=ExtremaConfig(min_interval_ms=600, prominence_ratio=0.05),
    )
    values = np.concatenate(
        [
            _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0),
            _make_breathing_values(800, amplitude=25.0),
        ]
    )
    samples, _ = _replay(values, cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    labels = [sample.extrema_event_label for sample in runtime_samples if sample.extrema_event_label is not None]

    assert "inhale_peak" in labels
    assert "exhale_trough" in labels


def test_pipeline_adaptive_mode_emits_inhale_and_exhale_events_for_breath_cycles() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        processing_mode="adaptive",
        extrema=ExtremaConfig(min_interval_ms=600, prominence_ratio=0.05),
    )
    values = np.concatenate(
        [
            _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0),
            _make_breathing_values(800, amplitude=25.0),
        ]
    )
    samples, _ = _replay(values, cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    labels = [sample.extrema_event_label for sample in runtime_samples if sample.extrema_event_label is not None]

    assert "inhale_peak" in labels
    assert "exhale_trough" in labels


def test_pipeline_extrema_events_match_with_or_without_output_smoothing() -> None:
    base_hold = HoldConfig(
        enabled=False,
        activity_window_ms=100,
        ratio_per_sec_enter=0.2,
        ratio_per_sec_exit=0.4,
        floor_per_sec=0.01,
        edge_margin_ratio=0.20,
    )
    cfg_smoothed = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=base_hold,
        extrema=ExtremaConfig(min_interval_ms=600, prominence_ratio=0.05),
    )
    cfg_unsmoothed = _make_pipeline_config(
        calibration_duration_s=5.0,
        hold=base_hold,
        output_smoothing=OutputSmoothingConfig(
            enabled=False,
            activity_window_ms=100,
            tau_active_s=0.25,
            tau_hold_s=5.0,
            activity_low_ratio_per_sec=0.10,
            activity_high_ratio_per_sec=0.50,
            activity_floor_per_sec=0.01,
        ),
        extrema=ExtremaConfig(min_interval_ms=600, prominence_ratio=0.05),
    )
    values = np.concatenate(
        [
            _make_breathing_values(cfg_smoothed.calibration_target_samples, amplitude=20.0),
            _make_breathing_values(800, amplitude=25.0),
        ]
    )

    smoothed_samples, _ = _replay(values, cfg_smoothed)
    unsmoothed_samples, _ = _replay(values, cfg_unsmoothed)
    smoothed_events = [
        (sample.sample_index, sample.extrema_event_label)
        for sample in smoothed_samples
        if sample.stage == "runtime" and sample.extrema_event_label is not None
    ]
    unsmoothed_events = [
        (sample.sample_index, sample.extrema_event_label)
        for sample in unsmoothed_samples
        if sample.stage == "runtime" and sample.extrema_event_label is not None
    ]

    assert smoothed_events == unsmoothed_events


def test_pipeline_rejects_low_prominence_noise_as_extrema() -> None:
    cfg = _make_pipeline_config(
        calibration_duration_s=5.0,
        extrema=ExtremaConfig(min_interval_ms=400, prominence_ratio=0.2),
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0)
    state = create_pipeline_state(cfg)
    for value in calibration_values:
        _, state = process_device_row(
            _make_row(float(value), processed_sensor_column=cfg.processed_sensor_column),
            state,
            cfg,
        )

    assert state.calibration_result is not None
    assert state.adaptive_state is not None

    state.filter_initialized = False
    state.sos_lp = None
    state.zi_lp = None
    state.previous_filtered_value = None
    state.previous_cleaned_value = None
    state.previous_delta_sign = 0
    state.last_event_sample_index = None
    state.last_peak_value = None
    state.last_trough_value = None
    state.hold_mode_active = False
    state.frozen_normalized_value = None
    state.recent_abs_velocity.clear()

    t_runtime = np.arange(500, dtype=float) / FS_HZ
    tiny_runtime = 512.0 + 1.0 * np.sin(2.0 * np.pi * 0.22 * t_runtime)
    samples = []
    for value in tiny_runtime:
        sample, state = process_device_row(
            _make_row(float(value), processed_sensor_column=cfg.processed_sensor_column),
            state,
            cfg,
        )
        samples.append(sample)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    assert all(sample.extrema_event_code == 0.0 for sample in runtime_samples)


def test_pipeline_raw_qc_reports_saturation_flatline_and_baseline_shift() -> None:
    cfg = _make_pipeline_config(
        raw_qc=RawQCConfig(
            enabled=True,
            raw_saturation_lo=1.0,
            raw_saturation_hi=1022.0,
            flatline_epsilon=0.1,
            flatline_duration_s=0.2,
            baseline_ema_tau_s=1.0,
            baseline_abs_dev_tau_s=0.5,
            baseline_shift_sigma=3.0,
            baseline_shift_floor=10.0,
            warmup_s=0.1,
        )
    )
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=10.0)
    runtime_values = np.concatenate(
        [
            np.full(30, 512.0),
            np.full(30, 1023.0),
            np.full(30, 512.0),
            np.full(30, 700.0),
        ]
    )
    samples, state = _replay(np.concatenate([calibration_values, runtime_values]), cfg)

    summary = raw_qc_summary(state.qc_state)
    event_types = {event.event_type for sample in samples for event in sample.qc_events}
    assert {"saturation", "flatline", "baseline_shift"} <= event_types
    assert summary["event_counts"]["saturation"] >= 1
    assert summary["event_counts"]["flatline"] >= 1
    assert summary["event_counts"]["baseline_shift"] >= 1
