"""End-to-end replay tests for the live breathing-belt pipeline."""

from __future__ import annotations

import numpy as np

from src.pipeline import (
    PipelineConfig,
    _normalize_runtime_sample,
    create_pipeline_state,
    process_device_row,
)
from src.quality import raw_qc_summary
from src.settings import (
    AdaptationSettings,
    ArtifactConfig,
    CalibrationSettings,
    FilterConfig,
    HoldConfig,
    RawQCConfig,
)


FS_HZ = 100


def _make_pipeline_config(
    *,
    processed_sensor_column: int = 5,
    calibration_duration_s: float = 0.2,
    adaptation: AdaptationSettings | None = None,
    hold: HoldConfig | None = None,
    raw_qc: RawQCConfig | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        sampling_rate_hz=FS_HZ,
        processed_sensor_column=processed_sensor_column,
        filter=FilterConfig(hp_cutoff_hz=0.005, hp_order=1, lp_cutoff_hz=1.5, lp_order=2),
        artifact=ArtifactConfig(spike_threshold=2.5, artifact_window=10),
        calibration=CalibrationSettings(
            duration_s=calibration_duration_s,
            percentile_lo=5.0,
            percentile_hi=95.0,
            amplitude_floor=1e-3,
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
        hold=hold or HoldConfig(activity_window_ms=100, ratio_per_sec_enter=0.2, ratio_per_sec_exit=0.4, floor_per_sec=0.01),
        raw_qc=raw_qc or RawQCConfig(),
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


def test_pipeline_amplitude_adapts_after_step_change() -> None:
    cfg = _make_pipeline_config()
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0)
    runtime_low = _make_breathing_values(120, amplitude=20.0)
    runtime_high = _make_breathing_values(240, amplitude=80.0)
    samples, state = _replay(
        np.concatenate([calibration_values, runtime_low, runtime_high]),
        cfg,
    )

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    initial_amplitude = runtime_samples[0].adaptive_amplitude
    assert initial_amplitude is not None
    assert state.adaptive_state is not None
    assert state.adaptive_state.amplitude > initial_amplitude * 1.5


def test_pipeline_hold_mode_freezes_adaptation() -> None:
    cfg = _make_pipeline_config()
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=25.0)
    hold_values = np.full(250, 512.0, dtype=float)
    samples, state = _replay(np.concatenate([calibration_values, hold_values]), cfg)

    runtime_samples = [sample for sample in samples if sample.stage == "runtime"]
    hold_start = next(idx for idx, sample in enumerate(runtime_samples) if sample.hold_mode_active)
    amplitude_at_hold = runtime_samples[hold_start].adaptive_amplitude
    amplitude_end = runtime_samples[-1].adaptive_amplitude

    assert runtime_samples[hold_start].hold_mode_active is True
    assert amplitude_at_hold is not None
    assert amplitude_end is not None
    assert np.isclose(amplitude_end, amplitude_at_hold)
    assert state.hold_mode_active is True


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
    event_types = {
        event.event_type
        for sample in samples
        for event in sample.qc_events
    }
    assert {"saturation", "flatline", "baseline_shift"} <= event_types
    assert summary["event_counts"]["saturation"] >= 1
    assert summary["event_counts"]["flatline"] >= 1
    assert summary["event_counts"]["baseline_shift"] >= 1


def test_runtime_artifact_gating_freezes_amplitude_update() -> None:
    cfg = _make_pipeline_config()
    state = create_pipeline_state(cfg)
    calibration_values = _make_breathing_values(cfg.calibration_target_samples, amplitude=20.0)
    for value in calibration_values:
        _, state = process_device_row(
            _make_row(float(value), processed_sensor_column=cfg.processed_sensor_column),
            state,
            cfg,
        )

    assert state.calibration_result is not None
    assert state.adaptive_state is not None

    amplitude_before = state.adaptive_state.amplitude
    _normalize_runtime_sample(cleaned_value=25.0, is_artifact=True, state=state, cfg=cfg)
    amplitude_after = state.adaptive_state.amplitude

    assert np.isclose(amplitude_after, amplitude_before)
