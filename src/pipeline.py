"""Pure live-processing pipeline for breathing-belt device rows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .calibration import (
    AdaptiveRangeConfig,
    AdaptiveRangeState,
    CalibrationConfig,
    CalibrationResult,
    initialize_adaptive_range,
    run_range_calibration,
    update_adaptive_range,
)
from .preprocessing import (
    get_high_pass_filter_coeffs,
    get_low_pass_filter_coeffs,
    high_pass_filter_sample,
    low_pass_filter_sample,
)
from .quality import RawQCEvent, RawQCState, create_raw_qc_state, update_raw_qc
from .settings import (
    AdaptationSettings,
    ArtifactConfig,
    CalibrationSettings,
    FilterConfig,
    HoldConfig,
    RawQCConfig,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for deterministic sample-by-sample live processing."""

    sampling_rate_hz: int
    processed_sensor_column: int
    filter: FilterConfig
    artifact: ArtifactConfig
    calibration: CalibrationSettings
    adaptation: AdaptationSettings
    hold: HoldConfig
    raw_qc: RawQCConfig

    @property
    def calibration_cfg(self) -> CalibrationConfig:
        return CalibrationConfig(
            duration_s=self.calibration.duration_s,
            fs_hz=float(self.sampling_rate_hz),
            percentile_lo=self.calibration.percentile_lo,
            percentile_hi=self.calibration.percentile_hi,
            saturation_lo=float("-inf"),
            saturation_hi=float("inf"),
            amplitude_floor=self.calibration.amplitude_floor,
        )

    @property
    def adaptive_cfg_startup(self) -> AdaptiveRangeConfig:
        return AdaptiveRangeConfig(
            fs_hz=float(self.sampling_rate_hz),
            center_tau_s=self.adaptation.startup_center_tau_s,
            amplitude_tau_s=self.adaptation.startup_amplitude_tau_s,
            amplitude_floor=self.calibration.amplitude_floor,
        )

    @property
    def adaptive_cfg_runtime(self) -> AdaptiveRangeConfig:
        return AdaptiveRangeConfig(
            fs_hz=float(self.sampling_rate_hz),
            center_tau_s=self.adaptation.center_tau_s,
            amplitude_tau_s=self.adaptation.amplitude_tau_s,
            amplitude_floor=self.calibration.amplitude_floor,
        )

    @property
    def calibration_target_samples(self) -> int:
        return max(1, int(round(self.calibration.duration_s * self.sampling_rate_hz)))

    @property
    def startup_target_samples(self) -> int:
        return max(1, int(round(self.adaptation.startup_duration_s * self.sampling_rate_hz)))

    @property
    def hold_activity_window_samples(self) -> int:
        return max(3, int(round((self.hold.activity_window_ms / 1000.0) * self.sampling_rate_hz)))


@dataclass(frozen=True)
class PipelineSample:
    """Fully processed information for one device row."""

    stage: str
    sample_index: int
    relative_time_s: float
    selected_sensor_raw: float
    filtered_value: float
    cleaned_value: float
    normalized_value: float | None
    is_artifact: bool
    hold_mode_active: bool
    adaptive_center: float | None
    adaptive_amplitude: float | None
    messages: tuple[str, ...] = ()
    qc_events: tuple[RawQCEvent, ...] = ()


@dataclass
class PipelineState:
    """Mutable state for the live-processing pipeline."""

    recent_raw_deltas: deque[float]
    recent_abs_velocity: deque[float]
    qc_state: RawQCState
    filter_initialized: bool = False
    sos_hp: np.ndarray | None = None
    zi_hp: np.ndarray | None = None
    sos_lp: np.ndarray | None = None
    zi_lp: np.ndarray | None = None
    previous_filtered_value: float | None = None
    previous_cleaned_value: float | None = None
    hold_mode_active: bool = False
    calibration_samples: list[float] = field(default_factory=list)
    calibration_result: CalibrationResult | None = None
    adaptive_state: AdaptiveRangeState | None = None
    runtime_processed_samples: int = 0
    startup_mode_active: bool = False
    calibration_last_reported_sec: int = -1
    stage_sample_index: int = 0

    @property
    def stage(self) -> str:
        return "calibration" if self.calibration_result is None else "runtime"


def create_pipeline_state(cfg: PipelineConfig) -> PipelineState:
    """Create a pipeline state object with config-dependent buffer sizes."""

    return PipelineState(
        recent_raw_deltas=deque(maxlen=max(cfg.artifact.artifact_window, 3)),
        recent_abs_velocity=deque(maxlen=cfg.hold_activity_window_samples),
        qc_state=create_raw_qc_state(),
    )


def process_device_row(
    device_row: np.ndarray,
    state: PipelineState,
    cfg: PipelineConfig,
) -> tuple[PipelineSample, PipelineState]:
    """Process one BITalino row into a normalized breathing sample."""

    stage = state.stage
    sample_index = state.stage_sample_index
    relative_time_s = sample_index / float(cfg.sampling_rate_hz)
    raw_sensor_value = float(device_row[cfg.processed_sensor_column])
    messages: list[str] = []

    filtered_value = _filter_sample(raw_sensor_value, state, cfg)
    cleaned_value, is_artifact = _suppress_reversal(filtered_value, state, cfg)
    qc_events, state.qc_state = update_raw_qc(
        raw_value=raw_sensor_value,
        stage=stage,
        sample_index=sample_index,
        relative_time_s=relative_time_s,
        state=state.qc_state,
        cfg=cfg.raw_qc,
        fs_hz=float(cfg.sampling_rate_hz),
    ) if cfg.raw_qc.enabled else ([], state.qc_state)

    normalized_value: float | None = None
    adaptive_center: float | None = None
    adaptive_amplitude: float | None = None

    if stage == "calibration":
        state.calibration_samples.append(cleaned_value)
        collected = len(state.calibration_samples)
        clipped_collected = min(collected, cfg.calibration_target_samples)
        reported_sec = int(clipped_collected / cfg.calibration_cfg.fs_hz)
        if reported_sec != state.calibration_last_reported_sec:
            state.calibration_last_reported_sec = reported_sec
            messages.append(
                f"Calibration progress: {clipped_collected}/{cfg.calibration_target_samples} samples"
            )

        if collected >= cfg.calibration_target_samples:
            state.calibration_samples = state.calibration_samples[: cfg.calibration_target_samples]
            state.calibration_result = run_range_calibration(
                state.calibration_samples,
                cfg.calibration_cfg,
            )
            state.adaptive_state = initialize_adaptive_range(
                state.calibration_samples,
                state.calibration_result,
                cfg.adaptive_cfg_startup,
            )
            adaptive_center = float(state.adaptive_state.center)
            adaptive_amplitude = float(state.adaptive_state.amplitude)
            messages.extend(
                [
                    "Calibration complete.",
                    (
                        "Calibration map: "
                        f"center={state.calibration_result.center:.6f}, "
                        f"amplitude={state.calibration_result.amplitude:.6f}, "
                        f"min={state.calibration_result.global_min:.6f}, "
                        f"max={state.calibration_result.global_max:.6f}"
                    ),
                    (
                        "Starting fast post-calibration adaptation for "
                        f"{cfg.adaptation.startup_duration_s:.1f}s."
                    ),
                ]
            )
            state.recent_abs_velocity.clear()
            state.recent_raw_deltas.clear()
            state.previous_cleaned_value = None
            state.previous_filtered_value = None
            state.hold_mode_active = False
            state.runtime_processed_samples = 0
            state.startup_mode_active = True
            state.stage_sample_index = 0
        else:
            state.stage_sample_index += 1
    else:
        normalized_value = _normalize_runtime_sample(cleaned_value, is_artifact, state, cfg)
        adaptive_center = float(state.adaptive_state.center) if state.adaptive_state else None
        adaptive_amplitude = float(state.adaptive_state.amplitude) if state.adaptive_state else None
        if state.startup_mode_active and state.runtime_processed_samples >= cfg.startup_target_samples:
            state.startup_mode_active = False
            messages.append(
                "Startup adaptation complete. Switched to slow runtime tracking."
            )
        state.stage_sample_index += 1

    sample = PipelineSample(
        stage=stage,
        sample_index=sample_index,
        relative_time_s=relative_time_s,
        selected_sensor_raw=raw_sensor_value,
        filtered_value=filtered_value,
        cleaned_value=cleaned_value,
        normalized_value=normalized_value,
        is_artifact=is_artifact,
        hold_mode_active=state.hold_mode_active if stage == "runtime" else False,
        adaptive_center=adaptive_center,
        adaptive_amplitude=adaptive_amplitude,
        messages=tuple(messages),
        qc_events=tuple(qc_events),
    )
    return sample, state


def _filter_sample(raw_sensor_value: float, state: PipelineState, cfg: PipelineConfig) -> float:
    if not state.filter_initialized:
        state.sos_hp, state.zi_hp = get_high_pass_filter_coeffs(
            cfg.filter.hp_cutoff_hz,
            cfg.sampling_rate_hz,
            cfg.filter.hp_order,
            initial_value=raw_sensor_value,
        )
        state.sos_lp, state.zi_lp = get_low_pass_filter_coeffs(
            cfg.filter.lp_cutoff_hz,
            cfg.sampling_rate_hz,
            cfg.filter.lp_order,
            initial_value=0.0,
        )
        state.filter_initialized = True

    high_passed_value, state.zi_hp = high_pass_filter_sample(
        raw_sensor_value,
        state.sos_hp,
        state.zi_hp,
    )
    low_passed_value, state.zi_lp = low_pass_filter_sample(
        high_passed_value,
        state.sos_lp,
        state.zi_lp,
    )
    return -float(low_passed_value)


def _suppress_reversal(
    filtered_value: float,
    state: PipelineState,
    cfg: PipelineConfig,
) -> tuple[float, bool]:
    sample_value = float(filtered_value)
    if state.previous_filtered_value is None:
        state.previous_filtered_value = sample_value
        state.recent_raw_deltas.clear()
        return sample_value, False

    raw_delta = sample_value - state.previous_filtered_value
    state.recent_raw_deltas.append(raw_delta)

    if len(state.recent_raw_deltas) >= 3:
        trend_value = float(np.mean(state.recent_raw_deltas))
        delta_scale = max(1e-3, float(np.std(state.recent_raw_deltas)))
    else:
        trend_value = raw_delta
        delta_scale = max(1e-3, abs(raw_delta))

    is_reversal = (
        abs(raw_delta) > cfg.artifact.spike_threshold * delta_scale
        and abs(trend_value) > 0.25 * delta_scale
        and np.sign(raw_delta) != np.sign(trend_value)
    )

    state.previous_filtered_value = sample_value
    cleaned_value = sample_value - raw_delta if is_reversal else sample_value
    return float(cleaned_value), bool(is_reversal)


def _normalize_runtime_sample(
    cleaned_value: float,
    is_artifact: bool,
    state: PipelineState,
    cfg: PipelineConfig,
) -> float:
    if state.adaptive_state is None:
        raise RuntimeError("Adaptive state must be initialized before runtime normalization.")

    value_float = float(cleaned_value)
    if state.previous_cleaned_value is None:
        abs_velocity = 0.0
    else:
        abs_velocity = abs(value_float - state.previous_cleaned_value) * cfg.sampling_rate_hz
    state.previous_cleaned_value = value_float
    state.recent_abs_velocity.append(abs_velocity)

    if len(state.recent_abs_velocity) < state.recent_abs_velocity.maxlen:
        activity_value = float("inf")
    else:
        activity_value = float(np.mean(state.recent_abs_velocity))

    enter_threshold = max(
        cfg.hold.floor_per_sec,
        state.adaptive_state.amplitude * cfg.hold.ratio_per_sec_enter,
    )
    exit_threshold = max(
        cfg.hold.floor_per_sec,
        state.adaptive_state.amplitude * cfg.hold.ratio_per_sec_exit,
    )

    if not state.hold_mode_active:
        if len(state.recent_abs_velocity) >= state.recent_abs_velocity.maxlen:
            state.hold_mode_active = activity_value < enter_threshold
    elif activity_value > exit_threshold:
        state.hold_mode_active = False

    allow_center_update = (
        cfg.adaptation.center_enabled and (not is_artifact) and (not state.hold_mode_active)
    )
    allow_amplitude_update = (
        cfg.adaptation.amplitude_enabled and (not is_artifact) and (not state.hold_mode_active)
    )
    active_cfg = cfg.adaptive_cfg_startup if state.startup_mode_active else cfg.adaptive_cfg_runtime
    normalized_value, state.adaptive_state = update_adaptive_range(
        x=value_float,
        state=state.adaptive_state,
        cfg=active_cfg,
        allow_update=allow_center_update or allow_amplitude_update,
        allow_center_update=allow_center_update,
        allow_amplitude_update=allow_amplitude_update,
    )
    state.runtime_processed_samples += 1
    return float(normalized_value)
