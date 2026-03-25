"""Pure live-processing pipeline for breathing-belt device rows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from .calibration import (
    AdaptiveRangeConfig,
    AdaptiveRangeState,
    CalibrationConfig,
    CalibrationResult,
    run_range_calibration,
)
from .preprocessing import get_low_pass_filter_coeffs, low_pass_filter_sample
from .quality import RawQCEvent, RawQCState, create_raw_qc_state, update_raw_qc
from .settings import (
    AdaptationSettings,
    ArtifactConfig,
    CalibrationSettings,
    ExtremaConfig,
    FilterConfig,
    HoldConfig,
    OutputSmoothingConfig,
    RawQCConfig,
)


_HOLD_RELEASE_DRIFT = 0.03


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for deterministic sample-by-sample live processing."""

    sampling_rate_hz: int
    processed_sensor_column: int
    invert_signal: bool
    filter: FilterConfig
    artifact: ArtifactConfig
    calibration: CalibrationSettings
    adaptation: AdaptationSettings
    hold: HoldConfig
    output_smoothing: OutputSmoothingConfig
    extrema: ExtremaConfig
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
            padding_ratio=self.calibration.padding_ratio,
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

    @property
    def output_smoothing_activity_window_samples(self) -> int:
        return max(
            3,
            int(
                round(
                    (self.output_smoothing.activity_window_ms / 1000.0)
                    * self.sampling_rate_hz
                )
            ),
        )

    @property
    def extrema_min_interval_samples(self) -> int:
        return max(1, int(round((self.extrema.min_interval_ms / 1000.0) * self.sampling_rate_hz)))


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
    extrema_event_code: float = 0.0
    extrema_event_label: str | None = None
    messages: tuple[str, ...] = ()
    qc_events: tuple[RawQCEvent, ...] = ()


@dataclass
class PipelineState:
    """Mutable state for the live-processing pipeline."""

    recent_raw_deltas: deque[float]
    recent_abs_velocity: deque[float]
    recent_output_abs_velocity: deque[float]
    qc_state: RawQCState
    filter_initialized: bool = False
    sos_hp: np.ndarray | None = None
    zi_hp: np.ndarray | None = None
    sos_lp: np.ndarray | None = None
    zi_lp: np.ndarray | None = None
    previous_filtered_value: float | None = None
    previous_cleaned_value: float | None = None
    hold_mode_active: bool = False
    frozen_normalized_value: float | None = None
    emitted_normalized_value: float | None = None
    calibration_samples: list[float] = field(default_factory=list)
    calibration_result: CalibrationResult | None = None
    adaptive_state: AdaptiveRangeState | None = None
    runtime_processed_samples: int = 0
    startup_mode_active: bool = False
    calibration_last_reported_sec: int = -1
    stage_sample_index: int = 0
    previous_delta_sign: int = 0
    last_event_sample_index: int | None = None
    last_peak_value: float | None = None
    last_trough_value: float | None = None

    @property
    def stage(self) -> str:
        return "calibration" if self.calibration_result is None else "runtime"


def create_pipeline_state(cfg: PipelineConfig) -> PipelineState:
    """Create a pipeline state object with config-dependent buffer sizes."""

    return PipelineState(
        recent_raw_deltas=deque(maxlen=max(cfg.artifact.artifact_window, 3)),
        recent_abs_velocity=deque(maxlen=cfg.hold_activity_window_samples),
        recent_output_abs_velocity=deque(maxlen=cfg.output_smoothing_activity_window_samples),
        qc_state=create_raw_qc_state(),
    )


def process_device_row(
    device_row: np.ndarray,
    state: PipelineState,
    cfg: PipelineConfig,
) -> tuple[PipelineSample, PipelineState]:
    """Process one BITalino row into a breathing-control sample."""

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
    extrema_event_code = 0.0
    extrema_event_label: str | None = None

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
            state.adaptive_state = _build_fixed_control_state(
                state.calibration_samples,
                state.calibration_result,
                cfg.calibration.amplitude_floor,
            )
            adaptive_center = float(state.adaptive_state.center)
            adaptive_amplitude = float(state.adaptive_state.amplitude)
            messages.extend(
                [
                    "Calibration complete.",
                    (
                        "Fixed control map: "
                        f"center={state.calibration_result.center:.6f}, "
                        f"amplitude={state.calibration_result.amplitude:.6f}, "
                        f"min={state.calibration_result.global_min:.6f}, "
                        f"max={state.calibration_result.global_max:.6f}, "
                        f"control_min={state.calibration_result.y_min:.6f}, "
                        f"control_max={state.calibration_result.y_max:.6f}"
                    ),
                ]
            )
            state.recent_abs_velocity.clear()
            state.recent_output_abs_velocity.clear()
            state.recent_raw_deltas.clear()
            state.previous_filtered_value = None
            state.previous_cleaned_value = None
            state.hold_mode_active = False
            state.frozen_normalized_value = None
            state.emitted_normalized_value = None
            state.runtime_processed_samples = 0
            state.startup_mode_active = False
            state.previous_delta_sign = 0
            state.last_event_sample_index = None
            state.last_peak_value = None
            state.last_trough_value = None
            state.stage_sample_index = 0
        else:
            state.stage_sample_index += 1
    else:
        normalized_value = _normalize_runtime_sample(cleaned_value, is_artifact, state, cfg)
        extrema_event_code, extrema_event_label = _detect_runtime_extremum(
            cleaned_value,
            sample_index,
            state,
            cfg,
        )
        adaptive_center = float(state.adaptive_state.center) if state.adaptive_state else None
        adaptive_amplitude = float(state.adaptive_state.amplitude) if state.adaptive_state else None
        state.runtime_processed_samples += 1
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
        extrema_event_code=extrema_event_code if stage == "runtime" else 0.0,
        extrema_event_label=extrema_event_label if stage == "runtime" else None,
        messages=tuple(messages),
        qc_events=tuple(qc_events),
    )
    return sample, state


def _filter_sample(raw_sensor_value: float, state: PipelineState, cfg: PipelineConfig) -> float:
    control_input = -raw_sensor_value if cfg.invert_signal else raw_sensor_value
    if not state.filter_initialized:
        state.sos_lp, state.zi_lp = get_low_pass_filter_coeffs(
            cfg.filter.lp_cutoff_hz,
            cfg.sampling_rate_hz,
            cfg.filter.lp_order,
            initial_value=control_input,
        )
        state.filter_initialized = True

    low_passed_value, state.zi_lp = low_pass_filter_sample(
        control_input,
        state.sos_lp,
        state.zi_lp,
    )
    return float(low_passed_value)


def _suppress_reversal(
    filtered_value: float,
    state: PipelineState,
    cfg: PipelineConfig,
) -> tuple[float, bool]:
    del state
    del cfg
    return float(filtered_value), False


def _normalize_runtime_sample(
    cleaned_value: float,
    is_artifact: bool,
    state: PipelineState,
    cfg: PipelineConfig,
) -> float:
    del is_artifact
    if state.adaptive_state is None or state.calibration_result is None:
        raise RuntimeError("Calibration state must be initialized before runtime normalization.")

    value_float = float(cleaned_value)
    previous_value = state.previous_cleaned_value
    if previous_value is None:
        abs_velocity = 0.0
    else:
        abs_velocity = abs(value_float - previous_value) * cfg.sampling_rate_hz
    state.recent_abs_velocity.append(abs_velocity)
    state.recent_output_abs_velocity.append(abs_velocity)
    state.previous_cleaned_value = value_float

    if len(state.recent_abs_velocity) < state.recent_abs_velocity.maxlen:
        activity_value = float("inf")
    else:
        activity_value = float(np.mean(state.recent_abs_velocity))

    amplitude = max(state.calibration_result.amplitude, cfg.calibration.amplitude_floor)
    enter_threshold = max(
        cfg.hold.floor_per_sec,
        amplitude * cfg.hold.ratio_per_sec_enter,
    )
    exit_threshold = max(
        cfg.hold.floor_per_sec,
        amplitude * cfg.hold.ratio_per_sec_exit,
    )

    normalized_candidate = _map_control_level(value_float, state.calibration_result)

    post_hold_level = normalized_candidate
    if not cfg.hold.enabled:
        state.hold_mode_active = False
        state.frozen_normalized_value = None
    else:
        if not state.hold_mode_active:
            if len(state.recent_abs_velocity) >= state.recent_abs_velocity.maxlen:
                state.hold_mode_active = (
                    activity_value < enter_threshold
                    and abs_velocity < enter_threshold
                    and _is_extrema_zone(normalized_candidate, cfg)
                )
                if state.hold_mode_active:
                    state.frozen_normalized_value = float(normalized_candidate)
        elif (
            abs_velocity > exit_threshold
            or (
                state.frozen_normalized_value is not None
                and abs(normalized_candidate - state.frozen_normalized_value)
                > _HOLD_RELEASE_DRIFT
            )
        ):
            state.hold_mode_active = False
            state.frozen_normalized_value = None

        if state.hold_mode_active:
            if state.frozen_normalized_value is None:
                state.frozen_normalized_value = float(normalized_candidate)
            post_hold_level = float(state.frozen_normalized_value)

    return _smooth_output_level(
        post_hold_level,
        state=state,
        cfg=cfg,
        candidate_level=normalized_candidate,
        amplitude=amplitude,
    )


def _map_control_level(
    value: float,
    calibration_result: CalibrationResult,
) -> float:
    control_min = float(calibration_result.y_min)
    control_max = float(calibration_result.y_max)
    if control_max <= control_min:
        raise RuntimeError("Control bounds must define a positive range.")

    mapped = (float(value) - control_min) / (control_max - control_min)
    return float(min(1.0, max(0.0, mapped)))


def _smooth_output_level(
    target_level: float,
    *,
    candidate_level: float,
    state: PipelineState,
    cfg: PipelineConfig,
    amplitude: float,
) -> float:
    if state.hold_mode_active and cfg.hold.enabled:
        state.emitted_normalized_value = float(target_level)
        return float(target_level)

    if not cfg.output_smoothing.enabled:
        state.emitted_normalized_value = float(target_level)
        return float(target_level)

    if state.emitted_normalized_value is None:
        state.emitted_normalized_value = float(target_level)
        return float(target_level)

    low_threshold = max(
        cfg.output_smoothing.activity_floor_per_sec,
        amplitude * cfg.output_smoothing.activity_low_ratio_per_sec,
    )
    high_threshold = max(
        cfg.output_smoothing.activity_floor_per_sec,
        amplitude * cfg.output_smoothing.activity_high_ratio_per_sec,
    )

    if len(state.recent_output_abs_velocity) < state.recent_output_abs_velocity.maxlen:
        activity_value = high_threshold
    else:
        activity_value = float(np.mean(state.recent_output_abs_velocity))

    activity_ratio = (activity_value - low_threshold) / (high_threshold - low_threshold)
    activity_ratio = float(min(1.0, max(0.0, activity_ratio)))
    base_tau_s = (
        cfg.output_smoothing.tau_hold_s
        + activity_ratio
        * (cfg.output_smoothing.tau_active_s - cfg.output_smoothing.tau_hold_s)
    )
    distance_to_edge = min(float(candidate_level), 1.0 - float(candidate_level))
    if distance_to_edge >= cfg.output_smoothing.edge_margin_ratio:
        edge_factor = 0.0
    else:
        edge_factor = 1.0 - (
            distance_to_edge / cfg.output_smoothing.edge_margin_ratio
        )
    extreme_tau_s = min(base_tau_s, cfg.output_smoothing.tau_extreme_s)
    tau_s = base_tau_s + edge_factor * (extreme_tau_s - base_tau_s)
    alpha = 1.0 - math.exp(-1.0 / (cfg.sampling_rate_hz * tau_s))

    state.emitted_normalized_value = (
        state.emitted_normalized_value
        + alpha * (float(target_level) - state.emitted_normalized_value)
    )
    return float(state.emitted_normalized_value)


def _is_extrema_zone(
    normalized_value: float,
    cfg: PipelineConfig,
) -> bool:
    edge_margin = cfg.hold.edge_margin_ratio
    return normalized_value <= edge_margin or normalized_value >= (1.0 - edge_margin)


def _detect_runtime_extremum(
    filtered_value: float,
    sample_index: int,
    state: PipelineState,
    cfg: PipelineConfig,
) -> tuple[float, str | None]:
    previous_value = state.previous_filtered_value
    state.previous_filtered_value = float(filtered_value)
    if previous_value is None:
        state.previous_delta_sign = 0
        return 0.0, None

    delta = float(filtered_value) - previous_value
    event_code = 0.0
    event_label: str | None = None
    candidate_index = max(sample_index - 1, 0)
    prominence_threshold = max(
        cfg.calibration.amplitude_floor,
        cfg.extrema.prominence_ratio * max(state.adaptive_state.amplitude, cfg.calibration.amplitude_floor),
    )

    if state.previous_delta_sign > 0 and delta <= 0.0:
        if _extremum_interval_elapsed(candidate_index, state, cfg):
            candidate_value = float(previous_value)
            reference_value = (
                state.last_trough_value
                if state.last_trough_value is not None
                else state.adaptive_state.center
            )
            if (
                candidate_value - reference_value >= prominence_threshold
            ):
                event_code = 1.0
                event_label = "inhale_peak"
                state.last_event_sample_index = candidate_index
                state.last_peak_value = candidate_value
    elif state.previous_delta_sign < 0 and delta >= 0.0:
        if _extremum_interval_elapsed(candidate_index, state, cfg):
            candidate_value = float(previous_value)
            reference_value = (
                state.last_peak_value
                if state.last_peak_value is not None
                else state.adaptive_state.center
            )
            if (
                reference_value - candidate_value >= prominence_threshold
            ):
                event_code = -1.0
                event_label = "exhale_trough"
                state.last_event_sample_index = candidate_index
                state.last_trough_value = candidate_value

    if delta > 0.0:
        state.previous_delta_sign = 1
    elif delta < 0.0:
        state.previous_delta_sign = -1
    else:
        state.previous_delta_sign = 0
    return event_code, event_label


def _extremum_interval_elapsed(
    candidate_index: int,
    state: PipelineState,
    cfg: PipelineConfig,
) -> bool:
    if state.last_event_sample_index is None:
        return True
    return (candidate_index - state.last_event_sample_index) >= cfg.extrema_min_interval_samples


def _build_fixed_control_state(
    calibration_samples: list[float] | np.ndarray,
    calibration_result: CalibrationResult,
    amplitude_floor: float,
) -> AdaptiveRangeState:
    samples = np.asarray(calibration_samples, dtype=float).reshape(-1)
    center = float(calibration_result.center)
    amplitude = max(float(calibration_result.amplitude), float(amplitude_floor))
    abs_dev_ema = float(np.mean(np.abs(samples - center)))
    abs_dev_to_amplitude_scale = amplitude / max(abs_dev_ema, 1e-12)
    return AdaptiveRangeState(
        center=center,
        amplitude=amplitude,
        abs_dev_ema=abs_dev_ema,
        abs_dev_to_amplitude_scale=abs_dev_to_amplitude_scale,
    )
