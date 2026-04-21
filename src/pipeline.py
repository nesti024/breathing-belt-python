"""Pure live-processing pipeline for breathing-belt device rows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Literal

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
    CalibrationSettings,
    ExtremaConfig,
    FilterConfig,
    HoldConfig,
    MovementConfig,
    OutputSmoothingConfig,
    RawQCConfig,
)


_HOLD_RELEASE_DRIFT = 0.03
ProcessingMode = Literal["control", "movement", "adaptive"]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for deterministic sample-by-sample live processing."""

    sampling_rate_hz: int
    processed_sensor_column: int
    invert_signal: bool
    filter: FilterConfig
    calibration: CalibrationSettings
    adaptation: AdaptationSettings
    hold: HoldConfig
    output_smoothing: OutputSmoothingConfig
    extrema: ExtremaConfig
    raw_qc: RawQCConfig
    processing_mode: ProcessingMode = "control"
    movement: MovementConfig = field(default_factory=MovementConfig)

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
    def movement_low_activity_window_samples(self) -> int:
        return max(
            3,
            int(
                round(
                    (self.movement.low_activity_window_ms / 1000.0)
                    * self.sampling_rate_hz
                )
            ),
        )

    @property
    def adaptation_low_activity_window_samples(self) -> int:
        return max(
            3,
            int(
                round(
                    (self.adaptation.low_activity_window_ms / 1000.0)
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
    processing_mode: ProcessingMode = "control"
    movement_value: float | None = None
    extrema_event_code: float = 0.0
    extrema_event_label: str | None = None
    messages: tuple[str, ...] = ()
    qc_events: tuple[RawQCEvent, ...] = ()


@dataclass
class PipelineState:
    """Mutable state for the live-processing pipeline."""

    recent_abs_velocity: deque[float]
    recent_output_abs_velocity: deque[float]
    recent_movement_abs_velocity: deque[float]
    recent_adaptive_abs_velocity: deque[float]
    qc_state: RawQCState
    filter_initialized: bool = False
    sos_hp: np.ndarray | None = None
    zi_hp: np.ndarray | None = None
    sos_lp: np.ndarray | None = None
    zi_lp: np.ndarray | None = None
    previous_filtered_value: float | None = None
    previous_cleaned_value: float | None = None
    previous_movement_activity_value: float | None = None
    previous_adaptive_value: float | None = None
    hold_mode_active: bool = False
    frozen_normalized_value: float | None = None
    emitted_normalized_value: float | None = None
    slowed_movement_value: float | None = None
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
        recent_abs_velocity=deque(maxlen=cfg.hold_activity_window_samples),
        recent_output_abs_velocity=deque(maxlen=cfg.output_smoothing_activity_window_samples),
        recent_movement_abs_velocity=deque(maxlen=cfg.movement_low_activity_window_samples),
        recent_adaptive_abs_velocity=deque(maxlen=cfg.adaptation_low_activity_window_samples),
        qc_state=create_raw_qc_state(),
    )


def reset_pipeline_state_for_source_gap(state: PipelineState) -> None:
    """Reset short-lived continuity-sensitive state after a source-sample gap.

    Calibration and adaptive reference state are preserved so processing can
    resume without re-running the full startup procedure.
    """

    _reset_continuity_sensitive_state(
        state,
        reset_runtime_progress=False,
        reset_stage_sample_index=False,
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
    cleaned_value, is_artifact = _suppress_reversal(filtered_value)
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
    movement_value: float | None = None
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
            if cfg.processing_mode == "movement":
                state.calibration_result = _run_movement_calibration(
                    state.calibration_samples,
                    cfg,
                )
                state.adaptive_state = _build_fixed_reference_state(
                    state.calibration_samples,
                    state.calibration_result,
                    cfg.calibration.amplitude_floor,
                )
            elif cfg.processing_mode == "adaptive":
                state.calibration_result = run_range_calibration(
                    state.calibration_samples,
                    cfg.calibration_cfg,
                )
                state.adaptive_state = initialize_adaptive_range(
                    state.calibration_samples,
                    state.calibration_result,
                    cfg.adaptive_cfg_startup,
                )
            else:
                state.calibration_result = run_range_calibration(
                    state.calibration_samples,
                    cfg.calibration_cfg,
                )
                state.adaptive_state = _build_fixed_reference_state(
                    state.calibration_samples,
                    state.calibration_result,
                    cfg.calibration.amplitude_floor,
                )
            adaptive_center = float(state.adaptive_state.center)
            adaptive_amplitude = float(state.adaptive_state.amplitude)
            if cfg.processing_mode == "movement":
                messages.extend(
                    [
                        "Movement-proxy calibration complete.",
                        (
                            "Movement-proxy reference: "
                            f"center={state.calibration_result.center:.6f}, "
                            f"reference_amplitude={state.calibration_result.amplitude:.6f}, "
                            f"percentile_lo={state.calibration_result.global_min:.6f}, "
                            f"percentile_hi={state.calibration_result.global_max:.6f}"
                        ),
                    ]
                )
            elif cfg.processing_mode == "adaptive":
                messages.extend(
                    [
                        "Adaptive calibration complete.",
                        (
                            "Adaptive range initialized: "
                            f"center={state.adaptive_state.center:.6f}, "
                            f"amplitude={state.adaptive_state.amplitude:.6f}, "
                            f"startup_duration_s={cfg.adaptation.startup_duration_s:.1f}"
                        ),
                    ]
                )
            else:
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
            _reset_continuity_sensitive_state(
                state,
                reset_runtime_progress=True,
                reset_stage_sample_index=True,
            )
        else:
            state.stage_sample_index += 1
    else:
        sample_adaptive_state = state.adaptive_state
        if cfg.processing_mode == "movement":
            movement_value = _compute_runtime_movement_value(cleaned_value, state)
            extrema_event_code, extrema_event_label = _detect_runtime_extremum(
                movement_value,
                sample_index,
                state,
                cfg,
            )
        elif cfg.processing_mode == "adaptive":
            # Export one coherent pre-update adaptive snapshot with each sample.
            normalized_value, movement_value = _normalize_runtime_adaptive_sample(
                cleaned_value,
                is_artifact,
                state,
                cfg,
            )
            extrema_event_code, extrema_event_label = _detect_runtime_extremum(
                movement_value,
                sample_index,
                state,
                cfg,
            )
        else:
            normalized_value = _normalize_runtime_sample(cleaned_value, is_artifact, state, cfg)
            extrema_event_code, extrema_event_label = _detect_runtime_extremum(
                cleaned_value,
                sample_index,
                state,
                cfg,
            )
        if cfg.processing_mode == "adaptive":
            adaptive_center = (
                float(sample_adaptive_state.center) if sample_adaptive_state else None
            )
            adaptive_amplitude = (
                float(sample_adaptive_state.amplitude) if sample_adaptive_state else None
            )
        else:
            adaptive_center = float(state.adaptive_state.center) if state.adaptive_state else None
            adaptive_amplitude = (
                float(state.adaptive_state.amplitude) if state.adaptive_state else None
            )
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
        hold_mode_active=(
            state.hold_mode_active if stage == "runtime" and cfg.processing_mode == "control" else False
        ),
        adaptive_center=adaptive_center,
        adaptive_amplitude=adaptive_amplitude,
        processing_mode=cfg.processing_mode,
        movement_value=movement_value,
        extrema_event_code=extrema_event_code if stage == "runtime" else 0.0,
        extrema_event_label=extrema_event_label if stage == "runtime" else None,
        messages=tuple(messages),
        qc_events=tuple(qc_events),
    )
    return sample, state


def _reset_continuity_sensitive_state(
    state: PipelineState,
    *,
    reset_runtime_progress: bool,
    reset_stage_sample_index: bool,
) -> None:
    state.recent_abs_velocity.clear()
    state.recent_output_abs_velocity.clear()
    state.recent_movement_abs_velocity.clear()
    state.recent_adaptive_abs_velocity.clear()
    state.filter_initialized = False
    state.sos_hp = None
    state.zi_hp = None
    state.sos_lp = None
    state.zi_lp = None
    state.previous_filtered_value = None
    state.previous_cleaned_value = None
    state.previous_movement_activity_value = None
    state.previous_adaptive_value = None
    state.hold_mode_active = False
    state.frozen_normalized_value = None
    state.emitted_normalized_value = None
    state.slowed_movement_value = None
    state.startup_mode_active = False
    state.previous_delta_sign = 0
    state.last_event_sample_index = None
    state.last_peak_value = None
    state.last_trough_value = None
    if reset_runtime_progress:
        state.runtime_processed_samples = 0
    if reset_stage_sample_index:
        state.stage_sample_index = 0


def _filter_sample(raw_sensor_value: float, state: PipelineState, cfg: PipelineConfig) -> float:
    control_input = -raw_sensor_value if cfg.invert_signal else raw_sensor_value
    if not state.filter_initialized:
        if cfg.processing_mode == "movement":
            state.sos_hp, state.zi_hp = get_high_pass_filter_coeffs(
                cfg.movement.hp_cutoff_hz,
                cfg.sampling_rate_hz,
                cfg.movement.hp_order,
                initial_value=control_input,
            )
            state.sos_lp, state.zi_lp = get_low_pass_filter_coeffs(
                cfg.movement.lp_cutoff_hz,
                cfg.sampling_rate_hz,
                cfg.movement.lp_order,
                initial_value=0.0,
            )
        else:
            state.sos_lp, state.zi_lp = get_low_pass_filter_coeffs(
                cfg.filter.lp_cutoff_hz,
                cfg.sampling_rate_hz,
                cfg.filter.lp_order,
                initial_value=control_input,
            )
        state.filter_initialized = True

    if state.sos_lp is None or state.zi_lp is None:
        raise RuntimeError("Low-pass filter state must be initialized before filtering.")

    if cfg.processing_mode == "movement":
        if state.sos_hp is None or state.zi_hp is None:
            raise RuntimeError("High-pass filter state must be initialized for movement-proxy mode.")
        high_passed_value, state.zi_hp = high_pass_filter_sample(
            control_input,
            state.sos_hp,
            state.zi_hp,
        )
        low_passed_value, state.zi_lp = low_pass_filter_sample(
            float(high_passed_value),
            state.sos_lp,
            state.zi_lp,
        )
        low_passed_value = _apply_movement_low_activity_slowdown(
            control_input=float(control_input),
            filtered_value=float(low_passed_value),
            state=state,
            cfg=cfg,
        )
    else:
        low_passed_value, state.zi_lp = low_pass_filter_sample(
            control_input,
            state.sos_lp,
            state.zi_lp,
        )
    return float(low_passed_value)


def _suppress_reversal(
    filtered_value: float,
) -> tuple[float, bool]:
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


def _compute_runtime_movement_value(
    cleaned_value: float,
    state: PipelineState,
) -> float:
    if state.calibration_result is None:
        raise RuntimeError("Calibration state must be initialized before runtime movement-proxy output.")
    return float(cleaned_value - state.calibration_result.center)


def _normalize_runtime_adaptive_sample(
    cleaned_value: float,
    is_artifact: bool,
    state: PipelineState,
    cfg: PipelineConfig,
) -> tuple[float, float]:
    del is_artifact
    if state.adaptive_state is None:
        raise RuntimeError("Adaptive state must be initialized before adaptive normalization.")

    current_state = state.adaptive_state
    movement_value = float(cleaned_value - current_state.center)
    abs_velocity = _append_abs_velocity(
        recent_abs_velocity=state.recent_adaptive_abs_velocity,
        current_value=float(cleaned_value),
        previous_value=state.previous_adaptive_value,
        fs_hz=cfg.sampling_rate_hz,
    )
    state.previous_adaptive_value = float(cleaned_value)
    in_startup = state.runtime_processed_samples < cfg.startup_target_samples
    state.startup_mode_active = in_startup
    adaptive_cfg = cfg.adaptive_cfg_startup if in_startup else cfg.adaptive_cfg_runtime
    low_activity = (
        cfg.adaptation.low_activity_gating_enabled
        and _is_low_activity(
            recent_abs_velocity=state.recent_adaptive_abs_velocity,
            abs_velocity=abs_velocity,
            amplitude=max(current_state.amplitude, cfg.calibration.amplitude_floor),
            ratio_per_sec=cfg.adaptation.low_activity_ratio_per_sec,
            floor_per_sec=cfg.adaptation.low_activity_floor_per_sec,
        )
    )
    allow_center_update = cfg.adaptation.center_enabled and not low_activity
    allow_amplitude_update = cfg.adaptation.amplitude_enabled and not low_activity
    normalized_value, state.adaptive_state = update_adaptive_range(
        x=float(cleaned_value),
        state=current_state,
        cfg=adaptive_cfg,
        allow_update=allow_center_update or allow_amplitude_update,
        allow_center_update=allow_center_update,
        allow_amplitude_update=allow_amplitude_update,
    )
    return float(normalized_value), movement_value


def _apply_movement_low_activity_slowdown(
    *,
    control_input: float,
    filtered_value: float,
    state: PipelineState,
    cfg: PipelineConfig,
) -> float:
    if state.stage != "runtime":
        state.slowed_movement_value = float(filtered_value)
        return float(filtered_value)

    abs_velocity = _append_abs_velocity(
        recent_abs_velocity=state.recent_movement_abs_velocity,
        current_value=float(control_input),
        previous_value=state.previous_movement_activity_value,
        fs_hz=cfg.sampling_rate_hz,
    )
    state.previous_movement_activity_value = float(control_input)

    if state.slowed_movement_value is None:
        state.slowed_movement_value = float(filtered_value)
        return float(filtered_value)

    low_activity = (
        cfg.movement.low_activity_slowdown_enabled
        and state.calibration_result is not None
        and _is_low_activity(
            recent_abs_velocity=state.recent_movement_abs_velocity,
            abs_velocity=abs_velocity,
            amplitude=max(state.calibration_result.amplitude, cfg.calibration.amplitude_floor),
            ratio_per_sec=cfg.movement.low_activity_ratio_per_sec,
            floor_per_sec=cfg.movement.low_activity_floor_per_sec,
        )
    )
    recentering_toward_zero = (
        abs(float(filtered_value)) < abs(state.slowed_movement_value)
        and float(filtered_value) * state.slowed_movement_value >= 0.0
    )
    drift_scale = (
        cfg.movement.low_activity_drift_scale
        if low_activity and recentering_toward_zero
        else 1.0
    )
    state.slowed_movement_value = state.slowed_movement_value + drift_scale * (
        float(filtered_value) - state.slowed_movement_value
    )
    return float(state.slowed_movement_value)


def _append_abs_velocity(
    *,
    recent_abs_velocity: deque[float],
    current_value: float,
    previous_value: float | None,
    fs_hz: int,
) -> float:
    if previous_value is None:
        abs_velocity = 0.0
    else:
        abs_velocity = abs(float(current_value) - float(previous_value)) * fs_hz
    recent_abs_velocity.append(abs_velocity)
    return float(abs_velocity)


def _is_low_activity(
    *,
    recent_abs_velocity: deque[float],
    abs_velocity: float,
    amplitude: float,
    ratio_per_sec: float,
    floor_per_sec: float,
) -> bool:
    if len(recent_abs_velocity) < recent_abs_velocity.maxlen:
        return False
    activity_threshold = max(float(floor_per_sec), float(amplitude) * float(ratio_per_sec))
    activity_value = float(np.mean(recent_abs_velocity))
    return activity_value < activity_threshold and float(abs_velocity) < activity_threshold


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
    signal_value: float,
    sample_index: int,
    state: PipelineState,
    cfg: PipelineConfig,
) -> tuple[float, str | None]:
    previous_value = state.previous_filtered_value
    state.previous_filtered_value = float(signal_value)
    if previous_value is None:
        state.previous_delta_sign = 0
        return 0.0, None

    delta = float(signal_value) - previous_value
    event_code = 0.0
    event_label: str | None = None
    candidate_index = max(sample_index - 1, 0)
    prominence_threshold = max(
        cfg.calibration.amplitude_floor,
        cfg.extrema.prominence_ratio * max(state.adaptive_state.amplitude, cfg.calibration.amplitude_floor),
    )
    baseline_value = (
        0.0
        if cfg.processing_mode in {"movement", "adaptive"}
        else state.adaptive_state.center
    )

    if state.previous_delta_sign > 0 and delta <= 0.0:
        if _extremum_interval_elapsed(candidate_index, state, cfg):
            candidate_value = float(previous_value)
            reference_value = (
                state.last_trough_value
                if state.last_trough_value is not None
                else baseline_value
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
                else baseline_value
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


def _run_movement_calibration(
    calibration_samples: list[float] | np.ndarray,
    cfg: PipelineConfig,
) -> CalibrationResult:
    samples = np.asarray(calibration_samples, dtype=float).reshape(-1)
    if samples.size == 0:
        raise ValueError("Movement-proxy calibration requires at least one sample.")

    sorted_samples = np.sort(samples)
    n_samples = int(sorted_samples.size)
    lo_idx = int(n_samples * cfg.calibration.percentile_lo / 100.0)
    hi_idx = int(n_samples * cfg.calibration.percentile_hi / 100.0) - 1
    lo_idx = max(0, min(lo_idx, n_samples - 1))
    hi_idx = max(0, min(hi_idx, n_samples - 1))
    if hi_idx < lo_idx:
        lo_idx = 0
        hi_idx = n_samples - 1

    percentile_lo = float(sorted_samples[lo_idx])
    percentile_hi = float(sorted_samples[hi_idx])
    center = float(np.median(samples))
    amplitude = max(
        0.5 * (percentile_hi - percentile_lo),
        float(cfg.calibration.amplitude_floor),
    )

    return CalibrationResult(
        global_min=percentile_lo,
        global_max=percentile_hi,
        center=center,
        amplitude=amplitude,
        y_min=percentile_lo,
        y_max=percentile_hi,
        saturated=False,
        n_samples=n_samples,
        saturated_count=0,
        lo_idx=lo_idx,
        hi_idx=hi_idx,
    )


def _build_fixed_reference_state(
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
