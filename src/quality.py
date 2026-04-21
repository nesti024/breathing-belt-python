"""Raw-signal quality control for respiration-belt acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .settings import RawQCConfig


@dataclass(frozen=True)
class RawQCEvent:
    """One raw-signal quality-control event."""

    event_type: str
    stage: str
    sample_index: int
    relative_time_s: float
    raw_value: float
    threshold: float
    message: str


@dataclass
class RawQCState:
    """Mutable state for live raw-signal quality control."""

    last_raw_value: float | None = None
    flatline_run_samples: int = 0
    baseline_ema: float | None = None
    abs_dev_ema: float = 0.0
    samples_seen: int = 0
    saturation_samples: int = 0
    saturation_active: bool = False
    flatline_active: bool = False
    baseline_shift_active: bool = False
    event_counts: dict[str, int] = field(
        default_factory=lambda: {
            "saturation": 0,
            "flatline": 0,
            "baseline_shift": 0,
        }
    )
    first_event_sample_index: int | None = None
    last_event_sample_index: int | None = None


def create_raw_qc_state() -> RawQCState:
    """Create an empty raw-QC state object."""

    return RawQCState()


def update_raw_qc(
    raw_value: float,
    stage: str,
    sample_index: int,
    relative_time_s: float,
    state: RawQCState,
    cfg: RawQCConfig,
    fs_hz: float,
) -> tuple[list[RawQCEvent], RawQCState]:
    """Update raw-signal QC state and emit any new QC episode events."""

    events: list[RawQCEvent] = []
    raw_float = float(raw_value)
    state.samples_seen += 1

    saturated = raw_float <= cfg.raw_saturation_lo or raw_float >= cfg.raw_saturation_hi
    if saturated:
        state.saturation_samples += 1
    if saturated and not state.saturation_active:
        threshold = cfg.raw_saturation_lo if raw_float <= cfg.raw_saturation_lo else cfg.raw_saturation_hi
        events.append(
            _record_event(
                state,
                RawQCEvent(
                    event_type="saturation",
                    stage=stage,
                    sample_index=sample_index,
                    relative_time_s=relative_time_s,
                    raw_value=raw_float,
                    threshold=float(threshold),
                    message=(
                        "Raw sample reached the configured saturation threshold. "
                        "This may indicate clipping or excessive belt tension."
                    ),
                ),
            )
        )
    state.saturation_active = saturated

    flatline_samples = max(1, int(round(cfg.flatline_duration_s * fs_hz)))
    if state.last_raw_value is None:
        state.flatline_run_samples = 1
    elif abs(raw_float - state.last_raw_value) <= cfg.flatline_epsilon:
        state.flatline_run_samples += 1
    else:
        state.flatline_run_samples = 1
        state.flatline_active = False
    if state.flatline_run_samples >= flatline_samples and not state.flatline_active:
        events.append(
            _record_event(
                state,
                RawQCEvent(
                    event_type="flatline",
                    stage=stage,
                    sample_index=sample_index,
                    relative_time_s=relative_time_s,
                    raw_value=raw_float,
                    threshold=float(cfg.flatline_epsilon),
                    message=(
                        "Raw signal remained effectively flat for the configured duration. "
                        "This may indicate sensor disconnection or a stalled signal."
                    ),
                ),
            )
        )
        state.flatline_active = True

    state = _update_baseline_shift_qc(
        raw_float=raw_float,
        stage=stage,
        sample_index=sample_index,
        relative_time_s=relative_time_s,
        state=state,
        cfg=cfg,
        fs_hz=fs_hz,
        events=events,
    )

    state.last_raw_value = raw_float
    return events, state


def raw_qc_summary(state: RawQCState) -> dict[str, object]:
    """Convert raw-QC state into a metadata-friendly summary mapping."""

    total_samples = max(1, state.samples_seen)
    return {
        "event_counts": dict(state.event_counts),
        "saturation_samples": int(state.saturation_samples),
        "total_samples": int(state.samples_seen),
        "saturation_fraction": float(state.saturation_samples / total_samples),
        "first_event_sample_index": state.first_event_sample_index,
        "last_event_sample_index": state.last_event_sample_index,
    }


def _update_baseline_shift_qc(
    raw_float: float,
    stage: str,
    sample_index: int,
    relative_time_s: float,
    state: RawQCState,
    cfg: RawQCConfig,
    fs_hz: float,
    events: list[RawQCEvent],
) -> RawQCState:
    if state.baseline_ema is None:
        state.baseline_ema = raw_float
        state.abs_dev_ema = 0.0
        return state

    deviation = abs(raw_float - state.baseline_ema)
    warmup_samples = max(1, int(round(cfg.warmup_s * fs_hz)))
    threshold = max(cfg.baseline_shift_sigma * max(state.abs_dev_ema, 1e-6), cfg.baseline_shift_floor)
    baseline_shift = state.samples_seen >= warmup_samples and deviation > threshold

    if baseline_shift and not state.baseline_shift_active:
        events.append(
            _record_event(
                state,
                RawQCEvent(
                    event_type="baseline_shift",
                    stage=stage,
                    sample_index=sample_index,
                    relative_time_s=relative_time_s,
                    raw_value=raw_float,
                    threshold=float(threshold),
                    message=(
                        "Raw baseline shifted abruptly beyond the configured threshold. "
                        "This may indicate belt slip or a sudden posture change."
                    ),
                ),
            )
        )

    state.baseline_shift_active = baseline_shift

    alpha_baseline = _tau_to_alpha(cfg.baseline_ema_tau_s, fs_hz)
    alpha_abs_dev = _tau_to_alpha(cfg.baseline_abs_dev_tau_s, fs_hz)
    state.baseline_ema = state.baseline_ema + alpha_baseline * (raw_float - state.baseline_ema)
    state.abs_dev_ema = state.abs_dev_ema + alpha_abs_dev * (deviation - state.abs_dev_ema)
    return state


def _record_event(state: RawQCState, event: RawQCEvent) -> RawQCEvent:
    state.event_counts[event.event_type] = state.event_counts.get(event.event_type, 0) + 1
    if state.first_event_sample_index is None:
        state.first_event_sample_index = event.sample_index
    state.last_event_sample_index = event.sample_index
    return event


def _tau_to_alpha(tau_s: float, fs_hz: float) -> float:
    if tau_s <= 0.0:
        return 1.0
    return 1.0 - math.exp(-1.0 / (fs_hz * tau_s))
