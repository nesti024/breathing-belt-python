"""Shared LSL stream and timing metadata builders."""

from __future__ import annotations

from typing import Any

from .settings import AppConfig


def build_control_lsl_metadata(config: AppConfig, processing_mode: str) -> dict[str, Any]:
    """Return canonical metadata for the control-value LSL stream."""

    if not config.lsl.enable:
        return {
            "enabled": False,
            "channel_count": 0,
            "stream_name": None,
            "stream_type": None,
            "source_id": None,
            "channel_names": [],
            "nominal_srate_hz": None,
        }

    if processing_mode == "movement":
        return {
            "enabled": True,
            "channel_count": 1,
            "stream_name": f"{config.lsl.stream_name}Movement",
            "stream_type": "BreathingMovement",
            "source_id": f"{config.lsl.source_id}_movement",
            "channel_names": ["movement_value"],
            "nominal_srate_hz": config.device.sampling_rate_hz,
        }
    if processing_mode == "adaptive":
        return {
            "enabled": True,
            "channel_count": 1,
            "stream_name": f"{config.lsl.stream_name}Adaptive",
            "stream_type": "BreathingAdaptive",
            "source_id": f"{config.lsl.source_id}_adaptive",
            "channel_names": ["breath_level"],
            "nominal_srate_hz": config.device.sampling_rate_hz,
        }
    return {
        "enabled": True,
        "channel_count": 1,
        "stream_name": config.lsl.stream_name,
        "stream_type": config.lsl.stream_type,
        "source_id": config.lsl.source_id,
        "channel_names": ["breath_level"],
        "nominal_srate_hz": config.device.sampling_rate_hz,
    }


def build_event_lsl_metadata(config: AppConfig, processing_mode: str) -> dict[str, Any]:
    """Return canonical metadata for the breath-event LSL stream."""

    if not config.lsl.enable:
        return {
            "enabled": False,
            "channel_count": 0,
            "stream_name": None,
            "stream_type": None,
            "source_id": None,
            "channel_names": [],
            "nominal_srate_hz": None,
            "event_code_map": {},
        }

    control_metadata = build_control_lsl_metadata(config, processing_mode)
    return {
        "enabled": True,
        "channel_count": 1,
        "stream_name": f"{control_metadata['stream_name']}Events",
        "stream_type": "BreathingEvents",
        "source_id": f"{control_metadata['source_id']}_events",
        "channel_names": ["event_code"],
        "nominal_srate_hz": 0.0,
        "event_code_map": {
            "1.0": "inhale_peak",
            "-1.0": "exhale_trough",
        },
    }


def build_lsl_timing_metadata(config: AppConfig) -> dict[str, str | float]:
    """Return canonical timing metadata shared by live senders and session export."""

    return {
        "timestamp_domain": "local_clock",
        "timestamp_origin": "host_estimated_segment_anchor",
        "chunk_backfill_policy": "nominal_fs_continuation_across_contiguous_reads",
        "constant_delay_s": config.lsl.constant_delay_s,
        "discontinuity_policy": "preserve_timestamp_gaps_after_loss",
    }
