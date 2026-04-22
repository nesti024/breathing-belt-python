"""Tests for shared LSL metadata builders."""

from __future__ import annotations

from dataclasses import replace

from src.lsl_metadata import (
    build_control_lsl_metadata,
    build_event_lsl_metadata,
    build_lsl_timing_metadata,
)
from src.settings import default_config


def test_build_control_lsl_metadata_varies_by_processing_mode() -> None:
    config = default_config()

    control_metadata = build_control_lsl_metadata(config, "control")
    movement_metadata = build_control_lsl_metadata(config, "movement")
    adaptive_metadata = build_control_lsl_metadata(config, "adaptive")

    assert control_metadata["stream_name"] == "BreathingBelt"
    assert control_metadata["stream_type"] == "Breathing"
    assert control_metadata["channel_names"] == ["breath_level"]
    assert movement_metadata["stream_name"] == "BreathingBeltMovement"
    assert movement_metadata["stream_type"] == "BreathingMovement"
    assert movement_metadata["channel_names"] == ["movement_value"]
    assert adaptive_metadata["stream_name"] == "BreathingBeltAdaptive"
    assert adaptive_metadata["stream_type"] == "BreathingAdaptive"


def test_build_event_lsl_metadata_and_timing_share_the_expected_contract() -> None:
    config = replace(
        default_config(),
        lsl=replace(default_config().lsl, constant_delay_s=0.25),
    )

    event_metadata = build_event_lsl_metadata(config, "movement")
    timing_metadata = build_lsl_timing_metadata(config)

    assert event_metadata["stream_name"] == "BreathingBeltMovementEvents"
    assert event_metadata["source_id"] == "breathingbelt001_movement_events"
    assert event_metadata["event_code_map"] == {
        "1.0": "inhale_peak",
        "-1.0": "exhale_trough",
    }
    assert timing_metadata["timestamp_domain"] == "local_clock"
    assert timing_metadata["constant_delay_s"] == 0.25


def test_build_lsl_metadata_returns_disabled_stubs_when_lsl_is_disabled() -> None:
    defaults = default_config()
    config = replace(defaults, lsl=replace(defaults.lsl, enable=False))

    control_metadata = build_control_lsl_metadata(config, "control")
    event_metadata = build_event_lsl_metadata(config, "control")

    assert control_metadata["enabled"] is False
    assert control_metadata["stream_name"] is None
    assert event_metadata["enabled"] is False
    assert event_metadata["event_code_map"] == {}
