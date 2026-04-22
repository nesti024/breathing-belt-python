"""Tests for configuration loading and live-acquisition validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from src.settings import default_config, load_config, validate_live_acquisition_config


def test_load_config_returns_defaults_when_file_is_missing() -> None:
    missing_path = Path(".codex-tmp") / f"missing-config-{uuid4().hex}.toml"

    assert load_config(missing_path) == default_config()


def test_validate_live_acquisition_config_rejects_blank_mac_address() -> None:
    defaults = default_config()
    config = replace(
        defaults,
        device=replace(defaults.device, mac_address="   "),
    )

    with pytest.raises(ValueError, match="device.mac_address must be set for live acquisition."):
        validate_live_acquisition_config(config)


def test_validate_live_acquisition_config_accepts_configured_mac_address() -> None:
    defaults = default_config()
    config = replace(
        defaults,
        device=replace(defaults.device, mac_address="00:11:22:33:44:55"),
    )

    validate_live_acquisition_config(config)


def test_load_config_rejects_processed_sensor_column_outside_expected_row_width() -> None:
    config_path = Path(".codex-tmp") / f"invalid-processed-sensor-column-{uuid4().hex}.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "[device]\nchannels = [0]\nprocessed_sensor_column = 6\n",
        encoding="utf-8",
    )

    try:
        with pytest.raises(
            ValueError,
            match="device.processed_sensor_column must be less than the expected BITalino row width",
        ):
            load_config(config_path)
    finally:
        config_path.unlink(missing_ok=True)
