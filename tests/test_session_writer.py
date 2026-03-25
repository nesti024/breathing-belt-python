"""Regression tests for session export persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from src.calibration import AdaptiveRangeState, CalibrationResult
from src.pipeline import PipelineSample
from src.quality import RawQCEvent
from src.session_writer import SessionWriter, build_session_metadata
from src.settings import AppConfig, default_config


def _make_config() -> AppConfig:
    defaults = default_config()
    return AppConfig(
        device=defaults.device.__class__(mac_address="00:00:00:00:00:00"),
        display=defaults.display,
        lsl=defaults.lsl,
        filter=defaults.filter,
        artifact=defaults.artifact,
        calibration=defaults.calibration,
        adaptation=defaults.adaptation,
        hold=defaults.hold,
        output_smoothing=defaults.output_smoothing,
        extrema=defaults.extrema,
        raw_qc=defaults.raw_qc,
        output=defaults.output.__class__(root_dir="ignored-in-test"),
    )


def test_session_writer_creates_expected_files_and_rows() -> None:
    config = _make_config()
    root_dir = Path(".codex-tmp") / f"session-writer-test-{uuid4().hex}"
    root_dir.mkdir(parents=True, exist_ok=False)
    try:
        writer = SessionWriter(root_dir, config)

        calibration_sample = PipelineSample(
            stage="calibration",
            sample_index=0,
            relative_time_s=0.0,
            selected_sensor_raw=512.0,
            filtered_value=0.1,
            cleaned_value=0.1,
            normalized_value=None,
            is_artifact=False,
            hold_mode_active=False,
            adaptive_center=None,
            adaptive_amplitude=None,
            extrema_event_code=0.0,
            extrema_event_label=None,
        )
        runtime_sample = PipelineSample(
            stage="runtime",
            sample_index=0,
            relative_time_s=0.0,
            selected_sensor_raw=500.0,
            filtered_value=1.2,
            cleaned_value=1.1,
            normalized_value=0.6,
            is_artifact=False,
            hold_mode_active=False,
            adaptive_center=0.0,
            adaptive_amplitude=1.0,
            extrema_event_code=1.0,
            extrema_event_label="inhale_peak",
        )
        qc_event = RawQCEvent(
            event_type="saturation",
            stage="runtime",
            sample_index=1,
            relative_time_s=0.01,
            raw_value=1023.0,
            threshold=1022.0,
            message="Clipping detected.",
        )

        writer.write_device_row(
            "calibration",
            0,
            0.0,
            np.array([0, 1, 2, 3, 4, 512, 0], dtype=float),
        )
        writer.write_signal_sample(calibration_sample)
        writer.write_device_row(
            "runtime",
            0,
            0.0,
            np.array([0, 1, 2, 3, 4, 500, 0], dtype=float),
        )
        writer.write_signal_sample(runtime_sample)
        writer.write_qc_event(qc_event)

        metadata = build_session_metadata(
            config=config,
            resolved_config_path=writer.resolved_config_path,
            software_version="0.1.0",
            started_at="2026-03-17T10:00:00+00:00",
            ended_at="2026-03-17T10:01:00+00:00",
            calibration_result=CalibrationResult(
                global_min=-1.0,
                global_max=1.0,
                center=0.0,
                amplitude=1.0,
                y_min=-1.2,
                y_max=1.2,
                saturated=False,
                n_samples=20,
                saturated_count=0,
                lo_idx=1,
                hi_idx=18,
            ),
            adaptive_state=AdaptiveRangeState(
                center=0.1,
                amplitude=1.1,
                abs_dev_ema=0.5,
                abs_dev_to_amplitude_scale=2.0,
            ),
            qc_summary={
                "event_counts": {"saturation": 1},
                "saturation_samples": 10,
                "total_samples": 50,
                "saturation_fraction": 0.2,
                "first_event_sample_index": 1,
                "last_event_sample_index": 1,
            },
        )
        writer.finalize(metadata)

        session_dir = writer.session_dir
        assert (session_dir / "resolved_config.toml").exists()
        assert (session_dir / "session_metadata.json").exists()
        assert (session_dir / "device_samples.csv").exists()
        assert (session_dir / "signal_trace.csv").exists()
        assert (session_dir / "qc_events.csv").exists()

        with (session_dir / "device_samples.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            device_rows = list(csv.DictReader(handle))
        assert device_rows[0]["stage"] == "calibration"
        assert device_rows[1]["stage"] == "runtime"
        assert "device_col_5" in device_rows[0]

        with (session_dir / "signal_trace.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            signal_rows = list(csv.DictReader(handle))
        assert signal_rows[0]["stage"] == "calibration"
        assert signal_rows[1]["stage"] == "runtime"
        assert signal_rows[1]["normalized_value"] == "0.600000"
        assert signal_rows[1]["extrema_event_code"] == "1.0"
        assert signal_rows[1]["extrema_event_label"] == "inhale_peak"

        with (session_dir / "qc_events.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            qc_rows = list(csv.DictReader(handle))
        assert qc_rows[0]["event_type"] == "saturation"

        stored_metadata = json.loads(
            (session_dir / "session_metadata.json").read_text(encoding="utf-8")
        )
        assert stored_metadata["software_version"] == "0.1.0"
        assert stored_metadata["calibration_result"]["amplitude"] == 1.0
        assert stored_metadata["control_model"]["mode"] == "fixed_calibration_padded_extrema_hold_output_smoothing"
        assert stored_metadata["control_model"]["calibration_padding_ratio"] == 0.2
        assert stored_metadata["control_model"]["control_min"] == -1.2
        assert stored_metadata["control_model"]["control_max"] == 1.2
        assert stored_metadata["control_model"]["hold_enabled"] is True
        assert stored_metadata["control_model"]["hold_edge_margin_ratio"] == 0.2
        assert stored_metadata["control_model"]["hold_release_drift"] == 0.03
        assert stored_metadata["control_model"]["output_smoothing_enabled"] is True
        assert stored_metadata["control_model"]["output_smoothing_activity_window_ms"] == 500
        assert stored_metadata["control_model"]["output_smoothing_tau_active_s"] == 0.25
        assert stored_metadata["control_model"]["output_smoothing_tau_extreme_s"] == 0.75
        assert stored_metadata["control_model"]["output_smoothing_tau_hold_s"] == 5.0
        assert (
            stored_metadata["control_model"]["output_smoothing_activity_low_ratio_per_sec"]
            == 0.1
        )
        assert (
            stored_metadata["control_model"]["output_smoothing_activity_high_ratio_per_sec"]
            == 0.5
        )
        assert (
            stored_metadata["control_model"]["output_smoothing_activity_floor_per_sec"]
            == 0.01
        )
        assert stored_metadata["control_model"]["output_smoothing_edge_margin_ratio"] == 0.1
        assert stored_metadata["lsl_stream"]["channel_count"] == 2
        assert stored_metadata["raw_qc_summary"]["event_counts"]["saturation"] == 1
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)
