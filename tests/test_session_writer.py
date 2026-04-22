"""Regression tests for session export persistence."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pytest

from src.calibration import AdaptiveRangeState, CalibrationResult
from src.pipeline import PipelineSample
from src.quality import RawQCEvent
from src.session_writer import SessionWriter, build_session_metadata
from src.settings import AppConfig, default_config, expected_bitalino_row_width


def _make_config() -> AppConfig:
    defaults = default_config()
    return AppConfig(
        device=defaults.device.__class__(mac_address="00:00:00:00:00:00"),
        display=defaults.display,
        lsl=defaults.lsl,
        filter=defaults.filter,
        movement=defaults.movement,
        calibration=defaults.calibration,
        adaptation=defaults.adaptation,
        hold=defaults.hold,
        output_smoothing=defaults.output_smoothing,
        extrema=defaults.extrema,
        raw_qc=defaults.raw_qc,
        output=defaults.output.__class__(root_dir="ignored-in-test"),
    )


def _device_sample_width(config: AppConfig) -> int:
    return expected_bitalino_row_width(config.device.channels)


def _timing_kwargs(
    *,
    source_sample_index: int,
    capture_time_lsl_s: float,
    lsl_timestamp_s: float,
    event_timestamp_lsl_s: float | None = None,
) -> dict[str, float | int | None]:
    return {
        "source_sample_index": source_sample_index,
        "capture_time_lsl_s": capture_time_lsl_s,
        "lsl_timestamp_s": lsl_timestamp_s,
        "event_timestamp_lsl_s": event_timestamp_lsl_s,
    }


def test_session_writer_creates_expected_files_and_rows() -> None:
    config = _make_config()
    root_dir = Path(".codex-tmp") / f"session-writer-test-{uuid4().hex}"
    root_dir.mkdir(parents=True, exist_ok=False)
    try:
        writer = SessionWriter(
            root_dir,
            config,
            device_sample_width=_device_sample_width(config),
        )

        device_path = writer.session_dir / "device_samples.csv"
        assert device_path.exists()
        assert device_path.read_text(encoding="utf-8").splitlines() == [
            "stage,sample_index,relative_time_s,source_sample_index,capture_time_lsl_s,lsl_timestamp_s,device_col_0,device_col_1,device_col_2,device_col_3,device_col_4,device_col_5,device_col_6"
        ]

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
            source_sample_index=0,
            capture_time_lsl_s=10.0,
            lsl_timestamp_s=9.95,
        )
        writer.write_signal_sample(
            calibration_sample,
            source_sample_index=0,
            capture_time_lsl_s=10.0,
            lsl_timestamp_s=9.95,
        )
        writer.write_device_row(
            "runtime",
            0,
            0.0,
            np.array([0, 1, 2, 3, 4, 500, 0], dtype=float),
            source_sample_index=1,
            capture_time_lsl_s=10.01,
            lsl_timestamp_s=9.96,
        )
        writer.write_signal_sample(
            runtime_sample,
            source_sample_index=1,
            capture_time_lsl_s=10.01,
            lsl_timestamp_s=9.96,
            event_timestamp_lsl_s=9.95,
        )
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
            lsl_run_stats={
                "control_samples_sent": 1,
                "control_samples_sent_via_chunks": 0,
                "control_samples_sent_individually": 1,
                "control_chunks_sent": 0,
                "event_samples_sent": 1,
                "queue_dropped_rows_total": 2,
                "observed_gap_count": 1,
            },
        )
        writer.finalize(metadata)

        session_dir = writer.session_dir
        with (session_dir / "device_samples.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            device_rows = list(csv.DictReader(handle))
        assert device_rows[0]["stage"] == "calibration"
        assert device_rows[1]["source_sample_index"] == "1"
        assert device_rows[1]["capture_time_lsl_s"] == "10.010000"
        assert device_rows[1]["lsl_timestamp_s"] == "9.960000"
        assert "device_col_5" in device_rows[0]

        with (session_dir / "signal_trace.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            signal_rows = list(csv.DictReader(handle))
        assert signal_rows[0]["stage"] == "calibration"
        assert signal_rows[1]["stage"] == "runtime"
        assert signal_rows[1]["processing_mode"] == "control"
        assert signal_rows[1]["source_sample_index"] == "1"
        assert signal_rows[1]["normalized_value"] == "0.600000"
        assert signal_rows[1]["movement_value"] == ""
        assert signal_rows[1]["extrema_event_code"] == "1.0"
        assert signal_rows[1]["extrema_event_label"] == "inhale_peak"
        assert signal_rows[1]["event_timestamp_lsl_s"] == "9.950000"

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
        assert stored_metadata["processing_mode"] == "control"
        assert stored_metadata["selected_mode_number"] == 1
        assert stored_metadata["calibration_result"]["amplitude"] == 1.0
        assert stored_metadata["control_model"]["active"] is True
        assert stored_metadata["adaptation_settings"]["low_activity_gating_enabled"] is True
        assert stored_metadata["lsl"]["control_stream"]["channel_count"] == 1
        assert stored_metadata["lsl"]["control_stream"]["channel_names"] == ["breath_level"]
        assert stored_metadata["lsl"]["event_stream"]["stream_name"] == "BreathingBeltEvents"
        assert stored_metadata["lsl"]["event_stream"]["nominal_srate_hz"] == 0.0
        assert stored_metadata["lsl"]["timing"]["timestamp_domain"] == "local_clock"
        assert stored_metadata["lsl"]["timing"]["timestamp_origin"] == "host_estimated_segment_anchor"
        assert (
            stored_metadata["lsl"]["timing"]["chunk_backfill_policy"]
            == "nominal_fs_continuation_across_contiguous_reads"
        )
        assert stored_metadata["lsl"]["timing"]["constant_delay_s"] == 0.0
        assert stored_metadata["lsl"]["run_stats"]["queue_dropped_rows_total"] == 2
        assert stored_metadata["lsl"]["run_stats"]["observed_gap_count"] == 1
        assert stored_metadata["raw_qc_summary"]["event_counts"]["saturation"] == 1
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)


def test_session_writer_keeps_header_only_raw_file_when_no_samples_are_written() -> None:
    config = _make_config()
    root_dir = Path(".codex-tmp") / f"session-writer-empty-raw-test-{uuid4().hex}"
    root_dir.mkdir(parents=True, exist_ok=False)
    try:
        writer = SessionWriter(
            root_dir,
            config,
            device_sample_width=_device_sample_width(config),
        )
        session_dir = writer.session_dir
        writer.close()

        device_path = session_dir / "device_samples.csv"
        assert device_path.exists()
        assert device_path.read_text(encoding="utf-8").splitlines() == [
            "stage,sample_index,relative_time_s,source_sample_index,capture_time_lsl_s,lsl_timestamp_s,device_col_0,device_col_1,device_col_2,device_col_3,device_col_4,device_col_5,device_col_6"
        ]
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)


def test_session_writer_rejects_raw_row_width_mismatch() -> None:
    config = _make_config()
    root_dir = Path(".codex-tmp") / f"session-writer-width-test-{uuid4().hex}"
    root_dir.mkdir(parents=True, exist_ok=False)
    try:
        writer = SessionWriter(
            root_dir,
            config,
            device_sample_width=_device_sample_width(config),
        )

        with pytest.raises(ValueError, match="expected 7, observed 6"):
            writer.write_device_row(
                "runtime",
                0,
                0.0,
                np.array([0, 1, 2, 3, 4, 500], dtype=float),
                source_sample_index=0,
                capture_time_lsl_s=1.0,
                lsl_timestamp_s=1.0,
            )

        writer.close()
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)


def test_session_writer_flush_incremental_fsyncs_all_csv_exports(monkeypatch) -> None:
    config = _make_config()
    root_dir = Path(".codex-tmp") / f"session-writer-flush-test-{uuid4().hex}"
    root_dir.mkdir(parents=True, exist_ok=False)
    try:
        writer = SessionWriter(
            root_dir,
            config,
            device_sample_width=_device_sample_width(config),
        )
        fsync_calls: list[int] = []
        monkeypatch.setattr("src.session_writer.os.fsync", lambda fd: fsync_calls.append(fd))

        writer.write_device_row(
            "runtime",
            0,
            0.0,
            np.array([0, 1, 2, 3, 4, 500, 0], dtype=float),
            source_sample_index=0,
            capture_time_lsl_s=1.0,
            lsl_timestamp_s=1.0,
        )
        writer.write_signal_sample(
            PipelineSample(
                stage="runtime",
                sample_index=0,
                relative_time_s=0.0,
                selected_sensor_raw=500.0,
                filtered_value=500.0,
                cleaned_value=500.0,
                normalized_value=0.5,
                is_artifact=False,
                hold_mode_active=False,
                adaptive_center=None,
                adaptive_amplitude=None,
            ),
            source_sample_index=0,
            capture_time_lsl_s=1.0,
            lsl_timestamp_s=1.0,
        )
        writer.write_qc_event(
            RawQCEvent(
                event_type="flatline",
                stage="runtime",
                sample_index=0,
                relative_time_s=0.0,
                raw_value=500.0,
                threshold=0.1,
                message="Test event.",
            )
        )
        writer.flush_incremental()

        assert fsync_calls == [
            writer._device_file.fileno(),
            writer._signal_file.fileno(),
            writer._qc_file.fileno(),
        ]
        writer.close()
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)


def test_session_writer_records_movement_mode_rows_and_metadata() -> None:
    config = _make_config()
    root_dir = Path(".codex-tmp") / f"session-writer-movement-test-{uuid4().hex}"
    root_dir.mkdir(parents=True, exist_ok=False)
    try:
        writer = SessionWriter(
            root_dir,
            config,
            device_sample_width=_device_sample_width(config),
        )

        movement_sample = PipelineSample(
            stage="runtime",
            sample_index=0,
            relative_time_s=0.0,
            selected_sensor_raw=500.0,
            filtered_value=1.8,
            cleaned_value=1.7,
            normalized_value=None,
            is_artifact=False,
            hold_mode_active=False,
            adaptive_center=0.2,
            adaptive_amplitude=2.0,
            processing_mode="movement",
            movement_value=1.5,
            extrema_event_code=-1.0,
            extrema_event_label="exhale_trough",
        )

        writer.write_device_row(
            "runtime",
            0,
            0.0,
            np.array([0, 1, 2, 3, 4, 500, 0], dtype=float),
            source_sample_index=5,
            capture_time_lsl_s=20.0,
            lsl_timestamp_s=19.9,
        )
        writer.write_signal_sample(
            movement_sample,
            source_sample_index=5,
            capture_time_lsl_s=20.0,
            lsl_timestamp_s=19.9,
            event_timestamp_lsl_s=19.89,
        )

        metadata = build_session_metadata(
            config=config,
            resolved_config_path=writer.resolved_config_path,
            software_version="0.1.0",
            started_at="2026-03-17T10:00:00+00:00",
            ended_at="2026-03-17T10:01:00+00:00",
            calibration_result=CalibrationResult(
                global_min=-2.0,
                global_max=2.0,
                center=0.1,
                amplitude=2.0,
                y_min=-2.0,
                y_max=2.0,
                saturated=False,
                n_samples=20,
                saturated_count=0,
                lo_idx=1,
                hi_idx=18,
            ),
            adaptive_state=AdaptiveRangeState(
                center=0.1,
                amplitude=2.0,
                abs_dev_ema=0.7,
                abs_dev_to_amplitude_scale=2.9,
            ),
            qc_summary={
                "event_counts": {},
                "saturation_samples": 0,
                "total_samples": 1,
                "saturation_fraction": 0.0,
                "first_event_sample_index": None,
                "last_event_sample_index": None,
            },
            processing_mode="movement",
            selected_mode_number=2,
        )
        writer.finalize(metadata)

        session_dir = writer.session_dir
        with (session_dir / "signal_trace.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            signal_rows = list(csv.DictReader(handle))
        assert signal_rows[0]["processing_mode"] == "movement"
        assert signal_rows[0]["normalized_value"] == ""
        assert signal_rows[0]["movement_value"] == "1.500000"
        assert signal_rows[0]["event_timestamp_lsl_s"] == "19.890000"

        stored_metadata = json.loads(
            (session_dir / "session_metadata.json").read_text(encoding="utf-8")
        )
        assert stored_metadata["processing_mode"] == "movement"
        assert stored_metadata["selected_mode_number"] == 2
        assert stored_metadata["movement_model"]["active"] is True
        assert stored_metadata["lsl"]["control_stream"]["stream_name"] == "BreathingBeltMovement"
        assert stored_metadata["lsl"]["control_stream"]["channel_names"] == ["movement_value"]
        assert stored_metadata["lsl"]["event_stream"]["stream_name"] == "BreathingBeltMovementEvents"
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)


def test_session_writer_records_adaptive_mode_rows_and_metadata() -> None:
    config = _make_config()
    root_dir = Path(".codex-tmp") / f"session-writer-adaptive-test-{uuid4().hex}"
    root_dir.mkdir(parents=True, exist_ok=False)
    try:
        writer = SessionWriter(
            root_dir,
            config,
            device_sample_width=_device_sample_width(config),
        )

        adaptive_sample = PipelineSample(
            stage="runtime",
            sample_index=0,
            relative_time_s=0.0,
            selected_sensor_raw=500.0,
            filtered_value=1.8,
            cleaned_value=1.7,
            normalized_value=0.62,
            is_artifact=False,
            hold_mode_active=False,
            adaptive_center=0.2,
            adaptive_amplitude=2.0,
            processing_mode="adaptive",
            movement_value=1.5,
            extrema_event_code=1.0,
            extrema_event_label="inhale_peak",
        )

        writer.write_device_row(
            "runtime",
            0,
            0.0,
            np.array([0, 1, 2, 3, 4, 500, 0], dtype=float),
            source_sample_index=8,
            capture_time_lsl_s=30.0,
            lsl_timestamp_s=29.95,
        )
        writer.write_signal_sample(
            adaptive_sample,
            source_sample_index=8,
            capture_time_lsl_s=30.0,
            lsl_timestamp_s=29.95,
            event_timestamp_lsl_s=29.94,
        )

        metadata = build_session_metadata(
            config=config,
            resolved_config_path=writer.resolved_config_path,
            software_version="0.1.0",
            started_at="2026-03-17T10:00:00+00:00",
            ended_at="2026-03-17T10:01:00+00:00",
            calibration_result=CalibrationResult(
                global_min=-1.5,
                global_max=1.5,
                center=0.1,
                amplitude=1.5,
                y_min=-1.8,
                y_max=1.8,
                saturated=False,
                n_samples=20,
                saturated_count=0,
                lo_idx=1,
                hi_idx=18,
            ),
            adaptive_state=AdaptiveRangeState(
                center=0.15,
                amplitude=1.7,
                abs_dev_ema=0.55,
                abs_dev_to_amplitude_scale=3.1,
            ),
            qc_summary={
                "event_counts": {},
                "saturation_samples": 0,
                "total_samples": 1,
                "saturation_fraction": 0.0,
                "first_event_sample_index": None,
                "last_event_sample_index": None,
            },
            processing_mode="adaptive",
            selected_mode_number=3,
        )
        writer.finalize(metadata)

        session_dir = writer.session_dir
        with (session_dir / "signal_trace.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            signal_rows = list(csv.DictReader(handle))
        assert signal_rows[0]["processing_mode"] == "adaptive"
        assert signal_rows[0]["normalized_value"] == "0.620000"
        assert signal_rows[0]["movement_value"] == "1.500000"

        stored_metadata = json.loads(
            (session_dir / "session_metadata.json").read_text(encoding="utf-8")
        )
        assert stored_metadata["processing_mode"] == "adaptive"
        assert stored_metadata["selected_mode_number"] == 3
        assert stored_metadata["adaptive_model"]["active"] is True
        assert stored_metadata["adaptive_model"]["initial_amplitude"] == 1.5
        assert stored_metadata["lsl"]["control_stream"]["stream_name"] == "BreathingBeltAdaptive"
        assert stored_metadata["lsl"]["control_stream"]["channel_names"] == ["breath_level"]
        assert stored_metadata["lsl"]["event_stream"]["stream_name"] == "BreathingBeltAdaptiveEvents"
    finally:
        shutil.rmtree(root_dir, ignore_errors=True)


def test_session_writer_uses_microseconds_to_avoid_same_second_directory_collisions(
    monkeypatch,
) -> None:
    config = _make_config()
    root_dir = Path(".codex-tmp") / f"session-writer-timestamp-test-{uuid4().hex}"
    root_dir.mkdir(parents=True, exist_ok=False)
    now_values = iter(
        [
            datetime(2026, 4, 21, 14, 30, 15, 123456),
            datetime(2026, 4, 21, 14, 30, 15, 654321),
        ]
    )

    class FakeDateTime:
        @classmethod
        def now(cls) -> datetime:
            del cls
            return next(now_values)

    monkeypatch.setattr("src.session_writer.datetime", FakeDateTime)

    writer_one = None
    writer_two = None
    try:
        writer_one = SessionWriter(
            root_dir,
            config,
            device_sample_width=_device_sample_width(config),
        )
        writer_two = SessionWriter(
            root_dir,
            config,
            device_sample_width=_device_sample_width(config),
        )

        assert writer_one.session_dir != writer_two.session_dir
        assert writer_one.session_dir.name == "20260421_143015_123456"
        assert writer_two.session_dir.name == "20260421_143015_654321"
    finally:
        if writer_one is not None:
            writer_one.close()
        if writer_two is not None:
            writer_two.close()
        shutil.rmtree(root_dir, ignore_errors=True)
