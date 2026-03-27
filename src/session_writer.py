"""Per-run persistence for publication-ready breathing-belt sessions."""

from __future__ import annotations

from csv import DictWriter
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import PipelineSample
from .quality import RawQCEvent
from .settings import AppConfig, write_config_toml


class SessionWriter:
    """Write raw, processed, and metadata artifacts for one acquisition run."""

    def __init__(self, root_dir: str | Path, config: AppConfig) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(root_dir) / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=False)

        self._device_file = None
        self._device_writer: DictWriter[str] | None = None
        self._signal_file = (self.session_dir / "signal_trace.csv").open("w", newline="", encoding="utf-8")
        self._signal_writer = DictWriter(
            self._signal_file,
            fieldnames=[
                "stage",
                "sample_index",
                "relative_time_s",
                "processing_mode",
                "selected_sensor_raw",
                "filtered_value",
                "cleaned_value",
                "normalized_value",
                "movement_value",
                "is_artifact",
                "hold_mode_active",
                "extrema_event_code",
                "extrema_event_label",
            ],
        )
        self._signal_writer.writeheader()
        self._qc_file = (self.session_dir / "qc_events.csv").open("w", newline="", encoding="utf-8")
        self._qc_writer = DictWriter(
            self._qc_file,
            fieldnames=[
                "event_type",
                "stage",
                "sample_index",
                "relative_time_s",
                "raw_value",
                "threshold",
                "message",
            ],
        )
        self._qc_writer.writeheader()

        self.resolved_config_path = self.session_dir / "resolved_config.toml"
        write_config_toml(self.resolved_config_path, config)
        self.metadata_path = self.session_dir / "session_metadata.json"

    def write_device_row(
        self,
        stage: str,
        sample_index: int,
        relative_time_s: float,
        device_row: np.ndarray,
    ) -> None:
        """Append one raw device row to the session export."""

        row_array = np.asarray(device_row, dtype=float).reshape(-1)
        if self._device_writer is None:
            device_columns = [f"device_col_{idx}" for idx in range(row_array.size)]
            self._device_file = (self.session_dir / "device_samples.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
            self._device_writer = DictWriter(
                self._device_file,
                fieldnames=["stage", "sample_index", "relative_time_s", *device_columns],
            )
            self._device_writer.writeheader()

        row_payload = {
            "stage": stage,
            "sample_index": sample_index,
            "relative_time_s": f"{relative_time_s:.6f}",
        }
        row_payload.update({f"device_col_{idx}": value for idx, value in enumerate(row_array)})
        self._device_writer.writerow(row_payload)

    def write_signal_sample(self, sample: PipelineSample) -> None:
        """Append one processed pipeline sample to the signal trace export."""

        self._signal_writer.writerow(
            {
                "stage": sample.stage,
                "sample_index": sample.sample_index,
                "relative_time_s": f"{sample.relative_time_s:.6f}",
                "processing_mode": sample.processing_mode,
                "selected_sensor_raw": f"{sample.selected_sensor_raw:.6f}",
                "filtered_value": f"{sample.filtered_value:.6f}",
                "cleaned_value": f"{sample.cleaned_value:.6f}",
                "normalized_value": (
                    "" if sample.normalized_value is None else f"{sample.normalized_value:.6f}"
                ),
                "movement_value": (
                    "" if sample.movement_value is None else f"{sample.movement_value:.6f}"
                ),
                "is_artifact": int(sample.is_artifact),
                "hold_mode_active": int(sample.hold_mode_active),
                "extrema_event_code": f"{sample.extrema_event_code:.1f}",
                "extrema_event_label": "" if sample.extrema_event_label is None else sample.extrema_event_label,
            }
        )

    def write_qc_event(self, event: RawQCEvent) -> None:
        """Append one QC episode event to the QC CSV export."""

        self._qc_writer.writerow(
            {
                "event_type": event.event_type,
                "stage": event.stage,
                "sample_index": event.sample_index,
                "relative_time_s": f"{event.relative_time_s:.6f}",
                "raw_value": f"{event.raw_value:.6f}",
                "threshold": f"{event.threshold:.6f}",
                "message": event.message,
            }
        )

    def finalize(self, metadata: dict[str, Any]) -> None:
        """Write session metadata and close all file handles."""

        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.close()

    def close(self) -> None:
        """Close any open session files."""

        if self._device_file is not None:
            self._device_file.close()
            self._device_file = None
        if self._signal_file is not None:
            self._signal_file.close()
            self._signal_file = None
        if self._qc_file is not None:
            self._qc_file.close()
            self._qc_file = None


def build_session_metadata(
    *,
    config: AppConfig,
    resolved_config_path: Path,
    software_version: str,
    started_at: str,
    ended_at: str,
    calibration_result: Any,
    adaptive_state: Any,
    qc_summary: dict[str, Any],
    processing_mode: str = "control",
    selected_mode_number: int = 1,
) -> dict[str, Any]:
    """Build a JSON-serializable metadata object for one session."""

    calibration_payload = None if calibration_result is None else asdict(calibration_result)
    adaptive_payload = None if adaptive_state is None else asdict(adaptive_state)
    control_active = processing_mode == "control"
    movement_active = processing_mode == "movement"
    adaptive_mode_active = processing_mode == "adaptive"
    control_min = (
        None if calibration_result is None or not control_active else float(calibration_result.y_min)
    )
    control_max = (
        None if calibration_result is None or not control_active else float(calibration_result.y_max)
    )
    metadata = {
        "started_at": started_at,
        "ended_at": ended_at,
        "software_version": software_version,
        "resolved_config_path": str(resolved_config_path),
        "processing_mode": processing_mode,
        "selected_mode_number": selected_mode_number,
        "acquired_channels": list(config.device.channels),
        "processed_sensor_column": config.device.processed_sensor_column,
        "invert_signal": config.device.invert_signal,
        "calibration_result": calibration_payload,
        "adaptation_settings": {
            "center_enabled": config.adaptation.center_enabled,
            "amplitude_enabled": config.adaptation.amplitude_enabled,
            "center_tau_s": config.adaptation.center_tau_s,
            "amplitude_tau_s": config.adaptation.amplitude_tau_s,
            "startup_duration_s": config.adaptation.startup_duration_s,
            "startup_center_tau_s": config.adaptation.startup_center_tau_s,
            "startup_amplitude_tau_s": config.adaptation.startup_amplitude_tau_s,
        },
        "control_model": {
            "active": control_active,
            "mode": "fixed_calibration_padded_extrema_hold_output_smoothing",
            "lp_cutoff_hz": config.filter.lp_cutoff_hz,
            "lp_order": config.filter.lp_order,
            "calibration_padding_ratio": config.calibration.padding_ratio,
            "control_min": control_min,
            "control_max": control_max,
            "hold_enabled": config.hold.enabled,
            "hold_activity_window_ms": config.hold.activity_window_ms,
            "freeze_enter_ratio_per_sec": config.hold.ratio_per_sec_enter,
            "freeze_exit_ratio_per_sec": config.hold.ratio_per_sec_exit,
            "freeze_floor_per_sec": config.hold.floor_per_sec,
            "hold_edge_margin_ratio": config.hold.edge_margin_ratio,
            "hold_release_drift": 0.03,
            "output_smoothing_enabled": config.output_smoothing.enabled,
            "output_smoothing_activity_window_ms": config.output_smoothing.activity_window_ms,
            "output_smoothing_tau_active_s": config.output_smoothing.tau_active_s,
            "output_smoothing_tau_extreme_s": config.output_smoothing.tau_extreme_s,
            "output_smoothing_tau_hold_s": config.output_smoothing.tau_hold_s,
            "output_smoothing_activity_low_ratio_per_sec": config.output_smoothing.activity_low_ratio_per_sec,
            "output_smoothing_activity_high_ratio_per_sec": config.output_smoothing.activity_high_ratio_per_sec,
            "output_smoothing_activity_floor_per_sec": config.output_smoothing.activity_floor_per_sec,
            "output_smoothing_edge_margin_ratio": config.output_smoothing.edge_margin_ratio,
            "extrema_min_interval_ms": config.extrema.min_interval_ms,
            "extrema_prominence_ratio": config.extrema.prominence_ratio,
        },
        "movement_model": {
            "active": movement_active,
            "mode": "realtime_centered_movement_proxy",
            "hp_cutoff_hz": config.movement.hp_cutoff_hz,
            "hp_order": config.movement.hp_order,
            "lp_cutoff_hz": config.movement.lp_cutoff_hz,
            "lp_order": config.movement.lp_order,
            "calibration_center": (
                None if calibration_result is None or not movement_active else float(calibration_result.center)
            ),
            "reference_amplitude": (
                None if calibration_result is None or not movement_active else float(calibration_result.amplitude)
            ),
            "percentile_lo": (
                None if calibration_result is None or not movement_active else float(calibration_result.global_min)
            ),
            "percentile_hi": (
                None if calibration_result is None or not movement_active else float(calibration_result.global_max)
            ),
            "extrema_min_interval_ms": config.extrema.min_interval_ms,
            "extrema_prominence_ratio": config.extrema.prominence_ratio,
        },
        "adaptive_model": {
            "active": adaptive_mode_active,
            "mode": "adaptive_live_control",
            "lp_cutoff_hz": config.filter.lp_cutoff_hz,
            "lp_order": config.filter.lp_order,
            "center_enabled": config.adaptation.center_enabled,
            "amplitude_enabled": config.adaptation.amplitude_enabled,
            "center_tau_s": config.adaptation.center_tau_s,
            "amplitude_tau_s": config.adaptation.amplitude_tau_s,
            "startup_duration_s": config.adaptation.startup_duration_s,
            "startup_center_tau_s": config.adaptation.startup_center_tau_s,
            "startup_amplitude_tau_s": config.adaptation.startup_amplitude_tau_s,
            "initial_center": (
                None if calibration_result is None or not adaptive_mode_active else float(calibration_result.center)
            ),
            "initial_amplitude": (
                None if calibration_result is None or not adaptive_mode_active else float(calibration_result.amplitude)
            ),
            "extrema_min_interval_ms": config.extrema.min_interval_ms,
            "extrema_prominence_ratio": config.extrema.prominence_ratio,
        },
        "lsl_stream": {
            "enabled": config.lsl.enable,
            "channel_count": 2 if config.lsl.enable else 0,
            "stream_name": (
                config.lsl.stream_name
                if processing_mode == "control"
                else (
                    f"{config.lsl.stream_name}Movement"
                    if processing_mode == "movement"
                    else f"{config.lsl.stream_name}Adaptive"
                )
            ) if config.lsl.enable else None,
            "stream_type": (
                config.lsl.stream_type
                if processing_mode == "control"
                else (
                    "BreathingMovement"
                    if processing_mode == "movement"
                    else "BreathingAdaptive"
                )
            ) if config.lsl.enable else None,
            "source_id": (
                config.lsl.source_id
                if processing_mode == "control"
                else (
                    f"{config.lsl.source_id}_movement"
                    if processing_mode == "movement"
                    else f"{config.lsl.source_id}_adaptive"
                )
            ) if config.lsl.enable else None,
            "channel_names": (
                ["breath_level", "event_code"]
                if processing_mode != "movement"
                else ["movement_value", "event_code"]
            ) if config.lsl.enable else [],
        },
        "final_adaptive_state": adaptive_payload,
        "raw_qc_summary": qc_summary,
    }
    return metadata
