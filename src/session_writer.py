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
                "selected_sensor_raw",
                "filtered_value",
                "cleaned_value",
                "normalized_value",
                "is_artifact",
                "hold_mode_active",
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
                "selected_sensor_raw": f"{sample.selected_sensor_raw:.6f}",
                "filtered_value": f"{sample.filtered_value:.6f}",
                "cleaned_value": f"{sample.cleaned_value:.6f}",
                "normalized_value": (
                    "" if sample.normalized_value is None else f"{sample.normalized_value:.6f}"
                ),
                "is_artifact": int(sample.is_artifact),
                "hold_mode_active": int(sample.hold_mode_active),
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
) -> dict[str, Any]:
    """Build a JSON-serializable metadata object for one session."""

    calibration_payload = None if calibration_result is None else asdict(calibration_result)
    adaptive_payload = None if adaptive_state is None else asdict(adaptive_state)
    metadata = {
        "started_at": started_at,
        "ended_at": ended_at,
        "software_version": software_version,
        "resolved_config_path": str(resolved_config_path),
        "acquired_channels": list(config.device.channels),
        "processed_sensor_column": config.device.processed_sensor_column,
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
        "final_adaptive_state": adaptive_payload,
        "raw_qc_summary": qc_summary,
    }
    return metadata
