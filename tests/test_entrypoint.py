"""CLI smoke tests for both supported entrypoint invocation styles."""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

import src.main as main_module
from src.main import prompt_processing_mode
from src.pipeline import PipelineSample
from src.settings import AppConfig, default_config


def test_script_entrypoint_help_succeeds() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "src/main.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Live breathing-belt acquisition and normalization" in result.stdout


def test_prompt_processing_mode_accepts_movement_choice() -> None:
    output = io.StringIO()
    selected_mode, processing_mode = prompt_processing_mode(
        input_func=lambda _: "2",
        output_stream=output,
    )

    assert selected_mode == 2
    assert processing_mode == "movement"


def test_prompt_processing_mode_accepts_adaptive_choice() -> None:
    output = io.StringIO()
    selected_mode, processing_mode = prompt_processing_mode(
        input_func=lambda _: "3",
        output_stream=output,
    )

    assert selected_mode == 3
    assert processing_mode == "adaptive"


def test_prompt_processing_mode_retries_after_invalid_input() -> None:
    answers = iter(["9", "1"])
    output = io.StringIO()
    selected_mode, processing_mode = prompt_processing_mode(
        input_func=lambda _: next(answers),
        output_stream=output,
    )

    assert selected_mode == 1
    assert processing_mode == "control"
    assert "Invalid selection. Enter 1, 2 or 3." in output.getvalue()


def test_prompt_processing_mode_defaults_to_control_on_empty_or_eof() -> None:
    empty_mode, empty_processing = prompt_processing_mode(
        input_func=lambda _: "",
        output_stream=io.StringIO(),
    )

    def _raise_eof(_: str) -> str:
        raise EOFError

    eof_mode, eof_processing = prompt_processing_mode(
        input_func=_raise_eof,
        output_stream=io.StringIO(),
    )

    assert (empty_mode, empty_processing) == (1, "control")
    assert (eof_mode, eof_processing) == (1, "control")


def test_run_acquisition_flushes_raw_export_once_per_chunk(monkeypatch) -> None:
    defaults = default_config()
    config = AppConfig(
        device=defaults.device.__class__(
            mac_address="00:00:00:00:00:00",
            chunk_size=2,
        ),
        display=defaults.display.__class__(
            enable_plot=False,
            plot_window_length=defaults.display.plot_window_length,
            debug_plot_window_bounds=defaults.display.debug_plot_window_bounds,
        ),
        lsl=defaults.lsl.__class__(
            enable=False,
            stream_name=defaults.lsl.stream_name,
            stream_type=defaults.lsl.stream_type,
            source_id=defaults.lsl.source_id,
        ),
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

    class FakeBelt:
        def __init__(self, **_: object) -> None:
            self._reads = 0

        def start(self) -> None:
            return None

        def get_all(self) -> np.ndarray:
            self._reads += 1
            if self._reads == 1:
                return np.array(
                    [
                        [0, 1, 2, 3, 4, 500, 0],
                        [1, 1, 2, 3, 4, 501, 0],
                    ],
                    dtype=float,
                )
            return np.empty((0, 7), dtype=float)

        def stop(self) -> None:
            return None

    writer_instances: list[object] = []

    class FakeSessionWriter:
        def __init__(
            self,
            root_dir: str,
            app_config: AppConfig,
            *,
            device_sample_width: int,
        ) -> None:
            self.root_dir = root_dir
            self.app_config = app_config
            self.device_sample_width = device_sample_width
            self.flush_calls = 0
            self.device_rows: list[tuple[str, int, float, np.ndarray]] = []
            self.signal_samples: list[PipelineSample] = []
            self.qc_events: list[object] = []
            self.resolved_config_path = Path("resolved_config.toml")
            writer_instances.append(self)

        def write_device_row(
            self,
            stage: str,
            sample_index: int,
            relative_time_s: float,
            device_row: np.ndarray,
        ) -> None:
            self.device_rows.append((stage, sample_index, relative_time_s, device_row.copy()))

        def write_signal_sample(self, sample: PipelineSample) -> None:
            self.signal_samples.append(sample)

        def write_qc_event(self, event: object) -> None:
            self.qc_events.append(event)

        def flush_raw(self) -> None:
            self.flush_calls += 1

        def finalize(self, metadata: dict[str, object]) -> None:
            self.metadata = metadata

    def fake_process_device_row(row: np.ndarray, state: object, cfg: object) -> tuple[PipelineSample, object]:
        sample_index = int(row[0])
        return (
            PipelineSample(
                stage="runtime",
                sample_index=sample_index,
                relative_time_s=sample_index / 100.0,
                selected_sensor_raw=float(row[5]),
                filtered_value=float(row[5]),
                cleaned_value=float(row[5]),
                normalized_value=0.5,
                is_artifact=False,
                hold_mode_active=False,
                adaptive_center=None,
                adaptive_amplitude=None,
            ),
            state,
        )

    pressed = iter([False, True])
    fake_keyboard = SimpleNamespace(is_pressed=lambda _: next(pressed))

    monkeypatch.setitem(sys.modules, "keyboard", fake_keyboard)
    monkeypatch.setattr(main_module, "prompt_processing_mode", lambda: (1, "control"))
    monkeypatch.setattr(main_module, "_import_breath_belt", lambda: FakeBelt)
    monkeypatch.setattr(main_module, "SessionWriter", FakeSessionWriter)
    monkeypatch.setattr(
        main_module,
        "create_pipeline_state",
        lambda _: SimpleNamespace(calibration_result=None, adaptive_state=None, qc_state=None),
    )
    monkeypatch.setattr(main_module, "process_device_row", fake_process_device_row)
    monkeypatch.setattr(main_module, "raw_qc_summary", lambda _: {})
    monkeypatch.setattr(main_module, "build_session_metadata", lambda **_: {})

    main_module.run_acquisition(config)

    writer = writer_instances[0]
    assert writer.device_sample_width == 7
    assert len(writer.device_rows) == 2
    assert writer.flush_calls == 1
