"""Command-line entrypoint for live breathing-belt acquisition."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import deque
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src import __version__
    from src.pipeline import PipelineConfig, create_pipeline_state, process_device_row
    from src.quality import raw_qc_summary
    from src.session_writer import SessionWriter, build_session_metadata
    from src.settings import AppConfig, load_config

    def _import_breath_belt():
        from src.connect import BreathBelt

        return BreathBelt

    def _import_plot_helpers():
        from src.plot import plot_breathing_channel, setup_live_plot

        return plot_breathing_channel, setup_live_plot

    def _import_lsl_sender():
        from src.lsl_out import LSLBreathingSender

        return LSLBreathingSender
else:
    from . import __version__
    from .pipeline import PipelineConfig, create_pipeline_state, process_device_row
    from .quality import raw_qc_summary
    from .session_writer import SessionWriter, build_session_metadata
    from .settings import AppConfig, load_config

    def _import_breath_belt():
        from .connect import BreathBelt

        return BreathBelt

    def _import_plot_helpers():
        from .plot import plot_breathing_channel, setup_live_plot

        return plot_breathing_channel, setup_live_plot

    def _import_lsl_sender():
        from .lsl_out import LSLBreathingSender

        return LSLBreathingSender


def main(argv: list[str] | None = None) -> int:
    """Run live acquisition from a TOML configuration file."""

    parser = ArgumentParser(description="Live breathing-belt acquisition and normalization")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to the TOML configuration file. Defaults to ./config.toml.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except Exception as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    try:
        run_acquisition(config=config)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def run_acquisition(config: AppConfig) -> None:
    """Acquire, normalize, plot, stream, and persist breathing-belt data."""

    import keyboard

    BreathBelt = _import_breath_belt()
    belt = None
    session_writer = None
    lsl_sender = None
    ax_final = None
    line_final = None
    blit_manager_final = None
    sample_indices = deque(maxlen=config.device.queue_max_samples)
    normalized_signal = deque(maxlen=config.device.queue_max_samples)
    pipeline_cfg = PipelineConfig(
        sampling_rate_hz=config.device.sampling_rate_hz,
        processed_sensor_column=config.device.processed_sensor_column,
        filter=config.filter,
        artifact=config.artifact,
        calibration=config.calibration,
        adaptation=config.adaptation,
        hold=config.hold,
        raw_qc=config.raw_qc,
    )
    pipeline_state = create_pipeline_state(pipeline_cfg)
    session_started_at = datetime.now().astimezone().isoformat()

    try:
        print("Starting acquisition...")
        belt = BreathBelt(
            mac_address=config.device.mac_address,
            sampling_rate=config.device.sampling_rate_hz,
            channels=config.device.channels,
            read_chunk_size=config.device.chunk_size,
            queue_max_samples=config.device.queue_max_samples,
            timeout_s=config.device.timeout_s,
            read_error_backoff_s=config.device.read_error_backoff_s,
            retries=config.device.retries,
            retry_delay_s=config.device.retry_delay_s,
        )
        belt.start()

        session_writer = SessionWriter(config.output.root_dir, config)

        if config.display.enable_plot:
            plot_breathing_channel, setup_live_plot = _import_plot_helpers()
            _, ax_final, line_final, blit_manager_final = setup_live_plot(
                "Normalized Breathing Signal (0-1 range)"
            )

        if config.lsl.enable:
            LSLBreathingSender = _import_lsl_sender()
            lsl_sender = LSLBreathingSender(
                name=config.lsl.stream_name,
                type=config.lsl.stream_type,
                nominal_srate=config.device.sampling_rate_hz,
                source_id=config.lsl.source_id,
            )

        print(
            f"Starting startup calibration for {config.calibration.duration_s:.1f}s "
            f"({pipeline_cfg.calibration_target_samples} processed samples)."
        )
        print("Breathe normally and include full inhale/exhale range.")
        print("Press 'c' to stop acquisition.")

        runtime_plot_started = False
        while not keyboard.is_pressed("c"):
            data = belt.get_all()
            if data.size == 0:
                time.sleep(0.001)
                continue

            for row in data:
                sample, pipeline_state = process_device_row(row, pipeline_state, pipeline_cfg)
                session_writer.write_device_row(
                    stage=sample.stage,
                    sample_index=sample.sample_index,
                    relative_time_s=sample.relative_time_s,
                    device_row=row,
                )
                session_writer.write_signal_sample(sample)

                for message in sample.messages:
                    print(message)
                for event in sample.qc_events:
                    print(f"WARNING [{event.event_type}]: {event.message}")
                    session_writer.write_qc_event(event)

                if sample.stage == "runtime":
                    if sample.sample_index == 0 and not runtime_plot_started:
                        sample_indices.clear()
                        normalized_signal.clear()
                        runtime_plot_started = True
                    sample_indices.append(sample.sample_index)
                    if sample.normalized_value is not None:
                        normalized_signal.append(sample.normalized_value)
                        if lsl_sender is not None:
                            lsl_sender.send(sample.normalized_value)
                        print(f"Normalized: {sample.normalized_value:.4f}")

            if config.display.enable_plot and normalized_signal:
                normalized_array = np.asarray(normalized_signal, dtype=float)
                n_plot = min(
                    config.display.plot_window_length,
                    len(normalized_array),
                    len(sample_indices),
                )
                plot_window = normalized_array[-n_plot:]
                window_min = float(np.min(plot_window))
                window_max = float(np.max(plot_window))
                if (
                    config.display.debug_plot_window_bounds
                    and sample_indices
                    and (sample_indices[-1] % config.device.sampling_rate_hz) < len(data)
                ):
                    print(
                        f"Plot window range check: min={window_min:.4f}, "
                        f"max={window_max:.4f}, points={n_plot}"
                    )
                if window_min < 0.0 or window_max > 1.0:
                    print(
                        "WARNING: plotted window out of [0,1] "
                        f"(min={window_min:.6f}, max={window_max:.6f})"
                    )
                plot_breathing_channel(
                    plot_window,
                    list(sample_indices)[-n_plot:],
                    live=True,
                    ax=ax_final,
                    line=line_final,
                    blit_manager=blit_manager_final,
                )
    finally:
        print("Stopping acquisition...")
        if belt is not None:
            belt.stop()

        session_ended_at = datetime.now().astimezone().isoformat()
        if session_writer is not None:
            metadata = build_session_metadata(
                config=config,
                resolved_config_path=session_writer.resolved_config_path,
                software_version=__version__,
                started_at=session_started_at,
                ended_at=session_ended_at,
                calibration_result=pipeline_state.calibration_result,
                adaptive_state=pipeline_state.adaptive_state,
                qc_summary=raw_qc_summary(pipeline_state.qc_state),
            )
            session_writer.finalize(metadata)
        print("Connection closed.")


if __name__ == "__main__":
    raise SystemExit(main())
