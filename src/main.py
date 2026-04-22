"""Command-line entrypoint for live breathing-belt acquisition."""

from __future__ import annotations

from collections import deque
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback
from typing import Callable, TextIO

import numpy as np

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src import __version__
    from src.pipeline import (
        PipelineConfig,
        ProcessingMode,
        create_pipeline_state,
        process_device_row,
        reset_pipeline_state_for_source_gap,
    )
    from src.quality import raw_qc_summary
    from src.session_writer import SessionWriter, build_session_metadata
    from src.settings import (
        AppConfig,
        expected_bitalino_row_width,
        load_config,
        validate_live_acquisition_config,
    )

    def _import_breath_belt():
        from src.connect import BreathBelt

        return BreathBelt

    def _import_plot_helpers():
        from src.plot import setup_live_plots, update_live_plots

        return setup_live_plots, update_live_plots

    def _import_lsl_sender():
        from src.lsl_out import LSLBreathingSender

        return LSLBreathingSender
else:
    from . import __version__
    from .pipeline import (
        PipelineConfig,
        ProcessingMode,
        create_pipeline_state,
        process_device_row,
        reset_pipeline_state_for_source_gap,
    )
    from .quality import raw_qc_summary
    from .session_writer import SessionWriter, build_session_metadata
    from .settings import (
        AppConfig,
        expected_bitalino_row_width,
        load_config,
        validate_live_acquisition_config,
    )

    def _import_breath_belt():
        from .connect import BreathBelt

        return BreathBelt

    def _import_plot_helpers():
        from .plot import setup_live_plots, update_live_plots

        return setup_live_plots, update_live_plots

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
        validate_live_acquisition_config(config)
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


def prompt_processing_mode(
    input_func: Callable[[str], str] = input,
    output_stream: TextIO | None = None,
) -> tuple[int, ProcessingMode]:
    """Prompt for a numbered live-processing mode and return the selection."""

    stream = sys.stdout if output_stream is None else output_stream
    while True:
        print("Select processing mode:", file=stream)
        print("1 = Legacy control (0..1, hold/smoothing)", file=stream)
        print("2 = Realtime movement proxy (centered, unclamped)", file=stream)
        print("3 = Adaptive live control (rhythm-flexible)", file=stream)
        try:
            selection = input_func("Enter mode number [1]: ").strip()
        except EOFError:
            return 1, "control"

        if selection == "":
            return 1, "control"
        if selection == "1":
            return 1, "control"
        if selection == "2":
            return 2, "movement"
        if selection == "3":
            return 3, "adaptive"
        print("Invalid selection. Enter 1, 2 or 3.", file=stream)


def _processing_mode_description(processing_mode: ProcessingMode) -> str:
    if processing_mode == "movement":
        return "Realtime movement proxy (centered, unclamped)"
    if processing_mode == "adaptive":
        return "Adaptive live control (rhythm-flexible)"
    return "Legacy control (0..1, hold/smoothing)"


def _plot_panel_config(processing_mode: ProcessingMode) -> tuple[str, str]:
    if processing_mode == "movement":
        return "Movement Proxy (Centered)", "Movement Proxy"
    if processing_mode == "adaptive":
        return "Adaptive Breath Level (0-1)", "Adaptive Breath Level"
    return "Breath Level (0-1)", "Breath Level"


def _lsl_config_for_mode(
    config: AppConfig,
    processing_mode: ProcessingMode,
) -> tuple[str, str, str, tuple[str, ...]]:
    if processing_mode == "movement":
        return (
            f"{config.lsl.stream_name}Movement",
            "BreathingMovement",
            f"{config.lsl.source_id}_movement",
            ("movement_value",),
        )
    if processing_mode == "adaptive":
        return (
            f"{config.lsl.stream_name}Adaptive",
            "BreathingAdaptive",
            f"{config.lsl.source_id}_adaptive",
            ("breath_level",),
        )
    return (
        config.lsl.stream_name,
        config.lsl.stream_type,
        config.lsl.source_id,
        ("breath_level",),
    )


def _lsl_event_config_for_mode(
    config: AppConfig,
    processing_mode: ProcessingMode,
) -> tuple[str, str, str, tuple[str]]:
    stream_name, _, source_id, _ = _lsl_config_for_mode(config, processing_mode)
    return (
        f"{stream_name}Events",
        "BreathingEvents",
        f"{source_id}_events",
        ("event_code",),
    )


def _lsl_timing_metadata(config: AppConfig) -> dict[str, str | float]:
    return {
        "timestamp_domain": "local_clock",
        "timestamp_origin": "host_estimated_segment_anchor",
        "chunk_backfill_policy": "nominal_fs_continuation_across_contiguous_reads",
        "constant_delay_s": config.lsl.constant_delay_s,
        "discontinuity_policy": "preserve_timestamp_gaps_after_loss",
    }


def _effective_lsl_timestamp(capture_time_lsl_s: float, constant_delay_s: float) -> float:
    return float(capture_time_lsl_s - constant_delay_s)


def _flush_control_span(
    sender,
    *,
    samples: list[float],
    timestamps: list[float],
    lsl_run_stats: dict[str, int | str],
) -> None:
    if sender is None or not samples:
        samples.clear()
        timestamps.clear()
        return

    if len(samples) == 1:
        sender.send(samples[0], timestamp=timestamps[0])
        lsl_run_stats["control_samples_sent"] += 1
        lsl_run_stats["control_samples_sent_individually"] += 1
    else:
        sender.send_chunk(samples, timestamps=timestamps)
        lsl_run_stats["control_samples_sent"] += len(samples)
        lsl_run_stats["control_samples_sent_via_chunks"] += len(samples)
        lsl_run_stats["control_chunks_sent"] += 1
    samples.clear()
    timestamps.clear()


def run_acquisition(config: AppConfig) -> None:
    """Acquire, normalize, plot, stream, and persist breathing-belt data."""

    import keyboard

    validate_live_acquisition_config(config)
    BreathBelt = _import_breath_belt()
    plot_window_samples = config.display.plot_window_length
    belt = None
    session_writer = None
    lsl_control_sender = None
    lsl_event_sender = None
    raw_ax = None
    raw_line = None
    normalized_ax = None
    normalized_line = None
    blit_manager = None
    raw_sample_indices: deque[int] = deque(maxlen=plot_window_samples)
    raw_signal: deque[float] = deque(maxlen=plot_window_samples)
    normalized_sample_indices: deque[int] = deque(maxlen=plot_window_samples)
    normalized_signal: deque[float] = deque(maxlen=plot_window_samples)
    peak_sample_indices: deque[int] = deque(maxlen=plot_window_samples)
    peak_raw_values: deque[float] = deque(maxlen=plot_window_samples)
    trough_sample_indices: deque[int] = deque(maxlen=plot_window_samples)
    trough_raw_values: deque[float] = deque(maxlen=plot_window_samples)
    previous_source_sample_index: int | None = None
    previous_runtime_lsl_timestamp: float | None = None
    reported_dropped_rows_total = 0
    lsl_run_stats: dict[str, int | str] = {
        "control_send_strategy": "hybrid_explicit_timestamps",
        "control_samples_sent": 0,
        "control_samples_sent_individually": 0,
        "control_samples_sent_via_chunks": 0,
        "control_chunks_sent": 0,
        "event_samples_sent": 0,
        "queue_dropped_rows_total": 0,
        "observed_gap_count": 0,
    }
    selected_mode_number, processing_mode = prompt_processing_mode()
    print(
        "Selected mode "
        f"{selected_mode_number}: "
        + _processing_mode_description(processing_mode)
    )
    pipeline_cfg = PipelineConfig(
        sampling_rate_hz=config.device.sampling_rate_hz,
        processed_sensor_column=config.device.processed_sensor_column,
        invert_signal=config.device.invert_signal,
        filter=config.filter,
        calibration=config.calibration,
        adaptation=config.adaptation,
        hold=config.hold,
        output_smoothing=config.output_smoothing,
        extrema=config.extrema,
        raw_qc=config.raw_qc,
        processing_mode=processing_mode,
        movement=config.movement,
    )
    pipeline_state = create_pipeline_state(pipeline_cfg)
    session_started_at = datetime.now().astimezone().isoformat()
    device_sample_width = expected_bitalino_row_width(config.device.channels)

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

        session_writer = SessionWriter(
            config.output.root_dir,
            config,
            device_sample_width=device_sample_width,
        )

        if config.display.enable_plot:
            setup_live_plots, update_live_plots = _import_plot_helpers()
            normalized_title, normalized_label = _plot_panel_config(processing_mode)
            _, raw_ax, raw_line, normalized_ax, normalized_line, blit_manager = (
                setup_live_plots(
                    normalized_title=normalized_title,
                    normalized_label=normalized_label,
                )
            )
        else:
            update_live_plots = None

        if config.lsl.enable:
            LSLBreathingSender = _import_lsl_sender()
            stream_name, stream_type, source_id, channel_labels = _lsl_config_for_mode(
                config,
                processing_mode,
            )
            lsl_control_sender = LSLBreathingSender(
                name=stream_name,
                type=stream_type,
                channel_count=1,
                nominal_srate=config.device.sampling_rate_hz,
                source_id=source_id,
                channel_labels=channel_labels,
                timing_metadata=_lsl_timing_metadata(config),
            )
            event_stream_name, event_stream_type, event_source_id, event_channel_labels = (
                _lsl_event_config_for_mode(config, processing_mode)
            )
            lsl_event_sender = LSLBreathingSender(
                name=event_stream_name,
                type=event_stream_type,
                channel_count=1,
                nominal_srate=0,
                source_id=event_source_id,
                channel_labels=event_channel_labels,
                timing_metadata=_lsl_timing_metadata(config),
                event_code_map={
                    1.0: "inhale_peak",
                    -1.0: "exhale_trough",
                },
            )

        print(
            f"Starting startup calibration for {config.calibration.duration_s:.1f}s "
            f"({pipeline_cfg.calibration_target_samples} processed samples)."
        )
        print("Breathe normally and include full inhale/exhale range.")
        print("Press 'c' to stop acquisition.")

        while not keyboard.is_pressed("c"):
            acquired_rows = belt.get_all()
            if len(acquired_rows) == 0:
                time.sleep(0.001)
                continue

            dropped_rows_total = int(getattr(belt, "dropped_rows_total", 0))
            if dropped_rows_total > reported_dropped_rows_total:
                dropped_delta = dropped_rows_total - reported_dropped_rows_total
                print(
                    "WARNING [queue_overflow]: "
                    f"dropped {dropped_delta} queued samples before processing."
                )
                reported_dropped_rows_total = dropped_rows_total
                lsl_run_stats["queue_dropped_rows_total"] = dropped_rows_total

            control_span_samples: list[float] = []
            control_span_timestamps: list[float] = []
            last_control_source_sample_index: int | None = None

            for acquired_row in acquired_rows:
                if (
                    previous_source_sample_index is not None
                    and acquired_row.source_sample_index != previous_source_sample_index + 1
                ):
                    missing_samples = max(
                        acquired_row.source_sample_index - previous_source_sample_index - 1,
                        0,
                    )
                    lsl_run_stats["observed_gap_count"] += 1
                    reset_pipeline_state_for_source_gap(pipeline_state)
                    previous_runtime_lsl_timestamp = None
                    print(
                        "WARNING [source_gap]: "
                        f"detected non-contiguous source samples ({missing_samples} "
                        "missing sample(s)); reset short-term pipeline state."
                    )
                previous_source_sample_index = acquired_row.source_sample_index

                sample, pipeline_state = process_device_row(
                    acquired_row.device_row,
                    pipeline_state,
                    pipeline_cfg,
                )
                lsl_timestamp_s = _effective_lsl_timestamp(
                    acquired_row.capture_time_lsl_s,
                    config.lsl.constant_delay_s,
                )
                session_writer.write_device_row(
                    stage=sample.stage,
                    sample_index=sample.sample_index,
                    relative_time_s=sample.relative_time_s,
                    device_row=acquired_row.device_row,
                    source_sample_index=acquired_row.source_sample_index,
                    capture_time_lsl_s=acquired_row.capture_time_lsl_s,
                    lsl_timestamp_s=lsl_timestamp_s,
                )

                for message in sample.messages:
                    print(message)
                for event in sample.qc_events:
                    print(f"WARNING [{event.event_type}]: {event.message}")
                    session_writer.write_qc_event(event)

                raw_sample_indices.append(acquired_row.source_sample_index)
                raw_signal.append(sample.selected_sensor_raw)

                event_timestamp_lsl_s: float | None = None
                if sample.stage == "runtime":
                    runtime_value = (
                        sample.movement_value if processing_mode == "movement" else sample.normalized_value
                    )
                    if runtime_value is not None:
                        normalized_sample_indices.append(acquired_row.source_sample_index)
                        normalized_signal.append(runtime_value)
                        if sample.extrema_event_code > 0.0:
                            peak_sample_indices.append(acquired_row.source_sample_index)
                            peak_raw_values.append(sample.selected_sensor_raw)
                        elif sample.extrema_event_code < 0.0:
                            trough_sample_indices.append(acquired_row.source_sample_index)
                            trough_raw_values.append(sample.selected_sensor_raw)
                        if lsl_control_sender is not None:
                            if (
                                last_control_source_sample_index is not None
                                and acquired_row.source_sample_index
                                != last_control_source_sample_index + 1
                            ):
                                _flush_control_span(
                                    lsl_control_sender,
                                    samples=control_span_samples,
                                    timestamps=control_span_timestamps,
                                    lsl_run_stats=lsl_run_stats,
                                )
                            control_span_samples.append(float(runtime_value))
                            control_span_timestamps.append(lsl_timestamp_s)
                            last_control_source_sample_index = acquired_row.source_sample_index

                        if (
                            lsl_event_sender is not None
                            and sample.extrema_event_code != 0.0
                            and previous_runtime_lsl_timestamp is not None
                        ):
                            event_timestamp_lsl_s = previous_runtime_lsl_timestamp
                            lsl_event_sender.send(
                                float(sample.extrema_event_code),
                                timestamp=event_timestamp_lsl_s,
                            )
                            lsl_run_stats["event_samples_sent"] += 1
                        if processing_mode == "movement":
                            print(f"Movement proxy: {runtime_value:.4f}")
                        elif processing_mode == "adaptive":
                            print(f"Adaptive normalized: {runtime_value:.4f}")
                        else:
                            print(f"Normalized: {runtime_value:.4f}")
                        if sample.extrema_event_label is not None:
                            print(f"Breath event: {sample.extrema_event_label}")
                        previous_runtime_lsl_timestamp = lsl_timestamp_s

                session_writer.write_signal_sample(
                    sample,
                    source_sample_index=acquired_row.source_sample_index,
                    capture_time_lsl_s=acquired_row.capture_time_lsl_s,
                    lsl_timestamp_s=lsl_timestamp_s,
                    event_timestamp_lsl_s=event_timestamp_lsl_s,
                )

            _flush_control_span(
                lsl_control_sender,
                samples=control_span_samples,
                timestamps=control_span_timestamps,
                lsl_run_stats=lsl_run_stats,
            )
            session_writer.flush_incremental()

            if config.display.enable_plot and raw_signal and update_live_plots is not None:
                if normalized_signal:
                    normalized_array = np.asarray(normalized_signal, dtype=float)
                    window_min = float(np.min(normalized_array))
                    window_max = float(np.max(normalized_array))
                else:
                    normalized_array = np.asarray([], dtype=float)
                    window_min = 0.0
                    window_max = 0.0

                if (
                    config.display.debug_plot_window_bounds
                    and normalized_sample_indices
                    and (
                        normalized_sample_indices[-1] % config.device.sampling_rate_hz
                    ) < len(acquired_rows)
                ):
                    print(
                        f"Plot window range check: min={window_min:.4f}, "
                        f"max={window_max:.4f}, points={len(normalized_array)}"
                    )
                if (
                    processing_mode != "movement"
                    and normalized_array.size > 0
                    and (window_min < 0.0 or window_max > 1.0)
                ):
                    print(
                        "WARNING: plotted window out of [0,1] "
                        f"(min={window_min:.6f}, max={window_max:.6f})"
                    )

                update_live_plots(
                    raw_signal,
                    raw_sample_indices,
                    normalized_signal,
                    normalized_sample_indices,
                    raw_ax=raw_ax,
                    raw_line=raw_line,
                    normalized_ax=normalized_ax,
                    normalized_line=normalized_line,
                    peak_times=peak_sample_indices,
                    peak_values=peak_raw_values,
                    trough_times=trough_sample_indices,
                    trough_values=trough_raw_values,
                    normalized_clip_range=(0.0, 1.0) if processing_mode != "movement" else None,
                    normalized_fixed_ylim=(0.0, 1.0) if processing_mode != "movement" else None,
                    normalized_autoscale_y=processing_mode == "movement",
                    blit_manager=blit_manager,
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
                processing_mode=processing_mode,
                selected_mode_number=selected_mode_number,
                lsl_run_stats=lsl_run_stats,
            )
            session_writer.finalize(metadata)
        print("Connection closed.")


if __name__ == "__main__":
    raise SystemExit(main())
