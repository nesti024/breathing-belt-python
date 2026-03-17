"""Live acquisition and adaptive normalization for respiration-belt recordings.

Signal path:
raw sensor sample -> high-pass filter -> low-pass filter -> sign inversion ->
reversal-only spike suppression -> robust calibration -> adaptive normalization
-> live plot and Lab Streaming Layer (LSL) output.
"""

from collections import deque
import time
import traceback

import keyboard
import numpy as np

from calibration import (
    AdaptiveRangeConfig,
    CalibrationConfig,
    initialize_adaptive_range,
    run_range_calibration,
    update_adaptive_range,
)
from connect import BreathBelt
from lsl_out import LSLBreathingSender
from plot import plot_breathing_channel, setup_live_plot
from preprocessing import (
    get_high_pass_filter_coeffs,
    get_low_pass_filter_coeffs,
    high_pass_filter_sample,
    low_pass_filter_sample,
)

# BITalino configuration.
mac_address = "98:D3:C1:FD:FF:DB"
sampling_rate = 100  # Hz
chunk_size = 10  # samples
sensor_column_index = 5

# Rolling display buffers.
buffer_length = 1000  # samples
plot_window_length = 200  # samples

# Real-time respiration filter.
# The high-pass stage attenuates quasi-static baseline drift and the low-pass
# stage limits the control signal to the respiratory band.
hp_cutoff_hz = 0.005
hp_order = 1
lp_cutoff_hz = 1.5
lp_order = 2

# Reversal-only artifact suppression.
spike_threshold = 2.5
artifact_window = 10  # samples

# Robust percentile calibration of the processed respiration signal.
calibration_duration_sec = 15.0
calibration_percentile_lo = 5.0
calibration_percentile_hi = 95.0
calibration_amplitude_floor = 1e-3

# Slow runtime adaptation of the normalization center and amplitude.
adaptive_center_enabled = True
adaptive_center_tau_sec = 600.0
adaptive_amplitude_tau_sec = 120.0

# Faster adaptation immediately after calibration to settle the live mapping.
adaptive_startup_duration_sec = 60.0
adaptive_startup_center_tau_sec = 25.0
adaptive_startup_amplitude_tau_sec = 25.0

# Breath-hold gating for adaptation.
# Activity is the short-window mean absolute derivative |dx/dt| of the
# processed signal in processed-signal units per second.
hold_activity_window_ms = 500
hold_activity_ratio_per_sec_enter = 1.0
hold_activity_ratio_per_sec_exit = 1.5
hold_activity_floor_per_sec = 0.01

# Optional display diagnostics.
debug_plot_window_bounds = True


def acquire_data() -> None:
    """Acquire, calibrate, normalize, plot, and stream the first belt channel."""
    belt = None
    try:
        print("Starting acquisition...")
        belt = BreathBelt(
            mac_address=mac_address,
            sampling_rate=sampling_rate,
            channels=(0, 1),
            read_chunk_size=chunk_size,
            queue_max_samples=buffer_length,
            timeout_s=0.25,
            read_error_backoff_s=0.05,
            retries=3,
            retry_delay_s=2.0,
        )
        belt.start()

        sample_indices = deque(maxlen=buffer_length)
        normalized_signal = deque(maxlen=buffer_length)
        sample_count = 0

        _, ax_final, line_final, blit_manager_final = setup_live_plot(
            "Normalized Breathing Signal (0-1 range)"
        )

        sos_hp, zi_hp = None, None
        sos_lp, zi_lp = None, None
        filter_initialized = False

        print("Press 'c' to stop acquisition.")

        lsl_sender = LSLBreathingSender(nominal_srate=sampling_rate)

        hold_activity_window_samples = int(
            round((hold_activity_window_ms / 1000.0) * sampling_rate)
        )
        hold_activity_window_samples = max(3, hold_activity_window_samples)
        recent_abs_velocity = deque(maxlen=hold_activity_window_samples)
        recent_raw_deltas = deque(maxlen=max(artifact_window, 3))
        previous_cleaned_value = None
        previous_filtered_value = None
        hold_mode_active = False

        calibration_cfg = CalibrationConfig(
            duration_s=calibration_duration_sec,
            fs_hz=float(sampling_rate),
            percentile_lo=calibration_percentile_lo,
            percentile_hi=calibration_percentile_hi,
            # The processed signal is not expressed in raw ADC units, so
            # saturation checks at the hardware rails are not meaningful here.
            saturation_lo=float("-inf"),
            saturation_hi=float("inf"),
            amplitude_floor=calibration_amplitude_floor,
        )
        adaptive_cfg_startup = AdaptiveRangeConfig(
            fs_hz=float(sampling_rate),
            center_tau_s=adaptive_startup_center_tau_sec,
            amplitude_tau_s=adaptive_startup_amplitude_tau_sec,
            amplitude_floor=calibration_amplitude_floor,
        )
        adaptive_cfg_runtime = AdaptiveRangeConfig(
            fs_hz=float(sampling_rate),
            center_tau_s=adaptive_center_tau_sec,
            amplitude_tau_s=adaptive_amplitude_tau_sec,
            amplitude_floor=calibration_amplitude_floor,
        )

        startup_target_samples = max(
            1, int(round(adaptive_startup_duration_sec * sampling_rate))
        )
        calibration_target_samples = max(
            1, int(round(calibration_cfg.duration_s * calibration_cfg.fs_hz))
        )

        runtime_processed_samples = 0
        startup_mode_active = False
        calibration_samples: list[float] = []
        calibration_result = None
        adaptive_state = None
        calibration_last_reported_sec = -1

        print(
            f"Starting startup calibration for {calibration_cfg.duration_s:.1f}s "
            f"({calibration_target_samples} processed samples)."
        )
        print("Breathe normally and include full inhale/exhale range.")

        def process_live_sample(filtered_value: float) -> tuple[float, bool]:
            """Suppress isolated directional reversals relative to the local trend."""
            nonlocal previous_filtered_value

            sample_value = float(filtered_value)
            if previous_filtered_value is None:
                previous_filtered_value = sample_value
                recent_raw_deltas.clear()
                return sample_value, False

            raw_delta = sample_value - previous_filtered_value
            recent_raw_deltas.append(raw_delta)

            if len(recent_raw_deltas) >= 3:
                trend_value = float(np.mean(recent_raw_deltas))
                delta_scale = max(1e-3, float(np.std(recent_raw_deltas)))
            else:
                trend_value = raw_delta
                delta_scale = max(1e-3, abs(raw_delta))

            is_reversal = (
                abs(raw_delta) > spike_threshold * delta_scale
                and abs(trend_value) > 0.25 * delta_scale
                and np.sign(raw_delta) != np.sign(trend_value)
            )

            previous_filtered_value = sample_value
            if is_reversal:
                cleaned_value = sample_value - raw_delta
            else:
                cleaned_value = sample_value

            return float(cleaned_value), bool(is_reversal)

        while not keyboard.is_pressed("c"):
            data = belt.get_all()
            if data.size == 0:
                time.sleep(0.001)
                continue

            filtered_chunk: list[float] = []
            for sample in data:
                raw_sensor_value = float(sample[sensor_column_index])
                sample_indices.append(sample_count)
                sample_count += 1

                if not filter_initialized:
                    sos_hp, zi_hp = get_high_pass_filter_coeffs(
                        hp_cutoff_hz,
                        sampling_rate,
                        hp_order,
                        initial_value=raw_sensor_value,
                    )
                    # The high-pass output is zero-centered by design, so the
                    # low-pass stage is initialized at zero to avoid a startup
                    # bias in the filtered trace.
                    sos_lp, zi_lp = get_low_pass_filter_coeffs(
                        lp_cutoff_hz,
                        sampling_rate,
                        lp_order,
                        initial_value=0.0,
                    )
                    filter_initialized = True

                high_passed_value, zi_hp = high_pass_filter_sample(
                    raw_sensor_value,
                    sos_hp,
                    zi_hp,
                )
                low_passed_value, zi_lp = low_pass_filter_sample(
                    high_passed_value,
                    sos_lp,
                    zi_lp,
                )
                filtered_chunk.append(-float(low_passed_value))

            cleaned_chunk = []
            artifact_mask_chunk = []
            for filtered_value in filtered_chunk:
                cleaned_value, is_artifact = process_live_sample(filtered_value)
                cleaned_chunk.append(cleaned_value)
                artifact_mask_chunk.append(is_artifact)

            if not cleaned_chunk:
                continue

            if calibration_result is None:
                calibration_samples.extend(float(value) for value in cleaned_chunk)
                collected = len(calibration_samples)
                clipped_collected = min(collected, calibration_target_samples)
                reported_sec = int(clipped_collected / calibration_cfg.fs_hz)

                if reported_sec != calibration_last_reported_sec:
                    calibration_last_reported_sec = reported_sec
                    print(
                        f"Calibration progress: {clipped_collected}/"
                        f"{calibration_target_samples} samples"
                    )

                if collected < calibration_target_samples:
                    continue

                calibration_samples = calibration_samples[:calibration_target_samples]
                calibration_result = run_range_calibration(
                    calibration_samples,
                    calibration_cfg,
                )
                adaptive_state = initialize_adaptive_range(
                    calibration_samples,
                    calibration_result,
                    adaptive_cfg_startup,
                )

                print("Calibration complete.")
                print(
                    "Calibration map: "
                    f"center={calibration_result.center:.6f}, "
                    f"amplitude={calibration_result.amplitude:.6f}, "
                    f"min={calibration_result.global_min:.6f}, "
                    f"max={calibration_result.global_max:.6f}"
                )
                if calibration_result.saturated:
                    print(
                        "WARNING: Calibration saturation detected "
                        f"({calibration_result.saturated_count}/"
                        f"{calibration_result.n_samples} samples at rails)."
                    )

                # Reset display and adaptation diagnostics so the runtime trace
                # begins immediately after the calibration interval.
                sample_indices.clear()
                normalized_signal.clear()
                recent_abs_velocity.clear()
                recent_raw_deltas.clear()
                previous_cleaned_value = None
                previous_filtered_value = None
                hold_mode_active = False
                sample_count = 0
                runtime_processed_samples = 0
                startup_mode_active = True
                print(
                    "Starting fast post-calibration adaptation for "
                    f"{adaptive_startup_duration_sec:.1f}s."
                )
                continue

            if adaptive_state is None:
                continue

            normalized_chunk = []
            for value, is_artifact in zip(cleaned_chunk, artifact_mask_chunk):
                value_float = float(value)
                if previous_cleaned_value is None:
                    abs_velocity = 0.0
                else:
                    abs_velocity = abs(value_float - previous_cleaned_value) * sampling_rate
                previous_cleaned_value = value_float
                recent_abs_velocity.append(abs_velocity)

                if len(recent_abs_velocity) < recent_abs_velocity.maxlen:
                    activity_value = float("inf")
                else:
                    activity_value = float(np.mean(recent_abs_velocity))

                enter_threshold = max(
                    hold_activity_floor_per_sec,
                    adaptive_state.amplitude * hold_activity_ratio_per_sec_enter,
                )
                exit_threshold = max(
                    hold_activity_floor_per_sec,
                    adaptive_state.amplitude * hold_activity_ratio_per_sec_exit,
                )

                # Hysteretic hold detection prevents amplitude or center updates
                # during near-static intervals, such as deliberate breath holds.
                if not hold_mode_active:
                    if len(recent_abs_velocity) >= recent_abs_velocity.maxlen:
                        hold_mode_active = activity_value < enter_threshold
                elif activity_value > exit_threshold:
                    hold_mode_active = False

                allow_amplitude_update = False
                allow_center_update = (
                    adaptive_center_enabled
                    and (not bool(is_artifact))
                    and (not hold_mode_active)
                )
                active_cfg = (
                    adaptive_cfg_startup if startup_mode_active else adaptive_cfg_runtime
                )
                normalized_value, adaptive_state = update_adaptive_range(
                    x=value_float,
                    state=adaptive_state,
                    cfg=active_cfg,
                    allow_update=allow_center_update,
                    allow_center_update=allow_center_update,
                    allow_amplitude_update=allow_amplitude_update,
                )
                normalized_chunk.append(normalized_value)
                runtime_processed_samples += 1

                if startup_mode_active and runtime_processed_samples >= startup_target_samples:
                    startup_mode_active = False
                    print(
                        "Startup adaptation complete. "
                        "Switched to slow runtime tracking."
                    )

            normalized_signal.extend(normalized_chunk)
            normalized_array = np.asarray(normalized_signal, dtype=float)

            n_plot = min(plot_window_length, len(normalized_array), len(sample_indices))
            if n_plot > 0:
                plot_window = normalized_array[-n_plot:]
                window_min = float(np.min(plot_window))
                window_max = float(np.max(plot_window))

                if debug_plot_window_bounds and (sample_count % sampling_rate) < len(cleaned_chunk):
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

            if normalized_chunk:
                newest_value = normalized_chunk[-1]
                print(f"Normalized: {newest_value:.4f}")
                for value in normalized_chunk:
                    lsl_sender.send(float(value))
    except Exception:
        traceback.print_exc()
    finally:
        print("Stopping acquisition...")
        if belt is not None:
            belt.stop()
        print("Connection closed.")


if __name__ == "__main__":
    acquire_data()
