import keyboard  # Import the keyboard module

import time
import traceback
from collections import deque

from calibration import (
    AdaptiveRangeConfig,
    CalibrationConfig,
    initialize_adaptive_range,
    run_range_calibration,
    update_adaptive_range,
)
from connect import *
from preprocessing import *
from plot import *
import numpy as np

# Replace with your BITalino's MAC address
mac_address = "98:D3:C1:FD:FF:DB"
sampling_rate = 100  # Sampling rate in Hz
# Lower chunk size to reduce end-to-end latency in the live signal.
chunk_size = 10  # Number of samples to read per chunk

# Two-stage real-time filtering for hold stability:
# 1) very-low high-pass for slow baseline drift suppression
# 2) low-pass for respiration band cleanup
hp_cutoff_hz = 0.005
hp_order = 1
lp_cutoff_hz = 1.5
lp_order = 2

# Artifact/spike processing parameters
spike_threshold = 2.5
artifact_window = 10
# One-time robust range calibration settings
calibration_duration_sec = 15.0
calibration_percentile_lo = 5.0
calibration_percentile_hi = 95.0
calibration_amplitude_floor = 1e-3

# Slow adaptive map update settings (minutes scale).
adaptive_center_enabled = True
adaptive_center_tau_sec = 600.0
adaptive_amplitude_tau_sec = 120.0

# Fast post-calibration adaptation settings (seconds scale).
adaptive_startup_duration_sec = 60.0
adaptive_startup_center_tau_sec = 25.0
adaptive_startup_amplitude_tau_sec = 25.0

# Freeze adaptation while signal activity stays very low (breath holds).
# Activity metric: smoothed |dx/dt| over a short rolling window.
hold_activity_window_ms = 500
hold_activity_ratio_per_sec_enter = 1.0
hold_activity_ratio_per_sec_exit = 1.5
hold_activity_floor_per_sec = 0.01

# Debug: print min/max of the exact plot window once per second.
debug_plot_window_bounds = True


def acquire_data():
    """
    Acquire data from two PLUX PZT sensors using a BITalino Corce.
    """
    belt = None
    try:
        print("Starting acquisition...")
        belt = BreathBelt(
            mac_address=mac_address,
            sampling_rate=sampling_rate,
            channels=(0, 1),
            read_chunk_size=chunk_size,
            queue_max_samples=1000,
            timeout_s=0.25,
            read_error_backoff_s=0.05,
            retries=3,
            retry_delay_s=2.0,
        )
        belt.start()

        # Prepare rolling buffers for data storage (auto-trim to max length)
        sample_indices = deque(maxlen=1000)
        channel_1_raw = deque(maxlen=1000)
        channel_1_filtered = deque(maxlen=1000)
        channel_1_filtered_processed = deque(maxlen=1000)  # For processed signal after artifact removal
        normalized_signal = deque(maxlen=1000)
        sample_count = 0
        # Store inhale/exhale events as (sample_index, event_type)
        breath_events = []  # event_type: 'in' or 'out'



        # Set up live plot for the final processed and normalized signal
        fig_final, ax_final, line_final, blit_manager_final = setup_live_plot('Normalized Breathing Signal (0-1 range)')

        # Filter coefficients will be initialized lazily with the first sample
        sos_hp, zi_1_hp = None, None
        sos_lp, zi_1_lp = None, None
        filter_initialized = False
        
        last_inhale = False
        last_exhale = False

        print("Press 'c' to stop acquisition.")

        # Initialize LSL sender
        from lslOut import LSLBreathingSender
        lsl_sender = LSLBreathingSender(nominal_srate=sampling_rate)

        hold_activity_window_samples = int(round((hold_activity_window_ms / 1000.0) * sampling_rate))
        hold_activity_window_samples = max(3, hold_activity_window_samples)
        recent_abs_velocity = deque(maxlen=hold_activity_window_samples)
        recent_raw_deltas = deque(maxlen=max(artifact_window, 3))
        previous_cleaned_value = None
        previous_filtered_value = None
        hold_mode_active = False

        # Calibrate on the same processed signal path used at runtime.
        calibration_cfg = CalibrationConfig(
            duration_s=calibration_duration_sec,
            fs_hz=float(sampling_rate),
            percentile_lo=calibration_percentile_lo,
            percentile_hi=calibration_percentile_hi,
            # Processed signal is not in ADC units; disable rail checks here.
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
        runtime_processed_samples = 0
        startup_mode_active = False
        calibration_target_samples = max(
            1,
            int(round(calibration_cfg.duration_s * calibration_cfg.fs_hz)),
        )
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
            """Apply reversal-only spike suppression without extra live smoothing."""
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
            cleaned_value = previous_filtered_value if not is_reversal else (
                previous_filtered_value - raw_delta
            )
            return float(cleaned_value), bool(is_reversal)

        while not keyboard.is_pressed('c'):
            # Check for inhale ('i') and exhale ('o') key presses
            if keyboard.is_pressed('i') and not last_inhale:
                breath_events.append((sample_count, 'in'))
                print(f"Inhale event at sample {sample_count}")
                last_inhale = True
            elif not keyboard.is_pressed('i'):
                last_inhale = False
            if keyboard.is_pressed('o') and not last_exhale:
                breath_events.append((sample_count, 'out'))
                print(f"Exhale event at sample {sample_count}")
                last_exhale = True
            elif not keyboard.is_pressed('o'):
                last_exhale = False

            data = belt.get_all()
            if data.size == 0:
                time.sleep(0.001)
                continue

            for sample in data:
                raw_sensor_1 = sample[5]
                sample_indices.append(sample_count)
                sample_count += 1

                # Lazy initialization: initialize filter with first sample value to avoid transients
                if not filter_initialized:
                    sos_hp, zi_1_hp = get_high_pass_filter_coeffs(
                        hp_cutoff_hz,
                        sampling_rate,
                        hp_order,
                        initial_value=raw_sensor_1,
                    )
                    # High-pass output is centered around zero, so seed low-pass at zero.
                    sos_lp, zi_1_lp = get_low_pass_filter_coeffs(
                        lp_cutoff_hz,
                        sampling_rate,
                        lp_order,
                        initial_value=0.0,
                    )
                    filter_initialized = True

                # Apply high-pass + low-pass cascade directly to the raw sample.
                filtered_sensor_1_hp, zi_1_hp = high_pass_filter_sample(raw_sensor_1, sos_hp, zi_1_hp)
                filtered_sensor_1_filtered, zi_1_lp = low_pass_filter_sample(
                    filtered_sensor_1_hp,
                    sos_lp,
                    zi_1_lp,
                )
                # Invert signal: sensor decreases when chest expands, so flip it
                filtered_sensor_1_filtered = -filtered_sensor_1_filtered
                channel_1_filtered.append(filtered_sensor_1_filtered)
                channel_1_raw.append(raw_sensor_1)

            cleaned_chunk = []
            artifact_mask_chunk = []
            for filtered_value in list(channel_1_filtered)[-len(data):]:
                cleaned_value, is_artifact = process_live_sample(filtered_value)
                cleaned_chunk.append(cleaned_value)
                artifact_mask_chunk.append(is_artifact)

            if not cleaned_chunk:
                continue

            # Startup robust calibration phase.
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

                # Start live output fresh after calibration.
                sample_indices.clear()
                channel_1_raw.clear()
                channel_1_filtered.clear()
                channel_1_filtered_processed.clear()
                normalized_signal.clear()
                breath_events.clear()
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

            # Runtime: adaptive normalization map with artifact-gated updates.
            channel_1_filtered_processed.extend(cleaned_chunk)
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

                # Hold detection with hysteresis:
                # - enter hold when activity stays below enter threshold
                # - exit hold only after crossing the higher exit threshold
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

            # Plot the normalized signal (last 200 points).
            n_plot = min(200, len(normalized_array), len(sample_indices))
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
                    live=True, ax=ax_final, line=line_final, blit_manager=blit_manager_final)
            
            # Send and print only the newest normalized data point.
            if normalized_chunk:
                newest_value = normalized_chunk[-1]
                print(f"Normalized: {newest_value:.4f}")
                # Send every newly processed point to reduce output stair-stepping.
                for value in normalized_chunk:
                    lsl_sender.send(float(value))
    except Exception:
        traceback.print_exc()
    finally:
        print("Stopping acquisition...")
        if belt is not None:
            belt.stop()
        print("Connection closed.")

if __name__ == '__main__':

    acquire_data()
