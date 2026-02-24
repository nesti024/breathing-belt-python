import keyboard  # Import the keyboard module

import traceback
from collections import deque

from connect import *
from preprocessing import *
from plot import *
import matplotlib.pyplot as plt
import numpy as np

# Replace with your BITalino's MAC address
mac_address = "98:D3:C1:FD:FF:DB"
sampling_rate = 100  # Sampling rate in Hz
chunk_size = 25  # Number of samples to read per chunk

# Low-pass filter parameters for level-style breathing display
# This keeps inhale/exhale plateaus visible and makes breath holds stay flat.
lp_order = 4
lp_cutoff = 1.0  # Low-pass cutoff frequency in Hz

#Normalization reset interval
reset_interval_sec = 120  # Reset min/max every x seconds


def acquire_data():
    """
    Acquire data from two PLUX PZT sensors using a BITalino Corce.
    """
    try:
        # Connect to the BITalino device
        device = connect_device(mac_address)

        # Start acquisition on channels 0 and 1 (assuming PZT sensors are connected to these channels)
        print("Starting acquisition...")
        start_acquisition(device, sampling_rate, [0, 1])


        # Prepare rolling buffers for data storage (auto-trim to max length)
        sample_indices = deque(maxlen=1000)
        channel_1_raw = deque(maxlen=1000)
        channel_1_level = deque(maxlen=1000)
        channel_1_level_processed = deque(maxlen=1000)  # For processed signal after artifact removal
        sample_count = 0
            # Store inhale/exhale events as (sample_index, event_type)
        breath_events = []  # event_type: 'in' or 'out'



        # Set up live plot for the final processed and normalized signal
        fig_final, ax_final, line_final, blit_manager_final = setup_live_plot('Normalized Breathing Level (0-1 range)')

        # Filter coefficients will be initialized lazily with the first sample
        sos_lp, zi_1_lp = None, None
        filter_initialized = False
        
        import time
        last_inhale = False
        last_exhale = False

        print("Press 'c' to stop acquisition.")

        # Initialize min/max tracker for normalization
        # Box breathing: use longer interval (120s = 7-10 cycles) for stable normalization
        minmax_tracker = MaxMinTracker(reset_interval_sec)

        # Initialize LSL sender
        from lslOut import LSLBreathingSender
        lsl_sender = LSLBreathingSender()

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

            data = read_samples(device, chunk_size)  # Read chunk_size samples at a time
            if data is None:
                continue
            for sample in data:
                raw_sensor_1 = sample[5]
                sample_indices.append(sample_count)
                sample_count += 1

                # Lazy initialization: initialize filter with first sample value to avoid transients
                if not filter_initialized:
                    sos_lp, zi_1_lp = get_low_pass_filter_coeffs(lp_cutoff, sampling_rate, lp_order, initial_value=raw_sensor_1)
                    filter_initialized = True

                # Apply stateful low-pass filtering for position-like breathing level
                filtered_sensor_1_level, zi_1_lp = low_pass_filter_sample(raw_sensor_1, sos_lp, zi_1_lp)
                # Invert signal: sensor decreases when chest expands, so flip it
                filtered_sensor_1_level = -filtered_sensor_1_level
                channel_1_level.append(filtered_sensor_1_level)
                channel_1_raw.append(raw_sensor_1)

                # Update min/max tracker with filtered breathing level
                minmax_tracker.update(filtered_sensor_1_level)

            # Apply spike removal and motion artifact detection before plotting
            if len(channel_1_level) >= 5:
                # Only process the latest chunk for spike removal and artifact detection
                chunk_len = min(chunk_size, len(channel_1_level))
                # Get the last chunk
                recent_level = np.array(list(channel_1_level)[-chunk_len:])
                # Spike removal on the chunk
                recent_level_processed = remove_spikes(recent_level, kernel_size=5, threshold=2)
                # Motion artifact detection on the chunk (lowered window and threshold for better sensitivity)
                artifact_mask = detect_motion_artifacts(recent_level_processed, window=5, threshold=2)
                print(f"Artifacts: {artifact_mask.sum()}/{len(artifact_mask)} detected in chunk.")
                # Mark artifacts as NaN
                recent_level_artifact = recent_level_processed.copy()
                recent_level_artifact[artifact_mask] = np.nan
                # Adaptive interpolation over artifacts
                recent_level_interp = interpolate_artifacts(recent_level_artifact)

                # Use rolling buffer - automatically handles trimming
                channel_1_level_processed.extend(recent_level_interp)

                # Vectorized normalization (much faster than list comprehension)
                max_val, min_val = minmax_tracker.get_max_min()
                processed_array = np.array(list(channel_1_level_processed))
                
                if max_val is not None and min_val is not None and (max_val - min_val) > 0.001:
                    normalized_interp = (processed_array - min_val) / (max_val - min_val)
                else:
                    # Fallback: return 0.5 (centered) if range is invalid
                    normalized_interp = np.full_like(processed_array, 0.5)

                # Plot the normalized, processed signal (last 200 points)
                n_plot = min(200, len(normalized_interp), len(sample_indices))
                plot_breathing_channel(
                    normalized_interp[-n_plot:],
                    list(sample_indices)[-n_plot:],
                    live=True, ax=ax_final, line=line_final, blit_manager=blit_manager_final)
                
                # Send and print only the newest normalized data point
                if len(normalized_interp) > 0:
                    newest_value = normalized_interp[-1]
                    print(f"Normalized: {newest_value:.4f}")
                    lsl_sender.send(float(newest_value))
            else:
                # Early stage: not enough data for artifact processing yet
                max_val, min_val = minmax_tracker.get_max_min()
                level_array = np.array(list(channel_1_level))
                
                if max_val is not None and min_val is not None and (max_val - min_val) > 0.001:
                    normalized_level = (level_array - min_val) / (max_val - min_val)
                else:
                    # Fallback: return 0.5 (centered) if range is invalid
                    normalized_level = np.full_like(level_array, 0.5)

                # Only plot the last 200 points
                n_plot = min(200, len(normalized_level), len(sample_indices))
                plot_breathing_channel(
                    normalized_level[-n_plot:],
                    list(sample_indices)[-n_plot:],
                    live=True, ax=ax_final, line=line_final, blit_manager=blit_manager_final)
                
                # Send and print only the newest normalized data point
                if len(normalized_level) > 0:
                    newest_value = normalized_level[-1]
                    print(f"Normalized: {newest_value:.4f}")
                    lsl_sender.send(float(newest_value))


        # Stop acquisition
        print("Stopping acquisition...")
        stop_acquisition(device)

        # Close the connection
        close_device(device)
        print("Connection closed.")

    

    except Exception:
        traceback.print_exc()

if __name__ == '__main__':

    acquire_data()
