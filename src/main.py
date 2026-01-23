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

# High-pass filter parameters for drift suppression and stable baseline
# Box breathing: 4s in + 4s hold + 4s out = 12s cycle = 0.083 Hz
# Use 0.03 Hz cutoff to preserve slow breathing while removing drift
hp_order = 2
hp_cutoff = 0.03  # High-pass cutoff frequency in Hz (lowered for box breathing)

#Normalization reset interval
reset_interval_sec = 20  # Reset min/max every x seconds


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
        channel_1_hp = deque(maxlen=1000)
        channel_1_hp_processed = deque(maxlen=1000)  # For processed signal after artifact removal
        sample_count = 0
            # Store inhale/exhale events as (sample_index, event_type)
        breath_events = []  # event_type: 'in' or 'out'



        # Set up live plot for the final processed and normalized signal
        fig_final, ax_final, line_final, blit_manager_final = setup_live_plot('Normalized Breathing Signal (0-1 range)')

        # Filter coefficients will be initialized lazily with the first sample
        sos_hp, zi_1_hp = None, None
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
            for sample in data:
                raw_sensor_1 = sample[5]
                sample_indices.append(sample_count)
                sample_count += 1

                # Lazy initialization: initialize filter with first sample value to avoid transients
                if not filter_initialized:
                    sos_hp, zi_1_hp = get_high_pass_filter_coeffs(hp_cutoff, sampling_rate, hp_order, initial_value=raw_sensor_1)
                    filter_initialized = True

                # Apply stateful high-pass filtering directly to the raw sample
                filtered_sensor_1_hp, zi_1_hp = high_pass_filter_sample(raw_sensor_1, sos_hp, zi_1_hp)
                # Invert signal: sensor decreases when chest expands, so flip it
                filtered_sensor_1_hp = -filtered_sensor_1_hp
                channel_1_hp.append(filtered_sensor_1_hp)
                channel_1_raw.append(raw_sensor_1)

                # Update min/max tracker with filtered value
                minmax_tracker.update(filtered_sensor_1_hp)

            # Apply spike removal and motion artifact detection before plotting
            if len(channel_1_hp) >= 5:
                # Only process the latest chunk for spike removal and artifact detection
                chunk_len = min(chunk_size, len(channel_1_hp))
                # Get the last chunk
                recent_hp = np.array(list(channel_1_hp)[-chunk_len:])
                # Spike removal on the chunk
                recent_hp_processed = remove_spikes(recent_hp, kernel_size=5, threshold=2)
                # Motion artifact detection on the chunk (lowered window and threshold for better sensitivity)
                artifact_mask = detect_motion_artifacts(recent_hp_processed, window=5, threshold=2)
                print(f"Artifacts: {artifact_mask.sum()}/{len(artifact_mask)} detected in chunk.")
                # Mark artifacts as NaN
                recent_hp_artifact = recent_hp_processed.copy()
                recent_hp_artifact[artifact_mask] = np.nan
                # Adaptive interpolation over artifacts
                recent_hp_interp = interpolate_artifacts(recent_hp_artifact)

                # Use rolling buffer - automatically handles trimming
                channel_1_hp_processed.extend(recent_hp_interp)

                # Vectorized normalization (much faster than list comprehension)
                max_val, min_val = minmax_tracker.get_max_min()
                processed_array = np.array(list(channel_1_hp_processed))
                
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
                hp_array = np.array(list(channel_1_hp))
                
                if max_val is not None and min_val is not None and (max_val - min_val) > 0.001:
                    normalized_hp = (hp_array - min_val) / (max_val - min_val)
                else:
                    # Fallback: return 0.5 (centered) if range is invalid
                    normalized_hp = np.full_like(hp_array, 0.5)

                # Only plot the last 200 points
                n_plot = min(200, len(normalized_hp), len(sample_indices))
                plot_breathing_channel(
                    normalized_hp[-n_plot:],
                    list(sample_indices)[-n_plot:],
                    live=True, ax=ax_final, line=line_final, blit_manager=blit_manager_final)
                
                # Send and print only the newest normalized data point
                if len(normalized_hp) > 0:
                    newest_value = normalized_hp[-1]
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
