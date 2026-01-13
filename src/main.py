import keyboard  # Import the keyboard module

import traceback

from connect import *
from preprocessing import *
from plot import *
import matplotlib.pyplot as plt
import numpy as np



# Replace with your BITalino's MAC address
mac_address = "98:D3:C1:FD:FF:DB"
sampling_rate = 100  # Sampling rate in Hz



# Band-pass filter parameters
bp_order = 4
bp_lowcut = 0.1  # Lower cutoff frequency in Hz
bp_highcut = 1  # Upper cutoff frequency in Hz





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


        # Prepare lists to store data for plotting

        sample_indices = []
        channel_1_raw = []
        channel_1_hp = []
        channel_1_bp = []
        sample_count = 0

        # Set up live plot for the final processed signal (HPF + BPF)
        fig_final, ax_final, line_final = setup_live_plot('Processed Breathing Signal (HPF + BPF)')

        # Initialize band-pass filter coefficients and state for both sensors
        b_bp, a_bp, zi_1_bp = get_band_pass_filter_coeffs(bp_lowcut, bp_highcut, sampling_rate, bp_order)
        _, _, zi_2_bp = get_band_pass_filter_coeffs(bp_lowcut, bp_highcut, sampling_rate, bp_order)

        print("Press 'c' to stop acquisition.")

        channel_1_bp_interp = None  # Store the final processed data for printing

        # Initialize min/max tracker for normalization
        minmax_tracker = MaxMinTracker(reset_interval_sec=60)

        # Initialize LSL sender
        from lslOut import LSLBreathingSender
        lsl_sender = LSLBreathingSender()

        while not keyboard.is_pressed('c'):
            data = read_samples(device, 10)  # Read 10 samples at a time
            for sample in data:
                raw_sensor_1 = sample[5]
                sample_indices.append(sample_count)
                sample_count += 1

                # Apply stateful band-pass filtering directly to the raw sample
                filtered_sensor_1_bp, zi_1_bp = band_pass_filter_sample(raw_sensor_1, b_bp, a_bp, zi_1_bp)
                channel_1_bp.append(filtered_sensor_1_bp)

                # Update min/max tracker and print normalized value
                minmax_tracker.update(filtered_sensor_1_bp)
                cur_max, cur_min = minmax_tracker.get_max_min()
                norm_val = normalize_value(filtered_sensor_1_bp, cur_min, cur_max)
                print(f"Normalized value: {norm_val}")
                # Send newest normalized value via LSL
                lsl_sender.send(norm_val)
                

            # Apply spike removal and motion artifact detection before plotting
            if len(channel_1_bp) >= 5:  # Only apply if enough samples
                channel_1_bp_np = np.array(channel_1_bp)
                # Spike removal
                channel_1_bp_np = remove_spikes(channel_1_bp_np, kernel_size=5, threshold=3)
                # Motion artifact detection
                artifact_mask = detect_motion_artifacts(channel_1_bp_np, window=10, threshold=5)
                # Mark artifacts as NaN
                channel_1_bp_artifact = channel_1_bp_np.copy()
                channel_1_bp_artifact[artifact_mask] = np.nan
                # Adaptive interpolation over artifacts
                channel_1_bp_interp = interpolate_artifacts(channel_1_bp_artifact)

                # Normalize the processed/interpolated signal using min/max from tracker
                cur_max, cur_min = minmax_tracker.get_max_min()
                if cur_max is not None and cur_min is not None:
                    normalized_interp = [normalize_value(val, cur_min, cur_max) for val in channel_1_bp_interp.tolist()]
                else:
                    normalized_interp = channel_1_bp_interp.tolist()

                # Plot the normalized, processed signal
                plot_breathing_channel(normalized_interp, sample_indices, live=True, ax=ax_final, line=line_final)
                # Print only the newest normalized data point
                if len(normalized_interp) > 0:
                    print(normalized_interp[-1])
            else:
                # Normalize the raw band-pass filtered data using min/max from tracker
                cur_max, cur_min = minmax_tracker.get_max_min()
                if cur_max is not None and cur_min is not None:
                    normalized_bp = [normalize_value(val, cur_min, cur_max) for val in channel_1_bp]
                else:
                    normalized_bp = channel_1_bp

                plot_breathing_channel(normalized_bp, sample_indices, live=True, ax=ax_final, line=line_final)
                # Print only the newest normalized data point
                if len(normalized_bp) > 0:
                    print(normalized_bp[-1])


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
