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
chunk_size = 25  # Number of samples to read per chunk

# High-pass filter parameters for drift suppression and stable baseline
hp_order = 1
hp_cutoff = 0.3 # High-pass cutoff frequency in Hz



def acquire_data():
    # Set this to True to plot raw vs filtered for phase distortion check
    SHOW_PHASE_DISTORTION = True  # <-- Set to False to disable
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



        # Set up live plot for the final processed signal (HPF only)
        fig_final, ax_final, line_final = setup_live_plot('Processed Breathing Signal (High-pass 0.1 Hz)')

        # Optional: Set up plot for raw vs filtered (phase distortion check)
        if SHOW_PHASE_DISTORTION:
            import matplotlib.pyplot as plt
            fig_phase, ax_phase = plt.subplots()
            ax_phase.set_title('Raw vs Filtered Signal (Phase Distortion Check)')
            ax_phase.set_xlabel('Sample')
            ax_phase.set_ylabel('Amplitude')
            line_raw, = ax_phase.plot([], [], label='Raw')
            line_filt, = ax_phase.plot([], [], label='Filtered')
            ax_phase.legend()

        # Initialize high-pass filter coefficients and state for both sensors
        b_hp, a_hp, zi_1_hp = get_high_pass_filter_coeffs(hp_cutoff, sampling_rate, hp_order)
        _, _, zi_2_hp = get_high_pass_filter_coeffs(hp_cutoff, sampling_rate, hp_order)

        print("Press 'c' to stop acquisition.")

        channel_1_bp_interp = None  # Store the final processed data for printing

        # Initialize min/max tracker for normalization
        minmax_tracker = MaxMinTracker(reset_interval_sec=60)

        # Initialize LSL sender
        from lslOut import LSLBreathingSender
        lsl_sender = LSLBreathingSender()

        while not keyboard.is_pressed('c'):
            data = read_samples(device, chunk_size)  # Read 10 samples at a time
            for sample in data:
                raw_sensor_1 = sample[5]
                sample_indices.append(sample_count)
                sample_count += 1

                # Apply stateful high-pass filtering directly to the raw sample
                filtered_sensor_1_hp, zi_1_hp = high_pass_filter_sample(raw_sensor_1, b_hp, a_hp, zi_1_hp)
                channel_1_hp.append(filtered_sensor_1_hp)
                channel_1_raw.append(raw_sensor_1)

                # Update min/max tracker and print normalized value
                minmax_tracker.update(filtered_sensor_1_hp)
                cur_max, cur_min = minmax_tracker.get_max_min()
                norm_val = normalize_value(filtered_sensor_1_hp, cur_min, cur_max)
                print(f"Normalized value: {norm_val}")
                # Send newest normalized value via LSL
                lsl_sender.send(norm_val)

            # Optional: Plot raw vs filtered for phase distortion check
            if SHOW_PHASE_DISTORTION and len(channel_1_raw) > 0 and len(channel_1_hp) > 0:
                # Only plot the last 500 points for clarity
                plot_len = 500
                ax_phase.set_xlim(max(0, sample_count - plot_len), sample_count)
                line_raw.set_data(sample_indices[-plot_len:], channel_1_raw[-plot_len:])
                line_filt.set_data(sample_indices[-plot_len:], channel_1_hp[-plot_len:])
                # Adjust y-limits automatically
                all_vals = channel_1_raw[-plot_len:] + channel_1_hp[-plot_len:]
                if all_vals:
                    ax_phase.set_ylim(min(all_vals), max(all_vals))
                fig_phase.canvas.draw()
                fig_phase.canvas.flush_events()
                

            # Apply spike removal and motion artifact detection before plotting
            if len(channel_1_hp) >= 5:  # Only apply if enough samples
                channel_1_hp_np = np.array(channel_1_hp)
                # Spike removal
                channel_1_hp_np = remove_spikes(channel_1_hp_np, kernel_size=5, threshold=3)
                # Motion artifact detection
                artifact_mask = detect_motion_artifacts(channel_1_hp_np, window=10, threshold=5)
                # Mark artifacts as NaN
                channel_1_hp_artifact = channel_1_hp_np.copy()
                channel_1_hp_artifact[artifact_mask] = np.nan
                # Adaptive interpolation over artifacts
                channel_1_hp_interp = interpolate_artifacts(channel_1_hp_artifact)

                # Normalize the processed/interpolated signal using min/max from tracker
                cur_max, cur_min = minmax_tracker.get_max_min()
                if cur_max is not None and cur_min is not None:
                    normalized_interp = [normalize_value(val, cur_min, cur_max) for val in channel_1_hp_interp.tolist()]
                else:
                    normalized_interp = channel_1_hp_interp.tolist()

                # Plot the normalized, processed signal
                # Only plot the last 200 points
                plot_breathing_channel(
                    normalized_interp[-200:],
                    sample_indices[-200:],
                    live=True, ax=ax_final, line=line_final)
                # Print only the newest normalized data point
                if len(normalized_interp) > 0:
                    print(normalized_interp[-1])
            else:
                # Normalize the raw high-pass filtered data using min/max from tracker
                cur_max, cur_min = minmax_tracker.get_max_min()
                if cur_max is not None and cur_min is not None:
                    normalized_hp = [normalize_value(val, cur_min, cur_max) for val in channel_1_hp]
                else:
                    normalized_hp = channel_1_hp

                # Only plot the last 200 points
                plot_breathing_channel(
                    normalized_hp[-200:],
                    sample_indices[-200:],
                    live=True, ax=ax_final, line=line_final)
                # Print only the newest normalized data point
                if len(normalized_hp) > 0:
                    print(normalized_hp[-1])


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
