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


# High-pass filter parameters
hp_order = 5
hp_cutoff = 0.05  # Cutoff frequency in Hz

# Band-pass filter parameters
bp_order = 4
bp_lowcut = 0.1  # Lower cutoff frequency in Hz
bp_highcut = 1.0  # Upper cutoff frequency in Hz





def acquire_data():
    """
    Acquire data from two PLUX PZT sensors using a BITalino Core.
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

        # Initialize filter coefficients and state for both sensors (high-pass)
        b_hp, a_hp, zi_1_hp = get_high_pass_filter_coeffs(hp_cutoff, sampling_rate, hp_order)
        _, _, zi_2_hp = get_high_pass_filter_coeffs(hp_cutoff, sampling_rate, hp_order)

        # Initialize band-pass filter coefficients and state for both sensors
        b_bp, a_bp, zi_1_bp = get_band_pass_filter_coeffs(bp_lowcut, bp_highcut, sampling_rate, bp_order)
        _, _, zi_2_bp = get_band_pass_filter_coeffs(bp_lowcut, bp_highcut, sampling_rate, bp_order)


        print("Press 'c' to stop acquisition.")

        while not keyboard.is_pressed('c'):
            data = read_samples(device, 10)  # Read 10 samples at a time
            for sample in data:
                raw_sensor_1 = sample[5]
                sample_indices.append(sample_count)
                sample_count += 1

                # Apply stateful high-pass filtering to each new sample
                filtered_sensor_1_hp, zi_1_hp = high_pass_filter_sample(raw_sensor_1, b_hp, a_hp, zi_1_hp)

                # Apply stateful band-pass filtering to the high-pass filtered sample
                filtered_sensor_1_bp, zi_1_bp = band_pass_filter_sample(filtered_sensor_1_hp, b_bp, a_bp, zi_1_bp)
                channel_1_bp.append(filtered_sensor_1_bp)

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
                # Plot the interpolated signal
                plot_breathing_channel(channel_1_bp_interp.tolist(), sample_indices, live=True, ax=ax_final, line=line_final)
            else:
                plot_breathing_channel(channel_1_bp, sample_indices, live=True, ax=ax_final, line=line_final)

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
