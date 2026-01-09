import keyboard  # Import the keyboard module



import traceback

from connect import *
from preprocessing import get_high_pass_filter_coeffs, high_pass_filter_sample
from plot import plot_breathing_channel, setup_live_plot
import matplotlib.pyplot as plt



# Replace with your BITalino's MAC address
mac_address = "98:D3:C1:FD:FF:DB"
sampling_rate = 100  # Sampling rate in Hz

# High-pass filter parameters
order = 5
cutoff = 0.05  # Cutoff frequency in Hz





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
        channel_1 = []
        sample_count = 0

        # Set up live plots for filtered and raw data
        fig_filt, ax_filt, line_filt = setup_live_plot('Filtered Breathing Signal')
        fig_raw, ax_raw, line_raw = setup_live_plot('Raw Breathing Signal')

        # Initialize filter coefficients and state for both sensors
        b, a, zi_1 = get_high_pass_filter_coeffs(cutoff, sampling_rate, order)
        _, _, zi_2 = get_high_pass_filter_coeffs(cutoff, sampling_rate, order)

        print("Press 'c' to stop acquisition.")
        while not keyboard.is_pressed('c'):
            data = read_samples(device, 10)  # Read 10 samples at a time
            for sample in data:
                raw_sensor_1 = sample[5]
                raw_sensor_2 = sample[6]
                sample_indices.append(sample_count)
                channel_1_raw.append(raw_sensor_1)
                sample_count += 1

                # Apply stateful high-pass filtering to each new sample
                filtered_sensor_1, zi_1 = high_pass_filter_sample(raw_sensor_1, b, a, zi_1)
                filtered_sensor_2, zi_2 = high_pass_filter_sample(raw_sensor_2, b, a, zi_2)
                channel_1.append(filtered_sensor_1)

                print(
                    f"Timestamp: {sample[0]}, Filtered Sensor 1: {filtered_sensor_1}, Filtered Sensor 2: {filtered_sensor_2}\n"
                    f"Timestamp: {sample[0]}, RAW Sensor 1:      {raw_sensor_1},         Raw Sensor 2:      {raw_sensor_2}")

            # Use plot_breathing_channel for live updating
            plot_breathing_channel(channel_1, sample_indices, live=True, ax=ax_filt, line=line_filt)
            plot_breathing_channel(channel_1_raw, sample_indices, live=True, ax=ax_raw, line=line_raw)

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
