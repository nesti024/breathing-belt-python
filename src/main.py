import keyboard  # Import the keyboard module



import traceback

from connect import *
from preprocessing import *



# Replace with your BITalino's MAC address
mac_address = "98:D3:C1:FD:FF:DB"
sampling_rate = 100  # Sampling rate in Hz

# High-pass filter parameters
order = 5
cutoff = 0.05  # Cutoff frequency in Hz





def acquire_data():
    """
    Acquire data from two PLUX PZT sensors using a BITalino Core.

    :param mac_address: The MAC address of the BITalino device.
    :param sampling_rate: Sampling rate in Hz (default: 100 Hz).
    """
    try:
        # Connect to the BITalino device
        device = connect_device(mac_address)

        # Start acquisition on channels 0 and 1 (assuming PZT sensors are connected to these channels)
        print("Starting acquisition...")
        start_acquisition(device, sampling_rate, [0, 1])



        # Collect data until the user presses 'c'
        print("Press 'c' to stop acquisition.")
        while not keyboard.is_pressed('c'):
            data = read_samples(device, 10)  # Read 10 samples at a time
            for sample in data:
                # Apply high-pass filtering to the sensor data
                filtered_sensor_1 = high_pass_filter([sample[5]], cutoff, sampling_rate, order)[-1]
                filtered_sensor_2 = high_pass_filter([sample[6]], cutoff, sampling_rate, order)[-1]
                #create timestamp:



                print(
                    f"Timestamp: {sample[0]}, Filtered Sensor 1: {filtered_sensor_1}, Filtered Sensor 2: {filtered_sensor_2}\n"
                    f"Timestamp: {sample[0]}, RAW Sensor 1:      {sample[5]},         Raw Sensor 2:      {sample[6]}")

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
