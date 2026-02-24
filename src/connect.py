import bitalino
import time
import numpy as np

def connect_device(mac_address, retries=3, retry_delay=2.0, timeout=None):
    """
    Connect to a BITalino device with retry logic.
    """
    for attempt in range(retries):
        try:
            device = bitalino.BITalino(mac_address, timeout=timeout)
            print(f"Connected to {mac_address} on attempt {attempt + 1}")
            return device
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                raise ConnectionError(f"Failed to connect after {retries} attempts") from e

def start_acquisition(device, sampling_rate, channels):
    """
    Start data acquisition on the given channels.
    """
    device.start(sampling_rate, channels)

def read_samples(device, sample_count):
    """
    Read a batch of samples from the device.
    Returns numpy array of shape (sample_count, n_columns) or None on failure.
    """
    try:
        data = device.read(sample_count)
        if data is None or len(data) == 0:
            return None
        return data
    except Exception as e:
        print(f"Read error: {e}")
        return None

def stop_acquisition(device):
    """
    Stop data acquisition.
    """
    device.stop()

def close_device(device):
    """
    Close the device connection.
    """
    device.close()
