import bitalino


def connect_device(mac_address):
    """
    Connect to a BITalino device.
    """
    return bitalino.BITalino(mac_address)


def start_acquisition(device, sampling_rate, channels):
    """
    Start data acquisition on the given channels.
    """
    device.start(sampling_rate, channels)


def read_samples(device, sample_count):
    """
    Read a batch of samples from the device.
    """
    return device.read(sample_count)


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
