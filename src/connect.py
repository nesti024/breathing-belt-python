import bitalino
import time
import threading
from collections import deque
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


class BreathBelt:
    """Threaded non-blocking BITalino reader with a bounded sample buffer."""

    def __init__(
        self,
        mac_address,
        sampling_rate,
        channels=(0, 1),
        read_chunk_size=10,
        queue_max_samples=1000,
        timeout_s=0.25,
        read_error_backoff_s=0.05,
        retries=3,
        retry_delay_s=2.0,
    ):
        if read_chunk_size <= 0:
            raise ValueError("read_chunk_size must be positive.")
        if queue_max_samples <= 0:
            raise ValueError("queue_max_samples must be positive.")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive.")
        if read_error_backoff_s < 0.0:
            raise ValueError("read_error_backoff_s must be non-negative.")

        self.mac_address = mac_address
        self.sampling_rate = int(sampling_rate)
        self.channels = tuple(channels)
        self.read_chunk_size = int(read_chunk_size)
        self.queue_max_samples = int(queue_max_samples)
        self.timeout_s = float(timeout_s)
        self.read_error_backoff_s = float(read_error_backoff_s)
        self.retries = int(retries)
        self.retry_delay_s = float(retry_delay_s)

        self._device = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._queue = deque(maxlen=self.queue_max_samples)
        self._latest = None
        self._sample_width = 0
        self._last_error = None
        self._started = False

    def _reader_loop(self):
        while not self._stop_event.is_set():
            device = self._device
            if device is None:
                break
            try:
                data = device.read(self.read_chunk_size)
                if data is None or len(data) == 0:
                    continue
                data_array = np.asarray(data)
                if data_array.ndim == 1:
                    data_array = data_array.reshape(1, -1)

                with self._lock:
                    self._sample_width = int(data_array.shape[1])
                    for row in data_array:
                        row_copy = np.asarray(row).copy()
                        self._queue.append(row_copy)
                        self._latest = row_copy
            except Exception as error:
                with self._lock:
                    self._last_error = error
                if self._stop_event.is_set():
                    break
                print(f"BreathBelt read error: {error}")
                if self.read_error_backoff_s > 0.0:
                    time.sleep(self.read_error_backoff_s)

    def start(self):
        if self._started:
            return

        with self._lock:
            self._queue.clear()
            self._latest = None
            self._sample_width = 0
            self._last_error = None

        self._stop_event.clear()
        device = connect_device(
            self.mac_address,
            retries=self.retries,
            retry_delay=self.retry_delay_s,
            timeout=self.timeout_s,
        )
        try:
            start_acquisition(device, self.sampling_rate, list(self.channels))
        except Exception:
            try:
                close_device(device)
            except Exception:
                pass
            raise

        self._device = device
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="BreathBeltReader",
            daemon=True,
        )
        self._started = True
        self._thread.start()

    def stop(self):
        if not self._started:
            return

        self._stop_event.set()

        thread = self._thread
        if thread is not None and thread.is_alive():
            join_timeout = max(0.5, self.timeout_s * 2.0)
            thread.join(timeout=join_timeout)

        device = self._device
        self._thread = None
        self._device = None
        self._started = False

        if device is None:
            return

        try:
            stop_acquisition(device)
        except Exception as error:
            with self._lock:
                self._last_error = error
        try:
            close_device(device)
        except Exception as error:
            with self._lock:
                self._last_error = error

    def get_latest(self):
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def get_all(self):
        with self._lock:
            if len(self._queue) == 0:
                return np.empty((0, self._sample_width), dtype=float)
            rows = list(self._queue)
            self._queue.clear()
        return np.vstack(rows)

    @property
    def last_error(self):
        with self._lock:
            return self._last_error

    @property
    def is_running(self):
        thread = self._thread
        return bool(self._started and thread is not None and thread.is_alive())
