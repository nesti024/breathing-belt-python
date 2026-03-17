"""Device-connection and asynchronous sample-acquisition utilities.

This module wraps the BITalino interface used by the breathing-belt
application. The public API separates one-shot device actions from the
``BreathBelt`` reader, which maintains a background acquisition thread and a
bounded in-memory sample buffer for non-blocking access from the main loop.
"""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any

import bitalino
import numpy as np


def connect_device(
    mac_address: str,
    retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float | None = None,
) -> Any:
    """Connect to a BITalino device with bounded retry behavior.

    Parameters
    ----------
    mac_address:
        Bluetooth MAC address of the target BITalino device.
    retries:
        Maximum number of connection attempts before raising an exception.
    retry_delay:
        Delay in seconds between failed connection attempts.
    timeout:
        Optional device timeout forwarded to the BITalino constructor.
    """

    for attempt in range(retries):
        try:
            device = bitalino.BITalino(mac_address, timeout=timeout)
            print(f"Connected to {mac_address} on attempt {attempt + 1}")
            return device
        except Exception as error:
            print(f"Connection attempt {attempt + 1} failed: {error}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                raise ConnectionError(
                    f"Failed to connect after {retries} attempts"
                ) from error


def start_acquisition(
    device: Any,
    sampling_rate: int,
    channels: list[int] | tuple[int, ...],
) -> None:
    """Start BITalino acquisition on the requested channels."""

    device.start(sampling_rate, channels)


def read_samples(device: Any, sample_count: int) -> np.ndarray | None:
    """Read one batch of samples from the BITalino device.

    Returns
    -------
    numpy.ndarray or None
        Array of shape ``(sample_count, n_columns)`` when data are available,
        otherwise ``None`` after an empty or failed read.
    """

    try:
        data = device.read(sample_count)
        if data is None or len(data) == 0:
            return None
        return data
    except Exception as error:
        print(f"Read error: {error}")
        return None


def stop_acquisition(device: Any) -> None:
    """Stop an active BITalino acquisition session."""

    device.stop()


def close_device(device: Any) -> None:
    """Close the underlying BITalino connection."""

    device.close()


class BreathBelt:
    """Asynchronous BITalino reader with a bounded sample queue.

    The reader continuously acquires data in a daemon thread and stores the
    newest samples in a deque. This design lets the main application consume
    all currently buffered samples without blocking on device I/O.
    """

    def __init__(
        self,
        mac_address: str,
        sampling_rate: int,
        channels: tuple[int, ...] = (0, 1),
        read_chunk_size: int = 10,
        queue_max_samples: int = 1000,
        timeout_s: float = 0.25,
        read_error_backoff_s: float = 0.05,
        retries: int = 3,
        retry_delay_s: float = 2.0,
    ) -> None:
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

        self._device: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._queue: deque[np.ndarray] = deque(maxlen=self.queue_max_samples)
        self._latest: np.ndarray | None = None
        self._sample_width = 0
        self._last_error: Exception | None = None
        self._started = False

    def _reader_loop(self) -> None:
        """Continuously acquire chunks and append them to the bounded queue."""

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

    def start(self) -> None:
        """Connect to the device, start acquisition, and launch the reader thread."""

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

    def stop(self) -> None:
        """Stop the reader thread and close the device safely."""

        if not self._started:
            return

        self._stop_event.set()

        thread = self._thread
        if thread is not None and thread.is_alive():
            # The join timeout is tied to the device timeout so shutdown remains
            # responsive even when a read is in progress.
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

    def get_latest(self) -> np.ndarray | None:
        """Return the most recent sample or ``None`` if no data are available."""

        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def get_all(self) -> np.ndarray:
        """Return and clear all currently buffered samples.

        Returns an empty array when the queue is empty so callers can keep a
        stable array-based interface.
        """

        with self._lock:
            if len(self._queue) == 0:
                return np.empty((0, self._sample_width), dtype=float)
            rows = list(self._queue)
            self._queue.clear()
        return np.vstack(rows)

    @property
    def last_error(self) -> Exception | None:
        """Most recent asynchronous read or shutdown error, if any."""

        with self._lock:
            return self._last_error

    @property
    def is_running(self) -> bool:
        """Whether the background acquisition thread is alive."""

        thread = self._thread
        return bool(self._started and thread is not None and thread.is_alive())
