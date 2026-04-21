"""Device-connection and asynchronous sample-acquisition utilities.

This module wraps the BITalino interface used by the breathing-belt
application. The public API separates one-shot device actions from the
``BreathBelt`` reader, which maintains a background acquisition thread and a
bounded in-memory sample buffer for non-blocking access from the main loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any

import bitalino
import numpy as np


def lsl_local_clock() -> float:
    """Return the current sender-side LSL clock time.

    ``pylsl`` is imported lazily so acquisition tests can run without the
    optional runtime dependency installed.
    """

    try:
        from pylsl import local_clock
    except ImportError:
        return float(time.perf_counter())
    return float(local_clock())


@dataclass(frozen=True)
class AcquiredRow:
    """One acquired device row with sender-side timing provenance."""

    device_row: np.ndarray
    source_sample_index: int
    capture_time_lsl_s: float

    def copy(self) -> "AcquiredRow":
        """Return a defensive copy suitable for external consumers."""

        return AcquiredRow(
            device_row=np.asarray(self.device_row).copy(),
            source_sample_index=int(self.source_sample_index),
            capture_time_lsl_s=float(self.capture_time_lsl_s),
        )


def _sequence_from_value(value: Any) -> int | None:
    """Return a BITalino sequence value when the input looks valid."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    integer_value = int(round(numeric_value))
    if abs(numeric_value - integer_value) > 1e-9:
        return None
    if not 0 <= integer_value <= 15:
        return None
    return integer_value


def _extract_chunk_sequences(data_array: np.ndarray) -> list[int] | None:
    """Return per-row sequence values when the first column is a valid BITalino counter."""

    contiguous_rows = np.asarray(data_array)
    if contiguous_rows.ndim != 2 or contiguous_rows.shape[1] == 0:
        return None

    sequences: list[int] = []
    for row in contiguous_rows:
        sequence = _sequence_from_value(row[0])
        if sequence is None:
            return None
        sequences.append(sequence)
    return sequences


def _sequences_are_contiguous(
    sequences: list[int],
    *,
    previous_sequence: int | None = None,
) -> bool:
    """Return whether sequence values form one contiguous BITalino span."""

    if not sequences:
        return False

    expected_sequence = previous_sequence
    for sequence in sequences:
        if expected_sequence is None:
            expected_sequence = sequence
            continue
        if sequence != ((expected_sequence + 1) % 16):
            return False
        expected_sequence = sequence
    return True


def timestamp_chunk_rows(
    data_array: np.ndarray,
    *,
    starting_source_sample_index: int,
    newest_capture_time_lsl_s: float,
    sampling_rate_hz: int,
) -> list[AcquiredRow]:
    """Assign per-row host-estimated capture times to one returned device chunk."""

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    contiguous_rows = np.asarray(data_array)
    if contiguous_rows.ndim != 2:
        raise ValueError("data_array must be two-dimensional.")

    dt_s = 1.0 / float(sampling_rate_hz)
    row_count = int(contiguous_rows.shape[0])
    acquired_rows: list[AcquiredRow] = []
    for offset, row in enumerate(contiguous_rows):
        samples_from_newest = row_count - 1 - offset
        acquired_rows.append(
            AcquiredRow(
                device_row=np.asarray(row).copy(),
                source_sample_index=int(starting_source_sample_index + offset),
                capture_time_lsl_s=float(
                    newest_capture_time_lsl_s - (samples_from_newest * dt_s)
                ),
            )
        )
    return acquired_rows


def timestamp_contiguous_rows(
    data_array: np.ndarray,
    *,
    starting_source_sample_index: int,
    first_capture_time_lsl_s: float,
    sampling_rate_hz: int,
) -> list[AcquiredRow]:
    """Assign timestamps to a contiguous span from a known first-sample time."""

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")

    contiguous_rows = np.asarray(data_array)
    if contiguous_rows.ndim != 2:
        raise ValueError("data_array must be two-dimensional.")

    dt_s = 1.0 / float(sampling_rate_hz)
    acquired_rows: list[AcquiredRow] = []
    for offset, row in enumerate(contiguous_rows):
        acquired_rows.append(
            AcquiredRow(
                device_row=np.asarray(row).copy(),
                source_sample_index=int(starting_source_sample_index + offset),
                capture_time_lsl_s=float(first_capture_time_lsl_s + (offset * dt_s)),
            )
        )
    return acquired_rows


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
        self._queue: deque[AcquiredRow] = deque(maxlen=self.queue_max_samples)
        self._latest: AcquiredRow | None = None
        self._sample_width = 0
        self._last_error: Exception | None = None
        self._started = False
        self._next_source_sample_index = 0
        self._dropped_rows_total = 0
        self._next_capture_time_lsl_s: float | None = None
        self._last_device_sequence: int | None = None
        self._segment_open = False

    def _reset_timing_state(self) -> None:
        """Reset reader-side timing provenance for a fresh acquisition segment."""

        self._next_capture_time_lsl_s = None
        self._last_device_sequence = None
        self._segment_open = False

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

                dt_s = 1.0 / float(self.sampling_rate)
                chunk_sequences = _extract_chunk_sequences(data_array)
                sequence_continuity_known = chunk_sequences is not None and _sequences_are_contiguous(
                    chunk_sequences
                )
                continue_existing_segment = (
                    self._segment_open
                    and self._next_capture_time_lsl_s is not None
                    and (
                        chunk_sequences is None
                        or (
                            sequence_continuity_known
                            and self._last_device_sequence is not None
                            and chunk_sequences[0] == ((self._last_device_sequence + 1) % 16)
                        )
                        or (
                            sequence_continuity_known and self._last_device_sequence is None
                        )
                    )
                )

                if continue_existing_segment:
                    acquired_rows = timestamp_contiguous_rows(
                        data_array,
                        starting_source_sample_index=self._next_source_sample_index,
                        first_capture_time_lsl_s=self._next_capture_time_lsl_s,
                        sampling_rate_hz=self.sampling_rate,
                    )
                else:
                    acquired_rows = timestamp_chunk_rows(
                        data_array,
                        starting_source_sample_index=self._next_source_sample_index,
                        newest_capture_time_lsl_s=lsl_local_clock(),
                        sampling_rate_hz=self.sampling_rate,
                    )

                last_capture_time_lsl_s = acquired_rows[-1].capture_time_lsl_s

                with self._lock:
                    self._sample_width = int(data_array.shape[1])
                    self._next_source_sample_index += len(acquired_rows)
                    self._next_capture_time_lsl_s = float(last_capture_time_lsl_s + dt_s)
                    self._segment_open = True
                    if sequence_continuity_known:
                        self._last_device_sequence = chunk_sequences[-1]
                    else:
                        self._last_device_sequence = None
                    for acquired_row in acquired_rows:
                        if len(self._queue) >= self.queue_max_samples:
                            self._queue.popleft()
                            self._dropped_rows_total += 1
                        self._queue.append(acquired_row)
                        self._latest = acquired_row
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
            self._next_source_sample_index = 0
            self._dropped_rows_total = 0
            self._reset_timing_state()

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

        with self._lock:
            self._reset_timing_state()

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

    def get_latest(self) -> AcquiredRow | None:
        """Return the most recent acquired row or ``None`` if no data are available."""

        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def get_all(self) -> list[AcquiredRow]:
        """Return and clear all currently buffered rows."""

        with self._lock:
            if len(self._queue) == 0:
                return []
            rows = [row.copy() for row in self._queue]
            self._queue.clear()
        return rows

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

    @property
    def dropped_rows_total(self) -> int:
        """Total number of queue-overflow rows dropped since acquisition start."""

        with self._lock:
            return int(self._dropped_rows_total)
