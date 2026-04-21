"""Regression tests for asynchronous BITalino sample acquisition."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Any, Callable

import numpy as np

import src.connect as connect_module


def _make_sample(sample_id: int) -> np.ndarray:
    """Create one synthetic BITalino-like sample row."""

    return np.array(
        [[sample_id, 0, 0, 0, 0, 100 + sample_id, 200 + sample_id]],
        dtype=int,
    )


def _wait_until(
    predicate: Callable[[], bool],
    timeout_s: float = 2.0,
    poll_s: float = 0.01,
) -> bool:
    """Poll a predicate until it becomes true or the timeout expires."""

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


class FakeDevice:
    """Minimal BITalino stand-in used to test reader behavior deterministically."""

    def __init__(self, responses: list[Any], idle_delay_s: float = 0.002) -> None:
        self._responses = deque(responses)
        self._lock = threading.Lock()
        self._idle_delay_s = idle_delay_s
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

    def start(self, sampling_rate: int, channels: list[int]) -> None:
        del sampling_rate, channels
        self.start_calls += 1

    def read(self, n_samples: int) -> np.ndarray:
        del n_samples
        time.sleep(self._idle_delay_s)
        with self._lock:
            if self._responses:
                item = self._responses.popleft()
            else:
                item = np.empty((0, 7), dtype=int)

        if isinstance(item, Exception):
            raise item
        return item

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _make_belt(
    monkeypatch,
    fake_device: FakeDevice,
    *,
    queue_max_samples: int = 100,
    clock_values: list[float] | None = None,
    read_chunk_size: int = 2,
) -> connect_module.BreathBelt:
    """Create a ``BreathBelt`` whose device connection is monkeypatched."""

    monkeypatch.setattr(
        connect_module,
        "connect_device",
        lambda *args, **kwargs: fake_device,
    )
    if clock_values is not None:
        times = iter(clock_values)
        monkeypatch.setattr(connect_module, "lsl_local_clock", lambda: float(next(times)))

    belt = connect_module.BreathBelt(
        mac_address="00:00:00:00:00:00",
        sampling_rate=100,
        channels=(0, 1),
        read_chunk_size=read_chunk_size,
        queue_max_samples=queue_max_samples,
        timeout_s=0.25,
        read_error_backoff_s=0.01,
        retries=1,
        retry_delay_s=0.0,
    )
    return belt


def test_timestamp_chunk_rows_backfills_from_newest_sample() -> None:
    rows = connect_module.timestamp_chunk_rows(
        np.asarray(
            [
                [1, 0, 0, 0, 0, 101, 201],
                [2, 0, 0, 0, 0, 102, 202],
            ],
            dtype=float,
        ),
        starting_source_sample_index=5,
        newest_capture_time_lsl_s=10.01,
        sampling_rate_hz=100,
    )

    assert [row.source_sample_index for row in rows] == [5, 6]
    assert [float(row.device_row[0]) for row in rows] == [1.0, 2.0]
    assert np.allclose([row.capture_time_lsl_s for row in rows], [10.0, 10.01])


def test_get_all_drains_queue(monkeypatch) -> None:
    """``get_all`` should return all buffered rows and then empty the queue."""

    fake = FakeDevice(
        [
            np.vstack([_make_sample(1), _make_sample(2)]),
            _make_sample(3),
        ]
    )
    belt = _make_belt(monkeypatch, fake, clock_values=[10.01, 10.02])
    belt.start()
    try:
        def latest_is_three() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 3

        assert _wait_until(latest_is_three)

        drained = belt.get_all()
        assert [int(row.device_row[0]) for row in drained] == [1, 2, 3]
        assert [row.source_sample_index for row in drained] == [0, 1, 2]
        assert np.allclose(
            [row.capture_time_lsl_s for row in drained],
            [10.0, 10.01, 10.02],
        )

        drained_again = belt.get_all()
        assert drained_again == []
    finally:
        belt.stop()


def test_get_latest_returns_newest_sample(monkeypatch) -> None:
    """``get_latest`` should expose the most recently acquired row only."""

    fake = FakeDevice(
        [
            _make_sample(1),
            np.vstack([_make_sample(2), _make_sample(3)]),
        ]
    )
    belt = _make_belt(monkeypatch, fake, clock_values=[1.0, 2.0])
    belt.start()
    try:
        def latest_is_three() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 3

        assert _wait_until(latest_is_three)

        latest = belt.get_latest()
        assert latest is not None
        assert latest.source_sample_index == 2
        assert int(latest.device_row[0]) == 3
        assert latest.capture_time_lsl_s == 1.02
    finally:
        belt.stop()


def test_contiguous_single_sample_reads_ignore_host_clock_jitter(monkeypatch) -> None:
    fake = FakeDevice([_make_sample(0), _make_sample(1), _make_sample(2), _make_sample(3)])
    belt = _make_belt(
        monkeypatch,
        fake,
        clock_values=[10.0, 40.0, 100.0, 1000.0],
        read_chunk_size=1,
    )
    belt.start()
    try:
        def latest_is_three() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 3

        assert _wait_until(latest_is_three)
        drained = belt.get_all()
        assert np.allclose(
            [row.capture_time_lsl_s for row in drained],
            [10.0, 10.01, 10.02, 10.03],
        )
    finally:
        belt.stop()


def test_contiguous_chunk_reads_stay_monotonic_across_read_boundaries(monkeypatch) -> None:
    fake = FakeDevice(
        [
            np.vstack([_make_sample(0), _make_sample(1)]),
            np.vstack([_make_sample(2), _make_sample(3)]),
        ]
    )
    belt = _make_belt(monkeypatch, fake, clock_values=[10.01, 99.99], read_chunk_size=2)
    belt.start()
    try:
        def latest_is_three() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 3

        assert _wait_until(latest_is_three)
        drained = belt.get_all()
        assert np.allclose(
            [row.capture_time_lsl_s for row in drained],
            [10.0, 10.01, 10.02, 10.03],
        )
    finally:
        belt.stop()


def test_sequence_wrap_preserves_one_contiguous_timeline(monkeypatch) -> None:
    fake = FakeDevice(
        [
            np.vstack([_make_sample(14), _make_sample(15)]),
            np.vstack([_make_sample(0), _make_sample(1)]),
        ]
    )
    belt = _make_belt(monkeypatch, fake, clock_values=[20.01, 80.01], read_chunk_size=2)
    belt.start()
    try:
        def latest_is_one() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 1

        assert _wait_until(latest_is_one)
        drained = belt.get_all()
        assert np.allclose(
            [row.capture_time_lsl_s for row in drained],
            [20.0, 20.01, 20.02, 20.03],
        )
    finally:
        belt.stop()


def test_sequence_jump_starts_a_new_timing_segment(monkeypatch) -> None:
    fake = FakeDevice(
        [
            np.vstack([_make_sample(0), _make_sample(1)]),
            np.vstack([_make_sample(4), _make_sample(5)]),
        ]
    )
    belt = _make_belt(monkeypatch, fake, clock_values=[10.01, 30.01], read_chunk_size=2)
    belt.start()
    try:
        def latest_is_five() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 5

        assert _wait_until(latest_is_five)
        drained = belt.get_all()
        assert np.allclose(
            [row.capture_time_lsl_s for row in drained],
            [10.0, 10.01, 30.0, 30.01],
        )
    finally:
        belt.stop()


def test_buffer_drops_oldest_when_full_and_counts_overflow(monkeypatch) -> None:
    """The bounded queue should retain only the newest rows when full."""

    fake = FakeDevice(
        [
            _make_sample(1),
            _make_sample(2),
            _make_sample(3),
            _make_sample(4),
            _make_sample(5),
        ]
    )
    belt = _make_belt(
        monkeypatch,
        fake,
        queue_max_samples=3,
        clock_values=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    belt.start()
    try:
        def latest_is_five() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 5

        assert _wait_until(latest_is_five)
        drained = belt.get_all()
        assert [int(row.device_row[0]) for row in drained] == [3, 4, 5]
        assert [row.source_sample_index for row in drained] == [2, 3, 4]
        assert belt.dropped_rows_total == 2
    finally:
        belt.stop()


def test_queue_overflow_preserves_a_natural_timestamp_gap_between_drains(monkeypatch) -> None:
    fake = FakeDevice(
        [
            np.vstack([_make_sample(0), _make_sample(1)]),
            np.vstack([_make_sample(2), _make_sample(3)]),
            np.vstack([_make_sample(4), _make_sample(5)]),
        ],
        idle_delay_s=0.05,
    )
    belt = _make_belt(
        monkeypatch,
        fake,
        queue_max_samples=3,
        clock_values=[10.01],
        read_chunk_size=2,
    )
    belt.start()
    try:
        def latest_is_one() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 1

        assert _wait_until(latest_is_one, timeout_s=1.0, poll_s=0.001)
        first_batch = belt.get_all()
        assert [int(row.device_row[0]) for row in first_batch] == [0, 1]

        def latest_is_five() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 5

        assert _wait_until(latest_is_five)
        second_batch = belt.get_all()
        assert [int(row.device_row[0]) for row in second_batch] == [3, 4, 5]
        assert belt.dropped_rows_total == 1
        assert np.isclose(
            second_batch[0].capture_time_lsl_s - first_batch[-1].capture_time_lsl_s,
            0.02,
        )
    finally:
        belt.stop()


def test_transient_read_error_does_not_kill_reader(monkeypatch) -> None:
    """A single read failure should be recorded without stopping acquisition."""

    fake = FakeDevice(
        [
            RuntimeError("transient read failure"),
            _make_sample(7),
        ]
    )
    belt = _make_belt(monkeypatch, fake, clock_values=[7.0])
    belt.start()
    try:
        assert _wait_until(lambda: belt.last_error is not None)

        def latest_is_seven() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest.device_row[0]) == 7

        assert _wait_until(latest_is_seven)
        assert belt.is_running is True
    finally:
        belt.stop()


def test_stop_is_idempotent_and_cleans_up_once(monkeypatch) -> None:
    """Repeated shutdown calls should not double-stop or double-close the device."""

    fake = FakeDevice([_make_sample(1)])
    belt = _make_belt(monkeypatch, fake, clock_values=[1.0])
    belt.start()

    assert _wait_until(lambda: belt.is_running)
    belt.stop()
    belt.stop()

    assert belt.is_running is False
    assert fake.stop_calls == 1
    assert fake.close_calls == 1
