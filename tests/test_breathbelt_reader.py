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
    queue_max_samples: int = 100,
) -> connect_module.BreathBelt:
    """Create a ``BreathBelt`` whose device connection is monkeypatched."""

    monkeypatch.setattr(
        connect_module,
        "connect_device",
        lambda *args, **kwargs: fake_device,
    )
    belt = connect_module.BreathBelt(
        mac_address="00:00:00:00:00:00",
        sampling_rate=100,
        channels=(0, 1),
        read_chunk_size=2,
        queue_max_samples=queue_max_samples,
        timeout_s=0.25,
        read_error_backoff_s=0.01,
        retries=1,
        retry_delay_s=0.0,
    )
    return belt


def test_get_all_drains_queue(monkeypatch) -> None:
    """``get_all`` should return all buffered samples and then empty the queue."""

    fake = FakeDevice(
        [
            np.vstack([_make_sample(1), _make_sample(2)]),
            _make_sample(3),
        ]
    )
    belt = _make_belt(monkeypatch, fake)
    belt.start()
    try:
        def latest_is_three() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest[0]) == 3

        assert _wait_until(latest_is_three)

        drained = belt.get_all()
        assert drained.shape == (3, 7)
        assert drained[:, 0].tolist() == [1, 2, 3]

        drained_again = belt.get_all()
        assert drained_again.shape == (0, 7)
    finally:
        belt.stop()


def test_get_latest_returns_newest_sample(monkeypatch) -> None:
    """``get_latest`` should expose the most recently acquired sample only."""

    fake = FakeDevice(
        [
            _make_sample(1),
            np.vstack([_make_sample(2), _make_sample(3)]),
        ]
    )
    belt = _make_belt(monkeypatch, fake)
    belt.start()
    try:
        def latest_is_three() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest[0]) == 3

        assert _wait_until(latest_is_three)

        latest = belt.get_latest()
        assert latest is not None
        assert int(latest[0]) == 3
    finally:
        belt.stop()


def test_buffer_drops_oldest_when_full(monkeypatch) -> None:
    """The bounded queue should retain only the newest samples when full."""

    fake = FakeDevice(
        [
            _make_sample(1),
            _make_sample(2),
            _make_sample(3),
            _make_sample(4),
            _make_sample(5),
        ]
    )
    belt = _make_belt(monkeypatch, fake, queue_max_samples=3)
    belt.start()
    try:
        def latest_is_five() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest[0]) == 5

        assert _wait_until(latest_is_five)
        drained = belt.get_all()
        assert drained.shape == (3, 7)
        assert drained[:, 0].tolist() == [3, 4, 5]
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
    belt = _make_belt(monkeypatch, fake)
    belt.start()
    try:
        assert _wait_until(lambda: belt.last_error is not None)

        def latest_is_seven() -> bool:
            latest = belt.get_latest()
            return latest is not None and int(latest[0]) == 7

        assert _wait_until(latest_is_seven)
        assert belt.is_running is True
    finally:
        belt.stop()


def test_stop_is_idempotent_and_cleans_up_once(monkeypatch) -> None:
    """Repeated shutdown calls should not double-stop or double-close the device."""

    fake = FakeDevice([_make_sample(1)])
    belt = _make_belt(monkeypatch, fake)
    belt.start()

    assert _wait_until(lambda: belt.is_running)
    belt.stop()
    belt.stop()

    assert belt.is_running is False
    assert fake.stop_calls == 1
    assert fake.close_calls == 1
