"""Tests for the LSL breathing sender wrapper."""

from __future__ import annotations

import importlib
import sys
import types


class _FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.children: list[_FakeNode] = []
        self.values: dict[str, str] = {}

    def append_child(self, name: str) -> "_FakeNode":
        child = _FakeNode(name)
        self.children.append(child)
        return child

    def append_child_value(self, key: str, value: str) -> "_FakeNode":
        self.values[key] = value
        return self


class _FakeStreamInfo:
    def __init__(
        self,
        name: str,
        type: str,
        channel_count: int,
        nominal_srate: float,
        channel_format: str,
        source_id: str,
    ) -> None:
        self.name = name
        self.type = type
        self.channel_count = channel_count
        self.nominal_srate = nominal_srate
        self.channel_format = channel_format
        self.source_id = source_id
        self._desc = _FakeNode("desc")

    def desc(self) -> _FakeNode:
        return self._desc


class _FakeStreamOutlet:
    def __init__(self, info: _FakeStreamInfo) -> None:
        self.info = info
        self.samples: list[list[float]] = []

    def push_sample(self, sample: list[float]) -> None:
        self.samples.append(sample)


def _reload_lsl_module(monkeypatch):
    fake_pylsl = types.SimpleNamespace(
        StreamInfo=_FakeStreamInfo,
        StreamOutlet=_FakeStreamOutlet,
    )
    monkeypatch.setitem(sys.modules, "pylsl", fake_pylsl)
    sys.modules.pop("src.lsl_out", None)
    import src.lsl_out as lsl_out

    return importlib.reload(lsl_out)


def test_lsl_sender_publishes_two_channel_samples_with_labels(monkeypatch) -> None:
    lsl_out = _reload_lsl_module(monkeypatch)
    sender = lsl_out.LSLBreathingSender()
    sender.send([0.25, 1.0])

    assert sender.info.channel_count == 2
    assert sender.outlet.samples == [[0.25, 1.0]]

    channels_node = sender.info.desc().children[0]
    labels = [child.values["label"] for child in channels_node.children]
    assert labels == ["breath_level", "event_code"]


def test_lsl_sender_supports_movement_stream_identity(monkeypatch) -> None:
    lsl_out = _reload_lsl_module(monkeypatch)
    sender = lsl_out.LSLBreathingSender(
        name="BreathingBeltMovement",
        type="BreathingMovement",
        source_id="breathingbelt001_movement",
        channel_labels=("movement_value", "event_code"),
    )
    sender.send([2.5, -1.0])

    assert sender.info.name == "BreathingBeltMovement"
    assert sender.info.type == "BreathingMovement"
    assert sender.info.source_id == "breathingbelt001_movement"
    assert sender.outlet.samples == [[2.5, -1.0]]

    channels_node = sender.info.desc().children[0]
    labels = [child.values["label"] for child in channels_node.children]
    assert labels == ["movement_value", "event_code"]


def test_lsl_sender_supports_adaptive_stream_identity(monkeypatch) -> None:
    lsl_out = _reload_lsl_module(monkeypatch)
    sender = lsl_out.LSLBreathingSender(
        name="BreathingBeltAdaptive",
        type="BreathingAdaptive",
        source_id="breathingbelt001_adaptive",
        channel_labels=("breath_level", "event_code"),
    )
    sender.send([0.65, 1.0])

    assert sender.info.name == "BreathingBeltAdaptive"
    assert sender.info.type == "BreathingAdaptive"
    assert sender.info.source_id == "breathingbelt001_adaptive"
    assert sender.outlet.samples == [[0.65, 1.0]]
