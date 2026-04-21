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
        self.timestamps: list[float | None] = []
        self.chunks: list[tuple[list[list[float]], list[float] | None]] = []

    def push_sample(self, sample: list[float], timestamp: float | None = None) -> None:
        self.samples.append(sample)
        self.timestamps.append(timestamp)

    def push_chunk(
        self,
        samples: list[list[float]],
        timestamps: list[float] | None = None,
    ) -> None:
        self.chunks.append((samples, timestamps))


def _reload_lsl_module(monkeypatch):
    fake_pylsl = types.SimpleNamespace(
        StreamInfo=_FakeStreamInfo,
        StreamOutlet=_FakeStreamOutlet,
        local_clock=lambda: 123.456,
    )
    monkeypatch.setitem(sys.modules, "pylsl", fake_pylsl)
    sys.modules.pop("src.lsl_out", None)
    import src.lsl_out as lsl_out

    return importlib.reload(lsl_out)


def test_lsl_sender_publishes_single_channel_samples_with_labels(monkeypatch) -> None:
    lsl_out = _reload_lsl_module(monkeypatch)
    sender = lsl_out.LSLBreathingSender()
    sender.send(0.25)

    assert sender.info.channel_count == 1
    assert sender.outlet.samples == [[0.25]]

    channels_node = sender.info.desc().children[0]
    labels = [child.values["label"] for child in channels_node.children]
    assert labels == ["breath_level"]
    assert sender.outlet.timestamps == [None]


def test_lsl_sender_supports_movement_stream_identity(monkeypatch) -> None:
    lsl_out = _reload_lsl_module(monkeypatch)
    sender = lsl_out.LSLBreathingSender(
        name="BreathingBeltMovement",
        type="BreathingMovement",
        source_id="breathingbelt001_movement",
        channel_labels=("movement_value",),
    )
    sender.send(2.5)

    assert sender.info.name == "BreathingBeltMovement"
    assert sender.info.type == "BreathingMovement"
    assert sender.info.source_id == "breathingbelt001_movement"
    assert sender.outlet.samples == [[2.5]]

    channels_node = sender.info.desc().children[0]
    labels = [child.values["label"] for child in channels_node.children]
    assert labels == ["movement_value"]


def test_lsl_sender_supports_adaptive_stream_identity(monkeypatch) -> None:
    lsl_out = _reload_lsl_module(monkeypatch)
    sender = lsl_out.LSLBreathingSender(
        name="BreathingBeltAdaptive",
        type="BreathingAdaptive",
        source_id="breathingbelt001_adaptive",
        channel_labels=("breath_level",),
    )
    sender.send(0.65)

    assert sender.info.name == "BreathingBeltAdaptive"
    assert sender.info.type == "BreathingAdaptive"
    assert sender.info.source_id == "breathingbelt001_adaptive"
    assert sender.outlet.samples == [[0.65]]


def test_lsl_sender_supports_explicit_timestamps(monkeypatch) -> None:
    lsl_out = _reload_lsl_module(monkeypatch)
    sender = lsl_out.LSLBreathingSender()

    assert sender.now() == 123.456
    sender.send(0.5, timestamp=321.0)

    assert sender.outlet.samples == [[0.5]]
    assert sender.outlet.timestamps == [321.0]


def test_lsl_sender_supports_explicit_timestamp_chunks(monkeypatch) -> None:
    lsl_out = _reload_lsl_module(monkeypatch)
    sender = lsl_out.LSLBreathingSender()
    sender.send_chunk([0.1, 0.2], timestamps=[1.0, 1.01])

    assert sender.outlet.chunks == [([ [0.1], [0.2] ], [1.0, 1.01])]


def test_lsl_sender_records_timing_and_event_metadata(monkeypatch) -> None:
    lsl_out = _reload_lsl_module(monkeypatch)
    sender = lsl_out.LSLBreathingSender(
        name="BreathingBeltEvents",
        type="BreathingEvents",
        source_id="breathingbelt001_events",
        channel_labels=("event_code",),
        timing_metadata={
            "timestamp_domain": "local_clock",
            "timestamp_origin": "host_estimated_from_chunk_return",
        },
        event_code_map={
            1.0: "inhale_peak",
            -1.0: "exhale_trough",
        },
    )

    desc_children = {child.name: child for child in sender.info.desc().children}
    assert desc_children["timing"].values["timestamp_domain"] == "local_clock"
    event_nodes = desc_children["event_codes"].children
    assert event_nodes[0].values == {"code": "1.0", "label": "inhale_peak"}
    assert event_nodes[1].values == {"code": "-1.0", "label": "exhale_trough"}
