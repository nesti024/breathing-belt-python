"""Lab Streaming Layer (LSL) output wrapper for normalized breathing data."""

from __future__ import annotations

from collections.abc import Iterable

from pylsl import StreamInfo, StreamOutlet, local_clock


class LSLBreathingSender:
    """Publish breathing control data as a multi-channel LSL stream.

    The stream is configured as ``float32`` because the breathing control
    signal and event codes are represented as numeric channels.
    """

    def __init__(
        self,
        name: str = "BreathingBelt",
        type: str = "Breathing",
        channel_count: int = 2,
        nominal_srate: float = 0,
        source_id: str = "breathingbelt001",
        channel_labels: Iterable[str] | None = ("breath_level", "event_code"),
    ) -> None:
        """Create the LSL outlet and its associated stream metadata."""

        labels = tuple(channel_labels or ())
        if labels and len(labels) != channel_count:
            raise ValueError("channel_labels must match channel_count when provided.")

        self.info = StreamInfo(
            name,
            type,
            channel_count,
            nominal_srate,
            "float32",
            source_id,
        )
        if labels:
            channels = self.info.desc().append_child("channels")
            for label in labels:
                channel = channels.append_child("channel")
                channel.append_child_value("label", str(label))
        self.outlet = StreamOutlet(self.info)

    def now(self) -> float:
        """Return the current LSL clock time."""

        return float(local_clock())

    def send(
        self,
        data: float | int | Iterable[float | int],
        timestamp: float | None = None,
    ) -> None:
        """Send one sample to the LSL outlet.

        Parameters
        ----------
        data:
            Either a scalar value or an iterable of channel values. The current
            breathing-belt application uses a two-channel stream by default,
            but the method accepts scalars to keep the wrapper generic.
        """

        if isinstance(data, (float, int)):
            sample = [float(data)]
        else:
            sample = [float(x) for x in data]

        if timestamp is None:
            self.outlet.push_sample(sample)
        else:
            self.outlet.push_sample(sample, float(timestamp))
