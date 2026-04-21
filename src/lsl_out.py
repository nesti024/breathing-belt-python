"""Lab Streaming Layer (LSL) output wrappers for breathing-belt streams."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pylsl import StreamInfo, StreamOutlet, local_clock


class LSLBreathingSender:
    """Publish breathing control or event data as a numeric LSL stream."""

    def __init__(
        self,
        name: str = "BreathingBelt",
        type: str = "Breathing",
        channel_count: int = 1,
        nominal_srate: float = 0,
        source_id: str = "breathingbelt001",
        channel_labels: Iterable[str] | None = ("breath_level",),
        *,
        timing_metadata: Mapping[str, str | float | int | bool] | None = None,
        event_code_map: Mapping[float | int, str] | None = None,
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
        desc = self.info.desc()
        if labels:
            channels = desc.append_child("channels")
            for label in labels:
                channel = channels.append_child("channel")
                channel.append_child_value("label", str(label))
        if timing_metadata:
            timing_node = desc.append_child("timing")
            for key, value in timing_metadata.items():
                timing_node.append_child_value(str(key), str(value))
        if event_code_map:
            event_codes_node = desc.append_child("event_codes")
            for code, label in event_code_map.items():
                event_node = event_codes_node.append_child("event")
                event_node.append_child_value("code", str(code))
                event_node.append_child_value("label", str(label))
        self.outlet = StreamOutlet(self.info)

    def now(self) -> float:
        """Return the current LSL clock time."""

        return float(local_clock())

    def send(
        self,
        data: float | int | Iterable[float | int],
        timestamp: float | None = None,
    ) -> None:
        """Send one sample to the LSL outlet."""

        sample = self._normalize_sample(data)
        if timestamp is None:
            self.outlet.push_sample(sample)
        else:
            self.outlet.push_sample(sample, float(timestamp))

    def send_chunk(
        self,
        samples: Sequence[float | int | Iterable[float | int]],
        *,
        timestamps: Sequence[float] | None = None,
    ) -> None:
        """Send a chunk of samples, preserving one timestamp per sample."""

        normalized_samples = [self._normalize_sample(sample) for sample in samples]
        if not normalized_samples:
            return

        if timestamps is not None and len(timestamps) != len(normalized_samples):
            raise ValueError("timestamps must match the number of samples in the chunk.")

        if timestamps is None:
            try:
                self.outlet.push_chunk(normalized_samples)
                return
            except AttributeError:
                for sample in normalized_samples:
                    self.outlet.push_sample(sample)
                return

        timestamp_list = [float(timestamp) for timestamp in timestamps]
        try:
            self.outlet.push_chunk(normalized_samples, timestamp_list)
        except (AttributeError, TypeError):
            for sample, timestamp in zip(normalized_samples, timestamp_list, strict=True):
                self.outlet.push_sample(sample, float(timestamp))

    @staticmethod
    def _normalize_sample(
        data: float | int | Iterable[float | int],
    ) -> list[float]:
        if isinstance(data, (float, int)):
            return [float(data)]
        return [float(value) for value in data]
