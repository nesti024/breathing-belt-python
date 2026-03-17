"""Lab Streaming Layer (LSL) output wrapper for normalized breathing data."""

from __future__ import annotations

from pylsl import StreamInfo, StreamOutlet


class LSLBreathingSender:
    """Publish normalized breathing samples as a single-channel LSL stream.

    The stream is configured as ``float32`` because the normalized breathing
    control signal is scalar and continuous in the range ``[0, 1]``.
    """

    def __init__(
        self,
        name="BreathingBelt",
        type="Breathing",
        channel_count=1,
        nominal_srate=0,
        source_id="breathingbelt001",
    ):
        """Create the LSL outlet and its associated stream metadata."""

        self.info = StreamInfo(
            name,
            type,
            channel_count,
            nominal_srate,
            "float32",
            source_id,
        )
        self.outlet = StreamOutlet(self.info)

    def send(self, data):
        """Send one sample to the LSL outlet.

        Parameters
        ----------
        data:
            Either a scalar value or an iterable of channel values. The current
            breathing-belt application uses a single-channel stream, but the
            method accepts iterables to keep the wrapper generic.
        """

        if isinstance(data, (float, int)):
            self.outlet.push_sample([float(data)])
        else:
            self.outlet.push_sample([float(x) for x in data])
