
from pylsl import StreamInfo, StreamOutlet

class LSLBreathingSender:
    """
    Class to send data via Lab Streaming Layer (LSL).
    Call send(data) to send a value (float or int) or a list/array of values.
    """
    def __init__(self, name="BreathingBelt", type="Breathing", channel_count=1, nominal_srate=0, source_id="breathingbelt001"):
        self.info = StreamInfo(name, type, channel_count, nominal_srate, 'float32', source_id)
        self.outlet = StreamOutlet(self.info)

    def send(self, data):
        """
        Send data via LSL. Data can be a single value or a list/array of values.
        """
        if isinstance(data, (float, int)):
            self.outlet.push_sample([float(data)])
        else:
            self.outlet.push_sample([float(x) for x in data])
