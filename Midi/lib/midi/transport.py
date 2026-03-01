from midi.midiMessages import MidiMessages
from typing import Optional

class MidiTransport:
    _global_tempo_bpm = 120  # Default global tempo

    """
    Class to manage MIDI transport controls (start, stop, continue, clock, etc.)
    and send the appropriate MIDI messages using MidiMessages, with support for channel and config.
    """
    def __init__(self, midi_messages: MidiMessages, channel: Optional[int] = None):
        """
        Args:
            midi_messages (MidiMessages): Instance to send MIDI messages.
            channel (Optional[int]): MIDI channel (0-15) or None for omni.
        """
        self.midi_messages = midi_messages
        self.channel = channel

    def set_channel(self, channel: Optional[int]):
        self.channel = channel

    def start(self):
        """Send MIDI Start message."""
        import mido
        msg = mido.Message('start')
        self.midi_messages.send_message(msg)

    def stop(self):
        """Send MIDI Stop message."""
        import mido
        msg = mido.Message('stop')
        self.midi_messages.send_message(msg)

    def continue_(self):
        """Send MIDI Continue message."""
        import mido
        msg = mido.Message('continue')
        self.midi_messages.send_message(msg)

    def clock(self):
        """Send MIDI Clock message."""
        import mido
        msg = mido.Message('clock')
        self.midi_messages.send_message(msg)

    def song_position(self, position: int):
        """Send MIDI Song Position Pointer message."""
        import mido
        msg = mido.Message('songpos', pos=position)
        self.midi_messages.send_message(msg)

    def song_select(self, song: int):
        """Send MIDI Song Select message."""
        import mido
        msg = mido.Message('song_select', song=song)
        self.midi_messages.send_message(msg)

    def send_custom(self, message_type: str, **kwargs):
        """Send a custom MIDI message type with extra kwargs."""
        import mido
        msg = mido.Message(message_type, **kwargs)
        self.midi_messages.send_message(msg)

    @classmethod
    def set_global_tempo(cls, bpm: float):
        """
        Set the global tempo in beats per minute (BPM).
        Args:
            bpm (float): Tempo in BPM
        """
        if bpm <= 0:
            raise ValueError("Tempo must be positive")
        cls._global_tempo_bpm = bpm

    @classmethod
    def get_global_tempo(cls) -> float:
        """
        Get the current global tempo in BPM.
        Returns:
            float: Current tempo in BPM
        """
        return cls._global_tempo_bpm

    def send_clock_for_tempo(self, duration_seconds: float = 1.0):
        """
        Send MIDI clock messages at the current global tempo for a given duration.
        Args:
            duration_seconds (float): How long to send clock messages (default 1 second)
        """
        import time
        bpm = self.get_global_tempo()
        clocks_per_beat = 24
        interval = 60.0 / (bpm * clocks_per_beat)
        end_time = time.time() + duration_seconds
        while time.time() < end_time:
            self.clock()
            time.sleep(interval)
