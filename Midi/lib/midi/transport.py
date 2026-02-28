from midi.midiMessages import MidiMessages
from typing import Optional

class MidiTransport:
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
        self.midi_messages.send_message('start', channel=self.channel)

    def stop(self):
        """Send MIDI Stop message."""
        self.midi_messages.send_message('stop', channel=self.channel)

    def continue_(self):
        """Send MIDI Continue message."""
        self.midi_messages.send_message('continue', channel=self.channel)

    def clock(self):
        """Send MIDI Clock message."""
        self.midi_messages.send_message('clock', channel=self.channel)

    def song_position(self, position: int):
        """Send MIDI Song Position Pointer message."""
        self.midi_messages.send_message('song_position', value=position, channel=self.channel)

    def song_select(self, song: int):
        """Send MIDI Song Select message."""
        self.midi_messages.send_message('song_select', value=song, channel=self.channel)

    def send_custom(self, message_type: str, **kwargs):
        """Send a custom MIDI message type with extra kwargs."""
        self.midi_messages.send_message(message_type, channel=self.channel, **kwargs)
