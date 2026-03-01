from midi.midiMessages import MidiMessages
from typing import Optional
import mido

class MidiTransport:
    """
    Class to manage MIDI transport controls (start, stop, continue, clock, etc.)
    and send the appropriate MIDI messages using MidiMessages, with support for channel and config.
    """
    _global_tempo_bpm = 120.0  # Default global tempo

    def __init__(self, midi_messages: MidiMessages, channel: Optional[int] = None):
        """
        Args:
            midi_messages (MidiMessages): Instance to send MIDI messages.
            channel (Optional[int]): MIDI channel (0-15) or None for omni.
        """
        self.midi_messages = midi_messages
        self.channel = channel

    def set_channel(self, channel: Optional[int]) -> None:
        self.channel = channel

    def start(self) -> None:
        """Send MIDI Start message."""
        msg = mido.Message('start')
        self.midi_messages.send_message(msg)

    def stop(self) -> None:
        """Send MIDI Stop message."""
        msg = mido.Message('stop')
        self.midi_messages.send_message(msg)

    def continue_(self) -> None:
        """Send MIDI Continue message."""
        msg = mido.Message('continue')
        self.midi_messages.send_message(msg)

    def clock(self) -> None:
        """Send MIDI Clock message."""
        msg = mido.Message('clock')
        self.midi_messages.send_message(msg)

    def song_position(self, position: int) -> None:
        """Send MIDI Song Position Pointer message."""
        msg = mido.Message('songpos', pos=position)
        self.midi_messages.send_message(msg)

    def song_select(self, song: int) -> None:
        """Send MIDI Song Select message."""
        msg = mido.Message('song_select', song=song)
        self.midi_messages.send_message(msg)

    def send_custom(self, message_type: str, **kwargs) -> None:
        """Send a custom MIDI message type with extra kwargs."""
        msg = mido.Message(message_type, **kwargs)
        self.midi_messages.send_message(msg)

    @classmethod
    def set_global_tempo(cls, bpm: float) -> None:
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

    def send_clock_for_tempo(self, duration_seconds: float = 1.0) -> None:
        """
        Send MIDI clock messages at the current global tempo for a given duration.
        Args:
            duration_seconds (float): How long to send clock messages (default 1 second)
        """
        import time
        if duration_seconds <= 0:
            return
        bpm = self.get_global_tempo()
        clocks_per_beat = 24
        interval = 60.0 / (bpm * clocks_per_beat)
        end_time = time.time() + duration_seconds
        while time.time() < end_time:
            self.clock()
            time.sleep(interval)
