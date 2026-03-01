"""
Sequencer playback engine.

Advances through steps using Tkinter's after() scheduler so it runs
on the UI thread — no threading required.

Timing is derived from the global tempo stored in MidiTransport.
Step duration = (60_000 ms / BPM) / 4  × step_length_multiplier
(base unit is a 1/16 note).
"""

import tkinter as tk
from typing import Callable, Optional
from midi.transport import MidiTransport


class SequencerEngine:
    """
    Drives the step sequencer: advances steps, sends note_on/note_off.

    Callbacks to wire up (set by SequencerFrameUi):
        get_active_notes(step) -> list[int]   MIDI note numbers to play at step
        get_step_length(step)  -> int         length multiplier (1, 2 or 4)
        get_num_steps()        -> int         total number of steps
        on_step_change(step)                  UI highlight callback (-1 to clear)
    """

    def __init__(self, widget: tk.Widget):
        """
        Args:
            widget: Any tkinter widget used to schedule after() calls.
        """
        self._widget = widget
        self.midi_messages = None
        self._channel: int = 0
        self._running: bool = False
        self._current_step: int = 0
        self._after_id: Optional[str] = None
        self.velocity: int = 100

        # Wired by SequencerFrameUi
        self.get_active_notes: Callable[[int], list[int]] = lambda step: []
        self.get_step_length:  Callable[[int], int]       = lambda step: 1
        self.get_num_steps:    Callable[[], int]          = lambda: 16
        self.on_step_change:   Optional[Callable[[int], None]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_midi_messages(self, midi_messages) -> None:
        """Inject (or clear) the MidiMessages instance."""
        self.midi_messages = midi_messages

    def set_channel(self, channel: int) -> None:
        """Set MIDI channel (0-based, 0–15)."""
        self._channel = max(0, min(15, int(channel)))

    def start(self) -> None:
        """Start the sequencer from step 0."""
        if self._running:
            return
        self._running = True
        self._current_step = 0
        self._tick()

    def stop(self) -> None:
        """Stop the sequencer and send all-notes-off."""
        self._running = False
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        self._all_notes_off()
        if self.on_step_change:
            self.on_step_change(-1)

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal tick loop
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if not self._running:
            return

        step = self._current_step

        # Collect and play active notes
        active = self.get_active_notes(step)
        if self.midi_messages and active:
            for note in active:
                try:
                    self.midi_messages.send_note_on(note, self.velocity, self._channel)
                except Exception as e:
                    print(f"[Sequencer] note_on error: {e}")

        # Notify UI
        if self.on_step_change:
            self.on_step_change(step)

        # Calculate step duration from current global tempo
        bpm = MidiTransport.get_global_tempo()
        base_ms = max(10, int(60_000 / bpm / 4))   # 1/16 note in ms
        multiplier = self.get_step_length(step)
        duration_ms = base_ms * multiplier

        # Schedule note_off just before the next step
        note_off_delay = max(1, duration_ms - 10)
        if active:
            self._widget.after(
                note_off_delay,
                lambda notes=active: self._send_note_off(notes),
            )

        # Advance to next step
        num_steps = max(1, self.get_num_steps())
        self._current_step = (step + 1) % num_steps

        # Schedule next tick
        self._after_id = self._widget.after(duration_ms, self._tick)

    def _send_note_off(self, notes: list[int]) -> None:
        if self.midi_messages:
            for note in notes:
                try:
                    self.midi_messages.send_note_off(note, 0, self._channel)
                except Exception as e:
                    print(f"[Sequencer] note_off error: {e}")

    def _all_notes_off(self) -> None:
        """Best-effort all-notes-off on the current channel."""
        if self.midi_messages:
            for note in range(128):
                try:
                    self.midi_messages.send_note_off(note, 0, self._channel)
                except Exception:
                    pass
