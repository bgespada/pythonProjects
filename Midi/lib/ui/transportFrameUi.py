import tkinter as tk
from tkinter import ttk
from typing import Callable
from midi.transport import MidiTransport

class TransportFrameUi(ttk.LabelFrame):
    """
    A reusable frame for MIDI transport controls (Start, Pause, Stop).
    Accepts a MidiTransport instance to send MIDI messages.
    """
    def __init__(self, parent, midi_transport: MidiTransport, **kwargs):
        super().__init__(parent, text="Transport Controls", padding="4", **kwargs)
        self.midi_transport = midi_transport
        self._clock_running = False
        self._clock_job = None
        self._start_callbacks: list[Callable] = []
        self._stop_callbacks:  list[Callable] = []
        self._build_ui()
        self.config(width=320, height=80)
        self.grid_propagate(False)

    def add_start_callback(self, callback: Callable) -> None:
        """Register a callback to be called when Start is pressed."""
        self._start_callbacks.append(callback)

    def add_stop_callback(self, callback: Callable) -> None:
        """Register a callback to be called when Stop is pressed."""
        self._stop_callbacks.append(callback)

    def _build_ui(self):
        # Label row (to match device frame)
        ttk.Label(self, text="Transport:", font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W)
        # Empty label for alignment
        ttk.Label(self, text="").grid(row=0, column=1, sticky=tk.W, padx=(6, 0))
        self.columnconfigure(1, weight=0)

        # Button frame (to match device frame layout)
        button_frame = ttk.Frame(self)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

        # Tempo spinbox
        ttk.Label(button_frame, text="Tempo:").grid(row=0, column=0, padx=(0, 2))
        self.tempo_var = tk.IntVar(value=120)
        self.tempo_spinbox = ttk.Spinbox(
            button_frame,
            from_=30, to=300,
            textvariable=self.tempo_var,
            width=5,
            increment=1,
            command=self._on_tempo_change
        )
        self.tempo_spinbox.grid(row=0, column=1, padx=(0, 8))
        self.tempo_spinbox.bind("<Return>", self._on_tempo_change)
        self.tempo_spinbox.bind("<FocusOut>", self._on_tempo_change)

        self.start_btn = ttk.Button(button_frame, text="Start", width=10, command=self._on_start)
        self.start_btn.grid(row=0, column=2, padx=(0, 2))

        self.pause_btn = ttk.Button(button_frame, text="Pause", width=10, command=self._on_pause)
        self.pause_btn.grid(row=0, column=3, padx=(0, 2))

        self.stop_btn = ttk.Button(button_frame, text="Stop", width=10, command=self._on_stop)
        self.stop_btn.grid(row=0, column=4)

    def _on_tempo_change(self, event=None):
        try:
            tempo = int(self.tempo_var.get())
            tempo = max(30, min(300, tempo))
            self.tempo_var.set(tempo)
            MidiTransport.set_global_tempo(tempo)
        except Exception as e:
            print(f"Failed to set tempo: {e}")

    def _on_start(self):
        if self.midi_transport:
            self.midi_transport.start()
            self._start_clock()
        for cb in self._start_callbacks:
            try:
                cb()
            except Exception as e:
                print(f"Start callback error: {e}")

    def _on_pause(self):
        if self.midi_transport:
            self.midi_transport.continue_()  # MIDI has Continue, not Pause; can be customized
            # Optionally, keep clock running or pause it here

    def _on_stop(self):
        if self.midi_transport:
            self.midi_transport.stop()
            self._stop_clock()
        for cb in self._stop_callbacks:
            try:
                cb()
            except Exception as e:
                print(f"Stop callback error: {e}")

    def _start_clock(self):
        if not self._clock_running:
            self._clock_running = True
            self._schedule_clock()

    def _stop_clock(self):
        self._clock_running = False
        if self._clock_job is not None:
            self.after_cancel(self._clock_job)
            self._clock_job = None

    def _schedule_clock(self):
        if not self._clock_running:
            return
        if self.midi_transport:
            self.midi_transport.clock()
        # Calculate interval based on tempo
        bpm = self.tempo_var.get() if hasattr(self, 'tempo_var') else 120
        clocks_per_beat = 24
        interval_ms = int(1000 * 60 / (bpm * clocks_per_beat))
        if interval_ms < 1:
            interval_ms = 1
        self._clock_job = self.after(interval_ms, self._schedule_clock)
