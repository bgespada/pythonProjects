import tkinter as tk
from tkinter import ttk
from .scaleTreeUi import ScaleTreeUi
from .pianoRollUi import PianoRollUi
from .sequencerEngine import SequencerEngine


class SequencerFrameUi(ttk.LabelFrame):
    """
    Main sequencer frame combining:
      - Toolbar  : step count selector, velocity control, hint label
      - Left     : ScaleTreeUi    – scale/root selector
      - Right    : PianoRollUi   – canvas piano roll
      - Engine   : SequencerEngine – playback timing

    Start/stop is driven externally by TransportFrameUi callbacks:
        engine.start()   ← called when Transport Start is pressed
        engine.stop()    ← called when Transport Stop is pressed
    """

    STEP_OPTIONS = [8, 16, 32]

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, text="Sequencer", padding="6", **kwargs)

        self.engine = SequencerEngine(widget=self)
        self._build_ui()
        self._wire_engine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_midi_messages(self, midi_messages) -> None:
        """Inject (or clear) the MidiMessages instance."""
        self.engine.set_midi_messages(midi_messages)

    def set_channel(self, channel: int) -> None:
        """Set MIDI channel (0-based)."""
        self.engine.set_channel(channel)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Toolbar ──────────────────────────────────────────────────
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        ttk.Label(toolbar, text="Steps:").pack(side=tk.LEFT)
        self._steps_var = tk.IntVar(value=16)
        steps_cb = ttk.Combobox(
            toolbar,
            textvariable=self._steps_var,
            values=self.STEP_OPTIONS,
            width=4,
            state="readonly",
        )
        steps_cb.pack(side=tk.LEFT, padx=(4, 0))
        steps_cb.bind("<<ComboboxSelected>>", self._on_steps_change)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=8, fill=tk.Y
        )

        ttk.Label(toolbar, text="Velocity:").pack(side=tk.LEFT)
        self._vel_var = tk.IntVar(value=100)
        vel_spin = ttk.Spinbox(
            toolbar,
            from_=1, to=127,
            textvariable=self._vel_var,
            width=4,
            command=self._on_velocity_change,
        )
        vel_spin.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=8, fill=tk.Y
        )

        ttk.Label(
            toolbar,
            text="Click cell: toggle  |  Click length row: cycle 1/16 → 1/8 → 1/4",
            foreground="gray",
            font=("Arial", 8),
        ).pack(side=tk.LEFT)

        # ── Scale tree (left) ─────────────────────────────────────────
        self.scale_tree = ScaleTreeUi(
            parent=self,
            on_scale_select=self._on_scale_select,
        )
        self.scale_tree.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 6))

        # ── Piano roll (right) ────────────────────────────────────────
        self.piano_roll = PianoRollUi(parent=self, num_steps=16)
        self.piano_roll.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

    def _wire_engine(self) -> None:
        """Connect engine callbacks to piano roll and toolbar variables."""
        self.engine.get_active_notes = self.piano_roll.get_active_notes_at_step
        self.engine.get_step_length  = self.piano_roll.get_step_length
        self.engine.get_num_steps    = lambda: self._steps_var.get()
        self.engine.on_step_change   = self._on_step_change

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_scale_select(self, notes: list[int], names: list[str]) -> None:
        self.piano_roll.set_scale(notes, names)

    def _on_steps_change(self, event=None) -> None:
        self.piano_roll.set_num_steps(self._steps_var.get())

    def _on_velocity_change(self) -> None:
        self.engine.velocity = self._vel_var.get()

    def _on_step_change(self, step: int) -> None:
        if step < 0:
            self.piano_roll.clear_highlight()
        else:
            self.piano_roll.highlight_step(step)
