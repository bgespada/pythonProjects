import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from .scaleTreeUi import ScaleTreeUi
from .pianoRollUi import PianoRollUi
from .sequencerEngine import SequencerEngine
from .presets import PRESET_NAMES, build_preset
from .patternStorage import save_pattern, load_pattern


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
        self._current_scale_notes: list[int] = []
        self._current_note_names: list[str] = []
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
        vel_spin.bind("<Return>", lambda _event: self._on_velocity_change())
        vel_spin.bind("<FocusOut>", lambda _event: self._on_velocity_change())
        self._vel_var.trace_add("write", lambda *_: self._on_velocity_change())

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=8, fill=tk.Y
        )

        ttk.Label(toolbar, text="Swing:").pack(side=tk.LEFT)
        self._swing_var = tk.IntVar(value=0)
        swing_spin = ttk.Spinbox(
            toolbar,
            from_=0, to=75,
            textvariable=self._swing_var,
            width=4,
            command=self._on_swing_change,
        )
        swing_spin.pack(side=tk.LEFT, padx=(4, 0))
        swing_spin.bind("<Return>", lambda _event: self._on_swing_change())
        swing_spin.bind("<FocusOut>", lambda _event: self._on_swing_change())

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=8, fill=tk.Y
        )

        ttk.Label(toolbar, text="Preset:").pack(side=tk.LEFT)
        self._preset_var = tk.StringVar(value=PRESET_NAMES[0])
        self._preset_combo = ttk.Combobox(
            toolbar,
            textvariable=self._preset_var,
            values=PRESET_NAMES,
            width=10,
            state="readonly",
        )
        self._preset_combo.pack(side=tk.LEFT, padx=(4, 0))

        ttk.Button(toolbar, text="Apply", command=self._apply_preset).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, padx=8, fill=tk.Y
        )

        ttk.Button(toolbar, text="Save", command=self._save_pattern).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Load", command=self._load_pattern).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(
            toolbar,
            text="Click cell: toggle | LEN: step timing | VEL: per-step velocity | GATE: note length 0..127",
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
        self.engine.get_step_gate = self.piano_roll.get_step_gate
        self.engine.get_step_velocity = self.piano_roll.get_step_velocity
        self.engine.get_num_steps    = lambda: self._steps_var.get()
        self.engine.on_step_change   = self._on_step_change

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_scale_select(self, notes: list[int], names: list[str]) -> None:
        self._current_scale_notes = list(notes)
        self._current_note_names = list(names)
        self.piano_roll.set_scale(notes, names)

    def _on_steps_change(self, _event=None) -> None:
        self.piano_roll.set_num_steps(self._steps_var.get())

    def _on_velocity_change(self) -> None:
        try:
            velocity = int(self._vel_var.get())
            velocity = max(1, min(127, velocity))
            self._vel_var.set(velocity)
            self.engine.velocity = velocity
            self.piano_roll.set_default_step_velocity(velocity)
        except Exception:
            pass

    def _on_swing_change(self) -> None:
        try:
            swing = int(self._swing_var.get())
            swing = max(0, min(75, swing))
            self._swing_var.set(swing)
            self.engine.set_swing_percent(swing)
        except Exception:
            pass

    def _apply_preset(self) -> None:
        if not self._current_scale_notes:
            messagebox.showinfo("Preset", "Select a scale first.")
            return

        pattern = build_preset(
            name=self._preset_var.get(),
            scale_notes=self._current_scale_notes,
            note_names=self._current_note_names,
            num_steps=self._steps_var.get(),
        )
        self.piano_roll.load_pattern(pattern)

    def _save_pattern(self) -> None:
        pattern = self.piano_roll.export_pattern()
        file_path = filedialog.asksaveasfilename(
            title="Save Sequencer Pattern",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            save_pattern(file_path, pattern)
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save pattern: {e}")

    def _load_pattern(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Load Sequencer Pattern",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            pattern = load_pattern(file_path)
            loaded_steps = int(pattern.get("num_steps", self._steps_var.get()))
            loaded_steps = max(8, min(32, loaded_steps))
            if loaded_steps in self.STEP_OPTIONS:
                self._steps_var.set(loaded_steps)
            self.piano_roll.load_pattern(pattern)
            self._current_scale_notes = list(pattern.get("scale_notes", self._current_scale_notes))
            self._current_note_names = list(pattern.get("note_names", self._current_note_names))
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load pattern: {e}")

    def _on_step_change(self, step: int) -> None:
        if step < 0:
            self.piano_roll.clear_highlight()
        else:
            self.piano_roll.highlight_step(step)
