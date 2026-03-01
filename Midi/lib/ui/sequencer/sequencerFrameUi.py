import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from .scaleTreeUi import ScaleTreeUi
from .pianoRollUi import PianoRollUi
from .sequencerEngine import SequencerEngine
from .presets import (
    UTILITY_PRESET_NAMES,
    MUSICAL_PRESET_NAMES,
    MUSICAL_PRESET_SCALES,
    MUSICAL_PRESET_STEPS,
    build_preset,
)
from .patternStorage import save_pattern, load_pattern
from midi.scales import SCALE_FAMILIES, ROOT_NOTES, generate_notes, note_name, MIN_MIDI_NOTE, MAX_MIDI_NOTE


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
        self._apply_musical_preset()

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
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # ── Global toolbar ───────────────────────────────────────────
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky=tk.W, pady=(0, 4))

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

        ttk.Button(toolbar, text="Save", command=self._save_pattern).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Load", command=self._load_pattern).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(
            toolbar,
            text="Click cell: toggle | LEN: step timing | VEL: per-step velocity | GATE: note length 0..127",
            foreground="gray",
            font=("Arial", 8),
        ).pack(side=tk.LEFT)

        # ── Notebook with two approaches ──────────────────────────────
        self._notebook = ttk.Notebook(self)
        self._notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._step_tab = ttk.Frame(self._notebook)
        self._musical_tab = ttk.Frame(self._notebook)
        self._step_tab.columnconfigure(1, weight=1)
        self._step_tab.rowconfigure(1, weight=1)
        self._musical_tab.columnconfigure(0, weight=1)
        self._musical_tab.rowconfigure(1, weight=1)

        self._notebook.add(self._step_tab, text="Step Sequencer")
        self._notebook.add(self._musical_tab, text="Musical Presets")

        # ── Step Sequencer tab ───────────────────────────────────────
        step_toolbar = ttk.Frame(self._step_tab)
        step_toolbar.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        ttk.Label(step_toolbar, text="Steps:").pack(side=tk.LEFT)
        self._steps_var = tk.IntVar(value=16)
        steps_cb = ttk.Combobox(
            step_toolbar,
            textvariable=self._steps_var,
            values=self.STEP_OPTIONS,
            width=4,
            state="readonly",
        )
        steps_cb.pack(side=tk.LEFT, padx=(4, 0))
        steps_cb.bind("<<ComboboxSelected>>", self._on_steps_change)

        ttk.Separator(step_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        ttk.Label(step_toolbar, text="Utility Preset:").pack(side=tk.LEFT)
        self._utility_preset_var = tk.StringVar(value=UTILITY_PRESET_NAMES[0])
        self._utility_preset_combo = ttk.Combobox(
            step_toolbar,
            textvariable=self._utility_preset_var,
            values=UTILITY_PRESET_NAMES,
            width=12,
            state="readonly",
        )
        self._utility_preset_combo.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(step_toolbar, text="Apply", command=self._apply_utility_preset).pack(side=tk.LEFT, padx=(4, 0))

        # ── Scale tree (left) ─────────────────────────────────────────
        self.scale_tree = ScaleTreeUi(
            parent=self._step_tab,
            on_scale_select=self._on_scale_select,
        )
        self.scale_tree.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 6))

        # ── Piano roll (right) ────────────────────────────────────────
        self.piano_roll = PianoRollUi(parent=self._step_tab, num_steps=16)
        self.piano_roll.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ── Musical Presets tab ──────────────────────────────────────
        musical_toolbar = ttk.Frame(self._musical_tab)
        musical_toolbar.grid(row=0, column=0, sticky=tk.W, pady=(0, 4))

        ttk.Label(musical_toolbar, text="Musical Preset:").pack(side=tk.LEFT)
        self._musical_preset_var = tk.StringVar(value=MUSICAL_PRESET_NAMES[0])
        self._musical_preset_combo = ttk.Combobox(
            musical_toolbar,
            textvariable=self._musical_preset_var,
            values=MUSICAL_PRESET_NAMES,
            width=24,
            state="readonly",
        )
        self._musical_preset_combo.pack(side=tk.LEFT, padx=(4, 0))
        self._musical_preset_combo.bind("<<ComboboxSelected>>", self._on_musical_preset_selected)
        ttk.Button(musical_toolbar, text="Apply", command=self._apply_musical_preset).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(musical_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        ttk.Label(musical_toolbar, text="Steps:").pack(side=tk.LEFT)
        self._musical_steps_var = tk.IntVar(value=MUSICAL_PRESET_STEPS.get(self._musical_preset_var.get(), 32))
        ttk.Label(musical_toolbar, textvariable=self._musical_steps_var).pack(side=tk.LEFT, padx=(4, 0))

        self.musical_piano_roll = PianoRollUi(parent=self._musical_tab, num_steps=32)
        self.musical_piano_roll.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def _wire_engine(self) -> None:
        """Connect engine callbacks to piano roll and toolbar variables."""
        self.engine.get_active_notes = lambda step: self._active_roll().get_active_notes_at_step(step)
        self.engine.get_step_length = lambda step: self._active_roll().get_step_length(step)
        self.engine.get_step_gate = lambda step: self._active_roll().get_step_gate(step)
        self.engine.get_step_velocity = lambda step: self._active_roll().get_step_velocity(step)
        self.engine.get_num_steps = lambda: self._active_steps_var().get()
        self.engine.on_step_change   = self._on_step_change

    def _is_musical_tab_active(self) -> bool:
        return str(self._notebook.select()) == str(self._musical_tab)

    def _active_roll(self) -> PianoRollUi:
        return self.musical_piano_roll if self._is_musical_tab_active() else self.piano_roll

    def _active_steps_var(self) -> tk.IntVar:
        return self._musical_steps_var if self._is_musical_tab_active() else self._steps_var

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

    def _apply_pattern_to_roll(
        self,
        preset_name: str,
        piano_roll: PianoRollUi,
        steps_var: tk.IntVar,
        scale_notes: list[int],
        note_names: list[str],
        enforce_any_step_count: bool = False,
    ) -> None:
        if not scale_notes:
            messagebox.showinfo("Preset", "Select a scale first.")
            return

        pattern = build_preset(
            name=preset_name,
            scale_notes=scale_notes,
            note_names=note_names,
            num_steps=steps_var.get(),
        )

        preset_steps = int(pattern.get("num_steps", steps_var.get()))
        preset_steps = max(8, min(32, preset_steps))
        if enforce_any_step_count or preset_steps in self.STEP_OPTIONS:
            steps_var.set(preset_steps)

        piano_roll.load_pattern(pattern)

    def _apply_utility_preset(self) -> None:
        if not self._current_scale_notes:
            messagebox.showinfo("Preset", "Select a scale first.")
            return
        self._apply_pattern_to_roll(
            preset_name=self._utility_preset_var.get(),
            piano_roll=self.piano_roll,
            steps_var=self._steps_var,
            scale_notes=self._current_scale_notes,
            note_names=self._current_note_names,
        )

    def _build_scale_for_musical_preset(self, preset_name: str) -> tuple[list[int], list[str]]:
        scale_def = MUSICAL_PRESET_SCALES.get(preset_name)
        if not scale_def:
            return [], []
        root = scale_def.get("root", "C")
        family = scale_def.get("family", "Diatonic")
        scale_name = scale_def.get("scale", "Natural Minor")

        root_midi = ROOT_NOTES.get(root, 0)
        intervals = SCALE_FAMILIES.get(family, {}).get(scale_name)
        if not intervals:
            return [], []

        notes = generate_notes(
            root_midi,
            intervals,
            min_midi_note=MIN_MIDI_NOTE,
            max_midi_note=MAX_MIDI_NOTE,
        )
        names = [note_name(n) for n in notes]
        return notes, names

    def _apply_musical_preset(self) -> None:
        preset_name = self._musical_preset_var.get()
        notes, names = self._build_scale_for_musical_preset(preset_name)
        if not notes:
            messagebox.showerror("Preset", "Could not build scale for selected musical preset.")
            return

        # Keep step tab scale selection in sync visually (optional) while ensuring
        # musical tab works without manual scale selection.
        target_scale = MUSICAL_PRESET_SCALES.get(preset_name)
        if target_scale:
            self.scale_tree.set_scale_selection(
                root=target_scale["root"],
                family=target_scale["family"],
                scale=target_scale["scale"],
                emit=True,
            )

        self._apply_pattern_to_roll(
            preset_name=preset_name,
            piano_roll=self.musical_piano_roll,
            steps_var=self._musical_steps_var,
            scale_notes=notes,
            note_names=names,
            enforce_any_step_count=True,
        )

    def _on_musical_preset_selected(self, _event=None) -> None:
        preset_name = self._musical_preset_var.get()
        suggested_steps = MUSICAL_PRESET_STEPS.get(preset_name)
        if suggested_steps is not None:
            self._musical_steps_var.set(max(8, min(32, int(suggested_steps))))

    def _on_tab_changed(self, _event=None) -> None:
        self.piano_roll.clear_highlight()
        self.musical_piano_roll.clear_highlight()

    def _save_pattern(self) -> None:
        pattern = self._active_roll().export_pattern()
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
            steps_var = self._active_steps_var()
            roll = self._active_roll()
            loaded_steps = int(pattern.get("num_steps", steps_var.get()))
            loaded_steps = max(8, min(32, loaded_steps))
            if loaded_steps in self.STEP_OPTIONS:
                steps_var.set(loaded_steps)
            roll.load_pattern(pattern)

            # Keep step-sequencer scale context updated from loaded content.
            if not self._is_musical_tab_active():
                self._current_scale_notes = list(pattern.get("scale_notes", self._current_scale_notes))
                self._current_note_names = list(pattern.get("note_names", self._current_note_names))
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load pattern: {e}")

    def _on_step_change(self, step: int) -> None:
        if step < 0:
            self.piano_roll.clear_highlight()
            self.musical_piano_roll.clear_highlight()
        else:
            if self._is_musical_tab_active():
                self.musical_piano_roll.highlight_step(step)
                self.piano_roll.clear_highlight()
            else:
                self.piano_roll.highlight_step(step)
                self.musical_piano_roll.clear_highlight()
