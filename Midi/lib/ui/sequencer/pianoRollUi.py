import tkinter as tk
from tkinter import ttk
from typing import Optional


class PianoRollUi:
    """
    Canvas-based piano roll grid.

    - Rows  = pitches from the selected scale (lowest at bottom, highest at top)
    - Columns = sequencer steps
    - Left-click a cell       → toggle note on/off
    - Click the length row    → cycle step length  1/16 → 1/8 → 1/4 → 1/16
    - Playback step is highlighted in a contrasting colour.
    """

    CELL_W      = 38
    CELL_H      = 22
    LABEL_W     = 50   # width of note-name column on the left
    HEADER_H    = 20   # height of step-number header row
    LEN_ROW_H   = 20   # height of step-length control row
    VEL_ROW_H   = 20   # height of per-step velocity row
    GATE_ROW_H  = 20   # height of per-step note-length (gate) row

    # Colour palette
    C_BG         = "#1a1a2e"
    C_GRID       = "#2a2a44"
    C_CELL_OFF_WHITE = "#d3d3d3"
    C_CELL_OFF_BLACK = "#111111"
    C_CELL_ON    = "#0f4c8a"
    C_CELL_PLAY  = "#e94560"   # active cell during playback
    C_HDR_BG     = "#0d0d1a"
    C_HDR_TEXT   = "#6677aa"
    C_HDR_PLAY   = "#e94560"   # current-step header highlight
    C_LBL_BG     = "#0d0d1a"
    C_LBL_TEXT   = "#8899bb"
    C_LEN_1_16   = "#1e2e1e"
    C_LEN_1_8    = "#2e2e10"
    C_LEN_1_4    = "#2e1010"
    C_LEN_TEXT   = "#aaaaaa"
    C_VEL_BG     = "#111a2d"
    C_VEL_TEXT   = "#9bb0ff"
    C_GATE_BG    = "#1a112d"
    C_GATE_TEXT  = "#d3a8ff"

    # Step-length options
    LEN_OPTS   = [1, 2, 4]
    LEN_LABELS = {1: "1/16", 2: "1/8", 4: "1/4"}
    LEN_COLORS = {1: "#1e2e1e", 2: "#2e2e10", 4: "#2e1010"}
    VEL_OPTS   = [20, 40, 60, 80, 100, 120, 127]
    GATE_OPTS  = [0, 16, 32, 48, 64, 80, 96, 112, 127]
    BLACK_KEY_PITCH_CLASSES = {1, 3, 6, 8, 10}

    def __init__(self, parent: tk.Widget, num_steps: int = 16):
        """
        Args:
            parent: Parent widget.
            num_steps: Initial number of steps (8, 16, or 32).
        """
        self.num_steps = num_steps
        self.scale_notes: list[int] = []    # MIDI note numbers, ascending
        self.note_names: list[str]  = []
        self._grid: dict[tuple[int, int], bool] = {}  # {(col, row_idx): active}
        self._step_lengths: list[int] = [1] * 32      # multiplier per step
        self._step_velocities: list[int] = [100] * 32 # velocity per step
        self._step_gates: list[int] = [64] * 32       # note length/gate per step (0..127)
        self._current_step: int = -1
        self._syncing_x = False
        self._param_drag_kind: Optional[str] = None   # "vel" | "gate" | None
        self._param_drag_col: Optional[int] = None
        self._param_drag_start_y: float = 0.0
        self._param_drag_start_value: int = 0
        self._value_editor = None
        self._value_editor_window = None
        self._value_editor_kind: Optional[str] = None
        self._value_editor_col: Optional[int] = None

        self._build_ui(parent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_scale(self, midi_notes: list[int], note_names: list[str]) -> None:
        """Replace the displayed scale rows and redraw the grid."""
        self.scale_notes = midi_notes
        self.note_names  = note_names
        self._grid.clear()
        self._current_step = -1
        self._draw()

    def set_num_steps(self, num_steps: int) -> None:
        """Change step count and redraw."""
        self.num_steps = num_steps
        self._draw()

    def set_default_step_velocity(self, velocity: int) -> None:
        """Set all currently visible step velocities to the same value."""
        clamped = max(1, min(127, int(velocity)))
        for col in range(self.num_steps):
            self._step_velocities[col] = clamped
        self._redraw_velocity_row()

    def highlight_step(self, step: int) -> None:
        """Highlight the currently-playing step column."""
        self._current_step = step
        self._redraw_headers()
        self._redraw_cells()

    def clear_highlight(self) -> None:
        """Remove playback highlight."""
        self._current_step = -1
        self._redraw_headers()
        self._redraw_cells()

    def get_active_notes_at_step(self, step: int) -> list[int]:
        """Return MIDI note numbers that are active at the given step."""
        n_rows = len(self.scale_notes)
        return [
            self.scale_notes[row]
            for row in range(n_rows)
            if self._grid.get((step, row), False)
        ]

    def get_step_length(self, step: int) -> int:
        """Return the length multiplier for a given step."""
        if step < len(self._step_lengths):
            return self._step_lengths[step]
        return 1

    def get_step_velocity(self, step: int) -> int:
        """Return velocity for a given step."""
        if step < len(self._step_velocities):
            return self._step_velocities[step]
        return 100

    def get_step_gate(self, step: int) -> int:
        """Return gate value (0..127) for a given step."""
        if step < len(self._step_gates):
            return self._step_gates[step]
        return 64

    def export_pattern(self) -> dict:
        """Export the current piano-roll pattern as a serializable dict."""
        notes_by_step: list[list[int]] = []
        n_rows = len(self.scale_notes)
        for col in range(self.num_steps):
            step_notes = [
                self.scale_notes[row]
                for row in range(n_rows)
                if self._grid.get((col, row), False)
            ]
            notes_by_step.append(step_notes)

        return {
            "version": 1,
            "num_steps": self.num_steps,
            "scale_notes": list(self.scale_notes),
            "note_names": list(self.note_names),
            "step_lengths": list(self._step_lengths[:self.num_steps]),
            "step_velocities": list(self._step_velocities[:self.num_steps]),
            "step_gates": list(self._step_gates[:self.num_steps]),
            "notes_by_step": notes_by_step,
        }

    def load_pattern(self, pattern: dict) -> None:
        """Load a pattern dict into the piano roll."""
        loaded_steps = int(pattern.get("num_steps", self.num_steps))
        self.num_steps = max(8, min(32, loaded_steps))

        if pattern.get("scale_notes") and pattern.get("note_names"):
            self.scale_notes = list(pattern.get("scale_notes", []))
            self.note_names = list(pattern.get("note_names", []))

        self._grid.clear()

        lengths = pattern.get("step_lengths", [])
        velocities = pattern.get("step_velocities", [])
        gates = pattern.get("step_gates", [])
        notes_by_step = pattern.get("notes_by_step", [])

        for col in range(self.num_steps):
            if col < len(lengths):
                value = int(lengths[col])
                self._step_lengths[col] = value if value in self.LEN_OPTS else 1
            else:
                self._step_lengths[col] = 1

            if col < len(velocities):
                self._step_velocities[col] = max(1, min(127, int(velocities[col])))
            else:
                self._step_velocities[col] = 100

            if col < len(gates):
                self._step_gates[col] = max(0, min(127, int(gates[col])))
            else:
                self._step_gates[col] = 64

            if col < len(notes_by_step):
                step_notes = notes_by_step[col]
                for note in step_notes:
                    if note in self.scale_notes:
                        row_idx = self.scale_notes.index(note)
                        self._grid[(col, row_idx)] = True

        self._current_step = -1
        self._draw()

    def grid(self, **kwargs) -> None:
        self.frame.grid(**kwargs)

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build_ui(self, parent: tk.Widget) -> None:
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        # Fixed top header canvas (step numbers)
        self._header_canvas = tk.Canvas(self.frame, bg=self.C_BG, height=self.HEADER_H, highlightthickness=0)
        self._header_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self._header_canvas.configure(xscrollcommand=self._on_header_xscroll)

        # Main note canvas + vertical scrollbar
        inner = tk.Frame(self.frame, bg=self.C_BG)
        inner.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        inner.columnconfigure(0, weight=1)
        inner.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(inner, bg=self.C_BG, highlightthickness=0)
        v_sb = ttk.Scrollbar(inner, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(
            xscrollcommand=self._on_main_xscroll,
            yscrollcommand=v_sb.set,
        )
        self._canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_sb.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Fixed bottom parameter canvas (horizontal only)
        self._param_canvas = tk.Canvas(self.frame, bg=self.C_BG, height=self._param_canvas_height(), highlightthickness=0)
        self._param_canvas.grid(row=2, column=0, sticky=(tk.W, tk.E))
        self._param_canvas.configure(xscrollcommand=self._on_param_xscroll)

        # Shared horizontal scrollbar for both canvases
        h_sb = ttk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self._on_hscroll)
        h_sb.grid(row=3, column=0, sticky=(tk.W, tk.E))
        self._h_scrollbar = h_sb

        self._canvas.bind("<Button-1>", self._on_left_click)
        self._param_canvas.bind("<Button-1>", self._on_param_button_press)
        self._param_canvas.bind("<B1-Motion>", self._on_param_drag_motion)
        self._param_canvas.bind("<ButtonRelease-1>", self._on_param_button_release)
        self._param_canvas.bind("<Double-Button-1>", self._on_param_double_click)
        self._header_canvas.bind("<Shift-MouseWheel>", self._on_header_shift_mouse_wheel)
        self._header_canvas.bind("<MouseWheel>", self._on_header_shift_mouse_wheel)
        self._canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self._canvas.bind("<Shift-MouseWheel>", self._on_shift_mouse_wheel)
        self._canvas.bind("<Button-4>", self._on_linux_scroll_up)
        self._canvas.bind("<Button-5>", self._on_linux_scroll_down)
        self._canvas.bind("<Shift-Button-4>", self._on_linux_shift_scroll_up)
        self._canvas.bind("<Shift-Button-5>", self._on_linux_shift_scroll_down)
        self._param_canvas.bind("<MouseWheel>", self._on_param_mouse_wheel)
        self._param_canvas.bind("<Shift-MouseWheel>", self._on_param_mouse_wheel)
        self._param_canvas.bind("<Button-4>", self._on_param_linux_scroll_left)
        self._param_canvas.bind("<Button-5>", self._on_param_linux_scroll_right)
        self._param_canvas.bind("<Shift-Button-4>", self._on_param_linux_scroll_left)
        self._param_canvas.bind("<Shift-Button-5>", self._on_param_linux_scroll_right)

        # Initial empty message
        self._canvas.create_text(
            10, 10, anchor=tk.NW,
            text="← Select a scale from the tree to begin",
            fill="#555577", font=("Arial", 9),
            tags="placeholder",
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _n_rows(self) -> int:
        return len(self.scale_notes)

    def _canvas_width(self) -> int:
        return self.LABEL_W + self.CELL_W * self.num_steps

    def _canvas_height(self) -> int:
        return self.CELL_H * self._n_rows()

    def _param_canvas_height(self) -> int:
        return self.LEN_ROW_H + self.VEL_ROW_H + self.GATE_ROW_H

    def _cell_x1(self, col: int) -> int:
        return self.LABEL_W + col * self.CELL_W

    def _row_y1(self, row: int) -> int:
        """row 0 = topmost = highest note"""
        return row * self.CELL_H

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        if not self.scale_notes:
            return

        self._canvas.delete("all")
        self._header_canvas.delete("all")
        self._param_canvas.delete("all")
        n_rows = self._n_rows()
        cw = self._canvas_width()
        ch = self._canvas_height()
        ph = self._param_canvas_height()
        self._canvas.config(scrollregion=(0, 0, cw, ch))
        self._header_canvas.config(scrollregion=(0, 0, cw, self.HEADER_H), height=self.HEADER_H)
        self._param_canvas.config(scrollregion=(0, 0, cw, ph), height=ph)

        # Background
        self._canvas.create_rectangle(0, 0, cw, ch, fill=self.C_BG, outline="")
        self._header_canvas.create_rectangle(0, 0, cw, self.HEADER_H, fill=self.C_BG, outline="")
        self._param_canvas.create_rectangle(0, 0, cw, ph, fill=self.C_BG, outline="")

        # Left label column background
        self._canvas.create_rectangle(0, 0, self.LABEL_W, ch, fill=self.C_LBL_BG, outline="")
        self._header_canvas.create_rectangle(0, 0, self.LABEL_W, self.HEADER_H, fill=self.C_LBL_BG, outline=self.C_GRID)
        self._header_canvas.create_rectangle(self.LABEL_W, 0, cw, self.HEADER_H, fill=self.C_HDR_BG, outline="")

        # Per-column: header number + cells + length row
        for col in range(self.num_steps):
            x1 = self._cell_x1(col)
            x2 = x1 + self.CELL_W

            # Step-number header
            hdr_bg = self.C_HDR_PLAY if col == self._current_step else self.C_HDR_BG
            self._header_canvas.create_rectangle(
                x1, 0, x2, self.HEADER_H,
                fill=hdr_bg, outline=self.C_GRID,
                tags=f"hdr_{col}",
            )
            self._header_canvas.create_text(
                x1 + self.CELL_W // 2, self.HEADER_H // 2,
                text=str(col + 1), fill=self.C_HDR_TEXT, font=("Arial", 7),
                tags=f"hdrtxt_{col}",
            )

            # Step-length row (in fixed parameter canvas)
            len_y1 = 0
            len_y2 = len_y1 + self.LEN_ROW_H
            lc = self.LEN_COLORS.get(self._step_lengths[col], self.C_LEN_1_16)
            self._param_canvas.create_rectangle(
                x1, len_y1, x2, len_y2,
                fill=lc, outline=self.C_GRID,
                tags=f"len_{col}",
            )
            self._param_canvas.create_text(
                x1 + self.CELL_W // 2, len_y1 + self.LEN_ROW_H // 2,
                text=self.LEN_LABELS.get(self._step_lengths[col], "1/16"),
                fill=self.C_LEN_TEXT, font=("Arial", 7),
                tags=f"lentxt_{col}",
            )

            # Step-velocity row (below length row)
            vel_y1 = len_y2
            vel_y2 = vel_y1 + self.VEL_ROW_H
            self._param_canvas.create_rectangle(
                x1, vel_y1, x2, vel_y2,
                fill=self.C_VEL_BG, outline=self.C_GRID,
                tags=f"vel_{col}",
            )
            self._param_canvas.create_text(
                x1 + self.CELL_W // 2, vel_y1 + self.VEL_ROW_H // 2,
                text=str(self._step_velocities[col]),
                fill=self.C_VEL_TEXT, font=("Arial", 7),
                tags=f"veltxt_{col}",
            )

            # Step-gate row (below velocity row)
            gate_y1 = vel_y2
            gate_y2 = gate_y1 + self.GATE_ROW_H
            self._param_canvas.create_rectangle(
                x1, gate_y1, x2, gate_y2,
                fill=self.C_GATE_BG, outline=self.C_GRID,
                tags=f"gate_{col}",
            )
            self._param_canvas.create_text(
                x1 + self.CELL_W // 2, gate_y1 + self.GATE_ROW_H // 2,
                text=str(self._step_gates[col]),
                fill=self.C_GATE_TEXT, font=("Arial", 7),
                tags=f"gatetxt_{col}",
            )

            # Cells (rows)
            for row in range(n_rows):
                note_idx = n_rows - 1 - row   # row 0 = highest note
                y1 = self._row_y1(row)
                y2 = y1 + self.CELL_H
                active  = self._grid.get((col, note_idx), False)
                playing = col == self._current_step
                color = self._cell_color(active, playing, note_idx)
                self._canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color, outline=self.C_GRID,
                    tags=f"cell_{col}_{note_idx}",
                )

        # Left labels for control rows (fixed parameter canvas)
        self._param_canvas.create_rectangle(0, 0, self.LABEL_W, self.LEN_ROW_H, fill=self.C_LBL_BG, outline=self.C_GRID)
        self._param_canvas.create_text(self.LABEL_W // 2, self.LEN_ROW_H // 2, text="LEN", fill=self.C_LBL_TEXT, font=("Arial", 8))
        self._param_canvas.create_rectangle(0, self.LEN_ROW_H, self.LABEL_W, self.LEN_ROW_H + self.VEL_ROW_H, fill=self.C_LBL_BG, outline=self.C_GRID)
        self._param_canvas.create_text(self.LABEL_W // 2, self.LEN_ROW_H + self.VEL_ROW_H // 2, text="VEL", fill=self.C_LBL_TEXT, font=("Arial", 8))
        self._param_canvas.create_rectangle(0, self.LEN_ROW_H + self.VEL_ROW_H, self.LABEL_W, self.LEN_ROW_H + self.VEL_ROW_H + self.GATE_ROW_H, fill=self.C_LBL_BG, outline=self.C_GRID)
        self._param_canvas.create_text(self.LABEL_W // 2, self.LEN_ROW_H + self.VEL_ROW_H + self.GATE_ROW_H // 2, text="GATE", fill=self.C_LBL_TEXT, font=("Arial", 8))

        # Per-row: note label
        for row in range(n_rows):
            note_idx = n_rows - 1 - row
            y1 = self._row_y1(row)
            y2 = y1 + self.CELL_H
            name = self.note_names[note_idx] if note_idx < len(self.note_names) else ""
            self._canvas.create_rectangle(
                0, y1, self.LABEL_W, y2,
                fill=self.C_LBL_BG, outline=self.C_GRID,
            )
            self._canvas.create_text(
                self.LABEL_W // 2, y1 + self.CELL_H // 2,
                text=name, fill=self.C_LBL_TEXT, font=("Arial", 8),
            )

    def _is_black_key(self, midi_note: int) -> bool:
        return (midi_note % 12) in self.BLACK_KEY_PITCH_CLASSES

    def _cell_color(self, active: bool, playing: bool, note_idx: int) -> str:
        if active and playing:
            return self.C_CELL_PLAY
        if active:
            return self.C_CELL_ON
        if 0 <= note_idx < len(self.scale_notes) and self._is_black_key(self.scale_notes[note_idx]):
            return self.C_CELL_OFF_BLACK
        return self.C_CELL_OFF_WHITE

    def _redraw_headers(self) -> None:
        if not self.scale_notes:
            return
        for col in range(self.num_steps):
            hdr_bg = self.C_HDR_PLAY if col == self._current_step else self.C_HDR_BG
            self._header_canvas.itemconfig(f"hdr_{col}", fill=hdr_bg)

    def _redraw_cells(self) -> None:
        if not self.scale_notes:
            return
        n_rows = self._n_rows()
        for col in range(self.num_steps):
            for row in range(n_rows):
                note_idx = n_rows - 1 - row
                active  = self._grid.get((col, note_idx), False)
                playing = col == self._current_step
                tag = f"cell_{col}_{note_idx}"
                if self._canvas.find_withtag(tag):
                    self._canvas.itemconfig(tag, fill=self._cell_color(active, playing, note_idx))

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _canvas_xy(self, event) -> tuple[float, float]:
        return self._canvas.canvasx(event.x), self._canvas.canvasy(event.y)

    def _hit_cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        """Return (col, note_idx) if the point is inside a cell, else None."""
        n_rows = self._n_rows()
        cells_y1 = 0
        if x < self.LABEL_W or y < cells_y1:
            return None
        cells_y2 = cells_y1 + n_rows * self.CELL_H
        if y >= cells_y2:
            return None
        col = int((x - self.LABEL_W) / self.CELL_W)
        row = int((y - cells_y1) / self.CELL_H)
        if not (0 <= col < self.num_steps and 0 <= row < n_rows):
            return None
        note_idx = n_rows - 1 - row
        return (col, note_idx)

    def _hit_length_row(self, x: float, y: float) -> Optional[int]:
        """Return col if the point is in the length row, else None."""
        len_y = 0
        if not (len_y <= y <= len_y + self.LEN_ROW_H and x >= self.LABEL_W):
            return None
        col = int((x - self.LABEL_W) / self.CELL_W)
        return col if 0 <= col < self.num_steps else None

    def _hit_velocity_row(self, x: float, y: float) -> Optional[int]:
        """Return col if the point is in the velocity row, else None."""
        vel_y = self.LEN_ROW_H
        if not (vel_y <= y <= vel_y + self.VEL_ROW_H and x >= self.LABEL_W):
            return None
        col = int((x - self.LABEL_W) / self.CELL_W)
        return col if 0 <= col < self.num_steps else None

    def _hit_gate_row(self, x: float, y: float) -> Optional[int]:
        """Return col if the point is in the gate row, else None."""
        gate_y = self.LEN_ROW_H + self.VEL_ROW_H
        if not (gate_y <= y <= gate_y + self.GATE_ROW_H and x >= self.LABEL_W):
            return None
        col = int((x - self.LABEL_W) / self.CELL_W)
        return col if 0 <= col < self.num_steps else None

    def _on_left_click(self, event) -> None:
        if not self.scale_notes:
            return
        x, y = self._canvas_xy(event)

        cell = self._hit_cell(x, y)
        if cell:
            col, note_idx = cell
            new_state = not self._grid.get((col, note_idx), False)
            self._grid[(col, note_idx)] = new_state
            self._redraw_cells()
            return

        col = self._hit_length_row(x, y)
        if col is not None:
            self._cycle_step_length(col)
            return

        col = self._hit_velocity_row(x, y)
        if col is not None:
            self._cycle_step_velocity(col)
            return

        col = self._hit_gate_row(x, y)
        if col is not None:
            self._cycle_step_gate(col)

    def _on_param_button_press(self, event) -> None:
        if not self.scale_notes:
            return
        x = self._param_canvas.canvasx(event.x)
        y = self._param_canvas.canvasy(event.y)

        # Any new click closes inline editor first
        self._close_value_editor(commit=True)

        col = self._hit_length_row(x, y)
        if col is not None:
            self._cycle_step_length(col)
            self._clear_param_drag_state()
            return

        col = self._hit_velocity_row(x, y)
        if col is not None:
            self._start_param_drag(kind="vel", col=col, y=y)
            return

        col = self._hit_gate_row(x, y)
        if col is not None:
            self._start_param_drag(kind="gate", col=col, y=y)

    def _on_param_drag_motion(self, event) -> None:
        if self._param_drag_kind is None or self._param_drag_col is None:
            return

        y = self._param_canvas.canvasy(event.y)
        delta = int((self._param_drag_start_y - y) / 2)  # 2 px per value step
        value = max(0, min(127, self._param_drag_start_value + delta))

        if self._param_drag_kind == "vel":
            self._set_step_velocity(self._param_drag_col, value)
        elif self._param_drag_kind == "gate":
            self._set_step_gate(self._param_drag_col, value)

    def _on_param_button_release(self, _event=None) -> None:
        self._clear_param_drag_state()

    def _on_param_double_click(self, event) -> None:
        if not self.scale_notes:
            return

        x = self._param_canvas.canvasx(event.x)
        y = self._param_canvas.canvasy(event.y)

        col = self._hit_velocity_row(x, y)
        if col is not None:
            x1, y1, x2, y2 = self._param_cell_rect("vel", col)
            self._open_value_editor("vel", col, x1, y1, x2, y2, self._step_velocities[col])
            return

        col = self._hit_gate_row(x, y)
        if col is not None:
            x1, y1, x2, y2 = self._param_cell_rect("gate", col)
            self._open_value_editor("gate", col, x1, y1, x2, y2, self._step_gates[col])

    def _cycle_step_length(self, col: int) -> None:
        cur = self._step_lengths[col]
        idx = self.LEN_OPTS.index(cur) if cur in self.LEN_OPTS else 0
        self._step_lengths[col] = self.LEN_OPTS[(idx + 1) % len(self.LEN_OPTS)]
        lc = self.LEN_COLORS.get(self._step_lengths[col], self.C_LEN_1_16)
        self._param_canvas.itemconfig(f"len_{col}", fill=lc)
        self._param_canvas.itemconfig(
            f"lentxt_{col}",
            text=self.LEN_LABELS.get(self._step_lengths[col], "1/16"),
        )

    def _cycle_step_velocity(self, col: int) -> None:
        cur = self._step_velocities[col]
        idx = self.VEL_OPTS.index(cur) if cur in self.VEL_OPTS else 0
        self._set_step_velocity(col, self.VEL_OPTS[(idx + 1) % len(self.VEL_OPTS)])

    def _cycle_step_gate(self, col: int) -> None:
        cur = self._step_gates[col]
        idx = self.GATE_OPTS.index(cur) if cur in self.GATE_OPTS else 0
        self._set_step_gate(col, self.GATE_OPTS[(idx + 1) % len(self.GATE_OPTS)])

    def _redraw_velocity_row(self) -> None:
        if not self.scale_notes:
            return
        for col in range(self.num_steps):
            tag = f"veltxt_{col}"
            if self._param_canvas.find_withtag(tag):
                self._param_canvas.itemconfig(tag, text=str(self._step_velocities[col]))

    def _set_step_velocity(self, col: int, value: int) -> None:
        self._step_velocities[col] = max(0, min(127, int(value)))
        self._param_canvas.itemconfig(f"veltxt_{col}", text=str(self._step_velocities[col]))

    def _set_step_gate(self, col: int, value: int) -> None:
        self._step_gates[col] = max(0, min(127, int(value)))
        self._param_canvas.itemconfig(f"gatetxt_{col}", text=str(self._step_gates[col]))

    def _start_param_drag(self, kind: str, col: int, y: float) -> None:
        self._param_drag_kind = kind
        self._param_drag_col = col
        self._param_drag_start_y = y
        if kind == "vel":
            self._param_drag_start_value = self._step_velocities[col]
        else:
            self._param_drag_start_value = self._step_gates[col]

    def _clear_param_drag_state(self) -> None:
        self._param_drag_kind = None
        self._param_drag_col = None

    def _param_cell_rect(self, kind: str, col: int) -> tuple[int, int, int, int]:
        x1 = self._cell_x1(col)
        x2 = x1 + self.CELL_W
        if kind == "vel":
            y1 = self.LEN_ROW_H
            y2 = y1 + self.VEL_ROW_H
        else:
            y1 = self.LEN_ROW_H + self.VEL_ROW_H
            y2 = y1 + self.GATE_ROW_H
        return x1, y1, x2, y2

    def _open_value_editor(self, kind: str, col: int, x1: int, y1: int, x2: int, y2: int, initial_value: int) -> None:
        self._close_value_editor(commit=True)

        self._value_editor_kind = kind
        self._value_editor_col = col
        value_var = tk.StringVar(value=str(initial_value))
        entry = ttk.Entry(self._param_canvas, textvariable=value_var, width=4)
        self._value_editor = entry
        self._value_editor_window = self._param_canvas.create_window(
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            window=entry,
            width=max(28, self.CELL_W - 6),
            height=max(16, (y2 - y1) - 4),
        )
        entry.focus_set()
        entry.selection_range(0, tk.END)
        entry.bind("<Return>", lambda _event: self._close_value_editor(commit=True))
        entry.bind("<Escape>", lambda _event: self._close_value_editor(commit=False))
        entry.bind("<FocusOut>", lambda _event: self._close_value_editor(commit=True))

    def _close_value_editor(self, commit: bool) -> None:
        if self._value_editor is None:
            return

        if commit and self._value_editor_kind is not None and self._value_editor_col is not None:
            try:
                value = int(self._value_editor.get())
                value = max(0, min(127, value))
                if self._value_editor_kind == "vel":
                    self._set_step_velocity(self._value_editor_col, value)
                elif self._value_editor_kind == "gate":
                    self._set_step_gate(self._value_editor_col, value)
            except Exception:
                pass

        if self._value_editor_window is not None:
            self._param_canvas.delete(self._value_editor_window)
        try:
            self._value_editor.destroy()
        except Exception:
            pass

        self._value_editor = None
        self._value_editor_window = None
        self._value_editor_kind = None
        self._value_editor_col = None

    # ------------------------------------------------------------------
    # Horizontal scroll sync
    # ------------------------------------------------------------------

    def _on_hscroll(self, *args) -> None:
        self._canvas.xview(*args)
        self._header_canvas.xview(*args)
        self._param_canvas.xview(*args)

    def _on_main_xscroll(self, first, last) -> None:
        self._h_scrollbar.set(first, last)
        if self._syncing_x:
            return
        self._syncing_x = True
        try:
            self._header_canvas.xview_moveto(first)
            self._param_canvas.xview_moveto(first)
        finally:
            self._syncing_x = False

    def _on_param_xscroll(self, first, last) -> None:
        self._h_scrollbar.set(first, last)
        if self._syncing_x:
            return
        self._syncing_x = True
        try:
            self._canvas.xview_moveto(first)
            self._header_canvas.xview_moveto(first)
        finally:
            self._syncing_x = False

    def _on_header_xscroll(self, first, last) -> None:
        self._h_scrollbar.set(first, last)
        if self._syncing_x:
            return
        self._syncing_x = True
        try:
            self._canvas.xview_moveto(first)
            self._param_canvas.xview_moveto(first)
        finally:
            self._syncing_x = False

    # ------------------------------------------------------------------
    # Scroll handlers
    # ------------------------------------------------------------------

    def _on_mouse_wheel(self, event) -> None:
        """Vertical scroll with mouse wheel."""
        step = -1 if event.delta > 0 else 1
        self._canvas.yview_scroll(step, "units")

    def _on_shift_mouse_wheel(self, event) -> None:
        """Horizontal scroll with Shift + mouse wheel."""
        step = -1 if event.delta > 0 else 1
        self._canvas.xview_scroll(step, "units")
        self._header_canvas.xview_scroll(step, "units")
        self._param_canvas.xview_scroll(step, "units")

    def _on_linux_scroll_up(self, _event=None) -> None:
        self._canvas.yview_scroll(-1, "units")

    def _on_linux_scroll_down(self, _event=None) -> None:
        self._canvas.yview_scroll(1, "units")

    def _on_linux_shift_scroll_up(self, _event=None) -> None:
        self._canvas.xview_scroll(-1, "units")
        self._header_canvas.xview_scroll(-1, "units")
        self._param_canvas.xview_scroll(-1, "units")

    def _on_linux_shift_scroll_down(self, _event=None) -> None:
        self._canvas.xview_scroll(1, "units")
        self._header_canvas.xview_scroll(1, "units")
        self._param_canvas.xview_scroll(1, "units")

    def _on_header_shift_mouse_wheel(self, event) -> None:
        step = -1 if event.delta > 0 else 1
        self._header_canvas.xview_scroll(step, "units")
        self._canvas.xview_scroll(step, "units")
        self._param_canvas.xview_scroll(step, "units")

    def _on_param_mouse_wheel(self, event) -> None:
        """Horizontal scroll in parameter grid with mouse wheel."""
        step = -1 if event.delta > 0 else 1
        self._param_canvas.xview_scroll(step, "units")
        self._header_canvas.xview_scroll(step, "units")
        self._canvas.xview_scroll(step, "units")

    def _on_param_linux_scroll_left(self, _event=None) -> None:
        self._param_canvas.xview_scroll(-1, "units")
        self._header_canvas.xview_scroll(-1, "units")
        self._canvas.xview_scroll(-1, "units")

    def _on_param_linux_scroll_right(self, _event=None) -> None:
        self._param_canvas.xview_scroll(1, "units")
        self._header_canvas.xview_scroll(1, "units")
        self._canvas.xview_scroll(1, "units")
