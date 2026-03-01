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
    LEN_ROW_H   = 20   # height of step-length row at the bottom

    # Colour palette
    C_BG         = "#1a1a2e"
    C_GRID       = "#2a2a44"
    C_CELL_OFF   = "#16213e"
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

    # Step-length options
    LEN_OPTS   = [1, 2, 4]
    LEN_LABELS = {1: "1/16", 2: "1/8", 4: "1/4"}
    LEN_COLORS = {1: "#1e2e1e", 2: "#2e2e10", 4: "#2e1010"}

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
        self._current_step: int = -1

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
        self.frame.rowconfigure(0, weight=1)

        # Canvas + scrollbars
        inner = tk.Frame(self.frame, bg=self.C_BG)
        inner.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        inner.columnconfigure(0, weight=1)
        inner.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(inner, bg=self.C_BG, highlightthickness=0)
        h_sb = ttk.Scrollbar(inner, orient=tk.HORIZONTAL, command=self._canvas.xview)
        v_sb = ttk.Scrollbar(inner, orient=tk.VERTICAL,   command=self._canvas.yview)
        self._canvas.configure(
            xscrollcommand=h_sb.set,
            yscrollcommand=v_sb.set,
        )
        self._canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        h_sb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        v_sb.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self._canvas.bind("<Button-1>", self._on_left_click)

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
        return self.HEADER_H + self.CELL_H * self._n_rows() + self.LEN_ROW_H

    def _cell_x1(self, col: int) -> int:
        return self.LABEL_W + col * self.CELL_W

    def _row_y1(self, row: int) -> int:
        """row 0 = topmost = highest note"""
        return self.HEADER_H + row * self.CELL_H

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        if not self.scale_notes:
            return

        self._canvas.delete("all")
        n_rows = self._n_rows()
        cw = self._canvas_width()
        ch = self._canvas_height()
        self._canvas.config(scrollregion=(0, 0, cw, ch))

        # Background
        self._canvas.create_rectangle(0, 0, cw, ch, fill=self.C_BG, outline="")

        # Left label column background
        self._canvas.create_rectangle(0, 0, self.LABEL_W, ch, fill=self.C_LBL_BG, outline="")

        # Header row background
        self._canvas.create_rectangle(self.LABEL_W, 0, cw, self.HEADER_H, fill=self.C_HDR_BG, outline="")

        # Per-column: header number + cells + length row
        for col in range(self.num_steps):
            x1 = self._cell_x1(col)
            x2 = x1 + self.CELL_W

            # Step-number header
            hdr_bg = self.C_HDR_PLAY if col == self._current_step else self.C_HDR_BG
            self._canvas.create_rectangle(
                x1, 0, x2, self.HEADER_H,
                fill=hdr_bg, outline=self.C_GRID,
                tags=f"hdr_{col}",
            )
            self._canvas.create_text(
                x1 + self.CELL_W // 2, self.HEADER_H // 2,
                text=str(col + 1), fill=self.C_HDR_TEXT, font=("Arial", 7),
                tags=f"hdrtxt_{col}",
            )

            # Cells (rows)
            for row in range(n_rows):
                note_idx = n_rows - 1 - row   # row 0 = highest note
                y1 = self._row_y1(row)
                y2 = y1 + self.CELL_H
                active  = self._grid.get((col, note_idx), False)
                playing = col == self._current_step
                color = self._cell_color(active, playing)
                self._canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color, outline=self.C_GRID,
                    tags=f"cell_{col}_{note_idx}",
                )

            # Step-length row at bottom
            len_y1 = self.HEADER_H + n_rows * self.CELL_H
            len_y2 = len_y1 + self.LEN_ROW_H
            lc = self.LEN_COLORS.get(self._step_lengths[col], self.C_LEN_1_16)
            self._canvas.create_rectangle(
                x1, len_y1, x2, len_y2,
                fill=lc, outline=self.C_GRID,
                tags=f"len_{col}",
            )
            self._canvas.create_text(
                x1 + self.CELL_W // 2, len_y1 + self.LEN_ROW_H // 2,
                text=self.LEN_LABELS.get(self._step_lengths[col], "1/16"),
                fill=self.C_LEN_TEXT, font=("Arial", 7),
                tags=f"lentxt_{col}",
            )

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

    def _cell_color(self, active: bool, playing: bool) -> str:
        if active and playing:
            return self.C_CELL_PLAY
        if active:
            return self.C_CELL_ON
        return self.C_CELL_OFF

    def _redraw_headers(self) -> None:
        if not self.scale_notes:
            return
        for col in range(self.num_steps):
            hdr_bg = self.C_HDR_PLAY if col == self._current_step else self.C_HDR_BG
            self._canvas.itemconfig(f"hdr_{col}", fill=hdr_bg)

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
                    self._canvas.itemconfig(tag, fill=self._cell_color(active, playing))

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _canvas_xy(self, event) -> tuple[float, float]:
        return self._canvas.canvasx(event.x), self._canvas.canvasy(event.y)

    def _hit_cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        """Return (col, note_idx) if the point is inside a cell, else None."""
        n_rows = self._n_rows()
        if x < self.LABEL_W or y < self.HEADER_H:
            return None
        length_y = self.HEADER_H + n_rows * self.CELL_H
        if y >= length_y:
            return None
        col = int((x - self.LABEL_W) / self.CELL_W)
        row = int((y - self.HEADER_H) / self.CELL_H)
        if not (0 <= col < self.num_steps and 0 <= row < n_rows):
            return None
        note_idx = n_rows - 1 - row
        return (col, note_idx)

    def _hit_length_row(self, x: float, y: float) -> Optional[int]:
        """Return col if the point is in the length row, else None."""
        n_rows = self._n_rows()
        len_y = self.HEADER_H + n_rows * self.CELL_H
        if not (len_y <= y <= len_y + self.LEN_ROW_H and x >= self.LABEL_W):
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
            self._grid[(col, note_idx)] = not self._grid.get((col, note_idx), False)
            self._redraw_cells()
            return

        col = self._hit_length_row(x, y)
        if col is not None:
            self._cycle_step_length(col)

    def _cycle_step_length(self, col: int) -> None:
        cur = self._step_lengths[col]
        idx = self.LEN_OPTS.index(cur) if cur in self.LEN_OPTS else 0
        self._step_lengths[col] = self.LEN_OPTS[(idx + 1) % len(self.LEN_OPTS)]
        lc = self.LEN_COLORS.get(self._step_lengths[col], self.C_LEN_1_16)
        self._canvas.itemconfig(f"len_{col}", fill=lc)
        self._canvas.itemconfig(
            f"lentxt_{col}",
            text=self.LEN_LABELS.get(self._step_lengths[col], "1/16"),
        )
