import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

class TransportUi(ttk.Frame):
    """
    A reusable UI class for MIDI transport controls (Start, Stop, Continue, etc.).
    Accepts callbacks for each transport action.
    """
    def __init__(self, parent, 
                 on_start: Optional[Callable[[], None]] = None,
                 on_stop: Optional[Callable[[], None]] = None,
                 on_continue: Optional[Callable[[], None]] = None,
                 on_clock: Optional[Callable[[], None]] = None,
                 on_song_position: Optional[Callable[[int], None]] = None,
                 on_song_select: Optional[Callable[[int], None]] = None):
        super().__init__(parent)
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_continue = on_continue
        self.on_clock = on_clock
        self.on_song_position = on_song_position
        self.on_song_select = on_song_select
        self._build_ui()

    def _build_ui(self):
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(btn_frame, text="Start", width=8, command=self._handle_start).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Stop", width=8, command=self._handle_stop).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Continue", width=8, command=self._handle_continue).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clock", width=8, command=self._handle_clock).pack(side=tk.LEFT, padx=2)

        # Song Position
        pos_frame = ttk.Frame(self)
        pos_frame.pack(fill=tk.X, padx=4, pady=(2, 0))
        ttk.Label(pos_frame, text="Song Position:").pack(side=tk.LEFT)
        self.song_position_var = tk.IntVar(value=0)
        pos_entry = ttk.Entry(pos_frame, textvariable=self.song_position_var, width=6)
        pos_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(pos_frame, text="Send", width=6, command=self._handle_song_position).pack(side=tk.LEFT, padx=2)

        # Song Select
        sel_frame = ttk.Frame(self)
        sel_frame.pack(fill=tk.X, padx=4, pady=(2, 4))
        ttk.Label(sel_frame, text="Song Select:").pack(side=tk.LEFT)
        self.song_select_var = tk.IntVar(value=0)
        sel_entry = ttk.Entry(sel_frame, textvariable=self.song_select_var, width=6)
        sel_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(sel_frame, text="Send", width=6, command=self._handle_song_select).pack(side=tk.LEFT, padx=2)

    def _handle_start(self):
        if self.on_start:
            self.on_start()

    def _handle_stop(self):
        if self.on_stop:
            self.on_stop()

    def _handle_continue(self):
        if self.on_continue:
            self.on_continue()

    def _handle_clock(self):
        if self.on_clock:
            self.on_clock()

    def _handle_song_position(self):
        if self.on_song_position:
            self.on_song_position(self.song_position_var.get())

    def _handle_song_select(self):
        if self.on_song_select:
            self.on_song_select(self.song_select_var.get())
