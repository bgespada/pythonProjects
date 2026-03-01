import tkinter as tk
from tkinter import ttk
from typing import Optional

class StatusBar(ttk.Frame):
    """
    A reusable status bar widget for displaying multiple pieces of information at the bottom of the UI.
    """
    def __init__(self, parent, initial_text="Ready", initial_color="blue"):
        super().__init__(parent)
        # Main info label (current data)
        self.info_label = ttk.Label(self, text=initial_text, foreground=initial_color)
        self.info_label.pack(side=tk.LEFT, padx=(0, 10))
        # MIDI channel label
        self.channel_label = ttk.Label(self, text="", foreground="gray")
        self.channel_label.pack(side=tk.LEFT, padx=(0, 10))

    def set_info(self, message: str, color: str = "blue"):
        self.info_label.config(text=message, foreground=color)

    def set_channel(self, channel: Optional[int] = None, color: str = "gray"):
        if channel is not None:
            self.channel_label.config(text=f"MIDI Channel: {channel}", foreground=color)
        else:
            self.channel_label.config(text="", foreground=color)

    def get_info(self) -> str:
        return self.info_label.cget("text")

    def get_channel(self) -> str:
        return self.channel_label.cget("text")
