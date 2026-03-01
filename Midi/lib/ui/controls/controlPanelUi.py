import tkinter as tk
from tkinter import ttk
from .parameterTreeUi import ParameterTreeUi
from .parameterControlsUi import ParameterControlsUi


class ControlPanelUi:
    """
    Main control panel that combines:
      - Left: ParameterTreeUi  — category navigation menu
      - Right: ParameterControlsUi — sliders for the selected category

    Usage:
        panel = ControlPanelUi(parent_frame)
        panel.pack(fill=tk.BOTH, expand=True)

        # After device connection:
        panel.set_midi_messages(midi_messages_instance)
    """

    def __init__(self, parent: tk.Widget):
        """
        Args:
            parent: Parent tkinter widget (e.g. the Controls LabelFrame).
        """
        # Root frame fills the parent
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=0)   # tree: fixed width
        self.frame.columnconfigure(1, weight=0)   # separator: fixed width
        self.frame.columnconfigure(2, weight=1)   # controls: stretches
        self.frame.rowconfigure(0, weight=1)

        # Vertical separator between tree and controls panel
        separator = ttk.Separator(self.frame, orient=tk.VERTICAL)
        separator.grid(row=0, column=1, sticky=(tk.N, tk.S), padx=(0, 4))

        # Left: category tree
        self.tree_ui = ParameterTreeUi(
            parent=self.frame,
            on_select=self._on_category_selected,
        )
        self.tree_ui.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 0))

        # Right: parameter sliders panel
        self.controls_ui = ParameterControlsUi(parent=self.frame)
        self.controls_ui.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.W, tk.E))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_midi_messages(self, midi_messages) -> None:
        """
        Inject the MidiMessages instance (call after device is connected).

        Args:
            midi_messages: MidiMessages instance or None.
        """
        self.controls_ui.set_midi_messages(midi_messages)

    def set_channel(self, channel: int) -> None:
        """
        Set the MIDI channel (0-based) for CC messages.

        Args:
            channel: MIDI channel number (0-15).
        """
        self.controls_ui.set_channel(channel)

    def grid(self, **kwargs) -> None:
        self.frame.grid(**kwargs)

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _on_category_selected(self, category: str) -> None:
        """Called when the user selects a category in the tree."""
        self.controls_ui.load_category(category)
