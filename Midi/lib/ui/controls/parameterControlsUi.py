import tkinter as tk
from tkinter import ttk
from midi.ccParameters import PARAMETER_CATEGORIES


class ParameterControlsUi:
    """
    Right-side panel that displays sliders for every parameter in the
    currently selected category.  Each slider sends a MIDI CC message
    on change when a MidiMessages instance is provided.
    """

    def __init__(self, parent: tk.Widget):
        """
        Args:
            parent: Parent tkinter widget.
        """
        self.midi_messages = None          # injected from outside after device connect
        self._channel: int = 0             # MIDI channel (0-based)

        # Outer frame
        self.frame = ttk.Frame(parent, padding="6")

        # Placeholder shown when no category is selected
        self._placeholder = ttk.Label(
            self.frame,
            text="Select a category from the menu",
            foreground="gray",
        )
        self._placeholder.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=8, pady=8)
        self.frame.columnconfigure(0, weight=0)   # label column: fixed
        self.frame.columnconfigure(1, weight=1)   # slider column: stretches
        self.frame.columnconfigure(2, weight=0)   # value column: fixed

        # Keep track of dynamic slider widgets so they can be cleared
        self._slider_widgets: list[tk.Widget] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_category(self, category: str) -> None:
        """
        Replace the current controls with sliders for category.

        Args:
            category: Key in PARAMETER_CATEGORIES.
        """
        self._clear()
        parameters = PARAMETER_CATEGORIES.get(category, [])
        if not parameters:
            self._show_placeholder(f"No parameters defined for '{category}'")
            return

        for row_idx, param in enumerate(parameters):
            self._add_slider(row_idx, param)

    def set_midi_messages(self, midi_messages) -> None:
        """Inject the MidiMessages instance used to send CC messages."""
        self.midi_messages = midi_messages

    def set_channel(self, channel: int) -> None:
        """Set the MIDI channel (0-based) used when sending CC messages."""
        self._channel = max(0, min(15, int(channel)))

    def grid(self, **kwargs) -> None:
        self.frame.grid(**kwargs)

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clear(self) -> None:
        """Remove all dynamic slider widgets."""
        self._placeholder.grid_remove()
        for widget in self._slider_widgets:
            widget.destroy()
        self._slider_widgets.clear()

    def _show_placeholder(self, text: str = "Select a category from the menu") -> None:
        self._placeholder.config(text=text)
        self._placeholder.grid()

    def _add_slider(self, row: int, param: dict) -> None:
        """
        Add a labelled slider row for a single parameter.

        Args:
            row:   Grid row index.
            param: Parameter dict with keys name, cc, min, max, default.
        """
        name    = param["name"]
        cc_num  = param["cc"]
        p_min   = param["min"]
        p_max   = param["max"]
        default = param["default"]

        # Parameter label (name + CC info)
        label = ttk.Label(
            self.frame,
            text=f"{name}  (CC {cc_num})",
            font=("Arial", 9),
            width=24,
            anchor=tk.W,
        )
        label.grid(row=row, column=0, sticky=tk.W, padx=(4, 8), pady=3)

        # Value variable
        var = tk.IntVar(value=default)

        # Value display label
        value_label = ttk.Label(self.frame, text=str(default), width=4)
        value_label.grid(row=row, column=2, sticky=tk.W, padx=(4, 0), pady=3)

        # Slider
        slider = ttk.Scale(
            self.frame,
            from_=p_min,
            to=p_max,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda val, v=var, vl=value_label, cc=cc_num: self._on_slider_change(val, v, vl, cc),
        )
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(0, 4), pady=3)

        self._slider_widgets.extend([label, slider, value_label])

    def _on_slider_change(self, val: str, var: tk.IntVar, value_label: ttk.Label, cc: int) -> None:
        """
        Called when a slider value changes.  Updates the value label and
        sends a MIDI CC message if a device is connected.
        """
        int_val = int(float(val))
        var.set(int_val)
        value_label.config(text=str(int_val))

        if self.midi_messages:
            try:
                self.midi_messages.send_control_change(
                    control=cc,
                    value=int_val,
                    channel=self._channel,
                )
            except Exception as e:
                print(f"Failed to send CC {cc}: {e}")
