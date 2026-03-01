import tkinter as tk
from tkinter import ttk
from midi.ccParameters import (
    PARAMETER_CATEGORIES,
    SOUND_ENGINES,
    get_sound_engine_parameters,
)


class ParameterControlsUi:
    """
    Right-side panel that displays controls for the currently selected
    category. Controls send MIDI CC messages when a MidiMessages
    instance is provided.
    """

    BULK_CC_DELAY_MS = 12

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

        # Keep track of dynamic widgets so they can be cleared
        self._control_widgets: list[tk.Widget] = []
        self._engine_combo: ttk.Combobox | None = None
        self._engine_name_var: tk.StringVar | None = None
        self._engine_value_label: ttk.Label | None = None
        self._current_category: str | None = None
        self._current_parameters: list[dict] = []
        self._active_state_key: str | None = None
        self._apply_defaults_button: ttk.Button | None = None
        self._selected_engine_name: str = SOUND_ENGINES[0]
        self._pending_cc_jobs: list[str] = []
        self._control_state: dict[str, dict[int, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_category(self, category: str) -> None:
        """
        Replace the current controls with widgets for category.

        Args:
            category: Key in PARAMETER_CATEGORIES.
        """
        self._clear()
        self._current_category = category
        if category == "Sound Engine":
            self._load_sound_engine_controls()
            return

        parameters = PARAMETER_CATEGORIES.get(category, [])
        self._current_parameters = list(parameters)
        self._active_state_key = self._state_key_for_category(category)
        if not parameters:
            self._show_placeholder(f"No parameters defined for '{category}'")
            return

        self._add_apply_defaults_button(row=0)

        for row_idx, param in enumerate(parameters, start=1):
            self._add_parameter_control(row_idx, param, self._active_state_key)

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
        """Remove all dynamic parameter widgets."""
        self._cancel_pending_cc_sequence()
        self._placeholder.grid_remove()
        for widget in self._control_widgets:
            widget.destroy()
        self._control_widgets.clear()
        self._engine_combo = None
        self._engine_name_var = None
        self._engine_value_label = None
        self._current_parameters = []
        self._active_state_key = None
        self._apply_defaults_button = None

    def _show_placeholder(self, text: str = "Select a category from the menu") -> None:
        self._placeholder.config(text=text)
        self._placeholder.grid()

    def _add_parameter_control(self, row: int, param: dict, state_key: str | None) -> None:
        """
        Add a labelled control row for a single parameter.

        Args:
            row:   Grid row index.
            param: Parameter dict with keys name, cc, min, max, default.
        """
        widget_type = param.get("widget", "slider")
        if widget_type == "boolean":
            self._add_boolean(row, param, state_key)
            return
        self._add_slider(row, param, state_key)

    def _add_slider(self, row: int, param: dict, state_key: str | None) -> None:
        name    = param["name"]
        cc_num  = param["cc"]
        p_min   = param["min"]
        p_max   = param["max"]
        value = self._resolve_param_value(param, state_key=state_key, use_defaults=False)

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
        var = tk.IntVar(value=value)

        # Value display label
        value_label = ttk.Label(self.frame, text=str(value), width=4)
        value_label.grid(row=row, column=2, sticky=tk.W, padx=(4, 0), pady=3)

        # Slider
        slider = ttk.Scale(
            self.frame,
            from_=p_min,
            to=p_max,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda val, v=var, vl=value_label, cc=cc_num, sk=state_key: self._on_slider_change(val, v, vl, cc, sk),
        )
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(0, 4), pady=3)

        self._control_widgets.extend([label, slider, value_label])

    def _add_boolean(self, row: int, param: dict, state_key: str | None) -> None:
        name = param["name"]
        cc_num = param["cc"]
        off_value = int(param.get("off_value", 0))
        on_value = int(param.get("on_value", 127))
        value = self._resolve_param_value(param, state_key=state_key, use_defaults=False)

        label = ttk.Label(
            self.frame,
            text=f"{name}  (CC {cc_num})",
            font=("Arial", 9),
            width=24,
            anchor=tk.W,
        )
        label.grid(row=row, column=0, sticky=tk.W, padx=(4, 8), pady=3)

        value_label = ttk.Label(self.frame, text=str(value), width=4)
        value_label.grid(row=row, column=2, sticky=tk.W, padx=(4, 0), pady=3)

        bool_var = tk.BooleanVar(value=(value == on_value))
        toggle = ttk.Checkbutton(
            self.frame,
            text="On",
            variable=bool_var,
            command=lambda v=bool_var, vl=value_label, cc=cc_num, off=off_value, on=on_value, sk=state_key: self._on_boolean_change(v, vl, cc, off, on, sk),
        )
        toggle.grid(row=row, column=1, sticky=tk.W, padx=(0, 4), pady=3)

        self._control_widgets.extend([label, toggle, value_label])

    def _load_sound_engine_controls(self) -> None:
        self._add_apply_defaults_button(row=0)
        self._add_engine_dropdown(row=1)
        selected_engine = self._selected_engine_name if self._selected_engine_name in SOUND_ENGINES else SOUND_ENGINES[0]
        self._render_sound_engine_params(selected_engine)
        self._send_engine_state(selected_engine)

    def _add_engine_dropdown(self, row: int) -> None:
        name = "Engine"
        cc_num = 0
        default_index = SOUND_ENGINES.index(self._selected_engine_name) if self._selected_engine_name in SOUND_ENGINES else 0

        label = ttk.Label(
            self.frame,
            text=f"{name}  (CC {cc_num})",
            font=("Arial", 9),
            width=24,
            anchor=tk.W,
        )
        label.grid(row=row, column=0, sticky=tk.W, padx=(4, 8), pady=3)

        self._engine_name_var = tk.StringVar(value=SOUND_ENGINES[default_index])
        combo = ttk.Combobox(
            self.frame,
            textvariable=self._engine_name_var,
            values=SOUND_ENGINES,
            state="readonly",
        )
        combo.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(0, 4), pady=3)
        combo.bind("<<ComboboxSelected>>", self._on_engine_selected)

        value_label = ttk.Label(self.frame, text=str(default_index), width=4)
        value_label.grid(row=row, column=2, sticky=tk.W, padx=(4, 0), pady=3)

        self._engine_combo = combo
        self._engine_value_label = value_label
        self._control_widgets.extend([label, combo, value_label])

    def _add_apply_defaults_button(self, row: int) -> None:
        button = ttk.Button(
            self.frame,
            text="Apply Defaults",
            command=self._on_apply_defaults,
        )
        button.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=4, pady=(3, 6))
        self._apply_defaults_button = button
        self._control_widgets.append(button)

    def _clear_engine_parameter_widgets(self) -> None:
        # Keep Apply Defaults button (row 0) and engine row widgets (row 1)
        for widget in self._control_widgets[4:]:
            widget.destroy()
        self._control_widgets = self._control_widgets[:4]

    def _render_sound_engine_params(self, engine_name: str) -> None:
        self._clear_engine_parameter_widgets()
        self._active_state_key = self._state_key_for_engine(engine_name)
        params = get_sound_engine_parameters(engine_name)
        self._current_parameters = list(params)
        if not params:
            no_params_label = ttk.Label(
                self.frame,
                text="No parameters for selected engine",
                foreground="gray",
            )
            no_params_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=4, pady=(6, 3))
            self._control_widgets.append(no_params_label)
            return

        for idx, param in enumerate(params, start=2):
            self._add_parameter_control(idx, param, self._active_state_key)

    def _on_engine_selected(self, _event=None) -> None:
        if self._engine_name_var is None or self._engine_value_label is None:
            return
        engine_name = self._engine_name_var.get()
        if engine_name not in SOUND_ENGINES:
            return

        engine_index = SOUND_ENGINES.index(engine_name)
        self._selected_engine_name = engine_name
        self._engine_value_label.config(text=str(engine_index))
        self._render_sound_engine_params(engine_name)
        self._send_engine_state(engine_name)

    def _send_engine_defaults(self, engine_name: str) -> None:
        if engine_name not in SOUND_ENGINES:
            return

        engine_index = SOUND_ENGINES.index(engine_name)
        messages = [(0, engine_index)]
        state_key = self._state_key_for_engine(engine_name)
        messages.extend(self._build_default_cc_messages(get_sound_engine_parameters(engine_name), state_key=state_key))
        self._send_cc_sequence(messages)

    def _send_engine_state(self, engine_name: str) -> None:
        if engine_name not in SOUND_ENGINES:
            return

        engine_index = SOUND_ENGINES.index(engine_name)
        state_key = self._state_key_for_engine(engine_name)
        messages = [(0, engine_index)]
        messages.extend(self._build_cc_messages(get_sound_engine_parameters(engine_name), state_key=state_key, use_defaults=False))
        self._send_cc_sequence(messages)

    def _send_parameter_defaults(self, parameters: list[dict]) -> None:
        messages = self._build_default_cc_messages(parameters, state_key=self._active_state_key)
        self._send_cc_sequence(messages)

    def _on_apply_defaults(self) -> None:
        if self._current_category == "Sound Engine":
            if self._engine_name_var is None:
                return
            engine_name = self._engine_name_var.get()
            self._control_state.pop(self._state_key_for_engine(engine_name), None)
            self._send_engine_defaults(engine_name)
            self._render_sound_engine_params(engine_name)
            return

        if self._current_category and self._current_parameters:
            # Reload category to restore control widgets to their default values,
            # then send matching CC defaults.
            current_category = self._current_category
            self._control_state.pop(self._state_key_for_category(current_category), None)
            self.load_category(current_category)
            self._send_parameter_defaults(self._current_parameters)

    def _on_slider_change(self, val: str, var: tk.IntVar, value_label: ttk.Label, cc: int, state_key: str | None) -> None:
        """
        Called when a slider value changes.  Updates the value label and
        sends a MIDI CC message if a device is connected.
        """
        int_val = int(float(val))
        var.set(int_val)
        value_label.config(text=str(int_val))
        self._store_param_value(state_key, cc, int_val)
        self._cancel_pending_cc_sequence()
        self._send_cc(cc=cc, value=int_val)

    def _on_boolean_change(
        self,
        var: tk.BooleanVar,
        value_label: ttk.Label,
        cc: int,
        off_value: int,
        on_value: int,
        state_key: str | None,
    ) -> None:
        value = on_value if var.get() else off_value
        value_label.config(text=str(value))
        self._store_param_value(state_key, cc, value)
        self._cancel_pending_cc_sequence()
        self._send_cc(cc=cc, value=value)

    def _state_key_for_category(self, category: str) -> str:
        return f"category:{category}"

    def _state_key_for_engine(self, engine_name: str) -> str:
        return f"sound_engine:{engine_name}"

    def _store_param_value(self, state_key: str | None, cc: int, value: int) -> None:
        if not state_key:
            return
        state = self._control_state.setdefault(state_key, {})
        state[int(cc)] = int(value)

    def _resolve_param_value(self, param: dict, state_key: str | None, use_defaults: bool) -> int:
        cc_num = int(param["cc"])
        if not use_defaults and state_key:
            state = self._control_state.get(state_key, {})
            if cc_num in state:
                return int(state[cc_num])

        widget_type = param.get("widget", "slider")
        if widget_type == "boolean":
            off_value = int(param.get("off_value", 0))
            on_value = int(param.get("on_value", 127))
            default = int(param.get("default", off_value))
            return on_value if default == on_value else off_value

        p_min = int(param.get("min", 0))
        p_max = int(param.get("max", 127))
        default = int(param.get("default", p_min))
        return max(p_min, min(p_max, default))

    def _build_cc_messages(self, parameters: list[dict], state_key: str | None, use_defaults: bool) -> list[tuple[int, int]]:
        messages: list[tuple[int, int]] = []
        for param in parameters:
            cc_num = int(param["cc"])
            value = self._resolve_param_value(param, state_key=state_key, use_defaults=use_defaults)
            self._store_param_value(state_key, cc_num, value)
            messages.append((cc_num, value))
        return messages

    def _build_default_cc_messages(self, parameters: list[dict], state_key: str | None) -> list[tuple[int, int]]:
        return self._build_cc_messages(parameters, state_key=state_key, use_defaults=True)

    def _cancel_pending_cc_sequence(self) -> None:
        if not self._pending_cc_jobs:
            return
        for job in self._pending_cc_jobs:
            try:
                self.frame.after_cancel(job)
            except Exception:
                pass
        self._pending_cc_jobs.clear()

    def _send_cc_sequence(self, messages: list[tuple[int, int]], delay_ms: int | None = None) -> None:
        self._cancel_pending_cc_sequence()
        if not messages:
            return

        step_ms = self.BULK_CC_DELAY_MS if delay_ms is None else max(0, int(delay_ms))
        if step_ms <= 0:
            for cc, value in messages:
                self._send_cc(cc=cc, value=value)
            return

        for idx, (cc, value) in enumerate(messages):
            job = self.frame.after(
                idx * step_ms,
                lambda c=cc, v=value: self._send_cc(cc=c, value=v),
            )
            self._pending_cc_jobs.append(job)

    def _send_cc(self, cc: int, value: int) -> None:
        if self.midi_messages:
            try:
                self.midi_messages.send_control_change(
                    control=cc,
                    value=value,
                    channel=self._channel,
                )
            except Exception as e:
                print(f"Failed to send CC {cc}: {e}")
