import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Tuple, Dict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from soundCard.midiDiscoverer import MidiDiscoverer
from soundCard.soundCardDiscoverer import SoundCardDiscoverer
from soundCard.soundCardConfig import SoundCardConfig, AudioConfig
from .midiConfigFrame import MidiConfigFrame, MidiConfig


class MidiDeviceSelector:
    """
    A tkinter-based GUI class for selecting MIDI input and/or output devices.
    Uses MidiDiscoverer to discover available devices.
    """
    
    def __init__(self, 
                 parent=None,
                 device_type: str = 'both',
                 callback: Optional[Callable[[str], None]] = None,
                 title: str = "MIDI Device Selector",
                 include_audio_config: bool = False):
        """
        Initialize MidiDeviceSelector.
        
        Args:
            parent: Parent tkinter widget (None creates root window)
            device_type (str): 'input', 'output', or 'both'
            callback (Optional[Callable]): Function to call when device is selected
            title (str): Window title
            include_audio_config (bool): Include sound card configuration options
        """
        self.device_type = device_type
        self.callback = callback
        self.selected_device: Optional[str] = None
        self.include_audio_config = include_audio_config
        self.audio_config: Optional[AudioConfig] = None
        self.midi_config: Optional[MidiConfig] = None
        
        self.discoverer = MidiDiscoverer()
        self.sound_discoverer = SoundCardDiscoverer() if include_audio_config else None
        self.sound_config = SoundCardConfig() if include_audio_config else None
        
        # Create window based on parent
        if parent is None:
            self.root = tk.Tk()
            self.is_root_owner = True
            self.is_toplevel = False
        else:
            self.root = tk.Toplevel(parent)
            self.is_root_owner = False
            self.is_toplevel = True
            # Make window modal
            self.root.transient(parent)
            self.root.grab_set()
        
        self.root.title(title)
        window_height = 700 if include_audio_config else 420
        self.root.geometry(f"380x{window_height}")
        self.root.resizable(False, False)
        
        self._create_widgets()
        self._load_devices()
    
    def _create_widgets(self) -> None:
        """Create the GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="4")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsiveness
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title label
        title_label = ttk.Label(
            main_frame,
            text=f"Select MIDI Device ({self.device_type.upper()})",
            font=("Arial", 10, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))
        
        # Device type selection (if 'both' is selected)
        if self.device_type == 'both':
            filter_frame = ttk.LabelFrame(main_frame, text="Device Type", padding="2")
            filter_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 4))
            filter_frame.columnconfigure(0, weight=1)
            
            self.device_filter = tk.StringVar(value='output')
            
            ttk.Radiobutton(
                filter_frame,
                text="Input Devices",
                variable=self.device_filter,
                value='input',
                command=self._load_devices
            ).pack(side=tk.LEFT, padx=5)
            
            ttk.Radiobutton(
                filter_frame,
                text="Output Devices",
                variable=self.device_filter,
                value='output',
                command=self._load_devices
            ).pack(side=tk.LEFT, padx=5)
            
            row_offset = 2
        else:
            self.device_filter = tk.StringVar(value=self.device_type)
            row_offset = 1
        
        # Listbox with scrollbar
        list_frame = ttk.LabelFrame(main_frame, text="Available Devices", padding="2")
        list_frame.grid(row=row_offset, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 4))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Listbox
        self.device_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            height=6,
            font=("Arial", 9)
        )
        self.device_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.device_listbox.yview)
        
        # Bind double-click event and single selection
        self.device_listbox.bind('<Double-Button-1>', self._on_device_double_click)
        self.device_listbox.bind('<<ListboxSelect>>', self._on_device_selected)
        
        # Audio configuration section (if enabled)
        if self.include_audio_config:
            self._create_audio_config_widgets(main_frame, row_offset + 1)
            midi_config_row = row_offset + 2
        else:
            midi_config_row = row_offset + 1
        
        # MIDI configuration section
        self.midi_config_frame = MidiConfigFrame(main_frame)
        self.midi_config_frame.grid(row=midi_config_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(4, 0))
        self._set_midi_config_enabled(False)  # Disabled initially
        
        # Button frame (below MIDI config)
        button_row = midi_config_row + 1
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=button_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(4, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        
        # Refresh button
        refresh_btn = ttk.Button(button_frame, text="Refresh", command=self._load_devices)
        refresh_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 2))

        # Select button
        select_btn = ttk.Button(button_frame, text="Select", command=self._on_select)
        select_btn.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 2))

        # Cancel button
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=self._on_cancel)
        cancel_btn.grid(row=0, column=2, sticky=(tk.W, tk.E))
        
        status_row = button_row + 1
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="", foreground="blue", font=("Arial", 8))
        self.status_label.grid(row=status_row, column=0, columnspan=2, sticky=tk.W)
    
    def _set_midi_config_enabled(self, enabled: bool) -> None:
        """
        Enable or disable MIDI configuration widgets.
        
        Args:
            enabled (bool): True to enable, False to disable
        """
        state = tk.NORMAL if enabled else tk.DISABLED
        
        # Recursively set state for all children in the frame
        def set_state(widget, desired_state):
            if isinstance(widget, tk.Widget):
                try:
                    widget.config(state=desired_state)
                except tk.TclError:
                    pass
                
                # Recursively set state for children
                for child in widget.winfo_children():
                    set_state(child, desired_state)
        
        set_state(self.midi_config_frame.frame, state)
    
    def _create_audio_config_widgets(self, parent: ttk.Frame, row: int) -> None:
        """Create audio configuration widgets."""
        # Audio configuration frame
        audio_frame = ttk.LabelFrame(parent, text="Audio Configuration", padding="5")
        audio_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        audio_frame.columnconfigure(1, weight=1)
        
        # Sound Card selection
        ttk.Label(audio_frame, text="Audio Device:").grid(row=0, column=0, sticky=tk.W)
        self.audio_device_var = tk.StringVar()
        audio_devices = self.sound_discoverer.get_device_info() if self.sound_discoverer else []
        audio_device_names = [f"{d['name']}" for d in audio_devices]
        
        self.audio_device_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.audio_device_var,
            values=audio_device_names,
            state='readonly',
            width=30
        )
        self.audio_device_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        if audio_device_names:
            self.audio_device_combo.current(0)
        
        # Sample Rate
        ttk.Label(audio_frame, text="Sample Rate:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.sample_rate_var = tk.StringVar(value="44100")
        sample_rates = ["44100", "48000", "96000", "192000"]
        
        self.sample_rate_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.sample_rate_var,
            values=sample_rates,
            state='readonly',
            width=12
        )
        self.sample_rate_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Channels
        ttk.Label(audio_frame, text="Channels:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.channels_var = tk.StringVar(value="2")
        channels = ["1 (Mono)", "2 (Stereo)", "4", "8"]
        
        self.channels_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.channels_var,
            values=channels,
            state='readonly',
            width=12
        )
        self.channels_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Block Size
        ttk.Label(audio_frame, text="Block Size:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.blocksize_var = tk.StringVar(value="2048")
        block_sizes = ["256", "512", "1024", "2048", "4096"]
        
        self.blocksize_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.blocksize_var,
            values=block_sizes,
            state='readonly',
            width=12
        )
        self.blocksize_combo.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # Latency
        ttk.Label(audio_frame, text="Latency:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.latency_var = tk.StringVar(value="low")
        latencies = ["low", "medium", "high"]
        
        self.latency_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.latency_var,
            values=latencies,
            state='readonly',
            width=12
        )
        self.latency_combo.grid(row=4, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
    
    def _load_devices(self) -> None:
        """Load and display MIDI devices based on selected type. Preserve selection if possible."""
        # Save current selection
        prev_selection = None
        selection = self.device_listbox.curselection()
        if selection:
            prev_selection = self.device_listbox.get(selection[0])
        elif hasattr(self, 'selected_device') and self.selected_device:
            prev_selection = self.selected_device

        self.device_listbox.delete(0, tk.END)

        device_type = self.device_filter.value if hasattr(self.device_filter, 'value') else self.device_filter.get()

        if device_type == 'input' or self.device_type == 'input':
            devices = self.discoverer.get_input_devices()
            device_label = "Input"
        else:
            devices = self.discoverer.get_output_devices()
            device_label = "Output"

        found_index = None
        if devices:
            for idx, device in enumerate(devices):
                self.device_listbox.insert(tk.END, device)
                if prev_selection and device == prev_selection:
                    found_index = idx
            self.status_label.config(
                text=f"Found {len(devices)} {device_label.lower()} device(s)",
                foreground="green"
            )
            # Restore previous selection if possible
            if found_index is not None:
                self.device_listbox.selection_set(found_index)
                self.device_listbox.see(found_index)
                self.selected_device = devices[found_index]
                self._set_midi_config_enabled(True)
        else:
            self.status_label.config(
                text=f"No {device_label.lower()} devices found",
                foreground="red"
            )
    
    def _on_device_selected(self, event) -> None:
        """Handle device selection in listbox."""
        selection = self.device_listbox.curselection()
        if selection:
            # Enable MIDI config frame when a device is selected
            self._set_midi_config_enabled(True)
            self.selected_device = self.device_listbox.get(selection[0])

    # Always remember the last selected device, even if focus changes
    # Do NOT clear self.selected_device unless user explicitly selects a different device or cancels
    
    def _on_device_double_click(self, event) -> None:
        """Handle double-click on device in listbox."""
        self._on_select()
    
    def _on_select(self) -> None:
        """Handle select button click."""
        selection = self.device_listbox.curselection()
        # Only require a selection if no device was ever selected
        if not selection and not self.selected_device:
            messagebox.showwarning("No Selection", "Please select a device from the list!")
            return
        if selection:
            self.selected_device = self.device_listbox.get(selection[0])

        # Capture MIDI configuration
        self.midi_config = self.midi_config_frame.get_config()

        # Capture audio configuration if enabled
        if self.include_audio_config:
            try:
                channels_str = self.channels_var.get().split()[0]
                self.audio_config = AudioConfig(
                    device_id=self.audio_device_combo.current(),
                    sample_rate=int(self.sample_rate_var.get()),
                    channels=int(channels_str),
                    blocksize=int(self.blocksize_var.get()),
                    latency=self.latency_var.get()
                )
            except Exception as e:
                messagebox.showerror("Configuration Error", f"Error reading audio config: {e}")
                return

        if self.callback:
            self.callback(self.selected_device)

        self.status_label.config(
            text=f"Selected: {self.selected_device}",
            foreground="green"
        )

        if self.is_root_owner:
            self.root.after(500, self.root.quit)
        else:
            # For Toplevel windows, just close the window
            self.root.destroy()
    
    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self.selected_device = None
        if self.is_root_owner:
            self.root.quit()
        else:
            # For Toplevel windows, just close the window
            self.root.destroy()
    
    def get_selected_device(self) -> Optional[str]:
        """
        Get the selected device.
        
        Returns:
            Optional[str]: Selected device name or None
        """
        return self.selected_device
    
    def get_audio_config(self) -> Optional[AudioConfig]:
        """
        Get the audio configuration (if included).
        
        Returns:
            Optional[AudioConfig]: Audio configuration or None
        """
        return self.audio_config
    
    def get_midi_config(self) -> Optional[MidiConfig]:
        """
        Get the MIDI configuration.
        
        Returns:
            Optional[MidiConfig]: MIDI configuration or None
        """
        return self.midi_config
    
    def get_selection(self) -> Tuple[Optional[str], Optional[AudioConfig], Optional[MidiConfig]]:
        """
        Get device, audio configuration, and MIDI configuration.
        
        Returns:
            Tuple: (device_name, audio_config, midi_config)
        """
        return (self.selected_device, self.audio_config, self.midi_config)
    
    def show(self) -> Optional[str]:
        """
        Display the window and return selected device (blocking).
        Only works when this class owns the root window.
        
        Returns:
            Optional[str]: Selected device name or None
        """
        if self.is_root_owner:
            self.root.mainloop()
            return self.selected_device
        else:
            # For Toplevel windows, wait for it to close
            self.root.wait_window()
            return self.selected_device
    
    def close(self) -> None:
        """Close the window."""
        if self.is_root_owner:
            self.root.quit()
        else:
            self.root.destroy()


def open_midi_device_selector(device_type: str = 'both') -> Optional[str]:
    """
    Convenience function to open MIDI device selector dialog.
    
    Args:
        device_type (str): 'input', 'output', or 'both'
    
    Returns:
        Optional[str]: Selected device name or None
    """
    selector = MidiDeviceSelector(device_type=device_type)
    return selector.show()


def open_midi_device_selector_with_audio(device_type: str = 'both') -> Tuple[Optional[str], Optional[AudioConfig], Optional[MidiConfig]]:
    """
    Convenience function to open MIDI device selector with audio and MIDI configuration.
    
    Args:
        device_type (str): 'input', 'output', or 'both'
    
    Returns:
        Tuple: (device_name, audio_config, midi_config)
    """
    selector = MidiDeviceSelector(device_type=device_type, include_audio_config=True)
    selector.show()
    return selector.get_selection()


# if __name__ == "__main__":
#     # Test the selector
#     selected = open_midi_device_selector(device_type='output')
    
#     if selected:
#         print(f"Selected device: {selected}")
#     else:
#         print("No device selected")
