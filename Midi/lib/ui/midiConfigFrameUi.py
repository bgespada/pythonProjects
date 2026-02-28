import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class MidiConfig:
    """Data class for MIDI configuration settings."""
    device_mode: str = 'output'  # 'input', 'output', or 'both'
    channel: int = 0  # 0-15 or special value for all channels
    default_velocity: int = 64  # 0-127
    default_octave: int = 5
    note_off_mode: str = 'note_off'  # 'note_off' or 'note_on_zero_velocity'


class MidiConfigFrameUi:
    """
    A reusable frame component for MIDI configuration.
    Allows users to configure MIDI channel, velocity, octave, and other settings.
    """
    
    def __init__(self, 
                 parent: tk.Widget,
                 on_config_changed: Optional[Callable[[MidiConfig], None]] = None):
        """
        Initialize the MIDI configuration frame.
        
        Args:
            parent: Parent tkinter widget
            on_config_changed: Optional callback when configuration changes
        """
        self.on_config_changed = on_config_changed
        self.config = MidiConfig()
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="MIDI Configuration", padding="10")
        
        # Device Mode (Input/Output/Both)
        ttk.Label(self.frame, text="Device Mode:").grid(row=0, column=0, sticky=tk.W)
        self.device_mode_var = tk.StringVar(value="Output")
        
        device_mode_options = ["Input", "Output", "Both"]
        self.device_mode_combo = ttk.Combobox(
            self.frame,
            textvariable=self.device_mode_var,
            values=device_mode_options,
            state='readonly',
            width=15
        )
        self.device_mode_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.device_mode_combo.bind('<<ComboboxSelected>>', lambda e: self._on_config_changed())
        self.frame.columnconfigure(1, weight=1)
        
        # Channel selection
        ttk.Label(self.frame, text="MIDI Channel:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.channel_var = tk.StringVar(value="All Channels")
        
        channel_options = ["All Channels"] + [str(i+1) for i in range(16)]
        self.channel_combo = ttk.Combobox(
            self.frame,
            textvariable=self.channel_var,
            values=channel_options,
            state='readonly',
            width=15
        )
        self.channel_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 0))
        self.channel_combo.bind('<<ComboboxSelected>>', lambda e: self._on_config_changed())
        
        # Velocity
        ttk.Label(self.frame, text="Default Velocity:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        
        velocity_frame = ttk.Frame(self.frame)
        velocity_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 0))
        velocity_frame.columnconfigure(1, weight=1)
        
        self.velocity_var = tk.IntVar(value=64)
        self.velocity_scale = ttk.Scale(
            velocity_frame,
            from_=0,
            to=127,
            orient=tk.HORIZONTAL,
            variable=self.velocity_var,
            command=self._on_velocity_changed
        )
        self.velocity_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.velocity_label = ttk.Label(velocity_frame, text="64", width=3)
        self.velocity_label.grid(row=0, column=1, padx=(10, 0))
        
        # Octave
        ttk.Label(self.frame, text="Default Octave:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        
        octave_frame = ttk.Frame(self.frame)
        octave_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 0))
        octave_frame.columnconfigure(1, weight=1)
        
        self.octave_var = tk.IntVar(value=5)
        self.octave_scale = ttk.Scale(
            octave_frame,
            from_=0,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.octave_var,
            command=self._on_octave_changed
        )
        self.octave_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.octave_label = ttk.Label(octave_frame, text="5", width=3)
        self.octave_label.grid(row=0, column=1, padx=(10, 0))
        
        # Note Off Mode
        ttk.Label(self.frame, text="Note Off Mode:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.note_off_var = tk.StringVar(value="note_off")
        
        noteoff_options = [
            ("Note Off Message", "note_off"),
            ("Note On (Velocity 0)", "note_on_zero_velocity")
        ]
        
        for i, (label, value) in enumerate(noteoff_options):
            ttk.Radiobutton(
                self.frame,
                text=label,
                variable=self.note_off_var,
                value=value,
                command=self._on_config_changed
            ).grid(row=4+i, column=1, sticky=tk.W, padx=(10, 0), pady=(10 if i == 0 else 0, 0))
    
    def grid(self, **kwargs) -> None:
        """
        Place the frame on grid.
        
        Args:
            **kwargs: Grid arguments
        """
        self.frame.grid(**kwargs)
    
    def pack(self, **kwargs) -> None:
        """
        Place the frame using pack.
        
        Args:
            **kwargs: Pack arguments
        """
        self.frame.pack(**kwargs)
    
    def _on_velocity_changed(self, value) -> None:
        """Handle velocity slider change."""
        velocity = int(float(value))
        self.velocity_label.config(text=str(velocity))
        self._on_config_changed()
    
    def _on_octave_changed(self, value) -> None:
        """Handle octave slider change."""
        octave = int(float(value))
        self.octave_label.config(text=str(octave))
        self._on_config_changed()
    
    def _on_config_changed(self) -> None:
        """Called when any configuration value changes."""
        # Update config object
        device_mode = self.device_mode_var.get()
        self.config.device_mode = device_mode.lower()  # Convert to lowercase
        
        channel_str = self.channel_var.get()
        if channel_str == "All Channels":
            self.config.channel = -1  # Special value for all channels
        else:
            self.config.channel = int(channel_str) - 1  # Convert 1-16 to 0-15
        
        self.config.default_velocity = self.velocity_var.get()
        self.config.default_octave = self.octave_var.get()
        self.config.note_off_mode = self.note_off_var.get()
        self.config.note_off_mode = self.note_off_var.get()
        
        # Call callback if provided
        if self.on_config_changed:
            self.on_config_changed(self.config)
    
    def get_config(self) -> MidiConfig:
        """
        Get the current MIDI configuration.
        
        Returns:
            MidiConfig: Current configuration
        """
        self._on_config_changed()  # Ensure config is up to date
        return self.config
    
    def set_config(self, config: MidiConfig) -> None:
        """
        Set the MIDI configuration.
        
        Args:
            config: MidiConfig object to set
        """
        # Set device mode
        device_mode_display = config.device_mode.capitalize()
        self.device_mode_var.set(device_mode_display)
        
        # Set channel
        if config.channel == -1:
            self.channel_var.set("All Channels")
        else:
            self.channel_var.set(str(config.channel + 1))
        
        # Set other values
        self.velocity_var.set(config.default_velocity)
        self.velocity_label.config(text=str(config.default_velocity))
        
        self.octave_var.set(config.default_octave)
        self.octave_label.config(text=str(config.default_octave))
        
        self.note_off_var.set(config.note_off_mode)
        
        self.config = config
    
    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        config = self.get_config()
        
        print("\n" + "="*50)
        print("MIDI CONFIGURATION")
        print("="*50)
        
        print(f"Device Mode:       {config.device_mode.capitalize()}")
        
        if config.channel == -1:
            print(f"Channel:           All Channels")
        else:
            print(f"Channel:           {config.channel + 1}")
        
        print(f"Default Velocity:  {config.default_velocity}")
        print(f"Default Octave:    {config.default_octave}")
        print(f"Note Off Mode:     {config.note_off_mode}")
        print("="*50 + "\n")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.set_config(MidiConfig())
