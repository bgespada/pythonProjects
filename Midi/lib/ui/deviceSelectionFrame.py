import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class DeviceSelectionFrame:
    """
    A reusable frame component for MIDI device selection.
    Displays current device and provides Select/Disconnect buttons.
    """
    
    def __init__(self, 
                 parent: tk.Widget,
                 on_select: Callable[[], None],
                 on_disconnect: Callable[[], None]):
        """
        Initialize the device selection frame.
        
        Args:
            parent: Parent tkinter widget
            on_select: Callback function when Select button is clicked
            on_disconnect: Callback function when Disconnect button is clicked
        """
        self.on_select = on_select
        self.on_disconnect = on_disconnect
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="MIDI Device", padding="10")
        
        # Device label
        ttk.Label(self.frame, text="Current Device:").grid(row=0, column=0, sticky=tk.W)
        self.device_label = ttk.Label(
            self.frame,
            text="No device selected",
            foreground="red",
            font=("Arial", 10, "bold")
        )
        self.device_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.frame.columnconfigure(1, weight=1)
        
        # Device selection buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        self.select_device_btn = ttk.Button(
            button_frame,
            text="Select Device",
            command=self._on_select_clicked
        )
        self.select_device_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.disconnect_btn = ttk.Button(
            button_frame,
            text="Disconnect",
            command=self._on_disconnect_clicked,
            state=tk.DISABLED
        )
        self.disconnect_btn.grid(row=0, column=1, sticky=(tk.W, tk.E))
    
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
    
    def _on_select_clicked(self) -> None:
        """Internal handler for select button."""
        self.on_select()
    
    def _on_disconnect_clicked(self) -> None:
        """Internal handler for disconnect button."""
        self.on_disconnect()
    
    def set_device_name(self, device_name: str, connected: bool = True) -> None:
        """
        Update the device label and button states.
        
        Args:
            device_name (str): Name of the device
            connected (bool): Whether the device is connected
        """
        if connected:
            self.device_label.config(text=device_name, foreground="green")
            self.disconnect_btn.config(state=tk.NORMAL)
            self.select_device_btn.config(state=tk.NORMAL)
        else:
            self.device_label.config(text="No device selected", foreground="red")
            self.disconnect_btn.config(state=tk.DISABLED)
            self.select_device_btn.config(state=tk.NORMAL)
    
    def disable_select_button(self) -> None:
        """Disable the select device button."""
        self.select_device_btn.config(state=tk.DISABLED)
    
    def enable_select_button(self) -> None:
        """Enable the select device button."""
        self.select_device_btn.config(state=tk.NORMAL)
