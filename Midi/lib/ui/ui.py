import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from soundCard.midiDiscoverer import MidiDiscoverer
from midi.midiMessages import MidiMessages
from .midiDeviceSelector import MidiDeviceSelector


class MidiUI:
    """
    Main UI class for managing MIDI device interactions.
    This is the primary interface that coordinates all UI components and MIDI operations.
    """
    
    def __init__(self):
        """Initialize the main MIDI UI application."""
        self.root = tk.Tk()
        self.root.title("MIDI Device Control")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # MIDI components
        self.discoverer: Optional[MidiDiscoverer] = None
        self.midi_messages: Optional[MidiMessages] = None
        self.selected_device: Optional[str] = None
        
        # UI components
        self.device_label: Optional[ttk.Label] = None
        self.status_label: Optional[ttk.Label] = None
        self.main_frame: Optional[ttk.Frame] = None
        
        self._create_widgets()
        self._initialize_midi()
    
    def _create_widgets(self) -> None:
        """Create the main UI widgets."""
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="MIDI Device Controller",
            font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 15))
        
        # Device selection frame
        device_frame = ttk.LabelFrame(self.main_frame, text="MIDI Device", padding="10")
        device_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        device_frame.columnconfigure(1, weight=1)
        
        # Device label
        ttk.Label(device_frame, text="Current Device:").grid(row=0, column=0, sticky=tk.W)
        self.device_label = ttk.Label(
            device_frame,
            text="No device selected",
            foreground="red",
            font=("Arial", 10, "bold")
        )
        self.device_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        
        # Device selection buttons
        button_frame = ttk.Frame(device_frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        select_device_btn = ttk.Button(
            button_frame,
            text="Select Device",
            command=self._on_select_device
        )
        select_device_btn.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        disconnect_btn = ttk.Button(
            button_frame,
            text="Disconnect",
            command=self._on_disconnect,
            state=tk.DISABLED
        )
        disconnect_btn.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.disconnect_btn = disconnect_btn
        
        # Content frame (for subclasses to add custom widgets)
        self.content_frame = ttk.LabelFrame(
            self.main_frame,
            text="Controls",
            padding="10"
        )
        self.content_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(0, weight=1)
        
        # Placeholder content
        placeholder = ttk.Label(
            self.content_frame,
            text="Add your MIDI controls here",
            foreground="gray"
        )
        placeholder.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            foreground="blue"
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _initialize_midi(self) -> None:
        """Initialize MIDI discoverer."""
        try:
            self.discoverer = MidiDiscoverer()
            self._update_status("MIDI discoverer initialized", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize MIDI: {e}")
            self._update_status(f"Error: {e}", "red")
    
    def _on_select_device(self) -> None:
        """Open device selector dialog."""
        try:
            selector = MidiDeviceSelector(
                parent=self.root,
                device_type='output',
                callback=self._on_device_selected,
                title="Select MIDI Output Device"
            )
            device = selector.show()
            
            if device:
                self._connect_device(device)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open device selector: {e}")
            self._update_status(f"Error: {e}", "red")
    
    def _on_device_selected(self, device: str) -> None:
        """Handle device selection callback."""
        self._connect_device(device)
    
    def _connect_device(self, device: str) -> None:
        """
        Connect to a MIDI device.
        
        Args:
            device (str): Device name to connect to
        """
        try:
            # Disconnect from previous device if any
            if self.midi_messages:
                self.midi_messages.close()
            
            # Connect to new device
            self.midi_messages = MidiMessages(device)
            self.selected_device = device
            
            # Update UI
            self.device_label.config(text=device, foreground="green")
            self.disconnect_btn.config(state=tk.NORMAL)
            self._update_status(f"Connected to: {device}", "green")
            
            # Trigger onConnect hook for subclasses
            self.on_device_connected()
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect to device: {e}")
            self._update_status(f"Connection failed", "red")
    
    def _on_disconnect(self) -> None:
        """Disconnect from current MIDI device."""
        try:
            if self.midi_messages:
                self.midi_messages.close()
                self.midi_messages = None
            
            self.selected_device = None
            self.device_label.config(text="No device selected", foreground="red")
            self.disconnect_btn.config(state=tk.DISABLED)
            self._update_status("Disconnected", "blue")
            
            # Trigger onDisconnect hook for subclasses
            self.on_device_disconnected()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to disconnect: {e}")
    
    def _update_status(self, message: str, color: str = "blue") -> None:
        """
        Update status bar message.
        
        Args:
            message (str): Status message
            color (str): Message color
        """
        if self.status_label:
            self.status_label.config(text=message, foreground=color)
    
    def _on_closing(self) -> None:
        """Handle window closing."""
        if self.midi_messages:
            self._on_disconnect()
        self.root.destroy()
    
    # Hook methods for subclass customization
    def on_device_connected(self) -> None:
        """
        Called when a device is successfully connected.
        Override in subclass to add custom behavior.
        """
        pass
    
    def on_device_disconnected(self) -> None:
        """
        Called when a device is disconnected.
        Override in subclass to add custom behavior.
        """
        pass
    
    def get_midi_messages(self) -> Optional[MidiMessages]:
        """
        Get the current MIDI messages handler.
        
        Returns:
            Optional[MidiMessages]: Current MIDI handler or None
        """
        return self.midi_messages
    
    def is_connected(self) -> bool:
        """
        Check if a MIDI device is connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.midi_messages is not None
    
    def run(self) -> None:
        """Start the UI application."""
        self.root.mainloop()


# if __name__ == "__main__":
#     app = MidiUI()
#     app.run()
