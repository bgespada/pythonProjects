
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
from pathlib import Path
import sys
from .statusBarUi import StatusBar


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from soundCard.midiDiscoverer import MidiDiscoverer
from midi.midiMessages import MidiMessages
from .midiDeviceSelectorUi import MidiDeviceSelectorUi
from .deviceSelectionFrameUi import DeviceSelectionFrameUi


class MidiUI:
    """
    Main UI class for managing MIDI device interactions.
    This is the primary interface that coordinates all UI components and MIDI operations.
    """
    
    def __init__(self):
        """Initialize the main MIDI UI application."""
        self.root = tk.Tk()
        self.root.title("MIDI Device Control")
        self.root.geometry("700x780")
        self.root.resizable(False, False)
        
        # MIDI components
        self.discoverer: Optional[MidiDiscoverer] = None
        self.midi_messages: Optional[MidiMessages] = None
        self.selected_device: Optional[str] = None
        
        # UI components
        self.device_label: Optional[ttk.Label] = None
        self.status_bar: Optional[StatusBar] = None
        self.main_frame: Optional[ttk.Frame] = None
        self.device_frame: Optional[DeviceSelectionFrameUi] = None
        self.control_panel = None
        self.sequencer_frame = None
        
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
        self.main_frame.rowconfigure(2, weight=0)   # Controls: compact
        self.main_frame.rowconfigure(3, weight=1)   # Sequencer: expands
        
        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="MIDI Device Controller",
            font=("Arial", 14, "bold")
        )
        title_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 15))
        

        # Container for device and transport frames (fixed width, no stretching)
        self.top_frames_container = ttk.Frame(self.main_frame)
        self.top_frames_container.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))

        # Device selection frame (left, top)
        self.device_frame = DeviceSelectionFrameUi(
            parent=self.top_frames_container,
            on_select=self._on_select_device,
            on_disconnect=self._on_disconnect
        )
        self.device_frame.frame.config(width=260, height=80)
        self.device_frame.frame.grid_propagate(False)
        self.device_frame.grid(row=0, column=0, sticky=tk.W, padx=(0, 12))

        # Transport frame (right, fixed after device frame)
        from .transportFrameUi import TransportFrameUi
        self.transport_frame = TransportFrameUi(
            parent=self.top_frames_container,
            midi_transport=None
        )
        self.transport_frame.config(width=340, height=80)
        self.transport_frame.grid_propagate(False)
        self.transport_frame.grid(row=0, column=1, sticky=tk.W)
        self.top_frames_container.columnconfigure(0, weight=0)
        self.top_frames_container.columnconfigure(1, weight=0)
        # Disable transport buttons initially
        for btn in (self.transport_frame.start_btn, self.transport_frame.pause_btn, self.transport_frame.stop_btn):
            btn.config(state=tk.DISABLED)

        # Controls frame (below device and transport, full width)
        self.content_frame = ttk.LabelFrame(
            self.main_frame,
            text="Controls",
            padding="10"
        )
        self.content_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(0, weight=1)
        
        # Control panel (tree + parameter sliders)
        from .controls import ControlPanelUi
        self.control_panel = ControlPanelUi(self.content_frame)
        self.control_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Sequencer frame (below controls, full width)
        from .sequencer import SequencerFrameUi
        self.sequencer_frame = SequencerFrameUi(self.main_frame)
        self.sequencer_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 6))

        # Wire transport start/stop → sequencer engine
        self.transport_frame.add_start_callback(self.sequencer_frame.engine.start)
        self.transport_frame.add_stop_callback(self.sequencer_frame.engine.stop)

        # Status bar
        self.status_bar = StatusBar(self.main_frame, initial_text="Ready", initial_color="blue")
        self.status_bar.grid(row=4, column=0, sticky=(tk.W, tk.E))
        
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
            # Disable select button to prevent multiple dialogs
            self.device_frame.disable_select_button()
            
            selector = MidiDeviceSelectorUi(
                parent=self.root,
                device_type='output',
                title="Select MIDI Output Device"
            )
            device = selector.show()
            
            if device:
                # Get MIDI config from selector
                midi_config = selector.get_midi_config()
                device_mode = midi_config.device_mode if midi_config else 'output'
                # Update MIDI channel in status bar
                if midi_config and hasattr(midi_config, 'channel') and midi_config.channel not in (None, "All Channels"):
                    try:
                        channel_num = int(midi_config.channel)
                    except Exception:
                        channel_num = None
                else:
                    channel_num = None
                self._update_midi_channel(channel_num)
                self._connect_device(device, device_mode=device_mode, channel=channel_num)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open device selector: {e}")
            self._update_status(f"Error: {e}", "red")
        finally:
            # Re-enable select button after dialog closes
            self.device_frame.enable_select_button()
    
    def _connect_device(self, device: str, device_mode: str = 'output', channel: Optional[int] = None) -> None:
        """
        Connect to a MIDI device.
        
        Args:
            device (str): Device name to connect to
            device_mode (str): Device mode ('input', 'output', or 'both')
            channel (Optional[int]): MIDI channel (1-based), or None
        """
        # Prevent reconnecting to same device
        if self.selected_device == device and self.midi_messages is not None:
            return
        
        try:
            # Disconnect from previous device if any
            if self.midi_messages:
                self.midi_messages.close()
            
            # Connect to new device with specified mode
            self.midi_messages = MidiMessages(device, device_type=device_mode)
            self.selected_device = device
            # Update UI
            self.device_frame.set_device_name(device, connected=True)
            self._update_status(f"Connected to: {device}", "green")
            # Enable and wire up transport frame
            from midi.transport import MidiTransport
            self.transport_frame.midi_transport = MidiTransport(self.midi_messages)
            for btn in (self.transport_frame.start_btn, self.transport_frame.pause_btn, self.transport_frame.stop_btn):
                btn.config(state=tk.NORMAL)
            # Wire up control panel
            if self.control_panel:
                self.control_panel.set_midi_messages(self.midi_messages)
                if channel is not None:
                    self.control_panel.set_channel(max(0, channel - 1))  # convert to 0-based, clamp
            # Wire up sequencer
            if self.sequencer_frame:
                self.sequencer_frame.set_midi_messages(self.midi_messages)
                if channel is not None:
                    self.sequencer_frame.set_channel(max(0, channel - 1))
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
            self.device_frame.set_device_name("No device selected", connected=False)
            self._update_status("Disconnected", "blue")
            # Disconnect control panel
            if self.control_panel:
                self.control_panel.set_midi_messages(None)
            # Disconnect sequencer
            if self.sequencer_frame:
                self.sequencer_frame.engine.stop()
                self.sequencer_frame.set_midi_messages(None)
            # Trigger onDisconnect hook for subclasses
            self.on_device_disconnected()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to disconnect: {e}")
    
    def _update_status(self, message: str, color: str = "blue") -> None:
        """
        Update status bar main info message.
        Args:
            message (str): Status message
            color (str): Message color
        """
        if self.status_bar:
            self.status_bar.set_info(message, color)

    def _update_midi_channel(self, channel: int = None):
        """
        Update the status bar MIDI channel section.
        Args:
            channel (int): MIDI channel number (1-based), or None to clear
        """
        if self.status_bar:
            self.status_bar.set_channel(channel)
    
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
