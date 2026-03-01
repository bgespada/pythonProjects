import mido
from typing import Optional, List, Callable
from mido import Message


class MidiMessages:
    """
    A class to handle MIDI message operations with a connected MIDI device.
    The MIDI device should be injected as a parameter (obtained from MidiDiscoverer).
    """
    
    def __init__(self, device: str, device_type: str = 'output'):
        """
        Initialize MidiMessages with a MIDI device.
        
        Args:
            device (str): Device name (from MidiDiscoverer) or mido connection object
            device_type (str): 'input', 'output', or 'both' (default: 'output')
        
        Raises:
            Exception: If device connection fails
        """
        self.device_name = device if isinstance(device, str) else str(device)
        self.device_type = device_type
        self.input_port: Optional[mido.ports.BaseInput] = None
        self.output_port: Optional[mido.ports.BaseOutput] = None
        self._connect_device(device)
    
    def _connect_device(self, device: str) -> None:
        """
        Establish connection to the MIDI device.
        
        Args:
            device (str): Device name to connect to
        
        Raises:
            Exception: If device connection fails
        """
        # Try to open as output device if requested
        if self.device_type in ('output', 'both'):
            try:
                self.output_port = mido.open_output(device)
                print(f"Connected to MIDI output device: {device}")
            except Exception as e:
                if self.device_type == 'output':
                    print(f"Warning: Could not open output port for {device}: {e}")
        
        # Try to open as input device if requested
        if self.device_type in ('input', 'both'):
            try:
                self.input_port = mido.open_input(device)
                print(f"Connected to MIDI input device: {device}")
            except Exception as e:
                if self.device_type == 'input':
                    print(f"Warning: Could not open input port for {device}: {e}")
        
        # Check if at least one connection succeeded based on device_type
        if self.device_type == 'output' and not self.output_port:
            raise Exception(f"Failed to connect to MIDI output device: {device}")
        elif self.device_type == 'input' and not self.input_port:
            raise Exception(f"Failed to connect to MIDI input device: {device}")
        elif self.device_type == 'both' and not self.output_port and not self.input_port:
            raise Exception(f"Failed to connect to MIDI device: {device}")
    
    def send_note_on(self, note: int, velocity: int = 64, channel: int = 0) -> None:
        """
        Send a Note On MIDI message.
        
        Args:
            note (int): Note number (0-127)
            velocity (int): Note velocity (0-127), default 64
            channel (int): MIDI channel (0-15), default 0
        """
        if self.output_port:
            msg = Message('note_on', note=note, velocity=velocity, channel=channel)
            self.output_port.send(msg)
    
    def send_note_off(self, note: int, velocity: int = 0, channel: int = 0) -> None:
        """
        Send a Note Off MIDI message.
        
        Args:
            note (int): Note number (0-127)
            velocity (int): Note velocity (0-127), default 0
            channel (int): MIDI channel (0-15), default 0
        """
        if self.output_port:
            msg = Message('note_off', note=note, velocity=velocity, channel=channel)
            self.output_port.send(msg)
    
    def send_control_change(self, control: int, value: int, channel: int = 0) -> None:
        """
        Send a Control Change (CC) MIDI message.
        
        Args:
            control (int): CC number (0-127)
            value (int): CC value (0-127)
            channel (int): MIDI channel (0-15), default 0
        """
        if self.output_port:
            msg = Message('control_change', control=control, value=value, channel=channel)
            self.output_port.send(msg)
    
    def send_program_change(self, program: int, channel: int = 0) -> None:
        """
        Send a Program Change MIDI message.
        
        Args:
            program (int): Program number (0-127)
            channel (int): MIDI channel (0-15), default 0
        """
        if self.output_port:
            msg = Message('program_change', program=program, channel=channel)
            self.output_port.send(msg)
    
    def send_pitch_bend(self, pitch: int = 8192, channel: int = 0) -> None:
        """
        Send a Pitch Bend MIDI message.
        
        Args:
            pitch (int): Pitch bend value (0-16383), 8192 is center, default 8192
            channel (int): MIDI channel (0-15), default 0
        """
        if self.output_port:
            msg = Message('pitchwheel', pitch=pitch, channel=channel)
            self.output_port.send(msg)
    
    def send_message(self, msg: Message) -> None:
        """
        Send a custom MIDI message.
        
        Args:
            msg (Message): mido Message object to send
        """
        if self.output_port:
            self.output_port.send(msg)
    
    def receive_messages(self, callback: Callable[[Message], None], timeout: Optional[float] = None) -> None:
        """
        Listen for incoming MIDI messages and process them with a callback.
        
        Args:
            callback (Callable): Function to call when a message is received
            timeout (Optional[float]): Timeout in seconds, None for infinite
        """
        if not self.input_port:
            raise Exception("Input port is not available")
        
        try:
            for msg in self.input_port.iter_pending():
                if msg:
                    callback(msg)
        except KeyboardInterrupt:
            print("MIDI message listening stopped.")
    
    def get_pending_messages(self) -> List[Message]:
        """
        Get all pending MIDI messages without blocking.
        
        Returns:
            List[Message]: List of pending MIDI messages
        """
        messages = []
        if self.input_port:
            for msg in self.input_port.iter_pending():
                if msg:
                    messages.append(msg)
        return messages
    
    def close(self) -> None:
        """Close the MIDI device connections."""
        if self.output_port:
            self.output_port.close()
            print(f"Closed MIDI output: {self.device_name}")
            self.output_port = None
        
        if self.input_port:
            self.input_port.close()
            print(f"Closed MIDI input: {self.device_name}")
            self.input_port = None
    
    def __del__(self) -> None:
        """Ensure ports are closed when object is destroyed."""
        try:
            self.close()
        except Exception:
            pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures device is closed."""
        self.close()
        return False
