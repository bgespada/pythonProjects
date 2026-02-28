import mido
from typing import List, Dict, Any, Optional


class MidiDiscoverer:
    """
    A class to discover and manage MIDI hardware devices connected to the laptop.
    """
    
    def __init__(self):
        """Initialize the MidiDiscoverer."""
        self.input_devices = None
        self.output_devices = None
    
    def get_input_devices(self) -> List[str]:
        """
        Returns a list of all available MIDI input devices.
        
        Returns:
            List[str]: List of MIDI input device names
        """
        try:
            self.input_devices = mido.get_input_names()
            return self.input_devices
        except Exception as e:
            print(f"Error querying MIDI input devices: {e}")
            return []
    
    def get_output_devices(self) -> List[str]:
        """
        Returns a list of all available MIDI output devices.
        
        Returns:
            List[str]: List of MIDI output device names
        """
        try:
            self.output_devices = mido.get_output_names()
            return self.output_devices
        except Exception as e:
            print(f"Error querying MIDI output devices: {e}")
            return []
    
    def get_all_devices(self) -> Dict[str, List[str]]:
        """
        Returns both input and output MIDI devices.
        
        Returns:
            Dict[str, List[str]]: Dictionary with 'input' and 'output' keys
        """
        return {
            'input': self.get_input_devices(),
            'output': self.get_output_devices()
        }
    
    def print_midi_devices(self) -> None:
        """Prints all available MIDI devices in a formatted manner."""
        input_devices = self.get_input_devices()
        output_devices = self.get_output_devices()
        
        has_devices = bool(input_devices or output_devices)
        
        if not has_devices:
            print("No MIDI devices found!")
            return
        
        print("Available MIDI Devices:")
        print("=" * 150)
        
        if input_devices:
            print("\nMIDI INPUT Devices:")
            print("-" * 150)
            for i, device in enumerate(input_devices):
                print(f"  ID: {i} | Name: {device}")
        else:
            print("\nNo MIDI Input devices found.")
        
        if output_devices:
            print("\nMIDI OUTPUT Devices:")
            print("-" * 150)
            for i, device in enumerate(output_devices):
                print(f"  ID: {i} | Name: {device}")
        else:
            print("\nNo MIDI Output devices found.")
        
        print()
    
    def get_input_device_by_name(self, name: str) -> str | None:
        """
        Returns MIDI input device by name (case-insensitive).
        
        Args:
            name: The name of the input device to search for
        
        Returns:
            str | None: Device name if found, None otherwise
        """
        devices = self.get_input_devices()
        for device in devices:
            if device.lower() == name.lower():
                return device
        return None
    
    def get_output_device_by_name(self, name: str) -> str | None:
        """
        Returns MIDI output device by name (case-insensitive).
        
        Args:
            name: The name of the output device to search for
        
        Returns:
            str | None: Device name if found, None otherwise
        """
        devices = self.get_output_devices()
        for device in devices:
            if device.lower() == name.lower():
                return device
        return None
    
    def get_input_device_by_id(self, device_id: int) -> str | None:
        """
        Returns MIDI input device by ID.
        
        Args:
            device_id: The ID of the input device
        
        Returns:
            str | None: Device name if found, None otherwise
        """
        devices = self.get_input_devices()
        if 0 <= device_id < len(devices):
            return devices[device_id]
        return None
    
    def get_output_device_by_id(self, device_id: int) -> str | None:
        """
        Returns MIDI output device by ID.
        
        Args:
            device_id: The ID of the output device
        
        Returns:
            str | None: Device name if found, None otherwise
        """
        devices = self.get_output_devices()
        if 0 <= device_id < len(devices):
            return devices[device_id]
        return None
    
    def open_input(self, device_name: str) -> Optional[Any]:
        """
        Opens a MIDI input device for reading messages.
        
        Args:
            device_name: The name of the input device to open
        
        Returns:
            Optional[Any]: MIDI input port object if successful, None otherwise
        """
        try:
            return mido.open_input(device_name)
        except Exception as e:
            print(f"Error opening MIDI input device '{device_name}': {e}")
            return None
    
    def open_output(self, device_name: str) -> Optional[Any]:
        """
        Opens a MIDI output device for sending messages.
        
        Args:
            device_name: The name of the output device to open
        
        Returns:
            Optional[Any]: MIDI output port object if successful, None otherwise
        """
        try:
            return mido.open_output(device_name)
        except Exception as e:
            print(f"Error opening MIDI output device '{device_name}': {e}")
            return None


# if __name__ == "__main__":
#     discoverer = MidiDiscoverer()
    
#     discoverer.print_midi_devices()
    
#     print("\nAll Devices Summary:")
#     all_devices = discoverer.get_all_devices()
#     print(f"Total MIDI Input Devices: {len(all_devices['input'])}")
#     print(f"Total MIDI Output Devices: {len(all_devices['output'])}")
