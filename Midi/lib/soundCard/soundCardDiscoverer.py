import sounddevice as sd
from typing import List, Dict, Any


class SoundCardDiscoverer:
    """
    A class to discover and manage audio devices connected to the laptop.
    """
    
    def __init__(self):
        """Initialize the SoundCardDiscoverer."""
        self.devices = None
        self.device_info = None
    
    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all audio devices connected to the laptop.
        
        Returns:
            List[Dict[str, Any]]: List of audio device dictionaries with properties
        """
        try:
            self.devices = sd.query_devices()
            return self.devices
        except Exception as e:
            print(f"Error querying audio devices: {e}")
            return []
    
    def get_device_info(self) -> List[Dict[str, Any]]:
        """
        Returns detailed information about all audio devices.
        
        Returns:
            List[Dict[str, Any]]: List of audio devices with formatted information
        """
        devices = self.get_audio_devices()
        self.device_info = []
        
        for i, device in enumerate(devices):
            info = {
                'id': i,
                'name': device['name'],
                'max_input_channels': device['max_input_channels'],
                'max_output_channels': device['max_output_channels'],
                'default_samplerate': device['default_samplerate'],
            }
            self.device_info.append(info)
        
        return self.device_info
    
    def print_audio_devices(self) -> None:
        """Prints all available audio devices in a formatted manner."""
        devices = self.get_device_info()
        
        if not devices:
            print("No audio devices found!")
            return
        
        print("Available Audio Devices:")
        print("-" * 150)
        for device in devices:
            print(f"ID: {device['id']}")
            print(f"  Name: {device['name']}")
            print(f"  Input Channels: {device['max_input_channels']}")
            print(f"  Output Channels: {device['max_output_channels']}")
            print(f"  Default Sample Rate: {device['default_samplerate']} Hz")
            print()
    
    def get_default_input_device(self) -> Dict[str, Any] | None:
        """
        Returns the default input audio device.
        
        Returns:
            Dict[str, Any] | None: Default input device info or None
        """
        try:
            default_device = sd.default.device
            if isinstance(default_device, (tuple, list)):
                device_id = default_device[0]
            else:
                device_id = default_device
            
            device = sd.query_devices(device_id)
            return {'id': device_id, **device}
        except Exception as e:
            print(f"Error getting default input device: {e}")
            return None
    
    def get_default_output_device(self) -> Dict[str, Any] | None:
        """
        Returns the default output audio device.
        
        Returns:
            Dict[str, Any] | None: Default output device info or None
        """
        try:
            default_device = sd.default.device
            if isinstance(default_device, (tuple, list)):
                device_id = default_device[1]
            else:
                device_id = default_device
            
            device = sd.query_devices(device_id)
            return {'id': device_id, **device}
        except Exception as e:
            print(f"Error getting default output device: {e}")
            return None
    
    def get_device_by_name(self, name: str) -> Dict[str, Any] | None:
        """
        Returns device info by device name.
        
        Args:
            name: The name of the device to search for
        
        Returns:
            Dict[str, Any] | None: Device info if found, None otherwise
        """
        devices = self.get_device_info()
        for device in devices:
            if device['name'].lower() == name.lower():
                return device
        return None


# if __name__ == "__main__":
#     discoverer = SoundCardDiscoverer()
    
#     discoverer.print_audio_devices()
#     print("\n")
#     print("Default Input Device:")
#     print(discoverer.get_default_input_device())
#     print("\nDefault Output Device:")
#     print(discoverer.get_default_output_device())
