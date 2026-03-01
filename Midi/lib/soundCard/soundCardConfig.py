import sounddevice as sd
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Data class for audio configuration settings."""
    device_id: Optional[int] = None
    sample_rate: int = 44100
    channels: int = 2
    blocksize: int = 2048
    latency: str = 'low'  # 'low', 'medium', 'high'
    dtype: str = 'float32'


class SoundCardConfig:
    """
    A class to configure and manage audio device settings.
    Handles device selection, sample rate, channels, and audio stream configuration.
    """
    
    def __init__(self):
        """Initialize the SoundCardConfig."""
        self.config = AudioConfig()
        self.active_stream: Optional[sd.Stream] = None
        self.default_device = sd.default.device
    
    def set_device(self, device_id: int) -> bool:
        """
        Set the active audio device.
        
        Args:
            device_id (int): Device ID to set as active
        
        Returns:
            bool: True if device was set successfully, False otherwise
        """
        try:
            device_info = sd.query_devices(device_id)
            if device_info is None:
                print(f"Error: Device ID {device_id} not found")
                return False
            
            self.config.device_id = device_id
            sd.default.device = device_id
            print(f"Audio device set to: {device_info['name']}")
            return True
        except Exception as e:
            print(f"Error setting audio device: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: int) -> bool:
        """
        Set the sample rate.
        
        Args:
            sample_rate (int): Sample rate in Hz (e.g., 44100, 48000, 96000)
        
        Returns:
            bool: True if valid, False otherwise
        """
        valid_rates = [8000, 11025, 16000, 22050, 44100, 48000, 96000, 192000]
        
        if sample_rate not in valid_rates:
            print(f"Error: Invalid sample rate. Valid rates: {valid_rates}")
            return False
        
        self.config.sample_rate = sample_rate
        print(f"Sample rate set to: {sample_rate} Hz")
        return True
    
    def set_channels(self, channels: int) -> bool:
        """
        Set the number of audio channels.
        
        Args:
            channels (int): Number of channels (1=mono, 2=stereo, etc.)
        
        Returns:
            bool: True if valid, False otherwise
        """
        if channels < 1 or channels > 32:
            print(f"Error: Invalid channel count. Must be between 1 and 32")
            return False
        
        self.config.channels = channels
        channel_name = {1: "Mono", 2: "Stereo"}.get(channels, f"{channels}-ch")
        print(f"Channels set to: {channel_name} ({channels})")
        return True
    
    def set_blocksize(self, blocksize: int) -> bool:
        """
        Set the audio block size (buffer size).
        
        Args:
            blocksize (int): Buffer size in samples (e.g., 256, 512, 1024, 2048, 4096)
        
        Returns:
            bool: True if valid, False otherwise
        """
        valid_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        
        if blocksize not in valid_sizes:
            print(f"Error: Invalid blocksize. Valid sizes: {valid_sizes}")
            return False
        
        self.config.blocksize = blocksize
        print(f"Block size set to: {blocksize} samples")
        return True
    
    def set_latency(self, latency: str) -> bool:
        """
        Set the audio latency mode.
        
        Args:
            latency (str): 'low', 'medium', or 'high'
        
        Returns:
            bool: True if valid, False otherwise
        """
        valid_latencies = ['low', 'medium', 'high']
        
        if latency not in valid_latencies:
            print(f"Error: Invalid latency. Valid options: {valid_latencies}")
            return False
        
        self.config.latency = latency
        print(f"Latency set to: {latency}")
        return True
    
    def get_config(self) -> AudioConfig:
        """
        Get the current audio configuration.
        
        Returns:
            AudioConfig: Current configuration
        """
        return self.config
    
    def get_device_capabilities(self, device_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the capabilities of a specific audio device.
        
        Args:
            device_id (int): Device ID to query
        
        Returns:
            Optional[Dict[str, Any]]: Device capabilities or None
        """
        try:
            device_info = sd.query_devices(device_id)
            if device_info is None:
                return None
            
            return {
                'name': device_info['name'],
                'max_input_channels': device_info['max_input_channels'],
                'max_output_channels': device_info['max_output_channels'],
                'default_samplerate': device_info['default_samplerate'],
                'default_latency': device_info['default_latency'],
            }
        except Exception as e:
            print(f"Error getting device capabilities: {e}")
            return None
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration against device capabilities.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        if self.config.device_id is None:
            print("Error: No audio device selected")
            return False
        
        capabilities = self.get_device_capabilities(self.config.device_id)
        if capabilities is None:
            return False
        
        # Check output channels
        if self.config.channels > capabilities['max_output_channels']:
            print(f"Error: Device supports max {capabilities['max_output_channels']} output channels")
            return False
        
        print("Configuration is valid")
        return True
    
    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("\n" + "="*50)
        print("AUDIO CONFIGURATION")
        print("="*50)
        
        if self.config.device_id is not None:
            device_info = sd.query_devices(self.config.device_id)
            print(f"Device:        {device_info['name']}")
        else:
            print(f"Device:        Not set")
        
        print(f"Sample Rate:   {self.config.sample_rate} Hz")
        
        channel_name = {1: "Mono", 2: "Stereo"}.get(self.config.channels, f"{self.config.channels}-ch")
        print(f"Channels:      {channel_name} ({self.config.channels})")
        print(f"Block Size:    {self.config.blocksize} samples")
        print(f"Latency:       {self.config.latency}")
        print(f"Data Type:     {self.config.dtype}")
        print("="*50 + "\n")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = AudioConfig()
        sd.default.device = self.default_device
        print("Configuration reset to defaults")


if __name__ == "__main__":
    # Example usage
    from soundCard.soundCardDiscoverer import SoundCardDiscoverer
    
    # Discover devices
    discoverer = SoundCardDiscoverer()
    devices = discoverer.get_device_info()
    
    print("Available Audio Devices:")
    for device in devices:
        print(f"  ID {device['id']}: {device['name']} ({device['max_output_channels']} out)")
    
    # Configure audio
    config = SoundCardConfig()
    if devices:
        config.set_device(devices[0]['id'])
        config.set_sample_rate(48000)
        config.set_channels(2)
        config.set_blocksize(512)
        config.set_latency('low')
        config.validate_config()
        config.print_config_summary()
