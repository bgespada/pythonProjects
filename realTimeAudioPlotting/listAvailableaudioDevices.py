import sounddevice as sd

# List all audio devices
def list_audio_devices():
    devices = sd.query_devices()
    print("Available Audio Devices:")
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (Input Channels: {device['max_input_channels']}, "
              f"Output Channels: {device['max_output_channels']}, "
              f"Sample Rates: {device['default_samplerate']})")

list_audio_devices()
