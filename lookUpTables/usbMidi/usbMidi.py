import mido
from mido import Message

# List available ports and prompt user to select
input_ports = mido.get_input_names()
print("Available MIDI input ports:")
for i, port in enumerate(input_ports):
    print(f"{i}: {port}")

print("Available MIDI output ports:")
output_ports = mido.get_output_names()
for i, port in enumerate(output_ports):
    print(f"{i}: {port}")

# Specify the input and output ports (replace with your device's name)
input_port_name = 'PicoDevice 0'
output_port_name = 'PicoDevice 1'

# Sending MIDI messages
def send_midi_message():
    with mido.open_output(output_port_name) as output_port:
        # Send a Note On message (Middle C, velocity 64)
        note_on = Message('note_on', note=60, velocity=64)
        output_port.send(note_on)
        print(f"Sent: {note_on}")

        # Send a Note Off message (Middle C, velocity 64)
        note_off = Message('note_off', note=60, velocity=64)
        output_port.send(note_off)
        print(f"Sent: {note_off}")

# Receiving MIDI messages
def receive_midi_message():
    try:
        with mido.open_input(input_port_name) as input_port:
            print(f"Listening for MIDI messages on '{input_port_name}'...")
            for message in input_port:
                print(f"Received: {message}")
    except Exception as e:
        print(f"Error opening MIDI input port '{input_port_name}': {e}")

# Uncomment to send or receive messages
# send_midi_message()
receive_midi_message()
