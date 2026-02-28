import os
import numpy as np
import librosa

# Constants
AUDIO_BLOCK_SIZE = 48  # Block size for Daisy
SAMPLE_RATE_TARGET = 48000  # Target sample rate for Daisy
OUTPUT_FOLDER = "generated_headers"  # Output folder for .h files

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_pitch_from_filename(filename):
    """Extract pitch from filename if it contains a number (e.g., 'C3_440Hz.wav')."""
    parts = filename.split("_")
    for part in parts:
        if "Hz" in part:
            try:
                return int(part.replace("Hz", ""))
            except ValueError:
                pass
    return 440  # Default pitch if not found

def process_wav_file(filepath):
    """Convert a WAV file into a C header file using Librosa and NumPy."""
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]  # Get filename without extension
    header_filename = os.path.join(OUTPUT_FOLDER, f"{name_no_ext}.h")

    # Load WAV file using librosa
    data, sample_rate = librosa.load(filepath, sr=SAMPLE_RATE_TARGET, mono=True)

    # Normalize to -1.0 to 1.0
    data = data / np.max(np.abs(data))

    # Get pitch
    pitch = get_pitch_from_filename(filename)

    # Generate header file content
    header_content = f"""#ifndef {name_no_ext.upper()}_H
#define {name_no_ext.upper()}_H

#define {name_no_ext.upper()}_LENGTH {len(data)}
#define {name_no_ext.upper()}_START 0
#define {name_no_ext.upper()}_END {len(data) - 1}
#define {name_no_ext.upper()}_PITCH {pitch}

const float {name_no_ext}[] = {{
    {', '.join(map(str, data))}
}};

#endif // {name_no_ext.upper()}_H
"""

    # Save to header file
    with open(header_filename, "w") as f:
        f.write(header_content)

    print(f"✅ Generated: {header_filename}")

# Process all WAV files in the folder
input_folder = "C:\\DaisyExamples\\MyProjects\\python\\convertWavToBuffer\\Roland TR-909"  # Change this to your actual folder
for file in os.listdir(input_folder):
    if file.lower().endswith(".wav"):
        process_wav_file(os.path.join(input_folder, file))

print("🎵 Header file generation complete!")
