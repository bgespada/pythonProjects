import soundfile as sf
import numpy as np
import librosa
import os

def estimate_pitch(data, samplerate):
    """Estimate the fundamental frequency (pitch) using librosa."""
    pitches, magnitudes = librosa.piptrack(y=data, sr=samplerate)
    
    # Extract the most dominant pitch
    pitch_values = []
    for t in range(pitches.shape[1]):  # Iterate over time frames
        index = magnitudes[:, t].argmax()  # Get max magnitude index
        pitch = pitches[index, t]
        if pitch > 20:  # Ignore very low frequencies (noise)
            pitch_values.append(pitch)

    if pitch_values:
        return np.median(pitch_values)  # Use median to reduce outliers
    return 0  # Return 0 if no pitch detected

def wav_to_daisy_header(filename, output_folder="converted_headers"):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Extract base name without extension for naming the buffer and file
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_filename = os.path.join(output_folder, f"{base_name}.h")
    var_name = base_name.replace(" ", "_")  # Ensure valid C variable name

    # Load WAV file
    data, samplerate = sf.read(filename, dtype="float32")

    # Resample if necessary
    target_sr = 48000
    if samplerate != target_sr:
        print(f"Resampling {base_name} from {samplerate} Hz to {target_sr} Hz...")
        data = librosa.resample(data.T, orig_sr=samplerate, target_sr=target_sr).T  # Preserve shape

    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # Convert to mono by averaging channels

    # Ensure float32 format
    buffer = data.astype(np.float32)
    buffer_size = len(buffer)

    # Estimate pitch
    pitch = estimate_pitch(buffer, target_sr)

    # Format as a C-style array
    c_array = ",\n    ".join(f"{sample:.6f}f" for sample in buffer)  # Convert all samples

    # Generate header file content
    header_content = f"""#ifndef {var_name.upper()}_H
#define {var_name.upper()}_H

#define {var_name.upper()}_SIZE {buffer_size}
#define {var_name.upper()}_PITCH {pitch:.2f}

const float {var_name.upper()}[{var_name.upper()}_SIZE] = {{
    {c_array}
}};

#endif // {var_name.upper()}_H
"""

    # Write to file
    with open(output_filename, "w") as f:
        f.write(header_content)

    print(f"✅ Converted '{filename}' → '{output_filename}' ({buffer_size} samples, Pitch: {pitch:.2f} Hz)")

def process_wav_folder(input_folder="wav_files", output_folder="converted_headers"):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all WAV files in the input folder
    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"No WAV files found in '{input_folder}'!")
        return

    print(f"🔹 Processing {len(wav_files)} WAV files in '{input_folder}'...")

    # Convert each WAV file
    for wav_file in wav_files:
        wav_to_daisy_header(os.path.join(input_folder, wav_file), output_folder)

    print(f"🎵 All WAV files processed! Headers saved in '{output_folder}'.")

# Run the script for a folder
# process_wav_folder("C:\\DaisyExamples\\MyProjects\\python\\convertWavToBuffer\\Roland TR-909", "converted_headers")
# input_folder = "C:\\Users\\BGE\\Desktop\\PROJECTES\\Roland TR-909 processed\\float"

# input_folder = "C:\\DaisyExamples\\MyProjects\\python\\Roland TR-909"
# input_folder = "C:\\Users\\BGE\\Desktop\\PROJECTES\\Roland TR-909 48kHz"

# input_folder = "C:\\Users\\BGE\\Desktop\\PROJECTES\\Samples\\kicks"
# process_wav_folder(input_folder, "snares")
input_folder = "C:\\Users\\BGE\\Desktop\\PROJECTES\\Samples\\snares"
process_wav_folder(input_folder, "snares")