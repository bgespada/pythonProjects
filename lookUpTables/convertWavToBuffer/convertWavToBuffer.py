import soundfile as sf
import numpy as np
import librosa
import os

def wav_to_daisy_header(filename):
    # Extract base name without extension for naming the buffer and file
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_filename = f"{base_name}.h"
    var_name = base_name.replace(" ", "_")  # Ensure valid C variable name

    # Load WAV file
    data, samplerate = sf.read(filename, dtype="float32")

    # Resample if necessary
    target_sr = 48000
    if samplerate != target_sr:
        print(f"Resampling from {samplerate} Hz to {target_sr} Hz...")
        data = librosa.resample(data.T, orig_sr=samplerate, target_sr=target_sr).T  # Preserve shape

    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # Convert to mono by averaging channels

    # Ensure float32 format
    buffer = data.astype(np.float32)
    buffer_size = len(buffer)

    # Format as a C-style array
    c_array = ",\n    ".join(f"{sample:.6f}f" for sample in buffer)  # Convert all samples

    # Generate header file content
    header_content = f"""#ifndef {var_name.upper()}_H
#define {var_name.upper()}_H

#define {var_name}_SIZE {buffer_size}

const float {var_name}[{var_name}_SIZE] = {{
    {c_array}
}};

#endif // {var_name.upper()}_H
"""

    # Write to file
    with open(output_filename, "w") as f:
        f.write(header_content)

    print(f"Header file '{output_filename}' generated successfully!")
    print(f"Array '{var_name}[{var_name}_size]' contains {buffer_size} samples.")

# Usage example
wav_to_daisy_header("C:\\DaisyExamples\\MyProjects\\python\\convertWavToBuffer\\Roland TR-909\\BT0A0A7.WAV")
