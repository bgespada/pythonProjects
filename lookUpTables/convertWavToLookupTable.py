import soundfile as sf
import numpy as np
import librosa
import os

OUTPUT_HEADER = "wav_lookup_table.h"
AUDIO_BLOCK_SIZE = 48  # Daisy uses 48-sample blocks
TARGET_SR = 48000  # Target sample rate

def estimate_pitch(data, samplerate):
    """Estimate pitch using librosa's piptrack method."""
    pitches, magnitudes = librosa.piptrack(y=data, sr=samplerate)
    pitch_values = [pitches[:, t].max() for t in range(pitches.shape[1]) if pitches[:, t].max() > 20]
    return np.median(pitch_values) if pitch_values else 0  # Return median pitch or 0 if no pitch detected

def process_wav_folder(input_folder="wav_files", output_header=OUTPUT_HEADER):
    os.makedirs(input_folder, exist_ok=True)

    wav_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]
    if not wav_files:
        print(f"No WAV files found in '{input_folder}'!")
        return

    print(f"🔹 Processing {len(wav_files)} WAV files in '{input_folder}'...")

    lookup_table = []
    metadata = []
    start_address = 0

    for wav_file in wav_files:
        file_path = os.path.join(input_folder, wav_file)
        base_name = os.path.splitext(wav_file)[0].replace(" ", "_")

        data, samplerate = sf.read(file_path, dtype="float32")

        if samplerate != TARGET_SR:
            print(f"Resampling {base_name} from {samplerate} Hz to {TARGET_SR} Hz...")
            data = librosa.resample(data.T, orig_sr=samplerate, target_sr=TARGET_SR).T

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # Convert stereo to mono

        data = data.astype(np.float32)

        # Ensure the length is a multiple of AUDIO_BLOCK_SIZE
        remainder = len(data) % AUDIO_BLOCK_SIZE
        if remainder > 0:
            padding = AUDIO_BLOCK_SIZE - remainder
            data = np.pad(data, (0, padding), mode="constant")

        end_address = start_address + len(data) - 1
        pitch = estimate_pitch(data, TARGET_SR)

        metadata.append(f"{{\"{base_name}\", {start_address}, {end_address}, {len(data)}, {pitch:.2f}}}")
        lookup_table.extend(data)

        start_address = end_address + 1  # Update start for the next file

    # Generate C++ header content
    lookup_table_data = ",\n    ".join(f"{sample:.6f}f" for sample in lookup_table)

    header_content = f"""#ifndef WAV_LOOKUP_TABLE_H
#define WAV_LOOKUP_TABLE_H

#define AUDIO_BLOCK_SIZE {AUDIO_BLOCK_SIZE}
#define WAV_TABLE_SIZE {len(lookup_table)}

static const float wav_lookup_table[WAV_TABLE_SIZE] = {{
    {lookup_table_data}
}};

// Metadata structure for each sample
struct WavSample {{
    const char* name;
    int start_address;
    int end_address;
    int length;
    float pitch;
}};

// List of WAV files in the lookup table
static const WavSample wav_samples[] = {{
    {",\n    ".join(metadata)}
}};

#define WAV_SAMPLE_COUNT {len(metadata)}

#endif // WAV_LOOKUP_TABLE_H
"""

    with open(output_header, "w") as f:
        f.write(header_content)

    print(f"✅ Generated '{output_header}' with {len(metadata)} WAV files!")

process_wav_folder("C:\\DaisyExamples\\MyProjects\\python\\convertWavToBuffer\\Roland TR-909", OUTPUT_HEADER)
