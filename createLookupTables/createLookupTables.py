import numpy as np
from pathlib import Path

# Generate basic waveforms
def generate_basic_waveforms(num_points=256):
    phase = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    waveforms = {
        "sine_wave": np.sin(phase),
        "square_wave": np.sign(np.sin(phase)),
        "triangle_wave": generate_triangle_wave(256),
        "sawtooth_wave": 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)),
        "reverse_sawtooth_wave": 1 - 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)) - 1,
    }
    # for key in waveforms:
    #     waveforms[key][0] = 0  # Ensure the first value is 0
    return waveforms

def generate_triangle_wave(num_points):
    phase = np.linspace(0, 1, num_points, endpoint=False)  # Phase from 0 to 1
    # Shift the waveform by 90 degrees (1/4 of the period)
    triangle_wave = (2 * np.abs(2 * ((phase + 0.25) % 1 - 0.5)) - 1) * -1
    return triangle_wave

def generate_half_triangle_wave(num_points):
    phase = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Proper triangle wave formula
    triangle_wave = 2 * np.abs(2 * (phase / (2 * np.pi)) - np.floor(2 * (phase / (2 * np.pi)) + 0.5)) - 1
    return triangle_wave

# Create additive synthesis waveforms
def create_additive_waveform(harmonics, num_points=256):
    phase = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    waveform = np.zeros_like(phase)
    for harmonic, amplitude in harmonics.items():
        waveform += amplitude * np.sin(harmonic * phase)
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

# Create Fourier synthesis waveforms
def create_fourier_waveform(num_harmonics, num_points=256):
    phase = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    waveform = np.zeros_like(phase)
    for n in range(1, num_harmonics + 1):
        waveform += (1 / n) * np.sin(n * phase)  # Harmonics with decreasing amplitude
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

# Mix multiple waveforms
def mix_waveforms(waveforms, weights):
    mixed_waveform = np.zeros_like(next(iter(waveforms.values())))
    for name, weight in weights.items():
        mixed_waveform += weight * waveforms[name]
    return mixed_waveform / np.max(np.abs(mixed_waveform))  # Normalize to [-1, 1]

# Save lookup table to a CSV file
def save_waveform_to_csv(waveform, filename):
    np.savetxt(filename, waveform, delimiter=",", fmt="%.6f")
    print(f"Saved waveform to {filename}")

# Example Usage
if __name__ == "__main__":
    num_points = 256

    # Generate basic waveforms
    basic_waveforms = generate_basic_waveforms(num_points)

    # Create an additive synthesis waveform (e.g., 1st, 2nd, and 3rd harmonics)
    additive_waveform = create_additive_waveform({1: 1.0, 2: 0.5, 3: 0.25}, num_points)

    # Create a Fourier synthesis waveform with 5 harmonics
    fourier_waveform = create_fourier_waveform(5, num_points)

    # Mix waveforms (e.g., sine and sawtooth)
    mixed_waveform = mix_waveforms(
        {"sine_wave": basic_waveforms["sine_wave"], "sawtooth_wave": basic_waveforms["sawtooth_wave"]},
        {"sine_wave": 0.7, "sawtooth_wave": 0.3}
    )

    # Save all waveforms to CSV
    appFolder = Path(__file__).parent.absolute()
    folder = f"{appFolder}\\lookupTables\\"
    for name, waveform in basic_waveforms.items():
        save_waveform_to_csv(waveform, f"{folder + name}_lookup_table.csv")
    save_waveform_to_csv(additive_waveform, f"{folder}additive_synthesis_lookup_table.csv")
    save_waveform_to_csv(fourier_waveform, f"{folder}fourier_synthesis_lookup_table.csv")
    save_waveform_to_csv(mixed_waveform, f"{folder}mixed_waveform_lookup_table.csv")
