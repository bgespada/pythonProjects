import numpy as np
import os
# from pathlib import Path

# # Function: Generate a Single Fourier Waveform
# def generate_fourier_waveform(samples, harmonics, amplitude_pattern=None, phase_pattern=None):
#     x = np.linspace(0, 1, samples, endpoint=False)
#     waveform = np.zeros(samples)
    
#     # Default amplitude and phase patterns
#     if amplitude_pattern is None:
#         amplitude_pattern = lambda n: 1 / n  # Amplitude decreases as 1/n
#     if phase_pattern is None:
#         phase_pattern = lambda n: 0          # No phase offset

#     # Sum harmonics
#     for n in range(1, harmonics + 1):
#         amplitude = amplitude_pattern(n)
#         phase = phase_pattern(n)
#         waveform += amplitude * np.sin(2 * np.pi * n * x + phase)
    
#     # Normalize waveform to [-1, 1]
#     waveform /= np.max(np.abs(waveform))
#     return waveform

# # Function: Generate Fourier Wavetable
# def generate_fourier_wavetable(samples, num_waveforms, max_harmonics):
#     wavetable = []
#     for i in range(num_waveforms):
#         harmonics = int((i + 1) / num_waveforms * max_harmonics)  # Gradually increase harmonics
#         amplitude_pattern = lambda n: 1 / n                      # Amplitude decreases as 1/n
#         phase_pattern = lambda n: np.random.uniform(0, 2 * np.pi) if i % 2 == 0 else 0  # Alternate phases
#         waveform = generate_fourier_waveform(samples, harmonics, amplitude_pattern, phase_pattern)
#         wavetable.append(waveform)
#         print(f"Generated waveform {i + 1}/{num_waveforms} with {harmonics} harmonics")
#     return np.array(wavetable)

def generate_fourier_wavetable(length=256, num_waveforms=8, max_harmonics=16, fundamental_freq=1.0):
    """
    Generates a Fourier wavetable where each waveform is a variation of the same form with
    more or fewer harmonics added.
    
    Parameters:
        length (int): Number of samples per waveform.
        num_waveforms (int): Number of waveforms in the wavetable.
        max_harmonics (int): Maximum number of harmonics to include in the waveforms.
        fundamental_freq (float): Fundamental frequency of the waveforms.
        
    Returns:
        np.ndarray: A 2D array representing the wavetable.
    """
    t = np.linspace(0, 1, length, endpoint=False)
    wavetable = []

    for waveform_idx in range(num_waveforms):
        harmonics = np.arange(1, max_harmonics + 1)  # Harmonic numbers
        amplitudes = np.random.rand(max_harmonics) * (1 / harmonics)  # Random amplitudes scaled by harmonic number
        
        # Gradually decrease the number of active harmonics for each waveform
        num_active_harmonics = max_harmonics - waveform_idx * (max_harmonics // num_waveforms)
        
        # Select only the active harmonics
        active_harmonics = harmonics[:num_active_harmonics]
        active_amplitudes = amplitudes[:num_active_harmonics]
        
        # Generate the waveform by summing active harmonics
        waveform = np.zeros(length)
        for h, a in zip(active_harmonics, active_amplitudes):
            waveform += a * np.sin(2 * np.pi * h * fundamental_freq * t)
        
        # Normalize to the range [-1, 1]
        waveform /= np.max(np.abs(waveform))
        wavetable.append(waveform)
    
    return np.array(wavetable)

# Function: Save Wavetable to CSV
def save_wavetable_to_csv(wavetable, filename):
    np.savetxt(filename, wavetable, delimiter=",", header="", comments="")
    print(f"Saved wavetable to {filename}")

# Function: Save Wavetable to C Header File
def save_wavetable_to_hpp(wavetable, filename):
    with open(filename, "w") as file:
        file.write("#ifndef FOURIER_WAVETABLE_HPP\n#define FOURIER_WAVETABLE_HPP\n\n")
        file.write(f"// Fourier Wavetable with {len(wavetable)} waveforms, {wavetable.shape[1]} samples each\n\n")
        file.write("const float FOURIER_WAVETABLE[][256] = {\n")
        for waveform in wavetable:
            file.write("    {" + ", ".join(f"{v:.6f}" for v in waveform) + "},\n")
        file.write("};\n\n#endif // FOURIER_WAVETABLE_HPP\n")
    print(f"Saved wavetable to {filename}")

def generate(folderFourierWavetable, folderHeader, num_points=256, num_waveforms=8, max_harmonics=16, fundamental_freq=1.0):
    # Generate Fourier Wavetable
    fourier_wavetable = generate_fourier_wavetable(num_points, num_waveforms, max_harmonics, fundamental_freq)
    # Save to CSV and HPP
    save_wavetable_to_csv(fourier_wavetable, os.path.join(folderFourierWavetable, "FOURIER_WAVETABLE.csv"))
    save_wavetable_to_hpp(fourier_wavetable, os.path.join(folderHeader, "fourierWavetable.hpp"))
