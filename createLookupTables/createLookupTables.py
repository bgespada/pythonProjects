import os
import numpy as np
from pathlib import Path

# Generate basic waveforms
def generate_basic_waveforms(num_points=256, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    waveforms = {
        "SINE": generate_sine_wave(num_points, amplitude),
        "SQUARE": generate_square_wave(num_points, amplitude),# np.sign(np.sin(phase)),
        "TRIANGLE": generate_triangle_wave(num_points, amplitude),
        "SAW": generate_saw_wave(phase, amplitude),
        "RAMP": generate_ramp_wave(phase, amplitude),
        "WHITE_NOISE": generate_white_noise(num_points, amplitude),
        "PINK_NOISE": generate_pink_noise(num_points, amplitude),
    }
    # for key in waveforms:
    #     waveforms[key][0] = 0  # Ensure the first value is 0
    return waveforms

def generate_sine_wave(samples=256, amplitude=1.0):
    return np.sin(np.linspace(0, 2 * np.pi, samples, endpoint=False)) * amplitude

def generate_triangle_wave(samples=256, amplitude=1.0):
    phase = np.linspace(0, 1, samples, endpoint=False)  # Phase from 0 to 1
    # Shift the waveform by 90 degrees (1/4 of the period)
    triangle_wave = ((2 * np.abs(2 * ((phase + 0.25) % 1 - 0.5)) - 1) * -1) * amplitude
    return triangle_wave

def generate_half_triangle_wave(samples=256, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    # Proper triangle wave formula
    triangle_wave = 2 * np.abs(2 * (phase / (2 * np.pi)) - np.floor(2 * (phase / (2 * np.pi)) + 0.5)) - 1
    return triangle_wave

def generate_square_wave(samples=256, amplitude=1.0):
    return np.sign(np.sin(np.linspace(0, 2 * np.pi, samples, endpoint=False))) * amplitude

def generate_saw_wave(phase, amplitude=1.0):
    return 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)) * amplitude

def generate_ramp_wave(phase, amplitude=1.0):
    return (1 - 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)) - 1) * amplitude

def generate_white_noise(samples=256, amplitude=1.0):
    """Generates white noise with a given number of samples."""
    # Normal distribution (mean = 0, stddev = 1)
    noise = np.random.normal(0, 1, samples) * amplitude
    # Normalize to range [-1, 1]
    noise = np.clip(noise, -1, 1)
    noise = noise / np.max(np.abs(noise))  # Normalize to the maximum amplitude
    return noise

def generate_pink_noise(samples=256, amplitude=1.0):
    """Generates pink noise using the Voss-McCartney algorithm."""
    # Generate white noise
    white_noise = generate_white_noise(samples)
    
    # Apply Voss-McCartney algorithm to simulate 1/f noise
    # Create a power spectrum with decreasing amplitude as frequency increases
    # This involves adding noise in bands of decreasing power
    pink_noise = np.zeros(samples)
    for i in range(1, samples + 1):
        pink_noise[i - 1] = np.sum(white_noise[:i]) / np.sqrt(i)  # Weighted sum for 1/f

    # Normalize the result
    pink_noise = (pink_noise - np.mean(pink_noise)) / np.max(np.abs(pink_noise))
     # Normalize the result to [-1, 1]
    pink_noise = np.clip(pink_noise, -1, 1)  # Clip values to [-1, 1]
    pink_noise = pink_noise / np.max(np.abs(pink_noise))  # Normalize the noise to the range [-1, 1]   
    return pink_noise

# Create additive synthesis waveforms
def create_additive_waveform(harmonics, samples=256, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    waveform = np.zeros_like(phase)
    for harmonic, amp in harmonics.items():
        waveform += amp * np.sin(harmonic * phase)
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

# Create Fourier synthesis waveforms
def create_fourier_waveform(num_harmonics, samples=256, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    waveform = np.zeros_like(phase)
    for n in range(1, num_harmonics + 1):
        waveform += (1 / n) * np.sin(n * phase)  # Harmonics with decreasing amplitude
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

def generate_exponential_decay(samples=256, decay_rate=5, amplitude=1.0):
    """Generates an exponentially decaying waveform."""
    x = np.linspace(0, 1, samples) * amplitude
    return np.exp(-decay_rate * x) * 2 - 1

def generate_gaussian_wave(samples=256, mean=0.5, stddev=0.1, amplitude=1.0):
    """Generates a Gaussian waveform."""
    x = np.linspace(0, 1, samples) * amplitude
    return np.exp(-0.5 * ((x - mean) / stddev) ** 2) * 2 - 1

def generate_pulse_train(samples=256, period=32):
    """Generates a periodic pulse train."""
    return np.tile([1] * (period // 2) + [-1] * (period // 2), samples // period + 1)[:samples]

def generate_polynomial_wave(samples=256, coefficients=[1, -2, 1], amplitude=1.0):
    """Generates a waveform based on a polynomial equation."""
    # Ensure coefficients are passed as a 1D array
    coefficients = np.array(coefficients)
    
    # Create an array of x values between -1 and 1
    x = np.linspace(-1, 1, samples)
    
    # Apply the polynomial to the x values using np.polyval
    y = np.polyval(coefficients, x)  # Apply polynomial evaluation
    
    # Subtract the value at x = 0 to shift the waveform to start at 0
    y -= np.polyval(coefficients, 0)  # Subtract the polynomial's value at x = 0
    
    # Normalize the output to [-1, 1] if needed
    y = np.clip(y, -1, 1)
    y = y / np.max(np.abs(y))  # Normalize to [-1, 1] range    
    return y

def generate_square_with_duty_cycle(samples, duty_cycle=0.5, amplitude=1.0):
    """Generates a square wave with a custom duty cycle."""
    x = np.linspace(0, 1, samples, endpoint=False) * amplitude
    return np.where((x % 1) < duty_cycle, 1, -1)

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
    additive_waveform_01 = create_additive_waveform({1: 1.0, 2: 0.5, 3: 0.25}, num_points)
    # Create a Fourier synthesis waveform with 5 harmonics
    fourier_waveform_01 = create_fourier_waveform(5, num_points)

    exponential_decay_waveform_01 = generate_exponential_decay(num_points)
    gaussian_waveform_01 = generate_gaussian_wave(num_points)
    pulse_train_wavwform_01 = generate_pulse_train(num_points)
    polynomial_wave_waveform_01 = generate_polynomial_wave(num_points, [0,-2,1, -3,1])
    square_with_duty_cycle_01 = generate_square_with_duty_cycle(num_points)

    # Mix waveforms (e.g., sine and sawtooth)
    mixed_waveform_01 = mix_waveforms(
        {"1": basic_waveforms["SINE"], "2": basic_waveforms["SAW"]},
        {"1": 0.7, "2": 0.3}
    )
    mixed_waveform_02 = mix_waveforms(
        {"1": basic_waveforms["SINE"], "2": pulse_train_wavwform_01},
        {"1": 0.7, "2": 0.3}
    )
    mixed_waveform_03 = mix_waveforms(
        {"1": basic_waveforms["SINE"], "2": basic_waveforms["RAMP"]},
        {"1": 0.7, "2": 0.3}
    )
    mixed_waveform_04 = mix_waveforms(
        {
            "1": basic_waveforms["SQUARE"],
            "2": pulse_train_wavwform_01,
            "3": polynomial_wave_waveform_01,
            "4": exponential_decay_waveform_01 * -1,
            "5": basic_waveforms["RAMP"],
            "6": exponential_decay_waveform_01
        },
        {
            "1": 0.05,
            "2": 0.05,
            "3": 0.05,
            "4": 0.4,
            "5": 0.1,
            "6": 0.45
        }
    )

    # Save all waveforms to CSV
    appFolder = Path(__file__).parent.absolute()
    folder = f"{appFolder}\\lookupTables\\"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for name, waveform in basic_waveforms.items():
        save_waveform_to_csv(waveform, f"{folder + name}.csv")
    save_waveform_to_csv(additive_waveform_01, f"{folder}ADDITIVE_01.csv")
    save_waveform_to_csv(fourier_waveform_01, f"{folder}FOURIER_01.csv")
    save_waveform_to_csv(exponential_decay_waveform_01, f"{folder}EXPONENTIAL_01.csv")
    save_waveform_to_csv(exponential_decay_waveform_01 * -1, f"{folder}EXPONENTIAL_02.csv")

    save_waveform_to_csv(gaussian_waveform_01, f"{folder}GAUSSIAN_01.csv")
    save_waveform_to_csv(pulse_train_wavwform_01, f"{folder}PULSE_TRAIN_01.csv")
    save_waveform_to_csv(polynomial_wave_waveform_01, f"{folder}POLYNOMIAL_01.csv")
    save_waveform_to_csv(square_with_duty_cycle_01, f"{folder}SQR_DC_01.csv")

    save_waveform_to_csv(mixed_waveform_01, f"{folder}MIXED_01.csv")
    save_waveform_to_csv(mixed_waveform_02, f"{folder}MIXED_02.csv")
    save_waveform_to_csv(mixed_waveform_03, f"{folder}MIXED_03.csv")
    save_waveform_to_csv(mixed_waveform_04, f"{folder}MIXED_04.csv")
