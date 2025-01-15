import os
import numpy as np
from pathlib import Path

import PlotLookupTablesFromFolderInTabs as plotAll
import GenerateHeaderFile
import GenerateFourierWavetable

# Generate basic waveforms
def generate_basic_waveforms(samples=256, amplitude=1.0, harmonics=20):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    waveforms = {
        "SINE": SineWave(samples, amplitude),
        "SQUARE": SquareWave(samples, amplitude),# np.sign(np.sin(phase)),
        "SQUARE_LIMITED": LimitedSquareWave(samples, harmonics, amplitude),
        "TRIANGLE": TriangleWave(samples, amplitude),
        "SAW": SawWave(phase, amplitude),
        "RAMP": RampWave(phase, amplitude),
        "WHITE_NOISE": WhiteNoise(samples, amplitude),
        "PINK_NOISE": PinkNoise(samples, amplitude),
        "WAVE_01": mix_waveforms({"1": WhiteNoise(samples, amplitude), "2": SineWave(samples, amplitude)}, {"1": 0.05, "2": 0.95}),
        "WAVE_02": mix_waveforms({"1": SineWave(samples, amplitude), "2": PulseTrain(samples)}, {"1": 0.7, "2": 0.3}),
        "WAVE_03": mix_waveforms({"1": SineWave(samples, amplitude), "2": PulseTrain(samples, 48)}, {"1": 0.7, "2": 0.3}),
        "WAVE_04": mix_waveforms({"1": SineWave(samples, amplitude), "2": RampWave(phase, amplitude)}, {"1": 0.7, "2": 0.3}),
        "WAVE_05": mix_waveforms({"1": SquareWave(samples, amplitude), "2": PulseTrain(), "3": PolynomialWave(), "4": ExponentialDecay() * -1, "5": RampWave(phase, amplitude), "6": ExponentialDecay()}, {"1": 0.05, "2": 0.05, "3": 0.05, "4": 0.4, "5": 0.1, "6": 0.45}),
        "ADDITIVE_01": additive_waveform(samples, {1: 1.0, 2: 0.5, 3: 0.25}),
        "FOURIER_01": fourier_waveform(samples, 5),
        "ADDITIVE_02": additive_waveform(samples, {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.15, 5: 0.05}),
        "FOURIER_02": fourier_waveform(samples, 55),
        "EXPONENTIAL_DECAY_01": ExponentialDecay(samples, 5, amplitude),
        "GAUSSIAN_01": gaussian_wave(samples, 0.5, 0.1, amplitude),
        "PULSE_TRAIN_01": PulseTrain(samples, 32),
        "PULSE_TRAIN_02": PulseTrain(samples, 32*2),
        "PULSE_TRAIN_03": PulseTrain(samples, 32*3),
        "PULSE_TRAIN_04": PulseTrain(samples, 32*4),
        "POLYNOMIAL_01": PolynomialWave(samples, [0, -2, 1, -3, 1]),
        "SQUARE_WITH_DUTY_CYCLE_01": square_with_duty_cycle(samples, 0.12, amplitude),
        "SQUARE_WITH_DUTY_CYCLE_02": square_with_duty_cycle(samples, 0.24, amplitude),
        "SQUARE_WITH_DUTY_CYCLE_03": square_with_duty_cycle(samples, 0.36, amplitude),
        "SQUARE_WITH_DUTY_CYCLE_04": square_with_duty_cycle(samples, 0.48, amplitude),
        "SQUARE_WITH_DUTY_CYCLE_05": square_with_duty_cycle(samples, 0.60, amplitude),
        "SQUARE_WITH_DUTY_CYCLE_06": square_with_duty_cycle(samples, 0.72, amplitude),
        "SQUARE_WITH_DUTY_CYCLE_07": square_with_duty_cycle(samples, 0.84, amplitude),
        "SQUARE_WITH_DUTY_CYCLE_08": square_with_duty_cycle(samples, 0.96, amplitude),
    }
    return waveforms

def SineWave(samples=256, amplitude=1.0):
    return np.sin(np.linspace(0, 2 * np.pi, samples, endpoint=False)) * amplitude

def TriangleWave(samples=256, amplitude=1.0):
    phase = np.linspace(0, 1, samples, endpoint=False)  # Phase from 0 to 1
    # Shift the waveform by 90 degrees (1/4 of the period)
    triangle_wave = ((2 * np.abs(2 * ((phase + 0.25) % 1 - 0.5)) - 1) * -1) * amplitude
    return triangle_wave

def half_triangle_wave(samples=256, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    # Proper triangle wave formula
    triangle_wave = 2 * np.abs(2 * (phase / (2 * np.pi)) - np.floor(2 * (phase / (2 * np.pi)) + 0.5)) - 1
    return triangle_wave

def SquareWave(samples=256, amplitude=1.0):
    return np.sign(np.sin(np.linspace(0, 2 * np.pi, samples, endpoint=False))) * amplitude

def SawWave(phase, amplitude=1.0):
    return 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)) * amplitude

def RampWave(phase, amplitude=1.0):
    return (1 - 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)) - 1) * amplitude

def WhiteNoise(samples=256, amplitude=1.0):
    """Generates white noise with a given number of samples."""
    # Normal distribution (mean = 0, stddev = 1)
    noise = np.random.normal(0, 1, samples) * amplitude
    # Normalize to range [-1, 1]
    noise = np.clip(noise, -1, 1)
    noise = noise / np.max(np.abs(noise))  # Normalize to the maximum amplitude
    return noise

def PinkNoise(samples=256, amplitude=1.0):
    """Generates pink noise using the Voss-McCartney algorithm."""
    # Generate white noise
    wn = WhiteNoise(samples)
    
    # Apply Voss-McCartney algorithm to simulate 1/f noise
    # Create a power spectrum with decreasing amplitude as frequency increases
    # This involves adding noise in bands of decreasing power
    pk = np.zeros(samples)
    for i in range(1, samples + 1):
        pk[i - 1] = np.sum(wn[:i]) / np.sqrt(i)  # Weighted sum for 1/f

    # Normalize the result
    pk = (pk - np.mean(pk)) / np.max(np.abs(pk))
     # Normalize the result to [-1, 1]
    pk = np.clip(pk, -1, 1)  # Clip values to [-1, 1]
    pk = pk / np.max(np.abs(pk))  # Normalize the noise to the range [-1, 1]   
    return pk

# Generate "Less than Square" Wave
def LimitedSquareWave(samples=256, harmonics=20, amplitude=1.0):
    x = np.linspace(0, 1, samples, endpoint=False)
    y = np.zeros(samples)
    for n in range(1, harmonics + 1, 2):  # Odd harmonics only
        y += (1 / n) * np.sin(2 * np.pi * n * x) * amplitude
    y /= np.max(np.abs(y))  # Normalize to [-1, 1]
    return y

# Create additive synthesis waveforms
def additive_waveform(samples=256, harmonics=1, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    waveform = np.zeros_like(phase)
    for harmonic, amp in harmonics.items():
        waveform += amp * np.sin(harmonic * phase)
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

# Create Fourier synthesis waveforms
def fourier_waveform(samples=256, harmonics=1, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    waveform = np.zeros_like(phase)
    for n in range(1, harmonics + 1):
        waveform += (1 / n) * np.sin(n * phase)  # Harmonics with decreasing amplitude
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

def ExponentialDecay(samples=256, decay_rate=5, amplitude=1.0):
    """Generates an exponentially decaying waveform."""
    x = np.linspace(0, 1, samples) * amplitude
    return np.exp(-decay_rate * x) * 2 - 1

def gaussian_wave(samples=256, mean=0.5, stddev=0.1, amplitude=1.0):
    """Generates a Gaussian waveform."""
    x = np.linspace(0, 1, samples) * amplitude
    return np.exp(-0.5 * ((x - mean) / stddev) ** 2) * 2 - 1

def PulseTrain(samples=256, period=32):
    """Generates a periodic pulse train."""
    return np.tile([1] * (period // 2) + [-1] * (period // 2), samples // period + 1)[:samples]

def PolynomialWave(samples=256, coefficients=[1, -2, 1], amplitude=1.0):
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

def square_with_duty_cycle(samples, duty_cycle=0.5, amplitude=1.0):
    """Generates a square wave with a custom duty cycle."""
    x = np.linspace(0, 1, samples, endpoint=False) * amplitude
    return np.where((x % 1) < duty_cycle, 1, -1)

# Mix multiple waveforms
def mix_waveforms(waveforms, weights):
    mixed_waveform = np.zeros_like(next(iter(waveforms.values())))
    for name, weight in weights.items():
        mixed_waveform += weight * waveforms[name]
    return mixed_waveform / np.max(np.abs(mixed_waveform))  # Normalize to [-1, 1]

# Method: Fourier Waveform
def random_fourier_wave(samples, harmonics):
    x = np.linspace(0, 1, samples, endpoint=False)
    y = np.zeros(samples)
    random_amplitudes = np.random.uniform(-1, 1, harmonics)  # Random amplitudes
    random_phases = np.random.uniform(0, 2 * np.pi, harmonics)  # Random phases
    for n in range(1, harmonics + 1):
        y += random_amplitudes[n - 1] * np.sin(2 * np.pi * n * x + random_phases[n - 1])
    y /= np.max(np.abs(y))  # Normalize to [-1, 1]
    return y, {"Amplitudes": random_amplitudes, "Phases": random_phases}

# Method: Additive Synthesis Waveform
def random_additive_wave(samples, harmonics):
    x = np.linspace(0, 1, samples, endpoint=False)
    y = np.zeros(samples)
    random_amplitudes = np.random.uniform(0, 1, harmonics)  # Random amplitudes
    for n in range(1, harmonics + 1):
        y += random_amplitudes[n - 1] * np.sin(2 * np.pi * n * x)
    y /= np.max(np.abs(y))  # Normalize to [-1, 1]
    return y, {"Amplitudes": random_amplitudes}

# Method: Polynomial Waveform
def random_polynomial_wave(samples, max_degree):
    coefficients = np.random.uniform(-1, 1, max_degree + 1)  # Random coefficients
    x = np.linspace(-1, 1, samples)
    y = np.polyval(coefficients, x)
    y -= np.polyval(coefficients, 0)  # Shift so waveform starts at 0
    y /= np.max(np.abs(y))  # Normalize to [-1, 1]
    return y, {"Coefficients": coefficients}

# Method: Mix Waveforms
def mix_random_waveforms(waveform1, waveform2, weight1=0.5, weight2=0.5):
    """
    Mix two waveforms with given weights.
    - waveform1: First waveform values.
    - waveform2: Second waveform values.
    - weight1: Weight for the first waveform (default 0.5).
    - weight2: Weight for the second waveform (default 0.5).
    """
    mixed = weight1 * waveform1 + weight2 * waveform2
    mixed /= np.max(np.abs(mixed))  # Normalize to [-1, 1]
    return mixed

def RandomWaveform(outputFolder, samples=256, iterations=10):
    for i in range(iterations):
        # Randomly choose a waveform type
        waveform_type = np.random.choice(['Fourier', 'Additive', 'Polynomial'])
        
        print(f"Iteration {i+1}: Generating {waveform_type} waveform")
        
        if waveform_type == 'Fourier':
            harmonics = np.random.randint(3, 8)  # Random number of harmonics
            y, params = random_fourier_wave(samples, harmonics)
            filename = os.path.join(outputFolder, f"random_fourier_wave_{i+1}.csv")
            print(f"  Parameters: {params}")
        
        elif waveform_type == 'Additive':
            harmonics = np.random.randint(3, 8)  # Random number of harmonics
            y, params = random_additive_wave(samples, harmonics)
            filename = os.path.join(outputFolder, f"random_additive_wave_{i+1}.csv")
            print(f"  Parameters: {params}")
        
        elif waveform_type == 'Polynomial':
            max_degree = np.random.randint(2, 6)  # Random polynomial degree
            y, params = random_polynomial_wave(samples, max_degree)
            filename = os.path.join(outputFolder, f"random_polynomial_wave_{i+1}.csv")
            print(f"  Parameters: {params}")
        
        # Save to CSV
        SaveWaveformToCsv(y, filename)

def RandomMixWaveform(outputFolder, samples=256, iterations=10):
    for i in range(iterations):
        print(f"\nIteration {i+1}: Generating waveforms...")
        
        # Generate two random waveforms
        waveform1_type = np.random.choice(['Fourier', 'Additive', 'Polynomial'])
        waveform2_type = np.random.choice(['Fourier', 'Additive', 'Polynomial'])
        
        # Generate Waveform 1
        if waveform1_type == 'Fourier':
            harmonics = np.random.randint(3, 8)
            waveform1, params1 = random_fourier_wave(samples, harmonics)
        elif waveform1_type == 'Additive':
            harmonics = np.random.randint(3, 8)
            waveform1, params1 = random_additive_wave(samples, harmonics)
        elif waveform1_type == 'Polynomial':
            max_degree = np.random.randint(2, 6)
            waveform1, params1 = random_polynomial_wave(samples, max_degree)
        
        # Generate Waveform 2
        if waveform2_type == 'Fourier':
            harmonics = np.random.randint(3, 8)
            waveform2, params2 = random_fourier_wave(samples, harmonics)
        elif waveform2_type == 'Additive':
            harmonics = np.random.randint(3, 8)
            waveform2, params2 = random_additive_wave(samples, harmonics)
        elif waveform2_type == 'Polynomial':
            max_degree = np.random.randint(2, 6)
            waveform2, params2 = random_polynomial_wave(samples, max_degree)
        
        # Save individual waveforms
        SaveWaveformToCsv(waveform1, os.path.join(outputFolder, f"waveform1_{i+1}.csv"))
        SaveWaveformToCsv(waveform2, os.path.join(outputFolder, f"waveform2_{i+1}.csv"))
        
        # Mix the two waveforms
        mixed_wave = mix_waveforms({"1": waveform1, "2": waveform2}, {"1": 0.7, "2": 0.3})
        SaveWaveformToCsv(mixed_wave, os.path.join(outputFolder, f"mixed_wave_{i+1}.csv"))
        
        # Print parameters
        print(f"  Waveform 1 ({waveform1_type}): {params1}")
        print(f"  Waveform 2 ({waveform2_type}): {params2}")
        print("  Mixed waveform saved.")    

def GenerateProgressiveWavetable(base_waveform, num_waveforms=8):
    """
    Generate a progressive wavetable by gradually transforming a base waveform.
    
    Parameters:
        base_waveform (np.ndarray): Original waveform to transform.
        num_waveforms (int): Number of variations to generate.
        
    Returns:
        np.ndarray: 2D array representing the wavetable.
    """
    length = len(base_waveform)
    wavetable = []

    for i in range(num_waveforms):
        # Progressively modify the waveform
        # Example transformations: smoothing, adding harmonics, scaling, etc.
        factor = i / (num_waveforms - 1)
        transformed_waveform = base_waveform * (1 - factor) + np.sin(2 * np.pi * np.linspace(0, 1, length)) * factor
        
        # Normalize to [-1, 1]
        transformed_waveform /= np.max(np.abs(transformed_waveform))
        wavetable.append(transformed_waveform)
    
    return np.array(wavetable)

def GenerateCustomProgressiveWavetable(base_waveform, num_waveforms=8, transformation="harmonics"):
    """
    Generate a progressive wavetable with custom transformations.
    
    Parameters:
        base_waveform (np.ndarray): Original waveform to transform.
        num_waveforms (int): Number of variations to generate.
        transformation (str): Type of transformation ("harmonics", "smoothing", "scaling").
        
    Returns:
        np.ndarray: 2D array representing the wavetable.
    """
    length = len(base_waveform)
    wavetable = []

    for i in range(num_waveforms):
        factor = i / (num_waveforms - 1)  # Progression factor (0 to 1)
        if transformation == "harmonics":
            # Add or remove harmonics progressively
            additional_wave = np.sin(2 * np.pi * np.linspace(0, 1, length) * (2 + int(factor * 4)))
            transformed_waveform = base_waveform + additional_wave * factor
        
        elif transformation == "smoothing":
            # Smooth waveform progressively using a moving average
            window_size = int(5 + factor * (length // 8))  # Larger window for more smoothing
            smoothed_waveform = np.convolve(base_waveform, np.ones(window_size) / window_size, mode='same')
            transformed_waveform = smoothed_waveform
        
        elif transformation == "scaling":
            return
            # Gradually reduce or increase amplitude
            scale_up = 1 + factor  # Amplitude increases progressively
            scale_down = 1 - factor  # Amplitude decreases progressively
            transformed_waveform = base_waveform * (scale_down if factor < 0.5 else scale_up)
        
        else:
            raise ValueError("Invalid transformation type. Choose 'harmonics', 'smoothing', or 'scaling'.")
        
        # Normalize to [-1, 1]
        transformed_waveform /= np.max(np.abs(transformed_waveform))
        wavetable.append(transformed_waveform)

    return np.array(wavetable)

def GenerateWaveformProgression(waveform1, waveform2, num_steps=8):
    """
    Generate a progressive wavetable transitioning from one waveform to another.
    
    Parameters:
        waveform1 (np.ndarray): Starting waveform.
        waveform2 (np.ndarray): Ending waveform.
        num_steps (int): Number of progressive steps between the waveforms.
        
    Returns:
        np.ndarray: 2D array representing the progression wavetable.
    """
    if len(waveform1) != len(waveform2):
        raise ValueError("Waveform lengths must be the same.")
    
    wavetable = []
    for i in range(num_steps):
        factor = i / (num_steps - 1)  # Progression factor (0 to 1)
        # Linear interpolation between the two waveforms
        interpolated_waveform = (1 - factor) * waveform1 + factor * waveform2
        # Normalize to [-1, 1]
        interpolated_waveform /= np.max(np.abs(interpolated_waveform))
        wavetable.append(interpolated_waveform)
    
    return np.array(wavetable)

def CreateWavetableFromWaveforms(waveforms):
    """
    Create a wavetable from multiple waveforms.
    
    Parameters:
        waveforms (list of np.ndarray): List of waveforms, each with the same number of samples.
        
    Returns:
        np.ndarray: 2D array representing the wavetable, where each row is a waveform.
    """
    if len(waveforms) == 0:
        raise ValueError("The waveforms list cannot be empty.")
    
    # Ensure all waveforms have the same number of samples
    length = len(waveforms[0])
    if not all(len(waveform) == length for waveform in waveforms):
        raise ValueError("All waveforms must have the same number of samples.")
    
    # Stack waveforms into a 2D array
    wavetable = np.stack(waveforms)
    
    # Normalize each waveform to the range [-1, 1]
    wavetable = np.array([waveform / np.max(np.abs(waveform)) for waveform in wavetable])
    
    return wavetable

def CreateWavetableFromCSVFiles(folder_path):
    """
    Create a wavetable from multiple CSV files in a folder.
    
    Parameters:
        folder_path (str): Path to the folder containing CSV files.
        
    Returns:
        np.ndarray: 2D array representing the wavetable.
    """
    # List CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV files found in the specified folder.")
    
    waveforms = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        waveform = np.loadtxt(file_path, delimiter=",")
        
        # Normalize waveform to [-1, 1]
        waveform /= np.max(np.abs(waveform))
        waveforms.append(waveform)
    
    # Ensure all waveforms have the same length
    length = len(waveforms[0])
    if not all(len(waveform) == length for waveform in waveforms):
        raise ValueError("All waveforms in CSV files must have the same number of samples.")
    
    # Stack waveforms into a 2D array
    wavetable = np.stack(waveforms)
    return wavetable

# Save lookup table to a CSV file
def SaveWaveformToCsv(waveform, filename):
    np.savetxt(filename, waveform, delimiter=",", fmt="%.6f")
    print(f"Saved waveform to {filename}")

# Function: Save Wavetable to CSV
def SaveWavetableToCsv(wavetable, filename):
    np.savetxt(filename, wavetable, delimiter=",", header="", comments="")
    print(f"Saved wavetable to {filename}")

def SaveBasicWaveforms(waveforms, folder):
    for name, waveform in waveforms.items():
        SaveWaveformToCsv(waveform, f"{folder + name}.csv")

# Function: Save Wavetable to C Header File
def SaveWavetableToHeader(wavetable, filename, name="WaveTable"):
    with open(filename, "w") as file:
        file.write(f"#ifndef {name.upper()}_HPP\n#define {name.upper()}_HPP\n\n")
        file.write(f"// Wavetable generated with {len(wavetable)} waveforms, {wavetable.shape[1]} samples each\n\n")
        file.write(f"constexpr float {name}[8][256] = {{\n")
        for waveform in wavetable:
            file.write("    {" + ", ".join(f"{v:.6f}" for v in waveform) + "},\n")
        file.write(f"}};\n\n#endif // {name.upper()}_HPP\n")
    print(f"Saved wavetable to {filename}")
    
# Example Usage
if __name__ == "__main__":

    appFolder = Path(__file__).parent.absolute()
    folderHeader = f"{appFolder}\\headerFile\\"
    if not os.path.exists(folderHeader):
        os.makedirs(folderHeader)
    folder = f"{appFolder}\\lookupTables\\"
    if not os.path.exists(folder):
        os.makedirs(folder)
    folderBasicWaveforms = f"{folder}basicWaveforms\\"
    if not os.path.exists(folderBasicWaveforms):
        os.makedirs(folderBasicWaveforms)
    folderRandomWaveforms = f"{folder}randomWaveforms\\"
    if not os.path.exists(folderRandomWaveforms):
        os.makedirs(folderRandomWaveforms)
    # folderAdditiveWavetable = f"{folder}additiveWavetable\\"
    # if not os.path.exists(folderAdditiveWavetable):
    #     os.makedirs(folderAdditiveWavetable)
    # folderFourierWaveforms = f"{folder}fourierWaveforms\\"
    # if not os.path.exists(folderFourierWaveforms):
    #     os.makedirs(folderFourierWaveforms)
    folderWavetables = f"{folder}wavetables\\"
    if not os.path.exists(folderWavetables):
        os.makedirs(folderWavetables)

    samples = 256

    # Generate basic waveforms
    basic_waveforms = generate_basic_waveforms(samples)
    SaveBasicWaveforms(basic_waveforms, folderBasicWaveforms)

    n = "WT_001"
    wt = CreateWavetableFromWaveforms([basic_waveforms["SINE"],
                                       basic_waveforms["TRIANGLE"],
                                       basic_waveforms["RAMP"],
                                       basic_waveforms["SAW"],
                                       basic_waveforms["SQUARE"],
                                       basic_waveforms["SQUARE_LIMITED"],
                                       basic_waveforms["WHITE_NOISE"],
                                       basic_waveforms["PINK_NOISE"]
                                       ])
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_002"
    wt = CreateWavetableFromWaveforms([basic_waveforms["ADDITIVE_01"],
                                       basic_waveforms["ADDITIVE_02"],
                                       basic_waveforms["FOURIER_01"],
                                       basic_waveforms["FOURIER_02"],
                                       basic_waveforms["WAVE_02"],
                                       basic_waveforms["WAVE_03"],
                                       basic_waveforms["WAVE_04"],
                                       basic_waveforms["WAVE_05"]
                                       ])
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_011"
    wt = GenerateProgressiveWavetable(basic_waveforms["SQUARE"])
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_012"
    wt = GenerateProgressiveWavetable(basic_waveforms["SQUARE_LIMITED"])
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_013"
    wt = GenerateProgressiveWavetable(basic_waveforms["SAW"])
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_014"
    wt = GenerateCustomProgressiveWavetable(basic_waveforms["SQUARE"], transformation="harmonics")
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_015"
    wt = GenerateCustomProgressiveWavetable(basic_waveforms["SQUARE_LIMITED"], transformation="harmonics")
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_016"
    wt = GenerateCustomProgressiveWavetable(basic_waveforms["SAW"], transformation="harmonics")
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_017"
    wt = GenerateCustomProgressiveWavetable(basic_waveforms["WAVE_02"], transformation="harmonics")
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    # n = "WT_018"
    # wt = GenerateCustomProgressiveWavetable(basic_waveforms["SQUARE"], transformation="smoothing")
    # SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    # n = "WT_019"
    # wt = GenerateCustomProgressiveWavetable(basic_waveforms["SQUARE_LIMITED"], transformation="smoothing")
    # SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_020"
    wt = GenerateCustomProgressiveWavetable(basic_waveforms["SAW"], transformation="smoothing")
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_021"
    wt = GenerateCustomProgressiveWavetable(basic_waveforms["WAVE_02"], transformation="smoothing")
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    # n = "WT_022"
    # wt = GenerateWaveformProgression(basic_waveforms["SQUARE"], basic_waveforms["SQUARE_LIMITED"])
    # SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_023"
    wt = GenerateWaveformProgression(basic_waveforms["SQUARE_LIMITED"], basic_waveforms["SAW"])
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_024"
    wt = GenerateWaveformProgression(basic_waveforms["SAW"], basic_waveforms["FOURIER_01"])
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_025"
    wt = GenerateWaveformProgression(basic_waveforms["WAVE_02"], basic_waveforms["WAVE_03"])
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_026"
    wt = CreateWavetableFromWaveforms([basic_waveforms["SQUARE_WITH_DUTY_CYCLE_01"],
                                       basic_waveforms["SQUARE_WITH_DUTY_CYCLE_02"],
                                       basic_waveforms["SQUARE_WITH_DUTY_CYCLE_03"],
                                       basic_waveforms["SQUARE_WITH_DUTY_CYCLE_04"],
                                       basic_waveforms["SQUARE_WITH_DUTY_CYCLE_05"],
                                       basic_waveforms["SQUARE_WITH_DUTY_CYCLE_06"],
                                       basic_waveforms["SQUARE_WITH_DUTY_CYCLE_07"],
                                       basic_waveforms["SQUARE_WITH_DUTY_CYCLE_08"]
                                       ])
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    n = "WT_027"
    wt = CreateWavetableFromCSVFiles(folder + "forWavetable\\")
    SaveWavetableToCsv(wt, f"{folderWavetables}{n}.csv")
    SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    
    # RandomWaveform(folderRandomWaveforms)
    # RandomMixWaveform(folderRandomWaveforms)

    # UNCOMMENT FOR GENERATE DIFFERENTS FOURIER WAVETABLE
    # Parameters
    # num_waveforms = 8  # Number of waveforms in the wavetable
    # max_harmonics = 16  # Maximum number of harmonics
    # generateFourierWavetable.generate(folderFourierWavetable, folderHeader, num_points, num_waveforms, max_harmonics)

    plotAll.plot_lookup_tables_in_tabs(folder)

