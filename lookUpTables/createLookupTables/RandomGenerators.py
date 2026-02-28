import numpy as np
import Distortion as dist

# Method: Fourier Waveform
def RandomFourierWave(samples, harmonics, distortion_type="clipping"):
    x = np.linspace(0, 1, samples, endpoint=False)
    y = np.zeros(samples)
    random_amplitudes = np.random.uniform(-1, 1, harmonics)  # Random amplitudes
    random_phases = np.random.uniform(0, 2 * np.pi, harmonics)  # Random phases
    for n in range(1, harmonics + 1):
        y += random_amplitudes[n - 1] * np.sin(2 * np.pi * n * x + random_phases[n - 1])
    y = dist.ApplyDistortion(y, distortion_type)
    y /= np.max(np.abs(y))  # Normalize to [-1, 1]
    # apply distortion
    return y, {"Amplitudes": random_amplitudes, "Phases": random_phases}

# Method: Additive Synthesis Waveform
def RandomAdditiveWave(samples, harmonics, distortion_type="clipping"):
    x = np.linspace(0, 1, samples, endpoint=False)
    y = np.zeros(samples)
    random_amplitudes = np.random.uniform(0, 1, harmonics)  # Random amplitudes
    for n in range(1, harmonics + 1):
        y += random_amplitudes[n - 1] * np.sin(2 * np.pi * n * x)
    # apply distortion
    y = dist.ApplyDistortion(y, distortion_type)
    y /= np.max(np.abs(y))  # Normalize to [-1, 1]
    return y, {"Amplitudes": random_amplitudes}

# Method: Polynomial Waveform
def RandomPolynomialWave(samples, max_degree, distortion_type="clipping"):
    coefficients = np.random.uniform(-1, 1, max_degree + 1)  # Random coefficients
    x = np.linspace(-1, 1, samples)
    y = np.polyval(coefficients, x)
    y -= np.polyval(coefficients, 0)  # Shift so waveform starts at 0
    # apply distortion
    y = dist.ApplyDistortion(y, distortion_type)
    y /= np.max(np.abs(y))  # Normalize to [-1, 1]
    return y, {"Coefficients": coefficients}

# Method: Mix Waveforms
def MixRandomWaveforms(waveform1, waveform2, weight1=0.5, weight2=0.5, distortion_type="clipping"):
    """
    Mix two waveforms with given weights.
    - waveform1: First waveform values.
    - waveform2: Second waveform values.
    - weight1: Weight for the first waveform (default 0.5).
    - weight2: Weight for the second waveform (default 0.5).
    """
    mixed = weight1 * waveform1 + weight2 * waveform2
    mixed /= np.max(np.abs(mixed))  # Normalize to [-1, 1]
    # apply distortion
    mixed = dist.ApplyDistortion(mixed, distortion_type)
    return mixed

