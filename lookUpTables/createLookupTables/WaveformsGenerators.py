import numpy as np
import Distortion as Dist

SAMPLES = 1024

# Helper: Generate an envelope
def envelope(signal, attack, decay, sustain_level, sustain_time, release, sample_rate):
    """
    Apply an ADSR envelope to a signal.
    """
    total_samples = len(signal)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    sustain_samples = int(sustain_time * sample_rate)
    release_samples = total_samples - (attack_samples + decay_samples + sustain_samples)

    if release_samples < 0:
        release_samples = 0

    env = np.zeros(total_samples)
    env[:attack_samples] = np.linspace(0, 1, attack_samples)
    env[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
    env[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain_level
    env[attack_samples + decay_samples + sustain_samples:] = np.linspace(sustain_level, 0, release_samples)

    return signal * env

# Helper: Generate noise
def generate_noise(length, color="white"):
    """
    Generate noise (white or pink).
    """
    if color == "white":
        return np.random.uniform(-1, 1, length)
    elif color == "pink":
        # Pink noise: 1/f distribution
        white = np.random.normal(0, 1, length)
        b, a = [0.02109238, 0.07113478, 0.68873558, -0.41499378, -0.05722678], [1, -1.73472577, 1.05249157, -0.26710995, 0.01588273]
        return np.convolve(white, b, mode='same')

def SineWave(samples=SAMPLES, amplitude=1.0):
    w = np.sin(np.linspace(0, 2 * np.pi, samples, endpoint=False)) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

def TriangleWave(samples=SAMPLES, amplitude=1.0):
    phase = np.linspace(0, 1, samples, endpoint=False)  # Phase from 0 to 1
    # Shift the waveform by 90 degrees (1/4 of the period)
    w = ((2 * np.abs(2 * ((phase + 0.25) % 1 - 0.5)) - 1) * -1) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

def half_triangle_wave(samples=SAMPLES, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    # Proper triangle wave formula
    w = 2 * np.abs(2 * (phase / (2 * np.pi)) - np.floor(2 * (phase / (2 * np.pi)) + 0.5)) - 1
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

def SquareWave(samples=SAMPLES, amplitude=1.0):
    w = np.sign(np.sin(np.linspace(0, 2 * np.pi, samples, endpoint=False))) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

def SawWave(phase, amplitude=1.0):
    w = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

def RampWave(phase, amplitude=1.0):
    w = (1 - 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5)) - 1) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

def WhiteNoise(samples=SAMPLES, amplitude=1.0):
    """Generates white noise with a given number of samples."""
    # Normal distribution (mean = 0, stddev = 1)
    noise = np.random.normal(0, 1, samples) * amplitude
    # Normalize to range [-1, 1]
    noise = np.clip(noise, -1, 1)
    noise = noise / np.max(np.abs(noise))  # Normalize to the maximum amplitude
    return noise

def PinkNoise(samples=SAMPLES, amplitude=1.0):
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
    pk = (pk - np.mean(pk)) / np.max(np.abs(pk)) * amplitude
     # Normalize the result to [-1, 1]
    pk = np.clip(pk, -1, 1)  # Clip values to [-1, 1]
    pk = pk / np.max(np.abs(pk))  # Normalize the noise to the range [-1, 1]   
    return pk

# Generate "Less than Square" Wave
def LimitedSquareWave(samples=SAMPLES, harmonics=20, amplitude=1.0):
    x = np.linspace(0, 1, samples, endpoint=False)
    y = np.zeros(samples)
    for n in range(1, harmonics + 1, 2):  # Odd harmonics only
        y += (1 / n) * np.sin(2 * np.pi * n * x) * amplitude
    y /= np.max(np.abs(y))  # Normalize to [-1, 1]
    return y

# Create additive synthesis waveforms
def AdditiveWaveform(samples=SAMPLES, harmonics=1, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    waveform = np.zeros_like(phase)
    for harmonic, amp in harmonics.items():
        waveform += amp * np.sin(harmonic * phase)
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

# Create Fourier synthesis waveforms
def FourierWaveform(samples=SAMPLES, harmonics=1, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    waveform = np.zeros_like(phase)
    for n in range(1, harmonics + 1):
        waveform += (1 / n) * np.sin(n * phase)  # Harmonics with decreasing amplitude
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

# Create Fourier synthesis square wave
def FourierSquareWave(samples=SAMPLES, harmonics=1, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    waveform = np.zeros_like(phase)
    for n in range(1, harmonics * 2, 2):  # Only odd harmonics
        waveform += (1 / n) * np.sin(n * phase)  # Add scaled sine wave of the harmonic
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

# Create Fourier synthesis triangle wave
def FourierTriangleWave(samples=SAMPLES, harmonics=1, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False) * amplitude
    waveform = np.zeros_like(phase)
    for n in range(1, harmonics * 2, 2):  # Only odd harmonics
        sign = (-1) ** ((n - 1) // 2)  # Alternating sign
        waveform += sign * (1 / n**2) * np.sin(n * phase)  # Add scaled sine wave
    return waveform / np.max(np.abs(waveform))  # Normalize to [-1, 1]

def ExponentialDecay(samples=SAMPLES, decay_rate=5, amplitude=1.0):
    """Generates an exponentially decaying waveform."""
    x = np.linspace(0, 1, samples) * amplitude
    w = np.exp(-decay_rate * x) * 2 - 1
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

def Gaussian(samples=SAMPLES, mean=0.5, stddev=0.1, amplitude=1.0):
    """Generates a Gaussian waveform."""
    x = np.linspace(0, 1, samples)
    w = (np.exp(-0.5 * ((x - mean) / stddev) ** 2) * 2 - 1) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

def PulseTrain(samples=SAMPLES, period=32):
    """Generates a periodic pulse train."""
    return np.tile([1] * (period // 2) + [-1] * (period // 2), samples // period + 1)[:samples]

def PolynomialWave(samples=SAMPLES, coefficients=[1, -2, 1], amplitude=1.0):
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
    y = np.clip(y * amplitude, -1, 1)
    y = y / np.max(np.abs(y))  # Normalize to [-1, 1] range    
    return y

def SquareWithDutyCycle(samples=SAMPLES, duty_cycle=0.5, amplitude=1.0):
    """Generates a square wave with a custom duty cycle."""
    x = np.linspace(0, 1, samples, endpoint=False) * amplitude
    return np.where((x % 1) < duty_cycle, 1, -1)

# 1. Triangle-Saw Hybrid Wave
def TriangleSaw(samples=SAMPLES, mix=0.5, amplitude=1.0):
    """
    Generate a hybrid wave blending triangle and sawtooth.
    
    Parameters:
        samples (int): Number of samples in the waveform.
        mix (float): Mix ratio (0.0 = pure triangle, 1.0 = pure sawtooth).
    
    Returns:
        np.ndarray: Hybrid waveform of exactly `samples` size.
    """
    # Generate a time array of `samples` points, strictly between 0 and 1
    t = np.linspace(0, 1, samples, endpoint=False)
    # Create a triangle wave
    triangle = 2 * np.abs(2 * (t - np.floor(t + 0.5))) - 1
    # Create a sawtooth wave
    sawtooth = 2 * (t - np.floor(t)) - 1   
    # Blend the waves based on `mix` value
    w = (1 - mix) * (triangle * amplitude) + mix * (sawtooth * amplitude)
    return w

# 2. Half-Sine Wave
def HalfSine(samples=SAMPLES, amplitude=1.0):
    """
    Generate a half-sine waveform with exactly `samples` points.
    
    Parameters:
        samples (int): Number of samples in the waveform.
    
    Returns:
        np.ndarray: Half-sine waveform of exactly `samples` size.
    """
    # Use np.arange to guarantee exact sample count
    t = np.arange(samples) / samples  # Fractional positions from 0 to 1
    # Scale to the half-sine wave range (0 to pi)
    t *= np.pi
    # Compute and return the waveform
    w = (np.abs(np.sin(t)) * 2 - 1) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

def HalfSineShift(samples=SAMPLES, peak_shift=0.0, amplitude=1.0):
    """
    Generate a half-sine waveform with adjustable peak position.

    Parameters:
        samples (int): Number of samples in the waveform.
        peak_shift (float): Shift of the peak position as a fraction of the period (0.0 to 1.0).
                           - 0.0: Peak at the start.
                           - 0.5: Peak at the middle (default).
                           - 1.0: Peak back to the start (wraps around).
    
    Returns:
        np.ndarray: Half-sine waveform with the peak position adjusted.
    """
    # Create the domain for one half sine wave
    t = np.linspace(0, np.pi, samples, endpoint=False)
    # Apply phase shift
    shifted_t = t + peak_shift * np.pi
    # Ensure values wrap within [0, pi]
    shifted_t = np.mod(shifted_t, np.pi)
    # Compute and return the waveform
    w = (np.abs(np.sin(shifted_t)) * 2 - 1) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

# 3. Step Waveform
def StepRamp(samples=SAMPLES, steps=8, amplitude=1.0):
    """
    Generate a staircase waveform with defined steps.
    :param steps: Number of steps in the waveform.
    :return: Step waveform.
    """
    t = np.linspace(0, 1, samples, endpoint=False)
    w = (np.floor(t * steps) / steps * 2 - 1) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

# 4. FM Modulated Wave
def Fm(samples=SAMPLES, carrier_freq=2, mod_freq=10, mod_index=5, amplitude=1.0):
    """
    Generate a frequency-modulated sinusoidal waveform.
    :param carrier_freq: Frequency of the carrier wave.
    :param mod_freq: Frequency of the modulating wave.
    :param mod_index: Modulation index.
    :return: FM modulated waveform.
    """
    t = np.linspace(0, 1, samples, endpoint=False)
    modulator = np.sin(2 * np.pi * mod_freq * t)
    w = np.sin(2 * np.pi * carrier_freq * t + mod_index * modulator) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

# 5. Band-Limited Impulse Train
def ImpulseTrain(samples=SAMPLES, interval=16):
    """
    Generate a band-limited impulse train.
    :return: Impulse train waveform.
    """
    impulse = np.zeros(samples)
    impulse[::samples // interval] = 1  # Impulses at regular intervals
    w = np.convolve(impulse, np.hamming(interval/2), mode="same")
    return w

def generate_kick(freq_start=150, freq_end=40, decay_rate=10, samples=SAMPLES, sample_rate=48000, duration=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration))
    env = np.exp(-t * decay_rate)
    freq = np.logspace(np.log10(freq_start), np.log10(freq_end), len(t))
    kick = np.sin(2 * np.pi * freq * t) * env
    kick /= np.max(np.abs(kick))
    indices = np.linspace(0, len(kick) - 1, samples, dtype=int)
    return kick[indices]

def generate_snare(decay_rate=15, noise_mix=0.8, resonance_freq=200, samples=SAMPLES, sample_rate=48000, duration=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration))
    white_noise = np.random.uniform(-1, 1, len(t))
    env = np.exp(-t * decay_rate)
    resonance = np.sin(2 * np.pi * resonance_freq * t) * env
    snare = (noise_mix * white_noise + (1 - noise_mix) * resonance) * env
    snare /= np.max(np.abs(snare))
    indices = np.linspace(0, len(snare) - 1, samples, dtype=int)
    return snare[indices]

def generate_closed_hihat(base_freq=8000, decay_rate=50, samples=SAMPLES, sample_rate=48000, duration=0.5):
    t = np.linspace(0, duration / 2, int(sample_rate * duration / 2))  # Shorter duration
    white_noise = np.random.uniform(-1, 1, len(t))
    env = np.exp(-t * decay_rate)
    resonance = np.sin(2 * np.pi * base_freq * t)
    closed_hihat = (0.6 * white_noise + 0.4 * resonance) * env
    closed_hihat /= np.max(np.abs(closed_hihat))
    indices = np.linspace(0, len(closed_hihat) - 1, samples, dtype=int)
    return closed_hihat[indices]

def generate_open_hihat(base_freqs=(8000, 8200, 8400), decay_rate=5, sample_rate=48000, duration=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration))
    white_noise = np.random.uniform(-1, 1, len(t))
    resonance = sum(np.sin(2 * np.pi * f * t) for f in base_freqs)
    filtered_noise = np.convolve(white_noise, np.ones(100) / 100, mode='same')
    open_hihat = 0.6 * filtered_noise + 0.4 * resonance
    decay = np.exp(-t * decay_rate)
    open_hihat *= decay
    open_hihat /= np.max(np.abs(open_hihat))
    indices = np.linspace(0, len(open_hihat) - 1, samples, dtype=int)
    return open_hihat[indices]

def generate_tom(base_freq=120, decay_rate=8, overtone_mix=0.6, samples=SAMPLES, sample_rate=48000, duration=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration))
    env = np.exp(-t * decay_rate)
    fundamental = np.sin(2 * np.pi * base_freq * t) * env
    overtone = np.sin(2 * np.pi * base_freq * 2 * t) * env
    tom = (1 - overtone_mix) * fundamental + overtone_mix * overtone
    tom /= np.max(np.abs(tom))
    indices = np.linspace(0, len(tom) - 1, samples, dtype=int)
    return tom[indices]

def generate_chime(base_freq=1000, decay_rate=5, overtone_ratios=(1, 2.5, 3.6), samples=SAMPLES, sample_rate=48000, duration=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration))
    env = np.exp(-t * decay_rate)
    chime = sum(np.sin(2 * np.pi * base_freq * r * t) for r in overtone_ratios) * env
    chime /= np.max(np.abs(chime))
    indices = np.linspace(0, len(chime) - 1, samples, dtype=int)
    return chime[indices]

# Generate short drum waveforms as lookup tables
def generate_kick_table(freq_start=150, freq_end=40, attack=0.01, decay=0.2, sustain_level=0, sustain_time=0, release=0.1, samples=SAMPLES, duration=1):
    t = np.linspace(0, duration, samples, endpoint=False)
    freq = np.linspace(freq_start, freq_end, len(t))
    sine_wave = np.sin(2 * np.pi * freq * t)
    return envelope(sine_wave, attack, decay, sustain_level, sustain_time, release, sample_rate=samples)

def generate_snare_table(samples=SAMPLES, duration=1):
    t = np.linspace(0, duration, samples, endpoint=False)
    noise = generate_noise(len(t))
    sine_wave = np.sin(2 * np.pi * 200 * t)
    combined = sine_wave * 0.3 + noise * 0.7
    return envelope(combined, attack=0.01, decay=0.15, sustain_level=0, sustain_time=0, release=0.1, sample_rate=samples)

def generate_hi_hat_table(samples=SAMPLES, duration=1):
    t = np.linspace(0, duration, samples, endpoint=False)
    noise = generate_noise(len(t), color="white")
    return envelope(noise, attack=0.005, decay=0.05, sustain_level=0, sustain_time=0, release=0.05, sample_rate=samples)

def generate_open_hihat(samples=SAMPLES, sample_rate=48000, duration=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration))
    white_noise = np.random.uniform(-1, 1, len(t))
    resonance_frequencies = [8000, 8200, 8400]
    resonance = sum(np.sin(2 * np.pi * f * t) for f in resonance_frequencies)
    filtered_noise = np.convolve(white_noise, np.ones(100) / 100, mode='same')
    open_hihat = 0.6 * filtered_noise + 0.4 * resonance
    decay = np.exp(-t * 5)
    open_hihat *= decay
    open_hihat /= np.max(np.abs(open_hihat))
    indices = np.linspace(0, len(open_hihat) - 1, samples, dtype=int)
    return open_hihat[indices]

def generate_tom_table(samples=SAMPLES, duration=1):
    t = np.linspace(0, duration, samples, endpoint=False)
    freq = np.linspace(250, 150, len(t))
    sine_wave = np.sin(2 * np.pi * freq * t)
    return envelope(sine_wave, attack=0.01, decay=0.3, sustain_level=0.2, sustain_time=0.2, release=0.1, sample_rate=TABLE_SIZE)

def generate_clap_table(samples=SAMPLES, duration=1):
    t = np.linspace(0, duration, samples, endpoint=False)
    noise = generate_noise(len(t))
    clap_pattern = np.sin(2 * np.pi * np.linspace(1, 5, len(t))) > 0.5  # Simulates bursts
    return envelope(noise * clap_pattern, attack=0.005, decay=0.1, sustain_level=0, sustain_time=0, release=0.05, sample_rate=TABLE_SIZE)

# Mix multiple waveforms
def MixWaveforms(waveforms, weights, amplitude=1.0):
    mixed_waveform = np.zeros_like(next(iter(waveforms.values())))
    for name, weight in weights.items():
        mixed_waveform += weight * waveforms[name]
    w = mixed_waveform / np.max(np.abs(mixed_waveform)) * amplitude
    w = np.clip(w, -1, 1)  # Clip values to [-1, 1]
    return w

