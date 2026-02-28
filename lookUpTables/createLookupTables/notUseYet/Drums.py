import numpy as np
import matplotlib.pyplot as plt


def generate_drum_wavetables(table_size=256, sample_rate=48000, duration=0.5, params=None):
    """
    Generates lookup tables for various drum and percussion sounds.

    Parameters:
        table_size (int): Size of the lookup table.
        sample_rate (int): Sampling rate in Hz.
        duration (float): Duration of each sound in seconds.
        params (dict): Dictionary of parameters for customization.

    Returns:
        dict: A dictionary containing drum and percussion wavetables.
    """
    params = params or {}

    def generate_kick(freq_start=150, freq_end=40, decay_rate=10):
        t = np.linspace(0, duration, int(sample_rate * duration))
        env = np.exp(-t * decay_rate)
        freq = np.logspace(np.log10(freq_start), np.log10(freq_end), len(t))
        kick = np.sin(2 * np.pi * freq * t) * env
        kick /= np.max(np.abs(kick))
        indices = np.linspace(0, len(kick) - 1, table_size, dtype=int)
        return kick[indices]

    def generate_snare(decay_rate=15, noise_mix=0.8, resonance_freq=200):
        t = np.linspace(0, duration, int(sample_rate * duration))
        white_noise = np.random.uniform(-1, 1, len(t))
        env = np.exp(-t * decay_rate)
        resonance = np.sin(2 * np.pi * resonance_freq * t) * env
        snare = (noise_mix * white_noise + (1 - noise_mix) * resonance) * env
        snare /= np.max(np.abs(snare))
        indices = np.linspace(0, len(snare) - 1, table_size, dtype=int)
        return snare[indices]

    def generate_closed_hihat(base_freq=8000, decay_rate=50):
        t = np.linspace(0, duration / 2, int(sample_rate * duration / 2))  # Shorter duration
        white_noise = np.random.uniform(-1, 1, len(t))
        env = np.exp(-t * decay_rate)
        resonance = np.sin(2 * np.pi * base_freq * t)
        closed_hihat = (0.6 * white_noise + 0.4 * resonance) * env
        closed_hihat /= np.max(np.abs(closed_hihat))
        indices = np.linspace(0, len(closed_hihat) - 1, table_size, dtype=int)
        return closed_hihat[indices]

    def generate_open_hihat(base_freqs=(8000, 8200, 8400), decay_rate=5):
        t = np.linspace(0, duration, int(sample_rate * duration))
        white_noise = np.random.uniform(-1, 1, len(t))
        resonance = sum(np.sin(2 * np.pi * f * t) for f in base_freqs)
        filtered_noise = np.convolve(white_noise, np.ones(100) / 100, mode='same')
        open_hihat = 0.6 * filtered_noise + 0.4 * resonance
        decay = np.exp(-t * decay_rate)
        open_hihat *= decay
        open_hihat /= np.max(np.abs(open_hihat))
        indices = np.linspace(0, len(open_hihat) - 1, table_size, dtype=int)
        return open_hihat[indices]

    def generate_tom(base_freq=120, decay_rate=8, overtone_mix=0.6):
        t = np.linspace(0, duration, int(sample_rate * duration))
        env = np.exp(-t * decay_rate)
        fundamental = np.sin(2 * np.pi * base_freq * t) * env
        overtone = np.sin(2 * np.pi * base_freq * 2 * t) * env
        tom = (1 - overtone_mix) * fundamental + overtone_mix * overtone
        tom /= np.max(np.abs(tom))
        indices = np.linspace(0, len(tom) - 1, table_size, dtype=int)
        return tom[indices]

    def generate_chime(base_freq=1000, decay_rate=5, overtone_ratios=(1, 2.5, 3.6)):
        t = np.linspace(0, duration, int(sample_rate * duration))
        env = np.exp(-t * decay_rate)
        chime = sum(np.sin(2 * np.pi * base_freq * r * t) for r in overtone_ratios) * env
        chime /= np.max(np.abs(chime))
        indices = np.linspace(0, len(chime) - 1, table_size, dtype=int)
        return chime[indices]

    return {
        "kick": generate_kick(**params.get("kick", {})),
        "snare": generate_snare(**params.get("snare", {})),
        "closed_hihat": generate_closed_hihat(**params.get("closed_hihat", {})),
        "open_hihat": generate_open_hihat(**params.get("open_hihat", {})),
        "tom": generate_tom(**params.get("tom", {})),
        "chime": generate_chime(**params.get("chime", {})),
    }

# Example usage with custom parameters
params = {
    "kick": {"freq_start": 200, "freq_end": 50, "decay_rate": 12},
    "snare": {"decay_rate": 20, "noise_mix": 0.7, "resonance_freq": 250},
    "closed_hihat": {"base_freq": 10000, "decay_rate": 60},
    "open_hihat": {"base_freqs": (9000, 9200, 9400), "decay_rate": 6},
    "tom": {"base_freq": 100, "decay_rate": 10, "overtone_mix": 0.5},
    "chime": {"base_freq": 1200, "decay_rate": 4, "overtone_ratios": (1, 2.5, 4.2)},
}

# Generate drum and percussion wavetables
drum_wavetables = generate_drum_wavetables(params=params)

# Plot the results
plt.figure(figsize=(14, 10))

for i, (name, waveform) in enumerate(drum_wavetables.items()):
    plt.subplot(3, 2, i + 1)
    plt.plot(waveform, label=name.capitalize())
    plt.title(f"{name.capitalize()} Lookup Table")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()
