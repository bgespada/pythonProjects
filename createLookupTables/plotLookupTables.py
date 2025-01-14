import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_csv_lookup_table(filename):
    # Load the waveform from the CSV file
    waveform = np.loadtxt(filename, delimiter=",")
    
    # Generate a sample index for x-axis
    num_points = len(waveform)
    x = np.arange(num_points)
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(x, waveform, label="Waveform")
    plt.title(f"Waveform from {filename}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example Usage
if __name__ == "__main__":
    appFolder = Path(__file__).parent.absolute()
    folder = f"{appFolder}\\lookupTables\\"

    # Replace 'your_lookup_table.csv' with the path to your CSV file
    plot_csv_lookup_table(f"{folder}sine_wave_lookup_table.csv")
    plot_csv_lookup_table(f"{folder}triangle_wave_lookup_table.csv")
    plot_csv_lookup_table(f"{folder}square_wave_lookup_table.csv")
    plot_csv_lookup_table(f"{folder}sawtooth_wave_lookup_table.csv")
    plot_csv_lookup_table(f"{folder}reverse_sawtooth_wave_lookup_table.csv")
    plot_csv_lookup_table(f"{folder}additive_synthesis_lookup_table.csv")
    plot_csv_lookup_table(f"{folder}fourier_synthesis_lookup_table.csv")
    plot_csv_lookup_table(f"{folder}mixed_waveform_lookup_table.csv")
