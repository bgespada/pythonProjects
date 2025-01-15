import numpy as np
import pandas as pd
import csv
import os
from pathlib import Path

def load_waveform_from_csv(filename):
    """
    Load a single waveform from a CSV file.

    Parameters:
        filename (str): Path to the CSV file.
        
    Returns:
        np.ndarray: 1D array representing the waveform.
    """
    data = pd.read_csv(filename, header=None).values.flatten()
    return data

def generate_progressive_wavetable(base_waveform, num_waveforms=8):
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

def save_wavetable_to_csv(wavetable, filename):
    """
    Save the generated wavetable to a CSV file.

    Parameters:
        wavetable (np.ndarray): A 2D array representing the wavetable.
        filename (str): Path to the output CSV file.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for waveform in wavetable:
            writer.writerow(waveform)

# Example usage
appFolder = Path(__file__).parent.absolute()
input_csv = f"{appFolder}\\lookupTables\\additiveWaveforms\\ADDITIVE_01.csv"
if not os.path.exists(input_csv):
    exit
folderAdditiveWavetable = f"{appFolder}\\lookupTables\\\\additiveWavetable\\"
if not os.path.exists(folderAdditiveWavetable):
    exit

output_csv = f"{folderAdditiveWavetable}ADDITIVE_WAVETABLE.csv"

base_waveform = load_waveform_from_csv(input_csv)
num_waveforms = 8

wavetable = generate_progressive_wavetable(base_waveform, num_waveforms)
save_wavetable_to_csv(wavetable, output_csv)
