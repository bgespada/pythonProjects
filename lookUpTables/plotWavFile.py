import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

# Folder containing .wav files (using raw string)
folder_path = 'C:\\DaisyExamples\\MyProjects\\python\\convertWavToBuffer\\Roland TR-909'

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"Error: The folder '{folder_path}' does not exist.")
else:
    print(f"Looking for .wav files in folder: {folder_path}")
    
    # Get list of all files in the folder
    files = os.listdir(folder_path)
    
    # Print all filenames
    print("All files in the folder:")
    for file_name in files:
        print(file_name)

    wav_files = [f for f in files if f.lower().endswith('.wav')]  # Case insensitive filter for .wav files
    
    if len(wav_files) == 0:
        print("No .wav files found in the folder.")
    else:
        print(f"Found {len(wav_files)} .wav files: {wav_files}")
        
        # Loop through each .wav file in the folder
        for file_name in wav_files:
            print(f"Processing file: {file_name}")  # Debugging: Check the files being processed
            file_path = os.path.join(folder_path, file_name)

            # Load the WAV file
            try:
                rate, data = wav.read(file_path)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue

            # Handle stereo by choosing one channel (e.g., left)
            if len(data.shape) > 1:
                data = data[:, 0]  # Take only the left channel

            # Generate time axis for the waveform
            time = np.linspace(0., len(data) / rate, num=len(data))

            # Plot the waveform
            plt.figure(figsize=(10, 6))
            plt.plot(time, data, label=f'Waveform of {file_name}')
            plt.title(f'Waveform of {file_name}')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude')
            plt.grid(True)

            # Save the plot with a unique name
            plot_file_path = os.path.join(folder_path, f"{file_name}_waveform.png")
            try:
                print(f"Saving plot for {file_name}...")  # Debugging: Check if the save is being reached
                plt.savefig(plot_file_path)
                plt.close()  # Close the plot to avoid overlapping with the next one
                print(f"Plot saved for {file_name} at {plot_file_path}")
            except Exception as e:
                print(f"Error saving plot for {file_name}: {e}")
                
    print("All plots processed (if no errors were encountered).")
