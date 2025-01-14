import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

def plot_all_lookup_tables_in_folder(folder_path):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the folder!")
        return
    
    plt.figure(figsize=(12, 6))  # Create a single figure for the plot

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        # Load the lookup table
        try:
            waveform = np.loadtxt(file_path, delimiter=",")
            # Generate x-axis indices
            x = np.arange(len(waveform))
            # Plot the waveform with a label
            plt.plot(x, waveform, label=csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Add titles, labels, and legend
    plt.title("Waveforms from Lookup Tables")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_with_toggle(folder_path):
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Store the line objects to toggle visibility
    lines = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        waveform = np.loadtxt(file_path, delimiter=",")
        num_points = len(waveform)
        x = np.arange(num_points)
        # Plot each waveform and store the line object
        line, = ax.plot(x, waveform, label=csv_file, picker=True)
        lines.append(line)

    # Add plot details
    ax.set_title("Lookup Tables with Toggle")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    # Define an on-pick event to toggle visibility
    def on_pick(event):
        line = event.artist
        visible = not line.get_visible()
        line.set_visible(visible)
        fig.canvas.draw()

    # Connect the pick event to the function
    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.tight_layout()
    plt.show()

def plot_all_separately(folder_path):
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    num_files = len(csv_files)
    fig, axes = plt.subplots(num_files, 1, figsize=(10, 3 * num_files))  # Adjusted height for smaller subplots

    if num_files == 1:
        axes = [axes]  # Ensure axes is iterable if only one file

    for ax, csv_file in zip(axes, csv_files):
        file_path = os.path.join(folder_path, csv_file)
        waveform = np.loadtxt(file_path, delimiter=",")
        num_points = len(waveform)
        x = np.arange(num_points)
        ax.plot(x, waveform, label=csv_file)
        
        # Adjust font sizes
        # ax.set_title(f"Waveform: {csv_file}", fontsize=10)  # Smaller title font
        # ax.set_xlabel("Sample Index", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)  # Smaller tick labels
        ax.grid(True)
        ax.legend(fontsize=8)  # Smaller legend font

    plt.tight_layout()
    plt.show()

def plot_in_columns(folder_path, num_columns=6):
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    num_files = len(csv_files)
    num_rows = math.ceil(num_files / num_columns)  # Calculate rows needed for the grid

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))  # Adjust figure size

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(folder_path, csv_file)
        waveform = np.loadtxt(file_path, delimiter=",")
        num_points = len(waveform)
        x = np.arange(num_points)

        ax = axes[i]
        ax.plot(x, waveform, label=csv_file)
        # ax.set_xlabel("Sample Index", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)  # Smaller tick labels
        ax.grid(True)
        ax.legend(fontsize=8, loc="upper right")

    # Hide any unused subplots
    for i in range(len(csv_files), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace 'your_folder_path' with the path to the folder containing CSV files
    appFolder = Path(__file__).parent.absolute()
    folder = f"{appFolder}\\lookupTables\\"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # plot_all_lookup_tables_in_folder(folder)
    # plot_all_with_toggle(folder)
    # plot_all_separately(folder)
    plot_in_columns(folder)
