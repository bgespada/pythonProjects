import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Function: Load Lookup Tables or Wavetable from a File
def load_lookup_tables(folder_path, wavetable_split=256):
    lookup_tables = {}
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                data = pd.read_csv(file_path, header=None).values.flatten()  # Flatten to 1D array
                # Check if the file represents a wavetable
                if len(data) > wavetable_split and len(data) % wavetable_split == 0:
                    # Split into multiple waveforms
                    num_waveforms = len(data) // wavetable_split
                    for i in range(num_waveforms):
                        segment = data[i * wavetable_split : (i + 1) * wavetable_split]
                        lookup_tables[f"{file} (Waveform {i + 1})"] = segment
                else:
                    # Treat as a single lookup table
                    lookup_tables[file] = data
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return lookup_tables

# Function: Plot Tables Distributed by Columns (with Scales and Values)
def plot_tab_with_columns(figure, lookup_tables, folder_name, num_columns=3):
    num_tables = len(lookup_tables)
    num_rows = -(-num_tables // num_columns)  # Ceiling division to calculate rows
    figure.clear()  # Clear any existing plots

    # Create subplots
    axes = figure.subplots(num_rows, num_columns, squeeze=False)
    axes = axes.flatten()  # Flatten to make it easier to iterate

    for i, (name, data) in enumerate(lookup_tables.items()):
        ax = axes[i]
        ax.plot(data)
        ax.set_title(name, fontsize=8)
        ax.grid(True)
        ax.set_xlabel("Sample Index", fontsize=7)
        ax.set_ylabel("Amplitude", fontsize=7)
        ax.tick_params(axis='both', labelsize=6)
    
    # Hide unused subplots
    for j in range(num_tables, len(axes)):
        figure.delaxes(axes[j])

    figure.tight_layout()

# Main Function: Create Tabs for Each Folder
def plot_lookup_tables_in_tabs(main_folder, wavetable_split=256):
    # Tkinter Window
    root = tk.Tk()
    root.title("Lookup Tables Viewer")
    root.geometry("1000x800")

    # Notebook for Tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Iterate over Subfolders
    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)
        if os.path.isdir(folder_path):
            lookup_tables = load_lookup_tables(folder_path, wavetable_split=wavetable_split)
            if lookup_tables:
                # Create a Tab
                tab = ttk.Frame(notebook)
                notebook.add(tab, text=folder_name)

                # Create a Matplotlib Figure
                figure = plt.Figure(figsize=(12, 8), dpi=100)
                plot_tab_with_columns(figure, lookup_tables, folder_name)

                # Embed the Plot in the Tab
                canvas = FigureCanvasTkAgg(figure, master=tab)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(fill=tk.BOTH, expand=True)

    root.mainloop()



# # Example Usage
# appFolder = Path(__file__).parent.absolute()
# folder = f"{appFolder}\\lookupTables\\"
# if not os.path.exists(folder):
#     os.makedirs(folder)
# plot_lookup_tables_in_tabs(folder)

