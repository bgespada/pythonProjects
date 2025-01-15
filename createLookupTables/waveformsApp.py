import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


# Waveform generation functions
def generate_sine_wave(length):
    t = np.linspace(0, 1, length, endpoint=False)
    return np.sin(2 * np.pi * t)


def generate_square_wave(length):
    t = np.linspace(0, 1, length, endpoint=False)
    return np.sign(np.sin(2 * np.pi * t))


def generate_triangle_wave(length):
    t = np.linspace(0, 1, length, endpoint=False)
    return 2 * np.abs(2 * (t % 0.5) - 0.5) - 1


def generate_fourier_waveform(length, max_harmonics=8):
    t = np.linspace(0, 1, length, endpoint=False)
    waveform = np.zeros(length)
    for h in range(1, max_harmonics + 1):
        waveform += (1 / h) * np.sin(2 * np.pi * h * t)
    return waveform / np.max(np.abs(waveform))


def generate_wavetable(base_waveform, num_tables=8):
    """Generates a wavetable with progressive harmonic richness."""
    length = len(base_waveform)
    wavetable = np.zeros((num_tables, length))
    for i in range(num_tables):
        harmonics = 1 + i * (8 // num_tables)  # Progressive increase in harmonics
        wavetable[i, :] = generate_fourier_waveform(length, harmonics)
    return wavetable


def save_to_header_file(data, filename, name="WaveTable"):
    with open(filename, 'w') as f:
        f.write(f"#ifndef {name.upper()}_HPP\n")
        f.write(f"#define {name.upper()}_HPP\n\n")
        for key, waveform in data.items():
            f.write(f"constexpr float {key}[] = {{")
            f.write(", ".join(f"{x:.6f}" for x in waveform))
            f.write("};\n\n")
        f.write(f"#endif\n")


def save_wavetable_to_header(wavetable, filename, name="WaveTable"):
    with open(filename, 'w') as f:
        f.write(f"#ifndef {name.upper()}_HPP\n")
        f.write(f"#define {name.upper()}_HPP\n\n")
        for i, table in enumerate(wavetable):
            f.write(f"constexpr float {name}_Table_{i}[] = {{")
            f.write(", ".join(f"{x:.6f}" for x in table))
            f.write("};\n\n")
        f.write(f"#endif\n")


# App Class
class WaveformApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Waveform Generator")
        self.geometry("1200x800")
        
        # Tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tabs for functionalities
        self.add_basic_waveforms_tab(notebook)
        self.add_extended_waveforms_tab(notebook)
        self.add_wavetable_tab(notebook)

    def add_basic_waveforms_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Basic Waveforms")
        
        frame = ttk.Frame(tab)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        canvas = plt.Figure(figsize=(8, 6))
        self.basic_axs = canvas.subplots(1, 3)
        self.basic_canvas = FigureCanvasTkAgg(canvas, frame)
        self.basic_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        buttons_frame = ttk.Frame(tab)
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(buttons_frame, text="Generate & Plot", command=self.plot_basic_waveforms).pack(side=tk.LEFT)
        ttk.Button(buttons_frame, text="Save Header File", command=self.save_basic_waveforms).pack(side=tk.LEFT)

    def plot_basic_waveforms(self):
        waveforms = {
            "Sine": generate_sine_wave(256),
            "Square": generate_square_wave(256),
            "Triangle": generate_triangle_wave(256)
        }
        for ax, (name, waveform) in zip(self.basic_axs, waveforms.items()):
            ax.clear()
            ax.plot(waveform)
            ax.set_title(name)
        self.basic_canvas.draw()

    def save_basic_waveforms(self):
        waveforms = {
            "Sine": generate_sine_wave(256),
            "Square": generate_square_wave(256),
            "Triangle": generate_triangle_wave(256)
        }
        save_to_header_file(waveforms, "basic_waveforms.hpp", "BasicWaveforms")

    def add_extended_waveforms_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Extended Waveforms")
        
        frame = ttk.Frame(tab)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        canvas = plt.Figure(figsize=(8, 6))
        self.extended_axs = canvas.subplots(1, 2)
        self.extended_canvas = FigureCanvasTkAgg(canvas, frame)
        self.extended_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        buttons_frame = ttk.Frame(tab)
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(buttons_frame, text="Generate & Plot", command=self.plot_extended_waveforms).pack(side=tk.LEFT)
        ttk.Button(buttons_frame, text="Save Header File", command=self.save_extended_waveforms).pack(side=tk.LEFT)

    def plot_extended_waveforms(self):
        waveforms = {
            "Fourier1": generate_fourier_waveform(256, max_harmonics=4),
            "Fourier2": generate_fourier_waveform(256, max_harmonics=8)
        }
        for ax, (name, waveform) in zip(self.extended_axs, waveforms.items()):
            ax.clear()
            ax.plot(waveform)
            ax.set_title(name)
        self.extended_canvas.draw()

    def save_extended_waveforms(self):
        waveforms = {
            "Fourier1": generate_fourier_waveform(256, max_harmonics=4),
            "Fourier2": generate_fourier_waveform(256, max_harmonics=8)
        }
        save_to_header_file(waveforms, "extended_waveforms.hpp", "ExtendedWaveforms")

    def add_wavetable_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Wavetable")
        
        frame = ttk.Frame(tab)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.wavetable_plot = plt.Figure(figsize=(8, 6))
        self.wavetable_ax = self.wavetable_plot.add_subplot(111)
        self.wavetable_canvas = FigureCanvasTkAgg(self.wavetable_plot, frame)
        self.wavetable_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(tab)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(control_frame, text="Waveform Type:").pack(side=tk.LEFT)
        self.wavetable_type = ttk.Combobox(control_frame, values=["Sine", "Square", "Triangle"])
        self.wavetable_type.current(0)
        self.wavetable_type.pack(side=tk.LEFT)
        
        buttons_frame = ttk.Frame(tab)
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Button(buttons_frame, text="Generate & Plot", command=self.plot_wavetable).pack(side=tk.LEFT)
        ttk.Button(buttons_frame, text="Save Header File", command=self.save_wavetable).pack(side=tk.LEFT)

    def plot_wavetable(self):
        length = 256
        waveform_type = self.wavetable_type.get()
        if waveform_type == "Sine":
            print("Sine selected")
            base_wave = generate_sine_wave(length)
        elif waveform_type == "Square":
            print("Square selected")
            base_wave = generate_square_wave(length)
        elif waveform_type == "Triangle":
            print("Triangle selected")
            base_wave = generate_triangle_wave(length)
        else:
            return

        wavetable = generate_wavetable(base_wave, num_tables=8)

        num_columns = 3
        num_rows = (len(wavetable) + num_columns - 1) // num_columns  # Calculate required rows

        # Clear the existing plot and reset the figure
        self.wavetable_plot.clear()
        self.wavetable_plot.set_size_inches(8, 6)  # Ensure consistent figure size
        axs = self.wavetable_plot.subplots(num_rows, num_columns, squeeze=False)  # Avoid errors for single row

        # Flatten axes for easier iteration and hide unused subplots
        axs = axs.flatten()
        for ax in axs[len(wavetable):]:
            ax.axis("off")

        # Plot each wavetable in its own subplot
        for i, table in enumerate(wavetable):
            axs[i].plot(table)
            axs[i].set_title(f"Table {i + 1}")
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        self.wavetable_plot.tight_layout()
        self.wavetable_canvas.draw()

    def save_wavetable(self):
        length = 256
        waveform_type = self.wavetable_type.get()
        if waveform_type == "Sine":
            base_wave = generate_sine_wave(length)
        elif waveform_type == "Square":
            base_wave = generate_square_wave(length)
        elif waveform_type == "Triangle":
            base_wave = generate_triangle_wave(length)
        else:
            return
        
        wavetable = generate_wavetable(base_wave, num_tables=8)
        save_wavetable_to_header(wavetable, f"{waveform_type.lower()}_wavetable.hpp", waveform_type)


# Run the app
if __name__ == "__main__":
    app = WaveformApp()
    app.mainloop()
