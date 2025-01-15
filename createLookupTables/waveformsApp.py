import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import CreateLookupTables as waveforms


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

    def plot_basic_waveforms(self, samples=256, amplitude=1.0, harmonics=20):
        phase = np.linspace(0, 2 * np.pi, samples, endpoint=False)
        wf = {
            "SINE": waveforms.SineWave(samples, amplitude),
            "SQUARE": waveforms.SquareWave(samples, amplitude),# np.sign(np.sin(phase)),
            "SQUARE_LIMITED": waveforms.LimitedSquareWave(samples, amplitude, harmonics),
            "TRIANGLE": waveforms.TriangleWave(samples, amplitude),
            "SAW": waveforms.SawWave(phase, amplitude),
            "RAMP": waveforms.RampWave(phase, amplitude),
            "WHITE_NOISE": waveforms.WhiteNoise(samples, amplitude),
            "PINK_NOISE": waveforms.PinkNoise(samples, amplitude)
        }
        for ax, (name, wf) in zip(self.basic_axs, wf.items()):
            ax.clear()
            ax.plot(wf)
            ax.set_title(name)
        self.basic_canvas.draw()

    def save_basic_waveforms(self):
        wf = {
            "Sine": waveforms.SineWave(256),
            "Square": waveforms.SquareWave(256),
            "Triangle": waveforms.TriangleWave(256)
        }
        save_to_header_file(wf, "basic_waveforms.hpp", "BasicWaveforms")

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
        wf = {
            "Fourier1": waveforms.fourier_waveform(256, harmonics=4),
            "Fourier2": waveforms.fourier_waveform(256, harmonics=8)
        }
        for ax, (name, wf) in zip(self.extended_axs, wf.items()):
            ax.clear()
            ax.plot(wf)
            ax.set_title(name)
        self.extended_canvas.draw()

    def save_extended_waveforms(self):
        wf = {
            "Fourier1": waveforms.fourier_waveform(256, harmonics=4),
            "Fourier2": waveforms.fourier_waveform(256, harmonics=8)
        }
        save_to_header_file(wf, "extended_waveforms.hpp", "ExtendedWaveforms")

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
            base_wave = waveforms.SineWave(length)
        elif waveform_type == "Square":
            print("Square selected")
            base_wave = waveforms.SquareWave(length)
        elif waveform_type == "Triangle":
            print("Triangle selected")
            base_wave = waveforms.TriangleWave(length)
        else:
            return

        wf = waveforms.generate_wavetable(base_wave, num_tables=8)

        num_columns = 3
        num_rows = (len(wf) + num_columns - 1) // num_columns  # Calculate required rows

        # Clear the existing plot and reset the figure
        self.wavetable_plot.clear()
        self.wavetable_plot.set_size_inches(8, 6)  # Ensure consistent figure size
        axs = self.wavetable_plot.subplots(num_rows, num_columns, squeeze=False)  # Avoid errors for single row

        # Flatten axes for easier iteration and hide unused subplots
        axs = axs.flatten()
        for ax in axs[len(wf):]:
            ax.axis("off")

        # Plot each wavetable in its own subplot
        for i, table in enumerate(wf):
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
            base_wave = waveforms.SineWave(length)
        elif waveform_type == "Square":
            base_wave = waveforms.SquareWave(length)
        elif waveform_type == "Triangle":
            base_wave = waveforms.TriangleWave(length)
        else:
            return
        
        wavetable = waveforms.GenerateWavetable(base_wave, num_tables=8)
        save_wavetable_to_header(wavetable, f"{waveform_type.lower()}_wavetable.hpp", waveform_type)


# Run the app
if __name__ == "__main__":
    app = WaveformApp()
    app.mainloop()
