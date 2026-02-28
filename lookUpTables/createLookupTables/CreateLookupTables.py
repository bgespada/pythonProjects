import os
import numpy as np
from pathlib import Path
import re

import WaveformsGenerators as wfGen
import RandomGenerators as randGen
import RenameFiles as rename
import PlotLookupTablesFromFolderInTabs as plotAll
import Folders as folders

def RandomWaveform(outputFolder, samples, iterations=16):
    for i in range(iterations):
        # Randomly choose a waveform type
        waveform_type = np.random.choice(['Fourier', 'Additive', 'Polynomial'])
        
        print(f"Iteration {i+1}: Generating {waveform_type} waveform")
        
        if waveform_type == 'Fourier':
            harmonics = np.random.randint(3, 8)  # Random number of harmonics
            y, params = randGen.RandomFourierWave(samples, harmonics)
            filename = os.path.join(outputFolder, f"random_fourier_wave_{i+1}.csv")
            print(f"  Parameters: {params}")
        
        elif waveform_type == 'Additive':
            harmonics = np.random.randint(3, 8)  # Random number of harmonics
            y, params = randGen.RandomAdditiveWave(samples, harmonics)
            filename = os.path.join(outputFolder, f"random_additive_wave_{i+1}.csv")
            print(f"  Parameters: {params}")
        
        elif waveform_type == 'Polynomial':
            max_degree = np.random.randint(2, 6)  # Random polynomial degree
            y, params = randGen.RandomPolynomialWave(samples, max_degree)
            filename = os.path.join(outputFolder, f"random_polynomial_wave_{i+1}.csv")
            print(f"  Parameters: {params}")
        
        # Save to CSV
        SaveWaveformToCsv(y, filename)

def RandomMixWaveform(outputFolder, samples, iterations=16):
    for i in range(iterations):
        print(f"\nIteration {i+1}: Generating waveforms...")
        
        # Generate two random waveforms
        waveform1_type = np.random.choice(['Fourier', 'Additive', 'Polynomial'])
        waveform2_type = np.random.choice(['Fourier', 'Additive', 'Polynomial'])
        
        # Generate Waveform 1
        if waveform1_type == 'Fourier':
            harmonics = np.random.randint(3, 8)
            waveform1, params1 = randGen.RandomFourierWave(samples, harmonics)
        elif waveform1_type == 'Additive':
            harmonics = np.random.randint(3, 8)
            waveform1, params1 = randGen.RandomAdditiveWave(samples, harmonics)
        elif waveform1_type == 'Polynomial':
            max_degree = np.random.randint(2, 6)
            waveform1, params1 = randGen.RandomPolynomialWave(samples, max_degree)
        
        # Generate Waveform 2
        if waveform2_type == 'Fourier':
            harmonics = np.random.randint(3, 8)
            waveform2, params2 = randGen.RandomFourierWave(samples, harmonics)
        elif waveform2_type == 'Additive':
            harmonics = np.random.randint(3, 8)
            waveform2, params2 = randGen.RandomAdditiveWave(samples, harmonics)
        elif waveform2_type == 'Polynomial':
            max_degree = np.random.randint(2, 6)
            waveform2, params2 = randGen.RandomPolynomialWave(samples, max_degree)
        
        # Save individual waveforms
        SaveWaveformToCsv(waveform1, os.path.join(outputFolder, f"waveform1_{i+1}.csv"))
        SaveWaveformToCsv(waveform2, os.path.join(outputFolder, f"waveform2_{i+1}.csv"))
        
        # Mix the two waveforms
        mixed_wave = wfGen.MixWaveforms({"1": waveform1, "2": waveform2}, {"1": 0.7, "2": 0.3})
        SaveWaveformToCsv(mixed_wave, os.path.join(outputFolder, f"mixed_wave_{i+1}.csv"))
        
        # Print parameters
        print(f"  Waveform 1 ({waveform1_type}): {params1}")
        print(f"  Waveform 2 ({waveform2_type}): {params2}")
        print("  Mixed waveform saved.")    

def GenerateProgressiveWavetable(base_waveform, num_waveforms=8):
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

def GenerateCustomProgressiveWavetable(base_waveform, num_waveforms=8, transformation="harmonics"):
    """
    Generate a progressive wavetable with custom transformations.
    
    Parameters:
        base_waveform (np.ndarray): Original waveform to transform.
        num_waveforms (int): Number of variations to generate.
        transformation (str): Type of transformation ("harmonics", "smoothing", "scaling").
        
    Returns:
        np.ndarray: 2D array representing the wavetable.
    """
    length = len(base_waveform)
    wavetable = []

    for i in range(num_waveforms):
        factor = i / (num_waveforms - 1)  # Progression factor (0 to 1)
        if transformation == "harmonics":
            # Add or remove harmonics progressively
            additional_wave = np.sin(2 * np.pi * np.linspace(0, 1, length) * (2 + int(factor * 4)))
            transformed_waveform = base_waveform + additional_wave * factor
        
        elif transformation == "smoothing":
            # Smooth waveform progressively using a moving average
            window_size = int(5 + factor * (length // 8))  # Larger window for more smoothing
            smoothed_waveform = np.convolve(base_waveform, np.ones(window_size) / window_size, mode='same')
            transformed_waveform = smoothed_waveform
        
        elif transformation == "scaling":
            return
            # Gradually reduce or increase amplitude
            scale_up = 1 + factor  # Amplitude increases progressively
            scale_down = 1 - factor  # Amplitude decreases progressively
            transformed_waveform = base_waveform * (scale_down if factor < 0.5 else scale_up)
        
        else:
            raise ValueError("Invalid transformation type. Choose 'harmonics', 'smoothing', or 'scaling'.")
        
        # Normalize to [-1, 1]
        transformed_waveform /= np.max(np.abs(transformed_waveform))
        wavetable.append(transformed_waveform)

    return np.array(wavetable)

def GenerateWaveformProgression(waveform1, waveform2, num_steps=8):
    """
    Generate a progressive wavetable transitioning from one waveform to another.
    
    Parameters:
        waveform1 (np.ndarray): Starting waveform.
        waveform2 (np.ndarray): Ending waveform.
        num_steps (int): Number of progressive steps between the waveforms.
        
    Returns:
        np.ndarray: 2D array representing the progression wavetable.
    """
    if len(waveform1) != len(waveform2):
        raise ValueError("Waveform lengths must be the same.")
    
    wavetable = []
    for i in range(num_steps):
        factor = i / (num_steps - 1)  # Progression factor (0 to 1)
        # Linear interpolation between the two waveforms
        interpolated_waveform = (1 - factor) * waveform1 + factor * waveform2
        # Normalize to [-1, 1]
        interpolated_waveform /= np.max(np.abs(interpolated_waveform))
        wavetable.append(interpolated_waveform)
    
    return np.array(wavetable)

def CreateWavetableFromWaveforms(waveforms):
    """
    Create a wavetable from multiple waveforms.
    
    Parameters:
        waveforms (list of np.ndarray): List of waveforms, each with the same number of samples.
        
    Returns:
        np.ndarray: 2D array representing the wavetable, where each row is a waveform.
    """
    if len(waveforms) == 0:
        raise ValueError("The waveforms list cannot be empty.")
    
    # Ensure all waveforms have the same number of samples
    length = len(waveforms[0])
    if not all(len(waveform) == length for waveform in waveforms):
        raise ValueError("All waveforms must have the same number of samples.")
    
    # Stack waveforms into a 2D array
    wavetable = np.stack(waveforms)
    
    # Normalize each waveform to the range [-1, 1]
    wavetable = np.array([waveform / np.max(np.abs(waveform)) for waveform in wavetable])
    
    return wavetable

def CreateWavetableFromCSVFiles(folder_path, filesNames):
    """
    Create a wavetable from multiple CSV files in a folder.
    
    Parameters:
        folder_path (str): Path to the folder containing CSV files.
        
    Returns:
        np.ndarray: 2D array representing the wavetable.
    """
    # List CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV files found in the specified folder.")
    
    waveforms = []
    for file in filesNames:
        file_path = os.path.join(folder_path, file)
        waveform = np.loadtxt(file_path, delimiter=",")
        
        # Normalize waveform to [-1, 1]
        waveform /= np.max(np.abs(waveform))
        waveforms.append(waveform)
    
    # Ensure all waveforms have the same length
    length = len(waveforms[0])
    if not all(len(waveform) == length for waveform in waveforms):
        raise ValueError("All waveforms in CSV files must have the same number of samples.")
    
    # Stack waveforms into a 2D array
    wavetable = np.stack(waveforms)
    return wavetable

# Save lookup table to a CSV file
def SaveWaveformToCsv(waveform, filename):
    np.savetxt(filename, waveform, delimiter=",", fmt="%.6f")
    # print(f"Saved waveform to {filename}")

# Function: Save Wavetable to CSV
def SaveWavetableToCsv(wavetable, filename):
    np.savetxt(filename, wavetable, delimiter=",", header="", comments="")
    # print(f"Saved wavetable to {filename}")

def SaveWaveformsToCsv(waveforms, folder):
    for name, waveform in waveforms.items():
        SaveWaveformToCsv(waveform, f"{folder + name}.csv")

# Function: Save Wavetable to C Header File
def GenerateHppFromCsv(folder_path, output_file):
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    with open(output_file, "w") as hpp_file:
        # Write header guards
        hpp_file.write("#ifndef LOOKUP_TABLES_HPP\n")
        hpp_file.write("#define LOOKUP_TABLES_HPP\n\n")
        
        hpp_file.write("// Lookup tables generated from CSV files\n\n")
        
        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            # Load the waveform data from CSV
            waveform = np.loadtxt(file_path, delimiter=",")
            
            # Generate a valid C++ variable name from the file name
            variable_name = os.path.splitext(csv_file)[0].replace(" ", "_").replace("-", "_")
            
            # Write the array definition
            # hpp_file.write(f"// {csv_file}\n")
            hpp_file.write(f"constexpr float WF_{variable_name}[256] = {{")
            # Write the waveform values
            formatted_values = ", ".join(f"{v:.6f}" for v in waveform)
            hpp_file.write(f"{formatted_values}\n")
            hpp_file.write("};\n")
        
        # Close header guards
        hpp_file.write("#endif // LOOKUP_TABLES_HPP\n")

def GenerateHppFromFolders(folders, output_file, samples):
    """
    Read all CSV files from a list of folders, process their contents, and generate a .hpp file.
    Each CSV will be turned into a constexpr float array.
    :param folders: List of folder paths to process.
    :param output_file: Output .hpp file name.
    """
    # header_content = "// Auto-generated header file with waveforms\n\n"
    # header_content += "#pragma once\n\n"
    # header_content += "#include <array>\n\n"
    header_content = "#ifndef LOOKUP_TABLES_HPP\n"
    header_content += "#define LOOKUP_TABLES_HPP\n\n"

    counter = 1  # Counter for naming waveforms (e.g., WF_001)
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".csv"):
                    filepath = os.path.join(root, file)
                    # data = process_csv_to_array(filepath)
                    # Load the waveform data from CSV
                    waveform = np.loadtxt(filepath, delimiter=",")
                    # Write the array definition
                    # hpp_file.write(f"// {csv_file}\n")
                    header_content += (f"const float WF_{counter:03d}[{samples}] = {{")
                    # Write the waveform values
                    formatted_values = ", ".join(f"{v:.6f}" for v in waveform)
                    header_content += (f"{formatted_values}")
                    header_content += ("};\n")
                    if counter % 8 == 0:
                        header_content += ("\n")
                    counter += 1

    # Write the output to the specified .hpp file
    with open(output_file, "w") as f:
        f.write(header_content)

    print(f"Header file '{output_file}' generated successfully with {counter-1} waveforms.")

def SaveWavetableToHeader(wavetable, filename, name="WaveTable"):
    with open(filename, "w") as file:
        file.write(f"#ifndef {name.upper()}_HPP\n#define {name.upper()}_HPP\n\n")
        file.write(f"// Wavetable generated with {len(wavetable)} waveforms, {wavetable.shape[1]} samples each\n\n")
        file.write(f"constexpr float {name}[8][256] = {{\n")
        for waveform in wavetable:
            file.write("    {" + ", ".join(f"{v:.6f}" for v in waveform) + "},\n")
        file.write(f"}};\n\n#endif // {name.upper()}_HPP\n")
    # print(f"Saved wavetable to {filename}")

def JoinHeaderFiles(source_folder, output_file):
    """
    Joins the content of all header files in the specified folder into a single header file.
    
    Args:
        source_folder (str): Path to the folder containing header files.
        output_file (str): Path to the output header file.
    """
    try:
        # List all files in the source folder
        files = [f for f in os.listdir(source_folder) if f.endswith('.hpp')]
        
        with open(output_file, 'w') as outfile:
            # Write a comment to indicate the beginning of the file
            outfile.write("/* Combined Header File */\n\n")
            
            for filename in files:
                filepath = os.path.join(source_folder, filename)
                with open(filepath, 'r') as infile:
                    # Write a comment indicating the start of a new file
                    outfile.write(f"/* Start of {filename} */\n")
                    outfile.write(infile.read())
                    outfile.write(f"\n/* End of {filename} */\n\n")
        
        print(f"All header files have been successfully combined into {output_file}.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def JoinHeaderFilesWithoutDirectives(source_folder, output_file):
    """
    Joins the content of all header files in the specified folder into a single header file,
    removing include guards (#ifdef, #ifndef, #define, #endif).

    Args:
        source_folder (str): Path to the folder containing header files.
        output_file (str): Path to the output header file.
    """
    try:
        # List all files in the source folder
        files = [f for f in os.listdir(source_folder) if f.endswith('.hpp')]
        
        with open(output_file, 'w') as outfile:
            for filename in files:
                filepath = os.path.join(source_folder, filename)
                with open(filepath, 'r') as infile:
                    # Read the file content
                    content = infile.read()
                    
                    # Remove include guards using regex
                    content = re.sub(r'#ifndef\s+.*?\n', '', content)  # Remove #ifndef lines
                    content = re.sub(r'#define\s+.*?\n', '', content)  # Remove #define lines
                    content = re.sub(r'#ifdef\s+.*?\n', '', content)   # Remove #ifdef lines
                    content = re.sub(r'#endif\s+.*?\n', '', content)   # Remove #endif lines
                    
                    # Write the cleaned content to the output file
                    outfile.write(content + "\n")
        
        print(f"All header files have been successfully combined into {output_file}.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Generate basic waveforms
def GenerateWaveforms01(samples, amplitude=1.0, harmonics=20):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    waveforms = {
        "01": wfGen.SineWave(samples, amplitude),
        "02": wfGen.SquareWave(samples, amplitude),# np.sign(np.sin(phase)),
        "03": wfGen.LimitedSquareWave(samples, harmonics, amplitude),
        "04": wfGen.TriangleWave(samples, amplitude),
        "05": wfGen.SawWave(phase, amplitude),
        "06": wfGen.RampWave(phase, amplitude),
        "07": wfGen.WhiteNoise(samples, amplitude),
        "08": wfGen.PinkNoise(samples, amplitude),
        "09": wfGen.MixWaveforms({"1": wfGen.WhiteNoise(samples, amplitude), "2": wfGen.SineWave(samples, amplitude)}, {"1": 0.05, "2": 0.95}),
        "10": wfGen.MixWaveforms({"1": wfGen.SineWave(samples, amplitude), "2": wfGen.PulseTrain(samples)}, {"1": 0.9, "2": 0.1}),
        "11": wfGen.MixWaveforms({"1": wfGen.SineWave(samples, amplitude), "2": wfGen.PulseTrain(samples, 48)}, {"1": 0.95, "2": 0.05}),
        "12": wfGen.MixWaveforms({"1": wfGen.SineWave(samples, amplitude), "2": wfGen.RampWave(phase, amplitude)}, {"1": 0.7, "2": 0.3}),
        "13": wfGen.MixWaveforms({"1": wfGen.SquareWave(samples, amplitude), "2": wfGen.PulseTrain(), "3": wfGen.PolynomialWave(), "4": wfGen.ExponentialDecay() * -1, "5": wfGen.RampWave(phase, amplitude), "6": wfGen.ExponentialDecay()}, {"1": 0.05, "2": 0.05, "3": 0.05, "4": 0.4, "5": 0.1, "6": 0.45}),
        "14": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.5, 3: 0.25}),
        "15": wfGen.FourierWaveform(samples, 5),
        "16": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.15, 5: 0.05}),
        "17": wfGen.FourierWaveform(samples, 55),
        "18": wfGen.ExponentialDecay(samples, 5, amplitude),
        "19": wfGen.Gaussian(samples, 0.5, 0.1, amplitude),
        "20": wfGen.PulseTrain(samples, 32),
        "21": wfGen.PulseTrain(samples, 32*2),
        "22": wfGen.PulseTrain(samples, 32*3),
        "23": wfGen.PulseTrain(samples, 32*4),
        "24": wfGen.PolynomialWave(samples, [0, -2, 1, -3, 1]),
        "25": wfGen.SquareWithDutyCycle(samples, 0.12, amplitude),
        "26": wfGen.SquareWithDutyCycle(samples, 0.24, amplitude),
        "27": wfGen.SquareWithDutyCycle(samples, 0.36, amplitude),
        "28": wfGen.SquareWithDutyCycle(samples, 0.48, amplitude),
        "29": wfGen.SquareWithDutyCycle(samples, 0.60, amplitude),
        "30": wfGen.SquareWithDutyCycle(samples, 0.72, amplitude),
        "31": wfGen.SquareWithDutyCycle(samples, 0.84, amplitude),
        "32": wfGen.SquareWithDutyCycle(samples, 0.96, amplitude),
        "33": wfGen.TriangleSaw(samples, 0.1),
        "34": wfGen.HalfSine(256),
        "35": wfGen.StepRamp(samples, 64),
        "36": wfGen.Fm(samples),
        "37": wfGen.ImpulseTrain(samples),
        "38": wfGen.MixWaveforms({"1": wfGen.PolynomialWave(samples, [0, -2, 1, -3, 1]), "2": wfGen.WhiteNoise(samples, amplitude)}, {"1": 0.9, "2": 0.1}),
        "39": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.2, 3: -0.7, 4: 1.0}),
        "40": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.1, 3: -0.8, 4: 1.0}),
        "41": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.8, 3: 0.1, 4: -1.0}),
        "42": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.7, 3: 0.2, 4: -1.0}),
        "43": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.6, 3: 0.3, 4: -1.0}),
        "44": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.5, 3: 0.4, 4: -1.0}),
        "45": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.4, 3: 0.5, 4: -1.0}),
        "46": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.3, 3: 0.6, 4: -1.0}),
        "47": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.2, 3: 0.7, 4: -1.0}),
        "48": wfGen.AdditiveWaveform(samples, {1: 1.0, 2: 0.1, 3: 0.8, 4: -1.0}),
    }
    return waveforms

# Generate basic waveforms
def GenerateWaveforms02(samples, amplitude=1.0, harmonics=20):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    waveforms = {
        "01": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.8, 3: -0.1, 4: 1.0}),
        "02": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.7, 3: -0.2, 4: 1.0}),
        "03": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.6, 3: -0.3, 4: 1.0}),
        "04": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.5, 3: -0.4, 4: 1.0}),
        "05": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.4, 3: -0.5, 4: 1.0}),
        "06": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.3, 3: -0.6, 4: 1.0}),
        "07": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.2, 3: -0.7, 4: 1.0}),
        "08": wfGen.AdditiveWaveform(samples, {1: -1.0, 2: -0.1, 3: -0.8, 4: 1.0}),
        "09": wfGen.HalfSineShift(peak_shift=0),
        "10": wfGen.HalfSineShift(peak_shift=0.1),
        "11": wfGen.HalfSineShift(peak_shift=0.2),
        "12": wfGen.HalfSineShift(peak_shift=0.3),
        "13": wfGen.HalfSineShift(peak_shift=0.4),
        "14": wfGen.HalfSineShift(peak_shift=0.6),
        "15": wfGen.HalfSineShift(peak_shift=0.8),
        "16": wfGen.HalfSineShift(peak_shift=1),
        "17": wfGen.ExponentialDecay(samples, 5, amplitude),
        "18": wfGen.ExponentialDecay(samples, 15, amplitude),
        "19": wfGen.ExponentialDecay(samples, 25, amplitude),
        "20": wfGen.ExponentialDecay(samples, 35, amplitude),
        "21": wfGen.ExponentialDecay(samples, 45, amplitude),
        "22": wfGen.ExponentialDecay(samples, 55, amplitude),
        "23": wfGen.ExponentialDecay(samples, 65, amplitude),
        "24": wfGen.ExponentialDecay(samples, 75, amplitude),
        # "17": fourier_waveform(samples, 55),
        # "18": ExponentialDecay(samples, 5, amplitude),
        # "19": gaussian_wave(samples, 0.5, 0.1, amplitude),
        # "20": PulseTrain(samples, 32),
        # "21": PulseTrain(samples, 32*2),
        # "22": PulseTrain(samples, 32*3),
        # "23": PulseTrain(samples, 32*4),
        # "24": PolynomialWave(samples, [0, -2, 1, -3, 1]),
        "25": wfGen.PolynomialWave(samples, [0, -2, 1, -3, 1]),
        "26": wfGen.PolynomialWave(samples, [0, -2, 2, -3, 1]),
        "27": wfGen.PolynomialWave(samples, [0, -2, 3, -3, 1]),
        "28": wfGen.PolynomialWave(samples, [0, -2, 4, -3, 1]),
        "29": wfGen.PolynomialWave(samples, [0, -2, 5, -3, 1]),
        "30": wfGen.PolynomialWave(samples, [0, -2, 6, -3, 1]),
        "31": wfGen.PolynomialWave(samples, [0, -2, 7, -3, 1]),
        "32": wfGen.PolynomialWave(samples, [0, -2, 8, -3, 1]),
        "33": wfGen.TriangleSaw(samples),
        "34": wfGen.MixWaveforms({"1": wfGen.TriangleSaw(samples), "2": wfGen.Fm()}, {"1": 0.85, "2": 0.15}),
        "35": wfGen.StepRamp(samples),
        "36": wfGen.Fm(samples),
        "37": wfGen.ImpulseTrain(samples),
        "38": wfGen.MixWaveforms({"1": wfGen.PolynomialWave(samples, [0, -2, 1, -3, 1]), "2": wfGen.WhiteNoise(samples, amplitude)}, {"1": 0.9, "2": 0.1}),
        "39": wfGen.PolynomialWave(samples, [0, 1, 0]),
        "40": wfGen.PolynomialWave(samples, [0, -1, 0]),
        "41": wfGen.PolynomialWave(samples, [0.8, 1, -0.1]),
        "42": wfGen.PolynomialWave(samples, [0.7, 1, -0.2]),
        "43": wfGen.PolynomialWave(samples, [0.6, 1, -0.3]),
        "44": wfGen.PolynomialWave(samples, [0.5, 1, -0.4]),
        "45": wfGen.PolynomialWave(samples, [0.4, 1, -0.5]),
        "46": wfGen.PolynomialWave(samples, [0.3, 1, -0.6]),
        "47": wfGen.PolynomialWave(samples, [0.2, 1, -0.7]),
        "48": wfGen.PolynomialWave(samples, [0.1, 1, -0.8]),
    }
    return waveforms

# Generate basic waveforms
def GenerateWaveforms03(samples, amplitude=1.0, harmonics=20):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    waveforms = {
        "01": wfGen.Gaussian(samples, mean=1.0, stddev=0.1, amplitude=amplitude),
        "02": wfGen.Gaussian(samples, mean=0.4, stddev=0.1, amplitude=amplitude),
        "03": wfGen.Gaussian(samples, mean=0.1, stddev=0.1, amplitude=amplitude),
        "04": wfGen.Gaussian(samples, mean=0.3, stddev=0.1, amplitude=amplitude),
        "05": wfGen.Gaussian(samples, mean=0.7, stddev=0.1, amplitude=amplitude),
        "06": wfGen.Gaussian(samples, mean=0.2, stddev=0.1, amplitude=amplitude),
        "07": wfGen.Gaussian(samples, mean=0.6, stddev=0.1, amplitude=amplitude),
        "08": wfGen.Gaussian(samples, mean=0.9, stddev=0.1, amplitude=amplitude),

        "09": wfGen.HalfSineShift(samples, peak_shift=0, amplitude=amplitude),
        "10": wfGen.HalfSineShift(samples, peak_shift=0.1, amplitude=amplitude),
        "11": wfGen.HalfSineShift(samples, peak_shift=0.2, amplitude=amplitude),
        "12": wfGen.HalfSineShift(samples, peak_shift=0.3, amplitude=amplitude),
        "13": wfGen.HalfSineShift(samples, peak_shift=0.4, amplitude=amplitude),
        "14": wfGen.HalfSineShift(samples, peak_shift=0.6, amplitude=amplitude),
        "15": wfGen.HalfSineShift(samples, peak_shift=0.8, amplitude=amplitude),
        "16": wfGen.HalfSineShift(samples, peak_shift=1, amplitude=amplitude),
        
        "17": wfGen.ExponentialDecay(samples, 5),
        "18": wfGen.ExponentialDecay(samples, 10),
        "19": wfGen.ExponentialDecay(samples, 15),
        "20": wfGen.ExponentialDecay(samples, 20),
        "21": wfGen.ExponentialDecay(samples, 25),
        "22": wfGen.ExponentialDecay(samples, 30),
        "23": wfGen.ExponentialDecay(samples, 35),
        "24": wfGen.ExponentialDecay(samples, 40),
        
        "25": wfGen.PolynomialWave(samples, [0, -2, 1], amplitude=amplitude),
        "26": wfGen.PolynomialWave(samples, [0, -2, 2], amplitude=amplitude),
        "27": wfGen.PolynomialWave(samples, [0, -2, 3], amplitude=amplitude),
        "28": wfGen.PolynomialWave(samples, [0, -2, 4], amplitude=amplitude),
        "29": wfGen.PolynomialWave(samples, [0, -2, 5], amplitude=amplitude),
        "30": wfGen.PolynomialWave(samples, [0, -2, 6], amplitude=amplitude),
        "31": wfGen.PolynomialWave(samples, [0, -2, 7], amplitude=amplitude),
        "32": wfGen.PolynomialWave(samples, [0, -2, 8], amplitude=amplitude),
        
        "33": wfGen.TriangleSaw(samples),
        "34": wfGen.MixWaveforms({"1": wfGen.TriangleSaw(samples), "2": wfGen.Fm()}, {"1": 0.85, "2": 0.15}),
        "35": wfGen.StepRamp(samples),
        "36": wfGen.Fm(samples),
        "37": wfGen.ImpulseTrain(samples),
        "38": wfGen.MixWaveforms({"1": wfGen.PolynomialWave(samples, [0, -2, 1, -3, 1]), "2": wfGen.WhiteNoise(samples, amplitude)}, {"1": 0.9, "2": 0.1}),
        "39": wfGen.PolynomialWave(samples, [0, 1, 0]),
        "40": wfGen.PolynomialWave(samples, [0, -1, 0]),
        
        "41": wfGen.PolynomialWave(samples, [0.8, 1, -0.1]),
        "42": wfGen.PolynomialWave(samples, [0.7, 1, -0.2]),
        "43": wfGen.PolynomialWave(samples, [0.6, 1, -0.3]),
        "44": wfGen.PolynomialWave(samples, [0.5, 1, -0.4]),
        "45": wfGen.PolynomialWave(samples, [0.4, 1, -0.5]),
        "46": wfGen.PolynomialWave(samples, [0.3, 1, -0.6]),
        "47": wfGen.PolynomialWave(samples, [0.2, 1, -0.7]),
        "48": wfGen.PolynomialWave(samples, [0.1, 1, -0.8]),
    }
    return waveforms

def GenerateWaveforms04(samples):
    kicks = {
        "01": wfGen.generate_kick(samples=samples),
        "02": wfGen.generate_kick(samples=samples, freq_start=10),
        "03": wfGen.generate_kick(samples=samples, freq_start=20),
        "04": wfGen.generate_kick(samples=samples, freq_start=30),
        "05": wfGen.generate_kick(samples=samples, freq_start=1000),
        "06": wfGen.generate_kick(samples=samples, freq_start=2000),
        "07": wfGen.generate_kick(samples=samples, freq_start=3000),
        "08": wfGen.generate_kick(samples=samples, freq_start=200, freq_end=150, decay_rate=100, duration=1),

        "09": wfGen.generate_kick(freq_start=10, freq_end=5, decay_rate=10, samples=samples, sample_rate=48000, duration=0.1),
        "10": wfGen.generate_kick(freq_start=10, freq_end=0.5, decay_rate=20, samples=samples, sample_rate=48000, duration=0.2),
        "11": wfGen.generate_kick(freq_start=0.3, freq_end=0.1, decay_rate=5, samples=samples, sample_rate=48000, duration=0.3),
        "12": wfGen.generate_kick(freq_start=0.1, freq_end=1, decay_rate=10, samples=samples, sample_rate=48000, duration=0.4),
        "13": wfGen.generate_kick(freq_start=10, freq_end=0.1, decay_rate=3, samples=samples, sample_rate=48000, duration=0.5),
        "14": wfGen.generate_kick(freq_start=1, freq_end=3, decay_rate=1, samples=samples, sample_rate=48000, duration=0.6),
        "15": wfGen.generate_kick(freq_start=2, freq_end=3, decay_rate=30, samples=samples, sample_rate=48000, duration=0.7),
        "16": wfGen.generate_kick(freq_start=3, freq_end=0.2, decay_rate=50, samples=samples, sample_rate=48000, duration=0.8),

        "17": wfGen.generate_kick_table(samples=samples, freq_start=10, freq_end=5, attack=0.1, decay=0.2, sustain_level=0, sustain_time=0, release=0.1, duration=1),
        "18": wfGen.generate_kick_table(samples=samples, freq_start=10),
        "19": wfGen.generate_kick_table(samples=samples, freq_start=20),
        "20": wfGen.generate_kick_table(samples=samples, freq_start=30),
        "21": wfGen.generate_kick_table(samples=samples, freq_start=1000),
        "22": wfGen.generate_kick_table(samples=samples, freq_start=2000),
        "23": wfGen.generate_kick_table(samples=samples, freq_start=3000),
        "24": wfGen.generate_kick_table(samples=samples, freq_start=200, freq_end=150, decay=0.5, duration=0.5),

        "25": wfGen.generate_kick_table(samples=samples),
        "26": wfGen.generate_kick_table(samples=samples, duration=0.5),
        "27": wfGen.generate_kick_table(samples=samples, duration=0.1),
        "28": wfGen.generate_kick_table(samples=samples, freq_start=30, freq_end=15, release=0.4, sustain_level=0.5, sustain_time=0.1),
        "29": wfGen.generate_kick_table(samples=samples, freq_start=110, freq_end=15, release=0.4, sustain_level=0.5, sustain_time=0.1),
        "30": wfGen.generate_kick_table(samples=samples, freq_start=2000),
        "31": wfGen.generate_kick_table(samples=samples, freq_start=3000),
        "32": wfGen.generate_kick_table(samples=samples, freq_start=200, freq_end=150, duration=0.25),

        "33": wfGen.generate_snare(decay_rate=15, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "34": wfGen.generate_snare(decay_rate=20, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "35": wfGen.generate_snare(decay_rate=25, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "36": wfGen.generate_snare(decay_rate=30, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "37": wfGen.generate_snare(decay_rate=35, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "38": wfGen.generate_snare(decay_rate=40, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "39": wfGen.generate_snare(decay_rate=45, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "40": wfGen.generate_snare(decay_rate=50, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),

        "41": wfGen.generate_snare(decay_rate=15, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "42": wfGen.generate_snare(decay_rate=11, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "43": wfGen.generate_snare(decay_rate=7, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.5),
        "44": wfGen.generate_snare(decay_rate=15, noise_mix=0.8, resonance_freq=100, samples=samples, sample_rate=48000, duration=0.5),
        "45": wfGen.generate_snare(decay_rate=15, noise_mix=0.8, resonance_freq=50, samples=samples, sample_rate=48000, duration=0.5),
        "46": wfGen.generate_snare(decay_rate=15, noise_mix=0.8, resonance_freq=10, samples=samples, sample_rate=48000, duration=0.5),
        "47": wfGen.generate_snare(decay_rate=15, noise_mix=0.8, resonance_freq=1000, samples=samples, sample_rate=48000, duration=0.5),
        "48": wfGen.generate_snare(decay_rate=15, noise_mix=0.8, resonance_freq=200, samples=samples, sample_rate=48000, duration=0.001)
    }
    return kicks

def GenerateWaveforms05(samples, amplitude=1.0):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    waveforms = {
        "01": wfGen.SineWave(samples=samples, amplitude=amplitude),
        "02": wfGen.TriangleWave(samples=samples, amplitude=amplitude),
        "03": wfGen.TriangleSaw(samples=samples, amplitude=amplitude),
        "04": wfGen.SawWave(phase=phase, amplitude=amplitude),
        "05": wfGen.RampWave(phase=phase, amplitude=amplitude),
        "06": wfGen.SquareWave(samples=samples, amplitude=amplitude),
        "07": wfGen.WhiteNoise(samples=samples, amplitude=amplitude),
        "08": wfGen.HalfSineShift(samples=samples, peak_shift=0.8, amplitude=amplitude),

        "09": wfGen.SineWave(samples=samples, amplitude=2.0),
        "10": wfGen.TriangleWave(samples=samples, amplitude=2.0),
        "11": wfGen.TriangleSaw(samples=samples, amplitude=2.0),
        "12": wfGen.SawWave(phase=phase, amplitude=2.0),
        "13": wfGen.RampWave(phase=phase, amplitude=2.0),
        "14": wfGen.SquareWave(samples=samples, amplitude=2.0),
        "15": wfGen.WhiteNoise(samples=samples, amplitude=2.0),
        "16": wfGen.HalfSineShift(samples=samples, peak_shift=0.8, amplitude=2.0),
    }
    return waveforms

# Generate basic waveforms
def GenerateWaveforms06(samples, amplitude=1.0, harmonics=20):
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    waveforms = {
        "01": wfGen.FourierWaveform(samples, harmonics, amplitude),
        "02": wfGen.FourierSquareWave(samples, 16, amplitude),
        "03": wfGen.FourierSquareWave(samples, harmonics, amplitude),
        "04": wfGen.LimitedSquareWave(samples, 16, amplitude),
        "05": wfGen.FourierTriangleWave(samples, harmonics, amplitude),
        "06": wfGen.MixWaveforms({"1": wfGen.FourierSquareWave(samples, 16, amplitude), "2": wfGen.FourierWaveform(samples, harmonics, amplitude), "3": wfGen.FourierTriangleWave(samples, harmonics, amplitude)}, {"1": 0.4, "2": 0.4, "3": 0.2}),
    }
    return waveforms


# Example Usage
if __name__ == "__main__":

    appFolder = folders.appFolder()
    folderHeader = folders.folder(f"{appFolder}\\headerFile\\")
    folderLookupTables = folders.folder(f"{appFolder}\\lookupTables\\")

    folderWaveforms01 = folders.folder(f"{folderLookupTables}waveforms01\\")
    folderWaveforms02 = folders.folder(f"{folderLookupTables}waveforms02\\")
    folderWaveforms03 = folders.folder(f"{folderLookupTables}waveforms03\\")
    folderWaveforms04 = folders.folder(f"{folderLookupTables}waveforms04\\")
    folderWaveforms05 = folders.folder(f"{folderLookupTables}waveforms05\\")
    folderRandomWf01 = folders.folder(f"{folderLookupTables}waveformsRandom01\\")
    folderRandomWf02 = folders.folder(f"{folderLookupTables}waveformsRandom02\\")
    folderRandomWf03 = folders.folder(f"{folderLookupTables}waveformsRandom03\\")
    # Folder with waveforms in csv format to generate header data
    foldersWithWaveforms = [folderWaveforms01, folderWaveforms02, folderWaveforms03, folderWaveforms04, folderWaveforms05, folderRandomWf01, folderRandomWf02, folderRandomWf03]

    folderGeneratedRandomWaveforms01 = folders.folder(f"{folderLookupTables}generatedRandomWaveforms01\\")

    folderWavetables01 = folders.folder(f"{folderLookupTables}wavetables01\\")
    folderWavetables02 = folders.folder(f"{folderLookupTables}wavetables02\\")
    folderWavetables03 = folders.folder(f"{folderLookupTables}wavetables03\\")
    folderWavetables04 = folders.folder(f"{folderLookupTables}wavetables04\\")
    folderWavetables05 = folders.folder(f"{folderLookupTables}wavetables05\\")
    folderWavetables06 = folders.folder(f"{folderLookupTables}wavetables06\\")
    folderWavetables07 = folders.folder(f"{folderLookupTables}wavetables07\\")

    samples = 1024

    # Generate basic waveforms
    waveforms01 = GenerateWaveforms01(samples, amplitude=1.0, harmonics=128)
    SaveWaveformsToCsv(waveforms01, folderWaveforms01)
    # Generate basic waveforms
    waveforms02 = GenerateWaveforms02(samples, amplitude=1.5)
    SaveWaveformsToCsv(waveforms02, folderWaveforms02)
    # Generate basic waveforms
    waveforms03 = GenerateWaveforms03(samples, amplitude=1.5)
    SaveWaveformsToCsv(waveforms03, folderWaveforms03)

    waveforms04 = GenerateWaveforms04(samples)
    SaveWaveformsToCsv(waveforms04, folderWaveforms04)

    # # Generate basic waveforms
    # waveforms05 = GenerateWaveforms05(samples, amplitude=1.0)
    # SaveWaveformsToCsv(waveforms05, folderWaveforms05)

    # RandomWaveform(folderGeneratedRandomWaveforms01)
    # RandomMixWaveform(folderGeneratedRandomWaveforms01)
    # rename.rename_files_to_numbers(folderGeneratedRandomWaveforms01, 1)

    GenerateHppFromFolders(foldersWithWaveforms, folderHeader + "waveforms.txt", samples)

    # # Wavetable folder 01
    # n = "WT_001"
    # wt = CreateWavetableFromWaveforms([waveforms01["01"],
    #                                    waveforms01["04"],
    #                                    waveforms01["06"],
    #                                    waveforms01["05"],
    #                                    waveforms01["02"],
    #                                    waveforms01["03"],
    #                                    waveforms01["07"],
    #                                    waveforms01["08"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables01}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_002"
    # wt = CreateWavetableFromWaveforms([waveforms01["09"],
    #                                    waveforms01["10"],
    #                                    waveforms01["11"],
    #                                    waveforms01["12"],
    #                                    waveforms01["13"],
    #                                    waveforms01["14"],
    #                                    waveforms01["15"],
    #                                    waveforms01["16"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables01}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_003"
    # wt = CreateWavetableFromWaveforms([waveforms01["17"],
    #                                    waveforms01["18"],
    #                                    waveforms01["19"],
    #                                    waveforms01["20"],
    #                                    waveforms01["21"],
    #                                    waveforms01["22"],
    #                                    waveforms01["23"],
    #                                    waveforms01["24"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables01}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_004"
    # wt = CreateWavetableFromWaveforms([waveforms01["33"],
    #                                    waveforms01["34"],
    #                                    waveforms01["35"],
    #                                    waveforms01["36"],
    #                                    waveforms01["37"],
    #                                    waveforms01["37"],
    #                                    waveforms01["39"],
    #                                    waveforms01["40"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables01}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_005"
    # wt = CreateWavetableFromWaveforms([waveforms02["17"],
    #                                    waveforms02["18"],
    #                                    waveforms02["19"],
    #                                    waveforms02["20"],
    #                                    waveforms02["21"],
    #                                    waveforms02["22"],
    #                                    waveforms02["23"],
    #                                    waveforms02["24"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables01}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_006"
    # wt = GenerateProgressiveWavetable(waveforms01["02"])
    # SaveWavetableToCsv(wt, f"{folderWavetables01}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_007"
    # wt = GenerateProgressiveWavetable(waveforms01["03"])
    # SaveWavetableToCsv(wt, f"{folderWavetables01}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_008"
    # wt = GenerateProgressiveWavetable(waveforms01["05"])
    # SaveWavetableToCsv(wt, f"{folderWavetables01}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    # # Wavetable folder 02
    # n = "WT_009"
    # wt = GenerateCustomProgressiveWavetable(waveforms01["02"], transformation="harmonics")
    # SaveWavetableToCsv(wt, f"{folderWavetables02}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_010"
    # wt = GenerateCustomProgressiveWavetable(waveforms01["03"], transformation="harmonics")
    # SaveWavetableToCsv(wt, f"{folderWavetables02}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_011"
    # wt = GenerateCustomProgressiveWavetable(waveforms01["05"], transformation="harmonics")
    # SaveWavetableToCsv(wt, f"{folderWavetables02}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_012"
    # wt = GenerateCustomProgressiveWavetable(waveforms01["10"], transformation="harmonics")
    # SaveWavetableToCsv(wt, f"{folderWavetables02}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_013"
    # wt = GenerateCustomProgressiveWavetable(waveforms01["02"], transformation="smoothing")
    # SaveWavetableToCsv(wt, f"{folderWavetables02}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_014"
    # wt = GenerateCustomProgressiveWavetable(waveforms01["03"], transformation="smoothing")
    # SaveWavetableToCsv(wt, f"{folderWavetables02}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_015"
    # wt = GenerateCustomProgressiveWavetable(waveforms01["05"], transformation="smoothing")
    # SaveWavetableToCsv(wt, f"{folderWavetables02}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_016"
    # wt = GenerateCustomProgressiveWavetable(waveforms01["10"], transformation="smoothing")
    # SaveWavetableToCsv(wt, f"{folderWavetables02}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    # # Wavetable folder 03
    # n = "WT_017"
    # wt = GenerateWaveformProgression(waveforms01["06"], waveforms01["03"])
    # SaveWavetableToCsv(wt, f"{folderWavetables03}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_018"
    # wt = GenerateWaveformProgression(waveforms01["03"], waveforms01["05"])
    # SaveWavetableToCsv(wt, f"{folderWavetables03}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_019"
    # wt = GenerateWaveformProgression(waveforms01["05"], waveforms01["15"])
    # SaveWavetableToCsv(wt, f"{folderWavetables03}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_020"
    # wt = GenerateWaveformProgression(waveforms01["10"], waveforms01["11"])
    # SaveWavetableToCsv(wt, f"{folderWavetables03}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_021"
    # wt = CreateWavetableFromWaveforms([waveforms01["25"],
    #                                    waveforms01["26"],
    #                                    waveforms01["27"],
    #                                    waveforms01["28"],
    #                                    waveforms01["29"],
    #                                    waveforms01["30"],
    #                                    waveforms01["31"],
    #                                    waveforms01["32"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables03}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_022"
    # wt = CreateWavetableFromWaveforms([waveforms01["41"],
    #                                    waveforms01["42"],
    #                                    waveforms01["43"],
    #                                    waveforms01["44"],
    #                                    waveforms01["45"],
    #                                    waveforms01["46"],
    #                                    waveforms01["47"],
    #                                    waveforms01["48"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables03}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_023"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf01, ["01.csv", "02.csv", "03.csv", "04.csv", "05.csv", "06.csv", "07.csv", "08.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables03}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_024"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf01, ["09.csv", "10.csv", "11.csv", "12.csv", "13.csv", "14.csv", "15.csv", "16.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables03}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    # # Wavetable folder 04
    # n = "WT_025"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf01, ["17.csv", "18.csv", "19.csv", "20.csv", "21.csv", "22.csv", "23.csv", "24.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables04}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_026"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf01, ["25.csv", "26.csv", "27.csv", "28.csv", "29.csv", "30.csv", "31.csv", "32.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables04}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_027"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf01, ["33.csv", "34.csv", "35.csv", "36.csv", "37.csv", "38.csv", "39.csv", "40.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables04}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_028"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf01, ["41.csv", "42.csv", "43.csv", "44.csv", "45.csv", "46.csv", "47.csv", "48.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables04}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_029"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf02, ["01.csv", "02.csv", "03.csv", "04.csv", "05.csv", "06.csv", "07.csv", "08.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables04}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_030"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf02, ["09.csv", "10.csv", "11.csv", "12.csv", "13.csv", "14.csv", "15.csv", "16.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables04}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_031"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf02, ["17.csv", "18.csv", "19.csv", "20.csv", "21.csv", "22.csv", "23.csv", "24.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables04}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_032"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf02, ["25.csv", "26.csv", "27.csv", "28.csv", "29.csv", "30.csv", "31.csv", "32.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables04}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    # # Wavetable folder 05
    # n = "WT_033"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf02, ["33.csv", "34.csv", "35.csv", "36.csv", "37.csv", "38.csv", "39.csv", "40.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables05}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_034"
    # wt = CreateWavetableFromCSVFiles(folderRandomWf02, ["41.csv", "42.csv", "43.csv", "44.csv", "45.csv", "46.csv", "47.csv", "48.csv"])
    # SaveWavetableToCsv(wt, f"{folderWavetables05}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_035"
    # wt = CreateWavetableFromWaveforms([waveforms04["01"],
    #                                    waveforms04["02"],
    #                                    waveforms04["03"],
    #                                    waveforms04["04"],
    #                                    waveforms04["05"],
    #                                    waveforms04["05"],
    #                                    waveforms04["07"],
    #                                    waveforms04["08"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables05}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_036"
    # wt = CreateWavetableFromWaveforms([waveforms04["09"],
    #                                    waveforms04["10"],
    #                                    waveforms04["11"],
    #                                    waveforms04["12"],
    #                                    waveforms04["13"],
    #                                    waveforms04["14"],
    #                                    waveforms04["15"],
    #                                    waveforms04["16"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables05}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_037"
    # wt = CreateWavetableFromWaveforms([waveforms04["17"],
    #                                    waveforms04["18"],
    #                                    waveforms04["19"],
    #                                    waveforms04["20"],
    #                                    waveforms04["21"],
    #                                    waveforms04["22"],
    #                                    waveforms04["23"],
    #                                    waveforms04["24"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables05}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_038"
    # wt = CreateWavetableFromWaveforms([waveforms04["25"],
    #                                    waveforms04["26"],
    #                                    waveforms04["27"],
    #                                    waveforms04["28"],
    #                                    waveforms04["29"],
    #                                    waveforms04["30"],
    #                                    waveforms04["31"],
    #                                    waveforms04["32"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables05}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_039"
    # wt = CreateWavetableFromWaveforms([waveforms03["01"],
    #                                    waveforms03["02"],
    #                                    waveforms03["03"],
    #                                    waveforms03["04"],
    #                                    waveforms03["05"],
    #                                    waveforms03["06"],
    #                                    waveforms03["07"],
    #                                    waveforms03["08"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables05}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_040"
    # wt = CreateWavetableFromWaveforms([waveforms03["09"],
    #                                    waveforms03["10"],
    #                                    waveforms03["11"],
    #                                    waveforms03["12"],
    #                                    waveforms03["13"],
    #                                    waveforms03["14"],
    #                                    waveforms03["15"],
    #                                    waveforms03["16"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables05}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    # # Wavetable folder 06
    # n = "WT_041"
    # wt = CreateWavetableFromWaveforms([wfGen.MixWaveforms({"1": waveforms03["09"], "2": waveforms03["41"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["10"], "2": waveforms03["42"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["11"], "2": waveforms03["43"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["12"], "2": waveforms03["44"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["13"], "2": waveforms03["45"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["14"], "2": waveforms03["46"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["15"], "2": waveforms03["47"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["16"], "2": waveforms03["48"]}, {"1": 0.5, "2": 0.5})
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables06}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_042"
    # wt = CreateWavetableFromWaveforms([wfGen.MixWaveforms({"1": waveforms03["09"], "2": waveforms03["17"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["10"], "2": waveforms03["18"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["11"], "2": waveforms03["19"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["12"], "2": waveforms03["20"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["13"], "2": waveforms03["21"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["14"], "2": waveforms03["22"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["15"], "2": waveforms03["23"]}, {"1": 0.5, "2": 0.5}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["16"], "2": waveforms03["24"]}, {"1": 0.5, "2": 0.5})
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables06}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_043"
    # wt = CreateWavetableFromWaveforms([wfGen.MixWaveforms({"1": waveforms03["25"], "2": waveforms03["33"]}, {"1": 0.3, "2": 0.7}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["26"], "2": waveforms03["34"]}, {"1": 0.3, "2": 0.7}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["27"], "2": waveforms03["35"]}, {"1": 0.3, "2": 0.7}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["28"], "2": waveforms03["36"]}, {"1": 0.3, "2": 0.7}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["29"], "2": waveforms03["37"]}, {"1": 0.3, "2": 0.7}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["30"], "2": waveforms03["38"]}, {"1": 0.3, "2": 0.7}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["31"], "2": waveforms03["39"]}, {"1": 0.3, "2": 0.7}) ,
    #                                    wfGen.MixWaveforms({"1": waveforms03["32"], "2": waveforms03["40"]}, {"1": 0.3, "2": 0.7})
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables06}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)
    # n = "WT_044"
    # wt = CreateWavetableFromWaveforms([waveforms05["01"],
    #                                    waveforms05["02"],
    #                                    waveforms05["03"],
    #                                    waveforms05["04"],
    #                                    waveforms05["05"],
    #                                    waveforms05["06"],
    #                                    waveforms05["07"],
    #                                    waveforms05["08"]
    #                                    ])
    # SaveWavetableToCsv(wt, f"{folderWavetables06}{n}.csv")
    # SaveWavetableToHeader(wt, f"{folderHeader}{n}.hpp", n)

    # Wavetable folder 07

    # JoinHeaderFilesWithoutDirectives(folderHeader, folderHeader + "wavetables.txt")

    plotAll.plot_lookup_tables_in_tabs(folderLookupTables, wavetable_split=samples)


