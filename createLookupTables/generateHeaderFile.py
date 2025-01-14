import os
import numpy as np
from pathlib import Path

def generate_hpp_from_csv(folder_path, output_file):
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
            hpp_file.write(f"// {csv_file}\n")
            hpp_file.write(f"const float {variable_name}[] = {{\n")
            
            # Write the waveform values
            formatted_values = ", ".join(f"{v:.6f}" for v in waveform)
            hpp_file.write(f"    {formatted_values}\n")
            
            hpp_file.write("};\n\n")
        
        # Close header guards
        hpp_file.write("#endif // LOOKUP_TABLES_HPP\n")

# Example usage
if __name__ == "__main__":
    appFolder = Path(__file__).parent.absolute()
    folderIn = f"{appFolder}\\lookupTables\\"
    if not os.path.exists(folderIn):
        os.makedirs(folderIn)
    folderOut = f"{appFolder}\\headerFile\\"
    if not os.path.exists(folderOut):
        os.makedirs(folderOut)
    generate_hpp_from_csv(folderIn, folderOut + "lookupTables.hpp")
