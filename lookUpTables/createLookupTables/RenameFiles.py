import os

def rename_files(directory, prefix="file", start_number=1):
    """
    Renames all files in a directory sequentially starting from `start_number`.
    
    Parameters:
        directory (str): Path to the directory containing files to rename.
        prefix (str): The prefix for the new file names.
        start_number (int): The starting number for the file sequence.
    """
    # Get a sorted list of all files in the directory
    files = sorted(os.listdir(directory))
    number_width = len(str(len(files) + start_number - 1))  # To handle leading zeros

    for index, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):  # Only rename files, skip directories
            new_name = f"{prefix}_{str(index + start_number).zfill(number_width)}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

def rename_files_to_numbers(directory, start_number=1):
    """
    Renames all files in a directory to sequential numbers starting from `start_number`.

    Parameters:
        directory (str): Path to the directory containing files to rename.
        start_number (int): The starting number for the file sequence.
    """
    # Get a sorted list of all files in the directory
    files = sorted(os.listdir(directory))
    number_width = len(str(len(files) + start_number - 1))  # To handle leading zeros

    for index, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):  # Only rename files, skip directories
            new_name = f"{str(index + start_number).zfill(number_width)}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
            
# # Example usage
# directory_path = "path/to/your/directory"
# rename_files(directory_path, prefix="file", start_number=1)
