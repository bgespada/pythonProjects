import os

# Set the path to your folder
folder_path = 'C:\\Users\\BGE\\Desktop\\PROJECTES\\DrumKitTR909 Project\\Samples\\Processed\\Consolidate'  # ← change this to your folder path

for filename in os.listdir(folder_path):
    if filename.endswith('.wav') and ' ' in filename:
        original_path = os.path.join(folder_path, filename)
        new_name = filename.split(' ')[0] + '_16.wav'
        new_path = os.path.join(folder_path, new_name)
        os.rename(original_path, new_path)
        print(f'Renamed: {filename} → {new_name}')
