import os

# Specify the paths of the folders
folders = [
    "Battery_installed",
    "Empty_board",
    "Manifold_installed",
    "Screws_installed",
    "with_U_clamp",
]

# Specify the destination folder (optional: keep files organized)
destination_folder = "Data"
os.makedirs(destination_folder, exist_ok=True)

# Initialize the starting number for progressive renaming
starting_number = 1

# Loop through each folder and process files
for folder in folders:
    if not os.path.exists(folder):
        print(f"Folder does not exist: {folder}")
        continue

    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)

        # Process only .jpg files
        if not file_name.lower().endswith(".jpg") or not os.path.isfile(file_path):
            continue

        # Generate the new file name for the .jpg file
        new_file_name = f"file_{starting_number:03d}.jpg"  # e.g., file_001.jpg
        new_file_path = os.path.join(destination_folder, new_file_name)

        # Check if a corresponding .txt file exists
        base_name = os.path.splitext(file_name)[0]  # Get the base name without extension
        txt_file_name = f"{base_name}.txt"
        txt_file_path = os.path.join(folder, txt_file_name)

        # Rename the .jpg file
        os.rename(file_path, new_file_path)
        print(f"Renamed: {file_path} -> {new_file_path}")

        # Rename the corresponding .txt file, if it exists
        if os.path.isfile(txt_file_path):
            new_txt_file_name = f"file_{starting_number:03d}.txt"
            new_txt_file_path = os.path.join(destination_folder, new_txt_file_name)
            os.rename(txt_file_path, new_txt_file_path)
            print(f"Renamed: {txt_file_path} -> {new_txt_file_path}")

        # Increment the counter
        starting_number += 1

print("Renaming completed.")
