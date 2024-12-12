import os
import shutil
import re

# Set paths
dataroot = 'datasets/VEDAI'  # Root directory for your dataset
phase = 'train'  # Set to 'train' or 'test'
dir_images = os.path.join(dataroot, phase)  # Directory containing both co and ir images
dir_A = os.path.join(dataroot, f"{phase}A")  # Directory to store co images (A images)
dir_B = os.path.join(dataroot, f"{phase}B")  # Directory to store ir images (B images)

# Create directories if they don’t exist
os.makedirs(dir_A, exist_ok=True)
os.makedirs(dir_B, exist_ok=True)

# Regular expression to match exact format '00000000_co' or '00000000_ir'
pattern = re.compile(r'^(\d{8})_(co|ir)\.(jpg|jpeg|png)$', re.IGNORECASE)

# Process each file
for img_path in sorted(os.listdir(dir_images)):
    match = pattern.match(img_path)
    if match:
        base_name, suffix, extension = match.groups()  # Extract base name, suffix, and extension
        file_name = f"{base_name}.{extension}"  # New file name without suffix
        
        # Determine destination based on suffix
        if suffix == 'co':
            dest_path = os.path.join(dir_A, file_name)
        elif suffix == 'ir':
            dest_path = os.path.join(dir_B, file_name)
        
        # Move and rename the file
        shutil.move(os.path.join(dir_images, img_path), dest_path)
        print(f"Moved {img_path} to {dest_path}")
    else:
        print(f"Skipped {img_path} as it does not match the required format.")
