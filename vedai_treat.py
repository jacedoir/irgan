import os
from PIL import Image

# Set paths
dataroot = 'datasets/VEDAI'  # Root directory for your dataset
phase = 'train'  # Set to 'train' or 'test'
dir_A = os.path.join(dataroot, f"{phase}A")  # Directory containing A images
dir_B = os.path.join(dataroot, f"{phase}B")  # Directory containing B images
dir_AB = os.path.join(dataroot, f"{phase}")  # Directory to save combined AB images

# Create the AB directory if it doesn't exist
os.makedirs(dir_AB, exist_ok=True)

# Get list of images in trainA and trainB
A_images = sorted([f for f in os.listdir(dir_A) if f.endswith(('.png', '.jpg', '.jpeg'))])
B_images = sorted([f for f in os.listdir(dir_B) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Process each pair of images
for file_name in A_images:
    # Check if corresponding file exists in trainB
    if file_name in B_images:
        A_path = os.path.join(dir_A, file_name)
        B_path = os.path.join(dir_B, file_name)
        
        # Open images
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # Ensure both images have the same height (resize if necessary)
        if A_img.size[1] != B_img.size[1]:
            new_height = min(A_img.size[1], B_img.size[1])
            A_img = A_img.resize((A_img.width, new_height), Image.BICUBIC)
            B_img = B_img.resize((B_img.width, new_height), Image.BICUBIC)
        
        # Concatenate images horizontally
        AB_img = Image.new('RGB', (A_img.width + B_img.width, A_img.height))
        AB_img.paste(A_img, (0, 0))
        AB_img.paste(B_img, (A_img.width, 0))
        
        # Save the combined image with naming convention: base_name_AB.jpg
        AB_path = os.path.join(dir_AB, f"{file_name.split('.')[0]}_AB.jpg")
        AB_img.save(AB_path)
        
        print(f"Saved {AB_path}")
    else:
        print(f"Skipped {file_name} as there is no matching image in trainB.")
