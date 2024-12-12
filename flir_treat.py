import os
from PIL import Image

# Define the base directory
base_dir = "datasets/FLIR"

# Define paths
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Function to process and concatenate images
def process_images(subfolder):
    # Define the output directory
    output_dir = os.path.join(base_dir, f"{subfolder}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define RGB and thermal directories
    rgb_dir = os.path.join(base_dir, subfolder, "RGB")
    thermal_dir = os.path.join(base_dir, subfolder, "thermal_8_bit")
    
    # Loop through RGB images
    for rgb_filename in os.listdir(rgb_dir):
        if rgb_filename.endswith(".jpg"):
            # Open RGB image
            rgb_path = os.path.join(rgb_dir, rgb_filename)
            rgb_image = Image.open(rgb_path)
            
            # Crop to centered square and resize to 512x512
            width, height = rgb_image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            rgb_cropped = rgb_image.crop((left, top, left + side, top + side))
            rgb_resized = rgb_cropped.resize((512, 512))
            
            # Get corresponding thermal image
            thermal_filename = rgb_filename.replace(".jpg", ".jpeg")
            thermal_path = os.path.join(thermal_dir, thermal_filename)
            if os.path.exists(thermal_path):
                thermal_image = Image.open(thermal_path)
                
                # Resize thermal image to 512x512
                thermal_resized = thermal_image.resize((512, 512))
                
                # Concatenate images side by side (RGB on left, thermal on right)
                combined_image = Image.new("RGB", (1024, 512))
                combined_image.paste(rgb_resized, (0, 0))
                combined_image.paste(thermal_resized, (512, 0))
                
                # Save combined image in output folder
                output_filename = os.path.join(output_dir, rgb_filename.replace(".jpg", "_AB.jpg"))
                combined_image.save(output_filename, "JPEG")
                
                print(f"Processed {output_filename}")
            else:
                print(f"Thermal image not found for {rgb_filename}")

# Process train and test subfolders
process_images("train")
process_images("test")
