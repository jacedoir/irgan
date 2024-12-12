import os
import random
import shutil
from PIL import Image
from tqdm import tqdm

# Define the base directory
base_dir = "datasets/KAIST"

# Define paths
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

#need to correct set05 that as no vid folder
temp_dir = os.path.join(base_dir, "set05")
#make a vid folder if not present
if not os.path.exists(os.path.join(temp_dir, "V000")):
    os.makedirs(os.path.join(temp_dir, "V000"), exist_ok=True)
    #move lwir and visible to vid
    shutil.move(os.path.join(temp_dir, "lwir"), os.path.join(temp_dir, "V000", "lwir"))
    shutil.move(os.path.join(temp_dir, "visible"), os.path.join(temp_dir, "V000", "visible"))

#create the directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
# Define the output directory
output_dir = os.path.join(base_dir, "temp")
os.makedirs(output_dir, exist_ok=True)

# Function to process and concatenate images
def process_images(subfolder):
    
    # Define RGB and thermal directories
    temp_trackbar = tqdm(os.listdir(os.path.join(base_dir, subfolder)), desc="Processing videos", leave=False)
    for video in temp_trackbar:
        rgb_dir = os.path.join(base_dir, subfolder, video, "visible")
        thermal_dir = os.path.join(base_dir, subfolder, video, "lwir")
    
        # Loop through RGB images
        temp_trackbar_2 = tqdm(os.listdir(rgb_dir), desc="Processing images", leave=False)
        for rgb_filename in temp_trackbar_2:
            if rgb_filename.endswith(".jpg"):
                # Open RGB image
                rgb_path = os.path.join(rgb_dir, rgb_filename)
                rgb_image = Image.open(rgb_path)
                
                # Crop to centered square and resize to 512x512
                width, height = rgb_image.size
                middle = width // 2
                rgb_cropped = rgb_image.crop((middle - 256, 0, middle + 256, 512))
                
                # Get corresponding thermal image
                thermal_filename = rgb_filename
                thermal_path = os.path.join(thermal_dir, thermal_filename)
                if os.path.exists(thermal_path):
                    thermal_image = Image.open(thermal_path)
                    
                    # Resize thermal image to 512x512
                    thermal_resized = thermal_image.crop((middle - 256, 0, middle + 256, 512))
                    
                    # Concatenate images side by side (RGB on left, thermal on right)
                    combined_image = Image.new("RGB", (1024, 512))
                    combined_image.paste(rgb_cropped, (0, 0))
                    combined_image.paste(thermal_resized, (512, 0))
                    
                    # Save combined image in output folder
                    file_name = subfolder + "_" + video + "_" + rgb_filename.replace(".jpg", "_AB.jpg")
                    output_filename = os.path.join(output_dir, file_name)
                    combined_image.save(output_filename, "JPEG")
                    
# Process sets
list_sets = os.listdir(base_dir)
list_sets.remove("train")
list_sets.remove("test")
list_sets.remove("temp")
trackbar = tqdm(list_sets, desc="Processing sets")
for set in trackbar:
    process_images(set)

#in temp take 80% of the images and move them to train and the rest to test then delete temp
trackbar = tqdm(os.listdir(os.path.join(base_dir, "temp")), desc="Moving images")
for image in trackbar:
    if random.random() < 0.8:
        shutil.move(os.path.join(base_dir, "temp", image), os.path.join(base_dir, "train", image))
    else:
        shutil.move(os.path.join(base_dir, "temp", image), os.path.join(base_dir, "test", image))
shutil.rmtree(os.path.join(base_dir, "temp"))
