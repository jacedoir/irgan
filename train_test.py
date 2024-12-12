import os
import random
import shutil

# Set paths
dataroot = 'datasets/KAIST'  # Root directory for your dataset
train_dir = os.path.join(dataroot, 'train')  # Directory containing all AB images
test_dir = os.path.join(dataroot, 'test')  # Directory for test images

# Create the test directory if it doesn’t exist
os.makedirs(test_dir, exist_ok=True)

# Get list of all AB images in the train directory
all_images = [f for f in os.listdir(train_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Calculate the number of images to move (20% of the total)
num_images_to_move = int(0.2 * len(all_images))

# Randomly select 20% of the images
test_images = random.sample(all_images, num_images_to_move)

# Move each selected image to the test directory
for img_name in test_images:
    src_path = os.path.join(train_dir, img_name)
    dest_path = os.path.join(test_dir, img_name)
    shutil.move(src_path, dest_path)
    print(f"Moved {img_name} to {test_dir}")

print(f"Moved {num_images_to_move} images to the test directory.")
