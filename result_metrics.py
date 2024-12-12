import argparse
import os
import re
import shutil
import subprocess
import numpy as np
import torch
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import LPIPS
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm
import cv2



# Function to compute LPIPS between GT and Fake images
def calculate_lpips(gt_image, fake_image, lpips_model):
    gt_image = gt_image.unsqueeze(0)
    fake_image = fake_image.unsqueeze(0)

    lpips_score = lpips_model(gt_image, fake_image)
    return lpips_score.item()

# Function to compute PSNR between GT and Fake images
def calculate_psnr(gt_image, fake_image):
    return psnr(np.array(gt_image), np.array(fake_image))

def compute_l1_score(gt_path, generated_path):
    """
    Computes the L1 score (mean absolute error) between a ground truth (GT) image and a generated image.
    
    Args:
        gt_path (str): Path to the ground truth image.
        generated_path (str): Path to the generated image.
        
    Returns:
        float: The L1 score (mean absolute error).
    """
    # Compute absolute difference
    abs_diff = np.abs(gt_image.astype(np.float32) - generated_image.astype(np.float32))
    
    # Compute mean absolute error
    l1_score = np.mean(abs_diff)
    return l1_score

def compute_ssim(gt, fake):
    """
    Computes the SSIM (Structural Similarity Index) between a ground truth (GT) image and a generated image.
    
    Args:
        gt_path (str): Path to the ground truth image.
        generated_path (str): Path to the generated image.
        
    Returns:
        float: The SSIM score.
    """
    # Convert images to grayscale
    gt_array = np.array(gt)
    fake_array = np.array(fake)
    gt_gray = cv2.cvtColor(gt_array, cv2.COLOR_BGR2GRAY)
    generated_gray = cv2.cvtColor(fake_array, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    score, _ = ssim(gt_gray, generated_gray, full=True)
    return score

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment', default=None, help='What experiment to score')

    opt = parser.parse_args()
    folder = "results/"+opt.experiment+"/test_latest/images/"

    #make two temp folders
    os.makedirs("gt", exist_ok=True)
    os.makedirs("fake", exist_ok=True)

    gt_folder = "gt"
    fake_folder = "fake"

    # Define the regex patterns
    real_b_pattern = r"real_B"
    fake_b_pattern = r"fake_B"

    counter = 0

    # Iterate through the files in the source folder
    for filename in os.listdir(folder):
        source_path = os.path.join(folder, filename)

        # Skip directories
        if os.path.isdir(source_path):
            continue

        # Copy and rename "real_B" images to the "gt" folder
        if re.search(real_b_pattern, filename):
            new_filename = re.sub(real_b_pattern, "", filename)
            destination_path = os.path.join(gt_folder, new_filename)
            shutil.copy(source_path, destination_path)
            counter += 1

        # Copy and rename "fake_B" images to the "fake" folder
        elif re.search(fake_b_pattern, filename):
            new_filename = re.sub(fake_b_pattern, "", filename)
            destination_path = os.path.join(fake_folder, new_filename)
            shutil.copy(source_path, destination_path)
            counter += 1

    print(f"Total number of images copied: {counter}, {counter//2} pairs")

    # Load the LPIPS model
    lpips_model = LPIPS(net='alex')  # You can choose 'vgg' or 'alex' based on preference
    
    batch_size = 50
    psnr_values = []
    lpips_values = []
    l1_values = []
    ssim_values = []

    gt_list = os.listdir(gt_folder)
    fake_list = os.listdir(fake_folder)
    nb_image = len(os.listdir(gt_folder))

    for img in tqdm(range(0, nb_image, batch_size)):
        # Load GT and Fake images
        gt_images = [Image.open(os.path.join(gt_folder, f)) for f in gt_list[img:img+batch_size]]
        fake_images = [Image.open(os.path.join(fake_folder, f)) for f in fake_list[img:img+batch_size]]

        # Ensure both folders contain the same number of images
        assert len(gt_images) == len(fake_images), "The number of GT and Fake images must be the same"
        
        # Compute PSNR, LPIPS for each image pair (GT vs Fake)
    
        for i, (gt, fake) in enumerate(zip(gt_images, fake_images)):
            # Compute PSNR
            psnr_value = calculate_psnr(gt, fake)
            psnr_values.append(psnr_value)

            # Compute SSIM
            ssim_value = compute_ssim(gt, fake)
            ssim_values.append(ssim_value)
            
            # Compute LPIPS
            gt_tensor = transforms.ToTensor()(gt)
            fake_tensor = transforms.ToTensor()(fake)
            lpips_value = calculate_lpips(gt_tensor, fake_tensor, lpips_model)
            lpips_values.append(lpips_value)

            # Compute L1
            l1_value = torch.nn.L1Loss()(gt_tensor, fake_tensor)
            l1_values.append(l1_value.item())

    
    # Calculate average PSNR and LPIPS
    avg_psnr = np.mean(psnr_values)
    avg_lpips = np.mean(lpips_values)
    avg_l1 = np.mean(l1_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Average PSNR: {avg_psnr}")
    print(f"Average LPIPS: {avg_lpips}")
    print(f"Average L1: {avg_l1}")
    print(f"Average SSIM: {avg_ssim}")
    # Calculate FID between GT and Fake images
    command = ["python", "-m", "pytorch_fid", fake_folder, gt_folder]
    subprocess.run(command, check=True)

    # Clean up the temporary folders
    shutil.rmtree(gt_folder)
    shutil.rmtree(fake_folder)
