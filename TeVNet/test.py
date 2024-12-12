import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

from models import TeVNet
from utils import TeVloss

# Set environment variable for CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def save_image(image_array, path):
    """Saves a numpy array as an image."""
    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(path)



def gray_to_rainbow(grayscale_value):
    colormap = cm.get_cmap('rainbow')
    normalized_value = np.clip(grayscale_value, 0, 1)
    color = colormap(normalized_value)
    return np.array(color[:3]) * 255


def load_model(weights_file, args, device):
    """Loads the model with the specified weights."""
    model = TeVNet(in_channels=3, out_channels=2 + args.vnums, args=args).to(device)
    model = nn.DataParallel(model, device_ids=[0])

    
    # checkpoint = torch.load(weights_file, map_location="cpu")
    
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)

    # # Check for mismatches
    # if missing_keys:
    #     print(f"Warning: Missing keys in model's state_dict: {missing_keys}")
    # if unexpected_keys:
    #     print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

    # Load model weights
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage, weights_only = True)['state_dict'].items():
        if n in state_dict:
            state_dict[n].copy_(p)
        else:
            raise KeyError(f"Key '{n}' not found in model state_dict.")

    model.to(device)
    model.eval()
    return model


def process_image(image_path, device):
    """Processes an image and returns it as a tensor."""
    image = Image.open(image_path).convert('RGB')
    input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    return input_tensor


def save_decomposed_images(preds, input_tensor, output_img_dir, img_name, lossmodule):
    """Saves decomposed images from model predictions."""
    preds_np = preds.cpu().numpy().squeeze(0)
    rec = lossmodule.rec(preds, input_tensor).mul(255.0).cpu().numpy().squeeze(0)
    e = lossmodule.rec_e(preds).mul(255.0).cpu().numpy().squeeze(0)
    T = lossmodule.rec_T(preds).mul(255.0).cpu().numpy().squeeze(0)
    env = lossmodule.rec_env(preds, torch.mean(input_tensor, dim=1)).mul(255.0).cpu().numpy().squeeze(0)


    # T_ready = T[0] / 255.0
    # colored_image = plt.cm.jet(T_ready)
    # colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    save_image(np.transpose(input_tensor.cpu().numpy().squeeze(0), (1, 2, 0)) * 255, os.path.join(output_img_dir, f'{img_name}_ori.png'))
    save_image(rec[0], os.path.join(output_img_dir, f'{img_name}_rec.png'))
    save_image(T[0], os.path.join(output_img_dir, f'{img_name}_T.png'))
    save_image(e[0], os.path.join(output_img_dir, f'{img_name}_e.png'))
    save_image(env[0], os.path.join(output_img_dir, f'{img_name}_env.png'))

    # Save each V component
    for i in range(2, 6):  # Assuming 4 V components
        save_image(preds_np[i] * 255, os.path.join(output_img_dir, f'{img_name}_V{i-1}.png'))


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model

    print("loading model")
    model = load_model(args.weights_file, args, device)
    print(f"model loaded to {device}")

    lossmodule = TeVloss(vnums=args.vnums)

    # Initialize loss tracking
    loss_list = []

    # Process images
    imglist = os.listdir(args.image_dir)
    with open(os.path.join(args.output_dir, 'test_loss.txt'), 'a+') as loss_file:
        for img_num, img in enumerate(tqdm(imglist)):
            img_name = img.replace('.png', '')
            output_img_dir = os.path.join(args.output_dir, img_name)
            os.makedirs(output_img_dir, exist_ok=True)

            # Load and process image
            image_path = os.path.join(args.image_dir, img)
            input_tensor = process_image(image_path, device)

            preds = model(input_tensor)

            print(f"image {img_num + 1} of {len(imglist)} processed")

            # Predict with the model
            with torch.no_grad():
                preds = model(input_tensor)

            print(f"predictions {img_num + 1} of {len(imglist)} loaded")

            T = preds[0, 1, :, :].cpu().numpy()

            colored_image = plt.cm.jet(T)
            colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

            image = Image.fromarray(colored_image.astype(np.uint8))

            # image.save(output_img_dir + ".png")

            # Calculate and log reconstruction loss
            loss = lossmodule.loss_rec(preds, input_tensor)
            loss_list.append(loss.cpu().item())
            loss_file.write(f'{img}, loss: {loss.item():.6f}\n')

            # Save decomposed images
            # save_decomposed_images(preds, input_tensor, output_img_dir, img_name, lossmodule)

        # Log mean loss
        mean_loss = np.mean(loss_list)
        loss_file.write(f'Mean loss: {mean_loss:.6f}\n')
        print(f'Mean loss: {mean_loss:.6f}')


if __name__ == '__main__':

    # torch.cuda.empty_cache()
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--vnums', type=int, default=4)
    parser.add_argument('--smp_model', type=str, default='PAN')
    parser.add_argument('--smp_encoder', type=str, default='resnet50')
    parser.add_argument('--smp_encoder_weights', type=str, default='imagenet')
    parser.add_argument('--output-dir', type=str, default='./output/')
    args = parser.parse_args()

    print(args)

    quit()

    main(args)
