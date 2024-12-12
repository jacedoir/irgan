import os
from PIL import Image
import json
from torch.utils.data import Dataset
from torchvision import transforms

# train_transforms = transforms.Compose([
#     transforms.CenterCrop(512),           # Center crop to 512x512
#     transforms.Resize(512),               # Resize to 512x512
#     transforms.RandomHorizontalFlip(),    # Random horizontal flip
#     transforms.RandomCrop(512),           # Random crop of 512x512
#     transforms.ToTensor(),                # Convert image to tensor
# ])

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to fixed size
    transforms.ToTensor()           # Convert to PyTorch tensor
])

eval_transforms = transforms.Compose([
    transforms.CenterCrop(512),           # Center crop to 512x512
    transforms.Resize(512),               # Resize to 512x512
    transforms.ToTensor(),                # Convert image to tensor
])

class TrainDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super(TrainDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform if transform is not None else train_transforms

        # List all image files in the directories
        self.images = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(('jpg', 'png'))])

    def __getitem__(self, idx):
        image_path = self.images[idx]

        # Load images
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor and normalization
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, image

    def __len__(self):
        return len(self.images)




class ConcatedImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        super(ConcatedImageDataset, self).__init__()  # Correct usage of super
        self.image_dir = image_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        
        # Load the image
        image = Image.open(image_path).convert('RGB')  # Ensure RGB mode
        width, height = image.size
        
        # Split the image into input (left) and output (right) halves
        input_image = image.crop((0, 0, width // 2, height))  # Left half
        output_image = image.crop((width // 2, 0, width, height))  # Right half
        
        # Apply transformations if provided
        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)
        
        return input_image, output_image

    def __len__(self):
        return len(self.image_filenames)



class PairedImageDataset(Dataset):
    def __init__(self, input_path, output_path, json_path, transform=None):
        """
        Args:
            json_path (str): Path to the JSON file containing input-output image pairs.
            transform (callable, optional): Optional transform to be applied on images.
        """
        super(PairedImageDataset, self).__init__()
        
        # Load image pairs from JSON file
        with open(json_path, 'r') as f:
            self.image_pairs = json.load(f)
        
        self.transform = transform if transform is not None else train_transforms
        self.images = list(self.image_pairs.items())  # List of (input_image, output_image) pairs
        self.input = input_path
        self.output = output_path

    def __getitem__(self, idx):

        input_image_path, output_image_path = self.images[idx]

        input_path = self.input + input_image_path
        # output_path = self.output + output_image_path


        # Load the input and output images
        input_image = Image.open(input_path).convert('RGB')
        output_image = Image.open(output_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return output_image, output_image

    def __len__(self):
        return len(self.images)




class EvalDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super(EvalDataset, self).__init__()
        self.img_dir = img_dir
        self.transform = transform if transform is not None else eval_transforms

        # List all image files in the directories
        self.images = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(('jpg', 'png'))])

    def __getitem__(self, idx):
        image_path = self.images[idx]

        # Load images
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor and normalization
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        return image, image

    def __len__(self):
        return len(self.images)
