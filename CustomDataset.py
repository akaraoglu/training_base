# custom_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2 as F
from PIL import Image
import numpy as np
import os
from GPUDataAugmentation import NumpyToCudaTensor

# Function to create DataLoaders
def create_dataloaders(data_dir, class_names, batch_size=4, num_workers=4, pin_memory=True):
    # Define the transformations for training and validation
    train_transform  = transforms.Compose([
        # NumpyToCudaTensor(),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        F.RandomResizedCrop(size=(256, 256), antialias=True),
        F.RandomHorizontalFlip(p=0.5),
        F.RandomVerticalFlip(p=0.5)
    ])
    

    val_transform   = F.Compose([
        # NumpyToCudaTensor(),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        F.RandomResizedCrop(size=(256, 256), antialias=True),
        F.RandomHorizontalFlip(p=0.5),
        F.RandomVerticalFlip(p=0.5),
    ])

    # Create datasets
    image_datasets = {
        'train': CustomImageDataset(os.path.join(data_dir, 'train'), class_names, transform=None),
        'val': CustomImageDataset(os.path.join(data_dir, 'val'), class_names, transform=None)
    }

    image_datasets['train'].print_dataset_summary()
    image_datasets['val'].print_dataset_summary()

    # Create DataLoaders with transformations applied at the batch level
    dataloaders = {
        x: CustomDataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=num_workers, collate_fn=None, pin_memory=pin_memory, transform=(train_transform if x == 'train' else val_transform))
        for x in ['train', 'val']
    }

    # Get dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    return dataloaders, dataset_sizes

import numpy as np

def custom_collate_fn(batch):
    """
    Custom collate function to process a batch of data using NumPy arrays.
    
    Args:
        batch (list): A list of tuples (image, label) where image is a NumPy array and label is an integer.
    
    Returns:
        tuple: A tuple of (images, labels) where images is a batch of images as NumPy arrays and labels as a NumPy array.
    """
    # Separate images and labels
    images, labels = zip(*batch)
    
    # Stack images into a single NumPy array
    images = np.array(images)
    
    # Convert labels to a NumPy array
    labels = np.array(labels)
    
    return images, labels


# Custom Image Dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.class_names = class_names
        self.transform = transform
        self.img_paths = []
        self.labels = []

        for class_name in class_names:
            class_dir = os.path.join(img_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(".jpg"):
                    self.img_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_names.index(class_name))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # Load the image using PIL
        image = Image.open(img_path).convert("RGB")

        # Convert the image to a NumPy array
        image = np.array(image)
        
        # if (image.shape[0] or image.shape[1]) != 150:
        #     print(img_path)
        #     print(image.shape)
            

        # # Convert image to a tensor and send to GPU
        image = transforms.ToTensor()(image).to('cuda')
        image = image / 255.0
        
        label = self.labels[idx]
        # if self.transform:
        #     image = self.transform(image)
        return image, label

    def print_dataset_summary(self):
        summary = {}
        total_images = 0

        for class_name in self.class_names:
            class_dir = os.path.join(self.img_dir, class_name)
            num_images = len([img_name for img_name in os.listdir(class_dir) if img_name.endswith(".jpg")])
            summary[class_name] = num_images
            total_images += num_images

        num_classes = len(self.class_names)

        print("Dataset Summary:")
        print("----------------")
        print(f"Total number of images: {total_images}")
        print(f"Number of classes: {num_classes}")
        print("\nNumber of images per class:")
        for class_name, num_images in summary.items():
            print(f" - {class_name}: {num_images} images")

# Custom DataLoader 
class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False, collate_fn=None, transform=None):
        super(CustomDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        self.transform = transform

    def __iter__(self):
        for images, labels in super(CustomDataLoader, self).__iter__():
            # Apply batch-level transformations
            if self.transform:
                images = self.transform(images)

            labels = torch.tensor(labels).to('cuda')

            yield images, labels
            
    
