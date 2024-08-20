import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2 as F
from PIL import Image
import numpy as np
import os
from GPUDataAugmentation import NumpyToCudaTensor, ToDeviceAndNormalize

# Function to create DataLoaders for training and validation datasets
def create_dataloaders(data_dir, class_names, batch_size=4, num_workers=4, pin_memory=True):
    """
    Create DataLoaders for training and validation datasets with batch-level transformations.

    Args:
        data_dir (str): Directory containing the data.
        class_names (list): List of class names corresponding to subdirectories in data_dir.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): Whether to copy tensors into CUDA pinned memory. 

    Returns:
        dict: Dictionary containing DataLoaders for 'train' and 'val' datasets.
        dict: Dictionary containing the sizes of the 'train' and 'val' datasets.
    """
    # Define the transformations for training and validation
    train_transform = transforms.Compose([
        ToDeviceAndNormalize(),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        F.RandomResizedCrop(size=(256, 256), antialias=True),
        F.RandomHorizontalFlip(p=0.5),
        F.RandomVerticalFlip(p=0.5)
    ])

    val_transform = transforms.Compose([
        ToDeviceAndNormalize(),
        F.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        F.Resize(size=(256, 256), antialias=True),  # Resize instead of RandomResizedCrop for validation
    ])

    # Create datasets without applying transformations initially
    image_datasets = {
        'train': CustomImageDataset(os.path.join(data_dir, 'train'), class_names),
        'val': CustomImageDataset(os.path.join(data_dir, 'val'), class_names)
    }

    # Create DataLoaders with transformations applied at the batch level
    dataloaders = {
        x: CustomDataLoader(
            dataset=image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn,
            transform=(train_transform if x == 'train' else val_transform)
        )
        for x in ['train', 'val']
    }

    # Get dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes

# Custom collate function to process a batch of data
def custom_collate_fn(batch):
    """
    Custom collate function to handle a batch of tensors where the images are not in channel-first order (C, H, W).

    Args:
        batch (list of tuples): A list of tuples where each tuple contains a tensor 
                                representing the image and a corresponding label.

    Returns:
        torch.Tensor: A batch tensor of images in (batch_size, C, H, W) format.
        torch.Tensor: A tensor containing the labels.
    """
    images, labels = zip(*batch)
    
    # Stack images into a batch and permute to (C, H, W)
    images = torch.stack(images)
    
    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return images, labels

# Custom Image Dataset for loading images and labels
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, class_names, transform=None):
        """
        Initialize the CustomImageDataset.

        Args:
            img_dir (str): Directory with all the images.
            class_names (list): List of class names (subdirectory names).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.class_names = class_names
        self.transform = transform
        self.img_paths = []
        self.labels = []

        # Collect image paths and corresponding labels
        for class_name in class_names:
            class_dir = os.path.join(img_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(".jpg"):
                    self.img_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_names.index(class_name))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            torch.Tensor: Image tensor.
            int: Corresponding label.
        """
        img_path = self.img_paths[idx]

        # Load and convert image to RGB format
        image = Image.open(img_path).convert("RGB")

        # Convert image to a NumPy array and then to a tensor
        image = transforms.ToTensor()(np.array(image))
        
        label = self.labels[idx]

        # Apply any transformations (if provided)
        if self.transform:
            image = self.transform(image)

        return image, label

    def print_dataset_summary(self):
        """
        Print a summary of the dataset including the number of images per class.
        """
        summary = {}
        total_images = 0

        # Count number of images per class
        for class_name in self.class_names:
            class_dir = os.path.join(self.img_dir, class_name)
            num_images = len([img_name for img_name in os.listdir(class_dir) if img_name.endswith(".jpg")])
            summary[class_name] = num_images
            total_images += num_images

        num_classes = len(self.class_names)

        # Print summary
        print("Dataset Summary:")
        print("----------------")
        print(f"Total number of images: {total_images}")
        print(f"Number of classes: {num_classes}")
        print("\nNumber of images per class:")
        for class_name, num_images in summary.items():
            print(f" - {class_name}: {num_images} images")

# Custom DataLoader with batch-level transformations
class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False, collate_fn=None, transform=None):
        """
        Initialize the CustomDataLoader.

        Args:
            dataset (Dataset): Dataset from which to load the data.
            batch_size (int, optional): How many samples per batch to load.
            shuffle (bool, optional): Set to True to have the data reshuffled at every epoch.
            num_workers (int, optional): How many subprocesses to use for data loading.
            pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory.
            collate_fn (callable, optional): Merges a list of samples to form a mini-batch.
            transform (callable, optional): Transformations to apply at the batch level.
        """
        super(CustomDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        self.transform = transform

    def __iter__(self):
        """
        Iterate over the dataset, applying the batch-level transformations.
        
        Yields:
            torch.Tensor: Transformed images batch.
            torch.Tensor: Corresponding labels.
        """
        for images, labels in super(CustomDataLoader, self).__iter__():
            # Apply batch-level transformations (if provided)
            if self.transform:
                images = self.transform(images)
            
            # Move labels to GPU
            yield images, labels.to("cuda:0")
