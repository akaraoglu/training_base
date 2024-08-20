import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import random
from PIL import Image
import numpy as np
import os
import  src.CustomDataAugmentation as CDA

# Function to set random seeds
def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function to create DataLoaders for training and validation datasets
def create_dataloaders(data_dir, class_names, batch_size=4, num_workers=4, pin_memory=True, seed=None):
    """
    Create DataLoaders for training and validation datasets with batch-level transformations.

    Args:
        data_dir (str): Directory containing the data.
        class_names (list): List of class names corresponding to subdirectories in data_dir.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): Whether to copy tensors into CUDA pinned memory. 
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: Dictionary containing DataLoaders for 'train' and 'val' datasets.
        dict: Dictionary containing the sizes of the 'train' and 'val' datasets.
    """
    if seed is not None:
        set_random_seeds(seed)
    else:
        set_random_seeds(1234)

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
            transform=x
        )
        for x in ['train', 'val']
    }

    # Get dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes

# Custom collate function to process a batch of data
def custom_collate_fn(batch):
    """
    Custom collate function to handle a batch of data.

    Args:
        batch (list of dicts): A list of dictionaries where each dictionary contains 
                               the 'input_image', 'gt_image', and 'label'.

    Returns:
        dict: A dictionary containing batched 'images_input'
    """
    input_images = torch.stack([item['images_input'] for item in batch])
    # gt_images = torch.stack([item['gt_image'] for item in batch])
    # labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    batch_dict = {'images_input': input_images 
                #   'gt_images': gt_images, 
                #   'labels': labels
                  }
    return batch_dict


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
        input_image = Image.open(img_path).convert("RGB")

        # Convert image to a NumPy array and then to a tensor
        input_image = transforms.ToTensor()(np.array(input_image))
        
        # Apply any transformations (if provided)
        if self.transform:
            input_image = self.transform(input_image)

        return {'images_input': input_image}

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
        # Define the transformations for training and validation
        self.train_transform_input = v2.Compose([
            CDA.ToDevice(),
            CDA.BoostBrightness(threshold=0.87, boost_factor=2), 
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomCrop((150,150)),
            v2.ColorJitter(brightness=.5, hue=.3),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define the transformations for training and validation
        self.val_transform_input = v2.Compose([
            CDA.ToDevice(),
            CDA.BoostBrightness(threshold=0.87, boost_factor=2), 
            # v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomVerticalFlip(p=0.5),
            # v2.RandomCrop((150,150)),
            # v2.ColorJitter(brightness=.5, hue=.3),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        # # Define the transformations for training and validation
        # self.transform_GT = v2.Compose([
        #     CDA.ToDevice(),
        #     CDA.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        self.transform_GT = v2.Compose([
            CDA.ToDevice(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transform

    def __iter__(self):
        """
        Iterate over the dataset, applying the batch-level transformations.
        
        Yields:
            dict: A dictionary containing 'images_input'
        """
        for batch in super(CustomDataLoader, self).__iter__():
            # Apply batch-level transformations (if provided)
            if self.transform == 'train':
                batch["images_gt"] = self.transform_GT(batch["images_input"])
                batch["images_input"] = self.train_transform_input(batch["images_input"])

            if self.transform == 'val':
                batch["images_gt"] = self.transform_GT(batch["images_input"])
                batch["images_input"] = self.val_transform_input(batch["images_input"])
                
            # Move images to GPU
            # batch['input_images'] = batch['input_images'].to("cuda:0")
            # batch['gt_images'] = batch['gt_images'].to("cuda:0")
            
            yield batch