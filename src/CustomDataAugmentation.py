import torch
import random
from torchvision.transforms import v2 as F
from torchvision.transforms import functional as TF
import numpy as np



# Custom ToDeviceAndNormalize transform for batch input
class ToDevice:
    def __init__(self, device='cuda:0'):
        self.device = device

    def __call__(self, batch):
        batch = batch.to(self.device)
        return batch


# Custom BoostBrightness transform for batch input
class BoostBrightness:
    def __init__(self, threshold=0.5, boost_factor=1.5):
        self.threshold = threshold
        self.boost_factor = boost_factor

    def __call__(self, batch):
        # Calculate the average brightness (mean of RGB channels)
        brightness = batch.mean(dim=1, keepdim=True)
        # Create a mask where the brightness exceeds the threshold
        mask = brightness > self.threshold
        # Increase brightness for those pixels
        batch = torch.where(mask, batch * self.boost_factor, batch)
        # Clip values to stay in the range [0, 1]
        batch = torch.clamp(batch, 0, 1)
        return batch

# Custom RandomResizedCrop transform for batch input
class RandomResizedCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, batch):
        # batch_size = batch['images_input'].size(0)
        crop_image = F.RandomCrop(size=(150, 150)) (batch['images_input'])

        batch['images_input'] = crop_image
        return batch

# Custom RandomHorizontalFlip transform for batch input
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, batch):
        if torch.rand(1).item() < self.p:
            batch['images_input'] = batch['images_input'].flip(-1)  # Flip horizontally
        return batch

# Custom RandomVerticalFlip transform for batch input
class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, batch):
        if torch.rand(1).item() < self.p:
            batch['images_input'] = batch['images_input'].flip(-2)  # Flip vertically
        return batch

# Custom Resize transform for batch input
class Resize:
    def __init__(self, size, antialias=True):
        self.size = size
        self.antialias = antialias

    def __call__(self, batch):
        batch['images_input'] = TF.resize(batch['images_input'], self.size, antialias=self.antialias)
        return batch

# Custom Normalize transform for batch input
class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, batch):
        batch['images_input'] = TF.normalize(batch['images_input'], self.mean, self.std)
        return batch
