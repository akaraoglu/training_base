import torch
from torchvision.transforms import v2 as F
import numpy as np

# Augmentation Classes
class NumpyToCudaTensor:
    def __call__(self, batch: np.ndarray) -> torch.Tensor:
        batch_tensor = torch.from_numpy(np.array(batch)).permute(0, 3, 1, 2).float()
        # batch_tensor = torch.as_tensor(np.array(batch).astype('float')).permute(0, 3, 1, 2)
        return batch_tensor.to('cuda')

class ToDeviceAndNormalize:
    def __call__(self, batch: np.ndarray) -> torch.Tensor:
        batch_tensor = batch.to('cuda').float() #/ torch.tensor(255.0).float()
        return batch_tensor
    

class BoostSaturation:
    def __init__(self, threshold=0.85, boost_factor=1.5):
        """
        Args:
            threshold (float): Threshold value (between 0 and 1) above which the pixel values will be boosted.
            boost_factor (float): Factor by which to boost the pixel values above the threshold.
        """
        self.threshold = threshold
        self.boost_factor = boost_factor

    def __call__(self, img):
        """
        Apply the BoostSaturation effect to an image tensor.
        
        Args:
            img (torch.Tensor): Input image tensor with pixel values normalized between 0 and 1. 
                                Expected shape is (C, H, W).
        
        Returns:
            torch.Tensor: Image tensor with boosted saturation.
        """
        # Ensure the image tensor is in the right range [0, 1]
        img = torch.clamp(img, 0.0, 1.0)

        # Create a mask for pixels above the threshold
        mask = img > self.threshold

        # Boost the pixel values above the threshold
        img = img + self.boost_factor * (img * mask.float())

        # Clamp the result to keep pixel values in the range [0, 1]
        img = torch.clamp(img, 0.0, 1.0)
        
        return img