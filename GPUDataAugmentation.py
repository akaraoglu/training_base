import torch
from torchvision.transforms import v2 as F
import numpy as np

# Augmentation Classes
class NumpyToCudaTensor:
    def __call__(self, batch: np.ndarray) -> torch.Tensor:
        batch_tensor = torch.from_numpy(np.array(batch)).permute(0, 3, 1, 2).float()
        # batch_tensor = torch.as_tensor(np.array(batch).astype('float')).permute(0, 3, 1, 2)
        return batch_tensor.to('cuda')
