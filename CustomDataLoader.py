import torch
from torch.utils.data import DataLoader

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False):
        super(CustomDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def __iter__(self):
        # Iterate over batches
        for batch in super(CustomDataLoader, self).__iter__():
            # Check if pin_memory is set and if the inputs are still on CPU
            inputs, labels = batch
            
            if self.pin_memory:
                # Ensure that pin_memory is applied only if tensors are on the CPU
                inputs = inputs.pin_memory().to('cuda', non_blocking=True)
                labels = labels.pin_memory().to('cuda', non_blocking=True)
            else:
                # If pin_memory is not used, just move the data to GPU
                inputs = inputs.to('cuda', non_blocking=True)
                labels = labels.to('cuda', non_blocking=True)
                
            yield inputs, labels


