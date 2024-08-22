import os
import time
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import freeze_support
from datetime import datetime

from src.CustomDataset import create_dataloaders
from src.ModelUnet import ShallowUNet
from toolset.DumpProjectFiles import SaveProjectFiles
from toolset.ConfigParser import Config
"""
 TODO:
 Verify the input, output and ground truth images.
 training verification.
  - Test class
  - Load the model and run a test image. 
  - Write test codes. (Automated smoke test)

Machine Learning Engineer Responsibilities
- Implementing machine learning algorithms
- Running AI systems experiments and tests
- Designing and developing machine learning systems
- Performing statistical analyses 

Fix the test code.
Load json before the class and send the config as dict.
Overwrite the variables in the config. 
Arrange the directories 
Limit the amount of images use in the test.

Automatize!!!!!

"""

class Trainator101:
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # Ensure the log_dir exists and create a date-time specific folder
        os.makedirs(self.config.log_dir, exist_ok=True)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.training_log_dir = os.path.join(self.config.log_dir, f'training_{current_time}')
        os.makedirs(self.training_log_dir, exist_ok=True)

        # Initialize the TensorBoard writer with the specific log directory
        self.writer = SummaryWriter(log_dir=self.training_log_dir)

        # Initialize the model
        self.model = self._initialize_model()

        # Initialize the loss function, optimizer, and scheduler
        self.criterion = nn.MSELoss()  # Using MSELoss for image-to-image tasks
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma)

        # Initialize DataLoaders
        self.dataloaders, self.dataset_sizes = self._create_dataloaders()
        
        self.dumpTrainingCodeAndParameters()
        
    def dumpTrainingCodeAndParameters(self):
        # Example usage:
        directories_to_dump = ["/src", "/parameters"] # or specify paths like ['/path/to/first/folder', ...]

        dumper = SaveProjectFiles(source_dirs=directories_to_dump, output_zip="training_files.zip",target_dir=self.training_log_dir)
        dumper.execute()


    def getLogDir(self):
        return self.training_log_dir
    
    def _initialize_model(self):
        """Initialize and modify the model to an image-to-image network."""
        model = ShallowUNet(in_channels=3, out_channels=3)  # U-Net with 3 input channels and 3 output channels
        model = model.to(self.device)
        return model
    
    def _create_dataloaders(self):
        """Create and return DataLoaders for training and validation."""
        return create_dataloaders(
            self.config.data_dir,
            self.config.class_names, 
            batch_size=self.config.batch_size, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory,
            device = self.device
        )

    def _denormalize(self, tensor):
        """Denormalize an image tensor from ImageNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(1, 3, 1, 1)
        tensor = tensor * std + mean
        return tensor

    def _log_to_tensorboard(self, epoch, phase, loss, inputs, outputs, targets):
        """Log training metrics and images to TensorBoard."""
        self.writer.add_scalar(f'{phase} Loss', loss, epoch)
        if phase == 'train':
            # Denormalize the images before logging
            inputs_denorm = self._denormalize(inputs.clone())
            outputs_denorm = self._denormalize(outputs.clone())
            targets_denorm = self._denormalize(targets.clone())

            # Log a grid of input images, outputs, and targets
            img_grid = make_grid(torch.cat((inputs_denorm, outputs_denorm, targets_denorm)))
            self.writer.add_image(f'{phase} Images (Inputs | Outputs | Targets)', img_grid, epoch)
    
    def _run_epoch(self, epoch):
        """Run both training and validation phases of a given epoch."""
        # Training phase
        train_loss = self._run_phase(epoch, 'train')

        # Validation phase (if applicable)
        if (epoch + 1) % self.config.val_interval == 0:
            val_loss = self._run_phase(epoch, 'val')

    def _run_phase(self, epoch, phase):
        """Run a single phase (train/val) of an epoch."""
        epoch_start_time = time.time()

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0

        for batch in self.dataloaders[phase]:
            
            images_gt = batch['images_gt'].to(self.device)
            images_boosted = batch['images_input'].to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(images_boosted)
                loss = self.criterion(outputs, images_gt)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * images_boosted.size(0)

        if phase == 'train':
            self.scheduler.step()

        epoch_loss = running_loss / self.dataset_sizes[phase]

        # Log to TensorBoard
        self._log_to_tensorboard(epoch, phase, epoch_loss, images_boosted, outputs, images_gt)
        epoch_time = time.time() - epoch_start_time
        print(f'[{phase}] | Epoch {epoch + 1}/{self.config.num_epochs}: | Loss: {epoch_loss:.4f}', end=' ')
        print(f"| Time: {epoch_time:.2f}s")

        return epoch_loss

    def train_model(self):
        """Train the model and return the best model based on validation accuracy."""
        best_model_wts = self.model.state_dict()
        best_loss = float('inf')

        for epoch in range(self.config.num_epochs):

            # Run both phases (train and optionally val)
            self._run_epoch(epoch)

            # Save model at specified intervals
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_model(self.model, f"model_epoch_{epoch + 1}.pth")

        return self.model

    def save_model(self, model, model_name):
        """Save the trained model to the specified path."""
        save_path = os.path.join(self.training_log_dir, model_name)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")
    
    def _get_latest_checkpoint(self):
        """Get the latest checkpoint file from the training log directory."""
        checkpoints = glob.glob(os.path.join(self.training_log_dir, '*.pth'))
        if not checkpoints:
            return None
        return max(checkpoints, key=os.path.getctime)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

if __name__ == '__main__':
    freeze_support()

    # Initialize the training class
    trainer = Trainator101()

    # Train the model
    best_model = trainer.train_model()

    # Close the TensorBoard writer
    trainer.close()
