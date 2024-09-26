import os
import sys
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
from src.neural_network.ModelUnet import ShallowUNet
from src.neural_network.LiteHDRNet import LiteHDRNet

from src.utils.DumpProjectFiles import SaveProjectFiles
from src.utils.ConfigParser import Config
from src.losses.loss_functions import LossSelector


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
        self.loss_config = {
            "mse": 1.0,
            "ssim": 0.5
        }

        # Initialize the LossSelector with multiple losses
        self.criterion = LossSelector(loss_config=self.loss_config)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=4)

        self.train_image_paths = [os.path.join(self.config.dataset_path, line.strip()) for line in open(self.config.train_image_paths)]
        self.val_image_paths = [os.path.join(self.config.dataset_path, line.strip()) for line in open(self.config.val_image_paths)]
        
        # Initialize DataLoaders
        self.dataloaders, self.dataset_sizes = self._create_dataloaders()
        
        # Dump the code and parameters
        self.dumpTrainingCodeAndParameters()
        
        # Print training information
        self.print_training_info()
        
        print("")
        print("Training starting...")
    
    def print_training_info(self):
        """Print important training information, including GPU, Python, PyTorch, CUDA, and cuDNN details."""
        developer = "Developer: Ali Karaoglu   ©2024 VisuAlysium"
        date_info = f"Date: {datetime.now().strftime('%Y-%m-%d')}"
        legal_info = "License: MIT"
        
        # GPU Information
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpu_info = []
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory_total = torch.cuda.get_device_properties(i).total_memory // (1024 ** 2)  # Convert to MB
                gpu_info.append(f"GPU {i}: {gpu_name} ({gpu_memory_total} MB memory)")
        else:
            gpu_info = ["No GPUs available, using CPU"]

        # Python, PyTorch, CUDA, cuDNN Versions
        python_version = sys.version.split(" ")[0]
        pytorch_version = torch.__version__
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "Not available"
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "Not available"

        # Training Information
        info_lines = [
            developer,
            date_info,
            legal_info,
            "-" * 50,  # A separator line
            f"Log Directory: {self.training_log_dir}",
            f"Device: {self.device}",
            f"Model: {self.model.__class__.__name__}",
            f"Loss Function: {self.criterion.__class__.__name__}",
            f"Loss Config: {self.loss_config}",
            f"Optimizer: {self.optimizer.__class__.__name__}",
            f"Learning Rate: {self.config.learning_rate}",
            f"Batch Size: {self.config.batch_size}",
            f"Number of Epochs: {self.config.num_epochs}",
            f"Training Images: {len(self.train_image_paths)}",
            f"Validation Images: {len(self.val_image_paths)}",
            f"Save Interval: {self.config.save_interval} epochs",
            f"Validation Interval: {self.config.val_interval} epochs",
            "-" * 50,  # Another separator line
            f"Python Version: {python_version}",
            f"PyTorch Version: {pytorch_version}",
            f"CUDA Version: {cuda_version}",
            f"cuDNN Version: {cudnn_version}",
            "-" * 50,  # Another separator line for GPU info
        ]

        # Add GPU information to the list
        info_lines.extend(gpu_info)

        # Format and print the info
        max_length = max(len(line) for line in info_lines)
        border = "+" + "-" * (max_length + 2) + "+"

        print("\n" + border)
        for line in info_lines:
            print(f"| {line.ljust(max_length)} |")
        print(border + "\n")

    def dumpTrainingCodeAndParameters(self):
        # Example usage:
        directories_to_dump = ["src", "parameters"] # or specify paths like ['/path/to/first/folder', ...]
        dumper = SaveProjectFiles(source_dirs=directories_to_dump, output_zip="training_files.zip",target_dir=self.training_log_dir)
        dumper.execute()

    def getLogDir(self):
        return self.training_log_dir
    
    def _initialize_model(self):
        """Initialize and modify the model to an image-to-image network."""
        model = ShallowUNet(3, 3, 32)  # U-Net with 3 input channels and 3 output channels
        model = model.to(self.device)
        return model
    
    def _create_dataloaders(self):
        """Create and return DataLoaders for training and validation."""
        return create_dataloaders(
            train_image_paths   = self.train_image_paths,
            val_image_paths     = self.val_image_paths,
            batch_size          = self.config.batch_size, 
            num_workers         = self.config.num_workers, 
            pin_memory          = self.config.pin_memory,
            device              = self.device
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
        
        # Denormalize the images before logging
        inputs_denorm = self._denormalize(inputs.clone())
        outputs_denorm = self._denormalize(outputs.clone())
        targets_denorm = targets #self._denormalize(targets.clone())

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
                loss = self.criterion(self._denormalize(outputs), images_gt)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * images_boosted.size(0)

        epoch_loss = running_loss / self.dataset_sizes[phase]

        if phase == 'train':
            self.scheduler.step(epoch_loss)

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
