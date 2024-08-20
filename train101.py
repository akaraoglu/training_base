import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from CustomDataset import create_dataloaders
from multiprocessing import freeze_support
import time
import json
import os
from datetime import datetime

#############
# TODO:
# Save the code and settings for each train as zip file. 

class Trainator101:
    def __init__(self, config_path='parameters/training_config_default.json'):
        # Load settings from the JSON file
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize device
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else "cpu")

        # Ensure the tensorboard_log_dir exists and create a date-time specific folder
        os.makedirs(self.config['tensorboard_log_dir'], exist_ok=True)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.training_log_dir = os.path.join(self.config['tensorboard_log_dir'], f'training_{current_time}')
        os.makedirs(self.training_log_dir, exist_ok=True)

        # Initialize the TensorBoard writer with the specific log directory
        self.writer = SummaryWriter(log_dir=self.training_log_dir)

        # Initialize the model
        self.model = self._initialize_model()

        # Initialize the loss function, optimizer, and scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=self.config['momentum'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['step_size'], gamma=self.config['gamma'])

        # Initialize DataLoaders
        self.dataloaders, self.dataset_sizes = self._create_dataloaders()

    def _initialize_model(self):
        """Initialize and modify the model to match the number of classes."""
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, self.config['num_classes'])
        model = model.to(self.device)
        return model

    def _create_dataloaders(self):
        """Create and return DataLoaders for training and validation."""
        return create_dataloaders(
            self.config['data_dir'], 
            self.config['class_names'], 
            batch_size=self.config['batch_size'], 
            num_workers=self.config['num_workers'], 
            pin_memory=self.config['pin_memory']
        )

    def _denormalize(self, tensor):
        """Denormalize an image tensor from ImageNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(1, 3, 1, 1)
        tensor = tensor * std + mean
        return tensor

    def _log_to_tensorboard(self, epoch, phase, loss, accuracy, inputs, labels):
        """Log training metrics and images to TensorBoard."""
        self.writer.add_scalar(f'{phase} Loss', loss, epoch)
        self.writer.add_scalar(f'{phase} Accuracy', accuracy, epoch)
        if phase == 'train':
            # Denormalize the images before logging
            inputs_denorm = self._denormalize(inputs.clone())
            # Log a grid of input images along with their labels
            img_grid = make_grid(inputs_denorm)
            self.writer.add_image(f'{phase} Images', img_grid, epoch)
            self.writer.add_text(f'{phase} Labels', str(labels.tolist()), epoch)
    
    def _run_epoch(self, epoch):
        """Run both training and validation phases of a given epoch."""
        # Training phase
        train_loss, train_acc = self._run_phase(epoch, 'train')

        # Validation phase (if applicable)
        if (epoch + 1) % self.config['val_interval'] == 0:
            val_loss, val_acc = self._run_phase(epoch, 'val')
            return val_acc
        return None

    def _run_phase(self, epoch, phase):
        """Run a single phase (train/val) of an epoch."""
        epoch_start_time = time.time()

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            self.scheduler.step()

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

        # Log to TensorBoard
        self._log_to_tensorboard(epoch, phase, epoch_loss, epoch_acc, inputs, labels)
        epoch_time = time.time() - epoch_start_time
        print(f'[{phase}] | Epoch {epoch + 1}/{self.config["num_epochs"]}: | Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', end=' ')
        print(f"| Time: {epoch_time:.2f}s")

        return epoch_loss, epoch_acc

    def train_model(self):
        """Train the model and return the best model based on validation accuracy."""
        best_model_wts = self.model.state_dict()
        best_acc = 0.0

        for epoch in range(self.config['num_epochs']):

            # Run both phases (train and optionally val)
            val_acc = self._run_epoch(epoch)

            # # Update the best model if validation accuracy improves
            # if val_acc is not None and val_acc > best_acc:
            #     best_acc = val_acc
            #     best_model_wts = self.model.state_dict()

            # Save model at specified intervals
            if (epoch + 1) % self.config['save_interval'] == 0:
                # save_path = os.path.join(self.training_log_dir, f"model_epoch_{epoch + 1}.pth")
                self.save_model(self.model, f"model_epoch_{epoch + 1}.pth")

        # print(f'Best val Acc: {best_acc:.4f}')

        # # Load best model weights
        # self.model.load_state_dict(best_model_wts)
        return self.model

    def save_model(self, model, model_name):
        """Save the trained model to the specified path."""
        save_path = os.path.join(self.training_log_dir, model_name)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

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
