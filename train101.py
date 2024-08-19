# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from CustomDataset import create_dataloaders
from multiprocessing import freeze_support
import time  # Import time module

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the MobileNetV3 model
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# Modify the classifier to match the number of classes in your dataset
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
num_ftrs = model.classifier[3].in_features  # Last layer's in_features
model.classifier[3] = nn.Linear(num_ftrs, len(class_names))

# Move the model to the GPU
model = model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Create DataLoaders
data_dir = r'E:/datasets/intel_obj'
dataloaders, dataset_sizes = create_dataloaders(data_dir, class_names, batch_size=32, num_workers=1, pin_memory=True)

# Training loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_start_time = time.time()  # Start timing the epoch

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                # inputs = inputs.to(device, dtype=torch.float)
                # labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        epoch_time = time.time() - epoch_start_time  # Calculate the time taken for the epoch
        print(f'Time elapsed for epoch {epoch}: {epoch_time:.2f} seconds')
        print()

    return model

if __name__ == '__main__':
    freeze_support()  # Support for multiprocessing on Windows

    # Train the model
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)

    # Save the model
    torch.save(model.state_dict(), 'mobilenet_v3_model.pth')
