import os
import glob
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
import json 
from src.neural_network.ModelUnet import ShallowUNet
from src.neural_network.LiteHDRNet import LiteHDRNet

from toolset.ConfigParser import Config

class Testification101:
    def __init__(self, config: Config, model_path: str):
        # Use the provided configuration dictionary
        self.config = config

        # Initialize device
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self.model = self._initialize_model(model_path)

        # Create a directory for saving test results
        self.test_results_dir = os.path.join(self.config.log_dir, 'test_results')
        os.makedirs(self.test_results_dir, exist_ok=True)

        # Transformations for test images
        self.transform = transforms.Compose([
            transforms.Resize((self.config.input_height, self.config.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _initialize_model(self, model_path):
        """Initialize the model and load the latest checkpoint."""
        model = LiteHDRNet(in_channels=3, out_channels=3)
        model = model.to(self.device)

        # Load the provided model checkpoint
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            latest_checkpoint = self._get_latest_checkpoint()
            if latest_checkpoint:
                model.load_state_dict(torch.load(latest_checkpoint, map_location=self.device))
                print(f"Loaded model from {latest_checkpoint}")
            else:
                print("No model checkpoint found. Please train the model first.")
                exit()

        model.eval()  # Set model to evaluation mode
        return model

    def test_model(self, test_images_dir, limit=None):
        """Test the model on the images in the given directory and save the outputs."""
        test_images = glob.glob(os.path.join(test_images_dir, '*.*'))
        
        if limit:
            test_images = test_images[:limit]
        
        for img_path in test_images:
            # Load and preprocess the image
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.model(image_tensor)

            # Denormalize the output
            output_denorm = self._denormalize(output.squeeze(0))

            # Save the result image
            img_name = os.path.basename(img_path)
            save_path = os.path.join(self.test_results_dir, img_name)
            save_image(output_denorm, save_path)
            print(f"Saved result image to {save_path}")

    def _denormalize(self, tensor):
        """Denormalize an image tensor from ImageNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(1, 3, 1, 1)
        tensor = tensor * std + mean
        return tensor

if __name__ == '__main__':
    # Load configuration from JSON file
    config_path = 'parameters/test_config_default.json'
    config = Config(config_path)

    # Overwrite configuration values if needed
    config.device = "cpu"
    config.log_dir = "log_train/training_20240821_130930/"  # Example overwrite

    # Directory containing test images
    test_images_dir = 'E:/datasets/intel_obj/pred'
    
    # Path to a specific model checkpoint (if any)
    model_path = "log_train/training_20240821_130930/model_epoch_1.pth"

    # Initialize the testing class
    tester = Testification101(config, model_path)    

    # Test the model and save the results with a limit on the number of images processed
    tester.test_model(test_images_dir, limit=10)  # Limit to 10 images
