import json
from typing import List

class Config:
    def __init__(self, config_path: str):
        # Initialize base variables with default values (could be None or some default)
        self.device: str = "cpu"
        self.model_name: str = "model_name"
        self.pretrained_weights: str = "pretrained_weights"
        self.train_image_paths: str = "empty"
        self.val_image_paths: str = "empty"
        self.data_dir: str = "data_dir"
        self.input_width: int = 224
        self.input_height: int = 224
        self.batch_size: int = 1
        self.num_workers: int = 0
        self.pin_memory: bool = False
        self.learning_rate: float = 0.001
        self.momentum: float = 0.9
        self.step_size: int = 7
        self.gamma: float = 0.1
        self.output_model_path: str = "output_model_path"
        self.log_dir: str = "log_dir"
        self.num_epochs: int = 1
        self.val_interval: int = 1
        self.save_interval: int = 1
        
        # Load the configuration from the JSON file
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Update the class attributes with the values from the JSON file
        self.__dict__.update(config_dict)

    def __repr__(self):
        return f'{self.__dict__}'
