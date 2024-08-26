import unittest
import torch
import os
from unittest.mock import MagicMock, patch
from src.train101 import Trainator101
from toolset.ConfigParser import Config
from src.neural_network.LiteHDRNet import LiteHDRNet

class TestTrainator101(unittest.TestCase):
    
    def setUp(self):
        # Mocking the Config object
        self.config = MagicMock(spec=Config)
        self.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config.log_dir = './test_logs'
        self.config.learning_rate = 0.001
        self.config.batch_size = 4
        self.config.num_epochs = 1
        self.config.step_size = 7
        self.config.gamma = 0.1
        self.config.train_image_paths = 'train_images.txt'
        self.config.val_image_paths = 'val_images.txt'
        self.config.save_interval = 1
        self.config.val_interval = 1
        self.config.num_workers = 0
        self.config.pin_memory = False

        # Write dummy image paths for training and validation
        with open('train_images.txt', 'w') as f:
            f.write('test_images/00007.JPG\n')
            f.write('test_images/000071.JPG\n')
        
        with open('val_images.txt', 'w') as f:
            f.write('test_images/00007.JPG\n')
            f.write('test_images/000071.JPG\n')

        # Mock the create_dataloaders function
        self.mock_dataloader = MagicMock()
        self.mock_dataloader.__len__.return_value = 2
        self.mock_dataloader.__iter__.return_value = iter([{'images_gt': torch.randn(4, 3, 256, 256), 
                                                            'images_input': torch.randn(4, 3, 256, 256)}])
        
        self.patcher = patch('src.CustomDataset.create_dataloaders', return_value=({'train': self.mock_dataloader, 'val': self.mock_dataloader}, {'train': 8, 'val': 8}))
        self.patcher.start()

    def tearDown(self):
        # Clean up any patches
        self.patcher.stop()

    def test_initialization(self):
        # Test if Trainator101 initializes without errors
        trainer = Trainator101(config=self.config)
        self.assertIsInstance(trainer, Trainator101)
        self.assertIsInstance(trainer.model, LiteHDRNet)
        self.assertEqual(trainer.config, self.config)

    def test_dataloader_creation(self):
        # Test if dataloaders are created correctly
        trainer = Trainator101(config=self.config)
        self.assertIn('train', trainer.dataloaders)
        self.assertIn('val', trainer.dataloaders)
        self.assertEqual(trainer.dataset_sizes['train'], 2)
        self.assertEqual(trainer.dataset_sizes['val'], 2)

    def test_training_step(self):
        # Test if a single epoch of training runs without errors
        trainer = Trainator101(config=self.config)
        trainer.train_model()
        self.assertTrue(True)  # If no exception is raised, the test passes

    def test_save_model(self):
        # Test model saving function
        trainer = Trainator101(config=self.config)
        trainer.save_model(trainer.model, "test_model.pth")
        save_path = trainer.getLogDir() + "/test_model.pth"
        self.assertTrue(os.path.exists(save_path))

    def test_tensorboard_logging(self):
        # Test TensorBoard logging functionality
        trainer = Trainator101(config=self.config)
        trainer._log_to_tensorboard(0, 'train', 0.1, torch.randn(4, 3, 256, 256), torch.randn(4, 3, 256, 256), torch.randn(4, 3, 256, 256))
        log_dir = trainer.getLogDir()
        self.assertTrue(os.path.exists(log_dir))

    def test_model_checkpoint_loading(self):
        # Test loading of the latest model checkpoint
        trainer = Trainator101(config=self.config)
        trainer.save_model(trainer.model, "test_checkpoint.pth")
        checkpoint = trainer._get_latest_checkpoint()
        self.assertIsNotNone(checkpoint)
        self.assertTrue(checkpoint.endswith("test_checkpoint.pth"))

if __name__ == '__main__':
    unittest.main()
