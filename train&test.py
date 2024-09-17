from src.test101 import Testification101
from src.train101 import Trainator101
from toolset.ConfigParser import Config
    
"""
TODO:

Machine Learning Engineer Responsibilities
- Implementing machine learning algorithms
- Running AI systems experiments and tests
- Designing and developing machine learning systems
- Performing statistical analyses 

Verify the input, output and ground truth images.
Limit the amount of images use in the training.
Use all the settings in the paramter file json. 

Docker integration? 

"""

if __name__ == '__main__':
    """
    Function to load settings, run training, and then run testing to save the results.
    """
    test_only = False

    # Define paths
    config_path_train = 'parameters/training_config_default.json'  # Path to the training configuration file
    config_path_test = 'parameters/test_config_default.json'  # Path to the training configuration file
    test_images_dir = 'path/to/test/images'  # Path to the directory containing test images

    # Step 1: Initialize and train the model using Trainator101
    print("Initializing training...")
    # Load settings from the JSON file
    config_train = Config(config_path_train)

    # Step 2: Initialize and run testing using Testification101
    print("Initializing testing...")
    config_path_test = 'parameters/test_config_default.json'
    config_test = Config(config_path_test)

    if test_only:
        # Overwrite configuration values if needed
        config_test.device = "cuda:0"
        config_test.log_dir = "log_train/training_20240828_133849/"  # Example overwrite

        # Directory containing test images
        test_images_dir = 'E:/datasets/imagenet/val'
        
        # Path to a specific model checkpoint (if any)
        model_path = "log_train/training_20240828_133849/model_epoch_100.pth"
    else:
       
        trainer = Trainator101(config=config_train)
        best_model = trainer.train_model()
        trainer.close()
        print("Training completed.")

        # Overwrite configuration values if needed
        config_test.device = "cpu"
        config_test.log_dir = trainer.getLogDir()
        # Directory containing test images
        test_images_dir = 'E:/datasets/intel_obj/pred'
        # Path to a specific model checkpoint (if any)
        model_path = trainer._get_latest_checkpoint() #"log_train/training_20240821_130930/model_epoch_1.pth"

        
    # Initialize the testing class
    tester = Testification101(config_test, model_path)    

    # Test the model and save the results with a limit on the number of images processed
    tester.test_model(test_images_dir, limit=100)  # Limit to 10 images