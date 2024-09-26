from src.test101 import Testification101
from src.train101 import Trainator101
from src.utils.ConfigParser import Config
import argparse
import os

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train and Test model.')
    
    # Add arguments and read from environment variables if set
    parser.add_argument('--config_path_train', 
                        type=str, 
                        default=os.getenv('CONFIG_PATH_TRAIN', 'parameters/training_config_default.json'), 
                        help='Path to training config file')
    
    parser.add_argument('--config_path_test', 
                        type=str, 
                        default=os.getenv('CONFIG_PATH_TEST', 'parameters/test_config_default.json'), 
                        help='Path to testing config file')
      
    # Handle 'test_only' as an environment variable (converted to boolean)
    test_only_env = os.getenv('TEST_ONLY', 'False').lower() in ['true', '1', 't', 'y', 'yes']
    parser.add_argument('--test_only', 
                        action='store_true', 
                        default=test_only_env, 
                        help='Flag to run test only')
    
    args = parser.parse_args()
    print("")
    print("********************************")
    print("Using parameter files:")
    print("-- For training: ",  args.config_path_train)
    print("-- For testing : ",args.config_path_test)
    print("-- Test only   : ",args.test_only)
    print("")
    return args

if __name__ == '__main__':
    """
    Function to load settings, run training, and then run testing to save the results.
    """
    test_only = False

    args = parse_args()
        
    # Step 1: Initialize and train the model using Trainator101
    print("Initializing training...")
    # Load settings from the JSON file
    config_train = Config(args.config_path_train)

    # Step 2: Initialize and run testing using Testification101
    print("Initializing testing...")
    config_test = Config(args.config_path_test)

    if test_only:
        # TODO: make these available in the test parameter file.

        # Overwrite configuration values if needed
        config_test.device = "cuda:0"
        config_test.log_dir = "log_train/training_20240828_133849/"  # Example overwrite

        # Directory containing test images
        test_images_dir = 'E:/datasets/imagenet/val'
        
        # Path to a specific model checkpoint (if any)
        model_path = "log_train/training_20240828_133849/model_epoch_100.pth" # Example model path
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