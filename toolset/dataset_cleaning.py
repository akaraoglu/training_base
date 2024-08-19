import os
from PIL import Image

def remove_images_with_different_shape(directory, target_shape=(150, 150)):
    """
    Traverse the directory and remove images that do not match the target shape.

    :param directory: Directory containing the image dataset.
    :param target_shape: Tuple representing the target image shape (width, height).
    """
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    if img.size != target_shape:
                        img.close()  # Ensure the image is closed before deleting
                        print(f"Removing {file_path}, as it is of shape {img.size}")
                        os.remove(file_path)
            except Exception as e:
                print(f"Could not process file {file_path}: {e}")

if __name__ == "__main__":
    # Specify the directory containing your image dataset
    dataset_directory = "E:/datasets/intel_obj"
    
    # Run the function to remove images that are not 150x150
    remove_images_with_different_shape(dataset_directory)
