import os

def is_image_file(filename):
    # Define a list of valid image file extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    # Check if the file extension is in the list of valid extensions
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def get_image_filenames(directory):
    image_filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                # Append the relative file path to the list
                image_filenames.append(os.path.join(root, file))
    return image_filenames

def save_to_file(image_filenames, output_file):
    with open(output_file, 'w') as file:
        for filename in image_filenames:
            file.write(filename + '\n')

if __name__ == "__main__":
    # Specify the directory to search and the output file name
    directory = "E:/datasets/intel_obj/val"
    output_file = 'val_image_paths.txt'

    # Get all image filenames recursively
    image_filenames = get_image_filenames(directory)

    # Save the filenames to a text file
    save_to_file(image_filenames, output_file)

    print(f"Image filenames saved to {output_file}")
