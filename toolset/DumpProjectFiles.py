import os
import zipfile

class SaveProjectFiles:
    def __init__(self, source_dirs=None, output_zip='dumped_files.zip', target_dir=None):
        """
        Initialize the ProjectDumper with the source directories, output .zip file name, 
        and target directory for the .zip file.
        
        :param source_dirs: A list of directories to search for .py and .json files. 
                            Defaults to ['project_folder/src'] if not provided.
        :param output_zip: The name of the output .zip file. 
                           Defaults to 'dumped_files.zip'.
        :param target_dir: The directory where the .zip file will be saved.
                           Defaults to the current working directory.
        """
        # Set the default directory to 'project_folder/src' if no directories are provided
        # default_path = os.path.join(os.getcwd(), 'project_folder/src')
        directories_to_dump = [
            os.path.join(os.getcwd(), 'src'),
            os.path.join(os.getcwd(), 'parameters')
        ]
        self.source_dirs = directories_to_dump# or [default_path]
        
        # Set the target directory
        self.target_dir = target_dir or os.getcwd()
        
        # Ensure the target directory exists
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Set the full path for the output zip file
        self.output_zip = os.path.join(self.target_dir, output_zip)

    def find_files(self, extensions=['.py', '.json']):
        """
        Find all files with the specified extensions in the specified source directories.
        
        :param extensions: A list of file extensions to search for.
        :return: A list of file paths that match the specified extensions.
        """
        matching_files = []
        for source_dir in self.source_dirs:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        matching_files.append(os.path.join(root, file))
        return matching_files

    def dump_to_zip(self):
        """
        Create a .zip file containing the found files, preserving the directory structure.
        """
        files_to_dump = self.find_files()

        with zipfile.ZipFile(self.output_zip, 'w') as zipf:
            for file_path in files_to_dump:
                # Calculate the base directory name to include in the zip structure
                for source_dir in self.source_dirs:
                    if file_path.startswith(source_dir):
                        base_dir = os.path.basename(source_dir)
                        relative_path = os.path.relpath(file_path, source_dir)
                        # Combine base_dir with the relative path to keep the folder structure in the zip
                        arcname = os.path.join(base_dir, relative_path)
                        # Add the file to the zip file with its adjusted path
                        zipf.write(file_path, arcname=arcname)
                        print(f"Added {file_path} as {arcname}")
                        break  # Stop after finding the correct source_dir

    def execute(self):
        """
        Execute the dump process.
        """
        print(f"Dumping .py and .json files from {self.source_dirs} to {self.output_zip}...")
        self.dump_to_zip()
        print("Dumping complete.")

# Example usage:
if __name__ == "__main__":
    # List of specific directories to include in the dump, or use default
    directories_to_dump = None  # or specify paths like ['/path/to/first/folder', ...]

    # Specify the target directory for the zip file
    target_directory = os.path.join(os.getcwd(), 'log_train')

    dumper = SaveProjectFiles(source_dirs=directories_to_dump, target_dir=target_directory)
    dumper.execute()
