# Use the official NVIDIA base image with CUDA 12.4
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set a working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
#COPY . /app
COPY requirements.txt           /app/
# Install any additional Python dependencies listed in requirements.txt
RUN pip install -r requirements.txt

# Copy only the specific folders you need into the Docker image
COPY test_images    /app/test_images
COPY parameters     /app/parameters
COPY src            /app/src

# (Optional) Copy other necessary files like requirements.txt
COPY train&test.py             /app/
COPY training_image_paths.txt   /app/
COPY val_image_paths.txt        /app/

# Set environment variables
ENV CONFIG_PATH_TRAIN=/app/parameters/training_config_docker.json
ENV CONFIG_PATH_TEST=/app/parameters/test_config_docker.json
ENV TEST_ONLY=False

# Set the default command to run the training script
CMD ["python3", "train&test.py"]