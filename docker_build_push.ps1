# Define Docker image name and tag
$imageName = "karaoglu/pytorch-cuda12.4-training"
$tag = "latest"

# Path to the dataset on the host machine
$datasetPath = "C:/Users/AliKaraoglu/Desktop/datasets/sihdr/input"
# Path to the results on the host machine
$resultsPath = "C:/Users/AliKaraoglu/Desktop/workspace/log_train"

# Build the Docker image
Write-Host "Building Docker image..."
docker build -t "$imageName`:$tag" .

# Push the Docker image to Docker Hub
Write-Host "Pushing Docker image to Docker Hub..."
docker push "$imageName`:$tag"

# Run the Docker container with GPU access and dataset mounting
Write-Host "Running Docker container..."
docker run -d --gpus all -it -v "$datasetPath`:/app/data" -v "$resultsPath`:/app/log_train/" --shm-size=8G "$imageName`:$tag"