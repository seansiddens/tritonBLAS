#!/bin/bash

# Script to run tritonBLAS development container interactively
# This replicates the docker-compose setup using plain docker commands

# Check if container already exists and remove it
if [ "$(docker ps -aq -f name=tritonBLAS-dev)" ]; then
    echo "Removing existing container..."
    docker rm -f tritonBLAS-dev
fi

# Build the image first
echo "Building tritonBLAS development image..."
docker build -t tritonblas-dev:latest .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Docker build failed. Trying alternative approach..."
    echo "Creating container with pre-installed Triton..."
    
    # Create a simpler container that installs Triton via pip instead of building from source
    docker run -it \
      --name tritonBLAS-dev \
      --network host \
      --device /dev/kfd \
      --device /dev/dri \
      --group-add video \
      --cap-add SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --shm-size 16g \
      --ulimit memlock=-1:-1 \
      --ulimit stack=67108864:67108864 \
      -v ${HOME}:${HOME} \
      -v $(pwd):/workspace \
      -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" \
      -w /workspace \
      rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_prealpha \
      /bin/bash -c "
        echo 'Installing Triton via pip...'
        pip install triton
        echo 'Triton installation complete!'
        echo 'You can now run:'
        echo '  pip3 install -e .'
        echo '  export PYTHONPATH=\$(pwd)/include/:\$PYTHONPATH'
        echo '  cd examples && python3 example_matmul.py'
        /bin/bash
      "
else
    # Run the container interactively with all the necessary configurations
    echo "Starting interactive container..."
    docker run -it \
      --name tritonBLAS-dev \
      --network host \
      --device /dev/kfd \
      --device /dev/dri \
      --group-add video \
      --cap-add SYS_PTRACE \
      --security-opt seccomp=unconfined \
      --shm-size 16g \
      --ulimit memlock=-1:-1 \
      --ulimit stack=67108864:67108864 \
      -v ${HOME}:${HOME} \
      -v $(pwd):/workspace \
      -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" \
      -w /workspace \
      tritonblas-dev:latest \
      /bin/bash

    echo "Container started. You can now run:"
    echo "  pip3 install -e ."
    echo "  export PYTHONPATH=\$(pwd)/include/:\$PYTHONPATH"
    echo "  cd examples && python3 example_matmul.py"
fi

# Try this stuff if the above doesn't work:
# apt-get update && apt-get install -y build-essential cmake ninja-build git
# pip3 install -U nanobind scikit-build-core cmake ninja
# python3 -c "import nanobind, sys; print('nanobind ok:', nanobind.__file__, sys.executable)"
# rm -rf _origami
# pip3 install --no-build-isolation -vvv -e .
