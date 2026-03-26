#!/bin/bash

# Assign arguments to variables
RUNTIME=$1
IMAGE_NAME="ece8930-sim"

# Validate inputs
if [ -z "$RUNTIME" ]; then
    echo "Usage: ./run_docker.sh [linux|mac]"
    exit 1
fi

if [ "$RUNTIME" == "linux" ]; then
    echo "Launching Linux container..."
    docker run -it --rm \
          --gpus all \
          -e NVIDIA_DRIVER_CAPABILITIES=all \
          -e NVIDIA_VISIBLE_DEVICES=all \
          --net=host \
          -e DISPLAY=$DISPLAY \
          -v /tmp/.X11-unix:/tmp/.X11-unix \
          -v ~/.Xauthority:/root/.Xauthority \
          -v $(pwd):/app \
          ${IMAGE_NAME}
        
elif [ "$RUNTIME" == "mac" ]; then
    echo "Launching Mac container..."
    docker run -it --rm \
        -e DISPLAY=host.docker.internal:0 \
        -v $(pwd):/app \
        ${IMAGE_NAME}
else
    echo "Error: Unknown runtime '$RUNTIME'. Use 'build' or 'run'."
fi