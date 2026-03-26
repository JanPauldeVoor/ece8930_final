#!/bin/bash

# Assign arguments to variables
RUNTIME=$1
IMAGE_NAME="ece8930-sim"

# Validate inputs
if [ -z "$RUNTIME" ]; then
    echo "Usage: ./build_docker.sh [linux|mac]"
    exit 1
fi


if [ "$RUNTIME" == "linux" ]; then
    echo "Building for WSL (NVIDIA CUDA enabled)..."
    docker build \
        --build-arg BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 \
        --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
        -t ${IMAGE_NAME} .
        
elif [ "$RUNTIME" == "mac" ]; then
    echo "Building for Mac (CPU/Apple Silicon MPS)..."
    docker build \
        --build-arg BASE_IMAGE=python:3.10-slim \
        -t ${IMAGE_NAME} .
else
    echo "Error: Unknown runtime '$RUNTIME'. Use 'linux' or 'mac'."
fi