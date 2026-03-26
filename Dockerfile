# Default to the lightweight Python image if no argument is provided
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (This syntax works for both Debian and Ubuntu base images)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    xvfb \
    libegl-dev \
    && rm -rf /var/lib/apt/lists/*

# alias python3 to python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip

# Accept a build argument for the PyTorch index URL (Used to pull specific CUDA versions)
ARG TORCH_INDEX_URL=""

# Conditionally install PyTorch based on the runtime architecture
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
      pip install --no-cache-dir torch torchvision torchaudio --index-url $TORCH_INDEX_URL; \
    else \
      pip install --no-cache-dir torch torchvision torchaudio; \
    fi

# Copy the rest of the requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]