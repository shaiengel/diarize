FROM nvidia/cuda:12.4.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Add NVIDIA CUDA repo for cuDNN 9
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    gnupg \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update

# Install Python 3.12 + CUDA libs + ffmpeg
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    libcudnn9-cuda-12 \
    libcublas-12-4 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy Linux-specific pyproject as pyproject.toml
COPY pyproject.linux.toml ./pyproject.toml
COPY README.md ./
COPY main.py diarize.py ./

# Install dependencies only, don't build the project (we just have scripts, not a package)
RUN uv sync --no-dev --no-install-project

# Create directories for models
RUN mkdir -p /opt/models /root/.cache/torch/pyannote

# Environment variables
ENV TORCH_HOME=/root/.cache/torch
ENV DEVICE=cuda
ENV PYTHONUNBUFFERED=1

# Default command (override as needed)
CMD ["uv", "run", "python", "diarize.py"]
