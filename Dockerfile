FROM nvidia/cuda:13.0.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 + ffmpeg + add CUDA 12 repo for compat libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    gnupg \
    ffmpeg \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && rm -f /etc/apt/sources.list.d/cuda*.list \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends libcublas-12-6 \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/*

# CUDA 12 libs needed by faster-whisper/ctranslate2
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

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
RUN mkdir -p /opt/models /opt/models/pyannote

# Environment variables
ENV TORCH_HOME=/root/.cache/torch
ENV DEVICE=cuda
ENV PYTHONUNBUFFERED=1

# or more precisely, if huggingface_hub expects a "hub" subfolder:
ENV PYANNOTE_CACHE=/opt/models/pyannote 
ENV HF_HUB_CACHE=/opt/models/pyannote    
ENV HUGGINGFACE_HUB_CACHE=/opt/models/pyannote
ENV HF_HUB_OFFLINE=1

# Model paths (can be HuggingFace repo IDs or local paths)
# When mounting cached models, override these with local paths
ENV WHISPER_MODEL=/opt/models/ivrit-ai--whisper-large-v3-turbo-ct2
ENV DIARIZATION_MODEL=pyannote/speaker-diarization-community-1
ENV EMBEDDING_MODEL_PATH=pyannote/wespeaker-voxceleb-resnet34-LM
# Default command (override as needed)
CMD ["uv", "run", "python", "diarize.py"]
