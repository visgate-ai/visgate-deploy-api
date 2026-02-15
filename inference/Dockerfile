# Use Runpod official image (newer version as per suggestions)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# s5cmd installation (for ultra-fast S3 downloads)
RUN apt-get update && apt-get install -y wget && \
    wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz && \
    tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz && \
    mv s5cmd /usr/local/bin/ && \
    rm s5cmd_2.2.2_Linux-64bit.tar.gz && \
    rm -rf /var/lib/apt/lists/*

# Runpod lib is already installed in base, but let's update it
# Install inference deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    "diffusers==0.30.3" \
    "transformers==4.44.2" \
    "accelerate==0.33.0" \
    safetensors \
    hf_transfer \
    Pillow

COPY app/ /app/app/
COPY pipelines/ /app/pipelines/

ENV PYTHONPATH=/app
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_MODEL_ID=""
ENV DEVICE=cuda

CMD ["python3", "-m", "app.worker"]
