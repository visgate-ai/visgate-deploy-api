# HF inference worker for Runpod: Flux, SDXL, diffusers text2image.
# Base has PyTorch+CUDA preinstalled to avoid large downloads during build.

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only inference deps (torch already in base image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    "diffusers>=0.30.0" \
    transformers \
    accelerate \
    safetensors \
    Pillow

COPY app/ /app/app/
COPY pipelines/ /app/pipelines/

ENV PYTHONPATH=/app
ENV HF_MODEL_ID=""
ENV DEVICE=cuda

CMD ["python3", "-m", "app.worker"]
