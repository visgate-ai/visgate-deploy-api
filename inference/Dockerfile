# Use Runpod official image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies for audio (ffmpeg, libsndfile) and s5cmd for storage
RUN apt-get update && apt-get install -y wget ffmpeg libsndfile1 && \
    wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz && \
    tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz && \
    mv s5cmd /usr/local/bin/ && \
    rm s5cmd_2.2.2_Linux-64bit.tar.gz && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install all universal dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
COPY pipelines/ /app/pipelines/

ENV PYTHONPATH=/app
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_MODEL_ID=""
ENV DEVICE=cuda
ENV VISGATE_WORKER_PROFILE=universal

EXPOSE 8000

CMD ["python3", "-m", "app.worker"]
