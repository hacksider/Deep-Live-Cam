# Headless Deep-Live-Cam for any NVIDIA cloud GPU host (RunPod, Vast.ai,
# Lambda, Paperspace, EC2/GCP/Azure GPU VMs, Kubernetes ...).
#
#   docker build -t deep-live-cam .
#   docker run --gpus all -p 8000:8000 -v dlc-models:/models deep-live-cam
#   curl -F source=@face.jpg -F target=@clip.mp4 http://localhost:8000/swap -o out.mp4
#
# One-off CLI job instead of the HTTP API:
#   docker run --gpus all -v dlc-models:/models -v "$PWD:/data" deep-live-cam \
#       python -m modules.headless -s /data/face.jpg -t /data/clip.mp4 -o /data/out.mp4
#
# Build with --build-arg PREFETCH_MODELS=1 to bake the ~2 GB of models into the
# image (no volume needed, slower push/pull).

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DLC_MODELS_DIR=/models/dlc \
    DLC_INSIGHTFACE_ROOT=/models/insightface \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv build-essential \
        ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-cloud.txt .
# Ubuntu 24.04 ships Python 3.12 (onnxruntime-gpu has no 3.10 wheels, so 22.04 won't do).
RUN python3 -m venv /opt/venv && pip install --upgrade pip && pip install -r requirements-cloud.txt

COPY . .

ARG PREFETCH_MODELS=0
RUN if [ "$PREFETCH_MODELS" = "1" ]; then python -m modules.headless --download-models; fi

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "cloud.server:app", "--host", "0.0.0.0", "--port", "8000"]
