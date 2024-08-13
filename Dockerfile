# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10-tk \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/hacksider/Deep-Live-Cam.git .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install ONNX Runtime GPU
RUN pip3 uninstall -y onnxruntime onnxruntime-gpu && \
    pip3 install --no-cache-dir onnxruntime-gpu==1.16.3

# Download required models
RUN mkdir -p models && \
    wget -O models/GFPGANv1.4.pth https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth && \
    wget -O models/inswapper_128_fp16.onnx https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx

# Set the entrypoint
ENTRYPOINT ["python3", "run.py", "--execution-provider", "cuda"]

# Default command (can be overridden)
CMD ["--help"]